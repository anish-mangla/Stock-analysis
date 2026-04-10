"""
ticker_selection.py

Selects candidate tickers based on recent weakness in Typical Price (TP),
where:

    TP = (High + Low + Close) / 3

The module downloads recent daily bars using yfinance, computes the 5-trading-day
TP change for each ticker, filters names below a configured threshold, and returns
a structured payload for downstream prompt composition and model execution.

This file is intentionally self-contained so it can be imported directly by main.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# Your stock universe
TICKERS: List[str] = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK.B","C",
    "CAT","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD","GM",
    "GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG","JNJ",
    "JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT",
    "MET","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE",
    "NOW","NVDA","ORCL","PEP","PFE","PG","PLTR","PM","PYPL","QCOM",
    "RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UBER","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM"
]

# Download enough recent bars to safely cover 5 trading days.
DEFAULT_PERIOD = "10d"
DEFAULT_INTERVAL = "1d"

# Filter rule:
# Select stocks whose 5-day TP % change is <= this threshold.
DEFAULT_DROP_THRESHOLD_PCT = -0.025  # -2.5%

# Optional cap to keep downstream model prompts manageable.
DEFAULT_MAX_SELECTED = 15


@dataclass
class TickerResult:
    """Structured result for a selected ticker."""
    ticker: str
    tp_change_abs: float
    tp_change_pct: float
    tp_old: float
    tp_new: float
    last_date: str
    trading_days_used: int
    rank: int


@dataclass
class TickerError:
    """Structured error for a ticker that could not be processed."""
    ticker: str
    reason: str


def _now_utc_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_download_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output so the rest of the code can consistently access:
    High, Low, Close

    yfinance sometimes returns:
    - a normal single-level column index, or
    - a MultiIndex column structure

    This function flattens the frame where needed.
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # Try to collapse to the field names like High / Low / Close
        flattened_cols: List[str] = []
        for col in df.columns:
            # Prefer the first non-empty string-like level
            parts = [str(part) for part in col if part not in ("", None)]
            flattened_cols.append(parts[0] if parts else str(col))
        df = df.copy()
        df.columns = flattened_cols

    return df


def get_5day_tp_change(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[pd.Timestamp], Optional[str]]:
    """
    For a given ticker, download recent data and compute Typical Price (TP):

        TP = (High + Low + Close) / 3

    Over the last 5 trading days:
        tp_old = TP 5 trading days ago
        tp_new = TP most recent day

    Returns:
        (
            abs_change,
            pct_change,
            tp_old,
            tp_new,
            last_date,
            error_reason
        )

    If computation fails or there is not enough valid data, fields are None
    and error_reason explains why.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:
        return None, None, None, None, None, f"download_error: {exc}"

    if df.empty:
        return None, None, None, None, None, "empty_data"

    df = _normalize_download_frame(df)

    required_cols = {"High", "Low", "Close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return None, None, None, None, None, f"missing_columns: {sorted(missing_cols)}"

    df = df.dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return None, None, None, None, None, "no_valid_rows_after_dropna"

    df_last5 = df.tail(5)
    if len(df_last5) < 5:
        return None, None, None, None, None, f"not_enough_data: only_{len(df_last5)}_days"

    tp = (df_last5["High"] + df_last5["Low"] + df_last5["Close"]) / 3.0

    try:
        tp_old = float(tp.iloc[0])
        tp_new = float(tp.iloc[-1])
    except Exception as exc:
        return None, None, None, None, None, f"tp_scalar_conversion_error: {exc}"

    if tp_old == 0:
        return None, None, None, None, None, "tp_old_zero_division_guard"

    abs_change = tp_new - tp_old
    pct_change = (tp_new / tp_old) - 1.0
    last_date = df_last5.index[-1]

    return float(abs_change), float(pct_change), tp_old, tp_new, last_date, None


def get_selected_tickers(
    tickers: Optional[List[str]] = None,
    drop_threshold_pct: float = DEFAULT_DROP_THRESHOLD_PCT,
    max_selected: Optional[int] = DEFAULT_MAX_SELECTED,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> Dict[str, Any]:
    """
    Run the full ticker selection workflow.

    Args:
        tickers:
            Universe of tickers to scan. If None, uses TICKERS.
        drop_threshold_pct:
            Select tickers where 5-day TP % change <= this threshold.
            Example: -0.025 means -2.5% or worse.
        max_selected:
            Maximum number of selected tickers to keep after sorting by worst
            pct change first. If None, keep all selected.
        period:
            yfinance period passed to download().
        interval:
            yfinance interval passed to download().

    Returns:
        Structured payload containing:
        - run metadata
        - selection rule
        - selected tickers
        - processing errors
        - summary stats
    """
    universe = tickers if tickers is not None else TICKERS

    selected_results: List[TickerResult] = []
    errors: List[TickerError] = []
    processed_count = 0

    for ticker in universe:
        processed_count += 1

        abs_chg, pct_chg, tp_old, tp_new, last_date, error_reason = get_5day_tp_change(
            ticker=ticker,
            period=period,
            interval=interval,
        )

        if error_reason is not None:
            errors.append(TickerError(ticker=ticker, reason=error_reason))
            continue

        assert abs_chg is not None
        assert pct_chg is not None
        assert tp_old is not None
        assert tp_new is not None
        assert last_date is not None

        # Keep only names that meet the weakness threshold.
        if pct_chg <= drop_threshold_pct:
            selected_results.append(
                TickerResult(
                    ticker=ticker,
                    tp_change_abs=round(abs_chg, 6),
                    tp_change_pct=round(pct_chg, 6),
                    tp_old=round(tp_old, 6),
                    tp_new=round(tp_new, 6),
                    last_date=str(pd.Timestamp(last_date).date()),
                    trading_days_used=5,
                    rank=0,  # assigned after sorting
                )
            )

    # Worst performers first
    selected_results.sort(key=lambda item: item.tp_change_pct)

    if max_selected is not None:
        selected_results = selected_results[:max_selected]

    # Assign final ranks after sorting/truncation
    for idx, item in enumerate(selected_results, start=1):
        item.rank = idx

    payload: Dict[str, Any] = {
        "run_timestamp_utc": _now_utc_iso(),
        "component": "ticker_selection",
        "selection_rule": {
            "metric": "5-day typical price percent change",
            "formula": "(High + Low + Close) / 3",
            "comparison": "<=",
            "threshold": drop_threshold_pct,
            "threshold_pct_display": f"{drop_threshold_pct:.2%}",
            "period": period,
            "interval": interval,
            "max_selected": max_selected,
        },
        "summary": {
            "universe_size": len(universe),
            "processed_count": processed_count,
            "selected_count": len(selected_results),
            "error_count": len(errors),
        },
        "tickers": [asdict(item) for item in selected_results],
        "errors": [asdict(err) for err in errors],
    }

    return payload


if __name__ == "__main__":
    result = get_selected_tickers()

    print("Ticker Selection Result")
    print("-" * 80)
    print(f"Run timestamp (UTC): {result['run_timestamp_utc']}")
    print(f"Universe size:       {result['summary']['universe_size']}")
    print(f"Processed count:     {result['summary']['processed_count']}")
    print(f"Selected count:      {result['summary']['selected_count']}")
    print(f"Error count:         {result['summary']['error_count']}")
    print(f"Threshold:           {result['selection_rule']['threshold_pct_display']}")
    print()

    if result["tickers"]:
        print("Selected tickers:")
        for item in result["tickers"]:
            print(
                f"{item['rank']:>2}. {item['ticker']}: "
                f"tp_change_abs={item['tp_change_abs']:.2f}, "
                f"tp_change_pct={item['tp_change_pct']:.2%}, "
                f"tp_old={item['tp_old']:.2f}, "
                f"tp_new={item['tp_new']:.2f}, "
                f"last_date={item['last_date']}"
            )
    else:
        print("No tickers met the selection rule.")

    if result["errors"]:
        print()
        print("Errors:")
        for err in result["errors"]:
            print(f"- {err['ticker']}: {err['reason']}")
