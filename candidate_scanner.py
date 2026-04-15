#!/usr/bin/env python3
"""
candidate_scanner.py

Scans the S&P 100 for mean-reversion candidates.

A stock becomes a candidate if EITHER:
1. It has drawn down ≥ 3% from its 20-day high (catches multi-day grinds)
2. It dropped ≥ 3% in a single day (catches sharp shocks)

For each candidate, computes basic price features needed by downstream modules.

This replaces the old ticker_selection.py which only looked at 5-day Typical Price.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


# S&P 100 universe
TICKERS: List[str] = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","C",
    "CAT","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS","CVX",
    "DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD","GM",
    "GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","ISRG","JNJ",
    "JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ","MDT",
    "MET","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE",
    "NOW","NVDA","ORCL","PEP","PFE","PG","PLTR","PM","PYPL","QCOM",
    "RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UBER","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM"
]

# Sector ETF mapping
TICKER_TO_SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "AMAT": "SMH", "LRCX": "SMH", "QCOM": "SMH", "TXN": "SMH", "MU": "SMH",
    "INTC": "SMH", "NOW": "XLK", "INTU": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "IBM": "XLK", "ACN": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "AMZN": "XLY", "META": "XLC", "NFLX": "XLC", "CMCSA": "XLC", "TMUS": "XLC",
    "T": "XLC", "VZ": "XLC", "DIS": "XLC", "UBER": "XLY", "TSLA": "XLY",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF",
    "MS": "XLF", "AXP": "XLF", "BLK": "XLF", "USB": "XLF", "SCHW": "XLF",
    "LLY": "XLV", "PFE": "XLV", "BMY": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "TMO": "XLV", "DHR": "XLV", "ABT": "XLV", "MDT": "XLV", "GILD": "XLV",
    "ISRG": "XLV", "UNH": "XLV", "CVS": "XLV", "AMGN": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    "CAT": "XLI", "DE": "XLI", "HON": "XLI", "GE": "XLI", "GD": "XLI",
    "LMT": "XLI", "UNP": "XLI", "UPS": "XLI", "RTX": "XLI", "BA": "XLI",
    "HD": "XLY", "LOW": "XLY", "MCD": "XLY", "NKE": "XLY", "SBUX": "XLY",
    "BKNG": "XLY", "COST": "XLP", "WMT": "XLP", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "PM": "XLP", "MDLZ": "XLP", "CL": "XLP",
    "DUK": "XLU", "NEE": "XLU", "SO": "XLU", "AMT": "XLRE",
    "SPG": "XLRE", "LIN": "XLB", "PLTR": "XLK", "GEV": "XLI",
    "MO": "XLP", "MMM": "XLI", "EMR": "XLI", "FDX": "XLI",
    "COF": "XLF", "BK": "XLF", "MET": "XLF", "AIG": "XLF",
    "JNJ": "XLV", "PYPL": "XLK", "MA": "XLK", "V": "XLK",
    "TGT": "XLY",
}

DEFAULT_SECTOR_ETF = "SPY"

# Thresholds
DRAWDOWN_THRESHOLD = -0.03       # 3% drawdown from 20-day high
SINGLE_DAY_DROP_THRESHOLD = -0.03  # 3% single-day drop
LOOKBACK_DAYS = 20                # for computing recent high
DOWNLOAD_PERIOD = "60d"           # enough for 20-day high + vol calculations
MAX_CANDIDATES = 20               # cap to keep Claude costs manageable


@dataclass
class Candidate:
    """A stock that meets the screening criteria."""
    ticker: str
    sector_etf: str
    close_price: float
    last_date: str

    # How the candidate was identified
    drawdown_from_20d_high: float   # negative number, e.g., -0.05 = 5% below high
    single_day_return: float        # today's return
    trigger_type: str               # "drawdown", "single_day_drop", or "both"

    # Drop bucket for lookup table compatibility
    drop_magnitude: float           # positive number, e.g., 0.05 = 5% drop
    drop_bucket: str                # "3-5%", "5-7%", "7-10%", "10%+"

    # Basic features computed during scanning
    high_20d: float
    vol_20d: float                  # 20-day rolling std of daily returns
    drop_vs_vol20: float            # drop_magnitude / vol_20d
    stock_return_5d: float
    stock_return_20d: float
    close_vs_20dma: float           # (close / 20dma) - 1
    close_in_range: float           # (close - low) / (high - low) for today


def _normalize_yf_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Handle yfinance MultiIndex column weirdness."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            df = df[ticker].copy()
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _get_drop_bucket(magnitude: float) -> str:
    """Map a positive drop magnitude to a bucket label."""
    if magnitude >= 0.10:
        return "10%+"
    elif magnitude >= 0.07:
        return "7-10%"
    elif magnitude >= 0.05:
        return "5-7%"
    elif magnitude >= 0.03:
        return "3-5%"
    return "< 3%"


def scan_candidates(
    tickers: Optional[List[str]] = None,
    drawdown_threshold: float = DRAWDOWN_THRESHOLD,
    single_day_threshold: float = SINGLE_DAY_DROP_THRESHOLD,
    max_candidates: Optional[int] = MAX_CANDIDATES,
) -> Dict[str, Any]:
    """
    Scan the universe and return candidates meeting the criteria.
    """
    universe = tickers if tickers is not None else TICKERS

    print(f"Scanning {len(universe)} tickers...")

    # Download all at once for speed
    raw = yf.download(
        tickers=universe,
        period=DOWNLOAD_PERIOD,
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        return {"candidates": [], "errors": [], "scan_timestamp": datetime.now(timezone.utc).isoformat()}

    candidates: List[Candidate] = []
    errors: List[Dict] = []

    for ticker in universe:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    errors.append({"ticker": ticker, "reason": "not_in_download"})
                    continue
                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = _normalize_yf_df(df, ticker)

            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(set(df.columns)):
                errors.append({"ticker": ticker, "reason": f"missing_columns: {required - set(df.columns)}"})
                continue

            df = df.dropna(subset=["Close"])
            if len(df) < LOOKBACK_DAYS:
                errors.append({"ticker": ticker, "reason": f"not_enough_data: {len(df)} days"})
                continue

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            # Current values
            current_close = float(close.iloc[-1])
            current_high = float(high.iloc[-1])
            current_low = float(low.iloc[-1])
            last_date = str(df.index[-1].date())

            # 20-day high
            high_20d = float(close.tail(LOOKBACK_DAYS).max())

            # Drawdown from 20-day high
            drawdown = (current_close / high_20d) - 1.0

            # Single-day return
            if len(close) >= 2:
                single_day_ret = (current_close / float(close.iloc[-2])) - 1.0
            else:
                single_day_ret = 0.0

            # Check if candidate
            is_drawdown = drawdown <= drawdown_threshold
            is_single_day = single_day_ret <= single_day_threshold

            if not is_drawdown and not is_single_day:
                continue

            # Determine trigger type
            if is_drawdown and is_single_day:
                trigger = "both"
            elif is_drawdown:
                trigger = "drawdown"
            else:
                trigger = "single_day_drop"

            # Drop magnitude (positive number) — use the larger of drawdown or single-day
            drop_mag = max(abs(drawdown), abs(single_day_ret))

            # Compute features
            daily_returns = close.pct_change().dropna()
            vol_20d = float(daily_returns.tail(20).std()) if len(daily_returns) >= 20 else 0.02

            drop_vs_vol = drop_mag / vol_20d if vol_20d > 0 else 0.0

            ret_5d = float((current_close / float(close.iloc[-6])) - 1.0) if len(close) >= 6 else 0.0
            ret_20d = float((current_close / float(close.iloc[-21])) - 1.0) if len(close) >= 21 else 0.0

            ma_20 = float(close.tail(20).mean())
            close_vs_20dma = (current_close / ma_20) - 1.0

            # Close in range: (close - low) / (high - low) for today
            day_range = current_high - current_low
            close_in_range = (current_close - current_low) / day_range if day_range > 0 else 0.5

            candidates.append(Candidate(
                ticker=ticker,
                sector_etf=TICKER_TO_SECTOR_ETF.get(ticker, DEFAULT_SECTOR_ETF),
                close_price=round(current_close, 4),
                last_date=last_date,
                drawdown_from_20d_high=round(drawdown, 6),
                single_day_return=round(single_day_ret, 6),
                trigger_type=trigger,
                drop_magnitude=round(drop_mag, 6),
                drop_bucket=_get_drop_bucket(drop_mag),
                high_20d=round(high_20d, 4),
                vol_20d=round(vol_20d, 6),
                drop_vs_vol20=round(drop_vs_vol, 4),
                stock_return_5d=round(ret_5d, 6),
                stock_return_20d=round(ret_20d, 6),
                close_vs_20dma=round(close_vs_20dma, 6),
                close_in_range=round(close_in_range, 4),
            ))

        except Exception as e:
            errors.append({"ticker": ticker, "reason": str(e)})

    # Sort by drop magnitude (worst first)
    candidates.sort(key=lambda c: -c.drop_magnitude)

    # Cap
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "candidates_found": len(candidates),
        "errors": len(errors),
        "thresholds": {
            "drawdown_threshold": drawdown_threshold,
            "single_day_threshold": single_day_threshold,
            "lookback_days": LOOKBACK_DAYS,
        },
        "candidates": [asdict(c) for c in candidates],
        "error_details": errors,
    }


if __name__ == "__main__":
    result = scan_candidates()

    print(f"\nScan Results")
    print(f"{'=' * 70}")
    print(f"Universe: {result['universe_size']}")
    print(f"Candidates: {result['candidates_found']}")
    print(f"Errors: {result['errors']}")
    print()

    if result["candidates"]:
        print(f"{'Rank':>4} {'Ticker':>6} {'Drop':>7} {'Bucket':>6} {'Trigger':>12} "
              f"{'DropVsVol':>9} {'Sector':>5} {'ClsInRng':>8}")
        print("-" * 70)
        for i, c in enumerate(result["candidates"], 1):
            print(f"{i:>4} {c['ticker']:>6} {c['drop_magnitude']*100:>6.2f}% "
                  f"{c['drop_bucket']:>6} {c['trigger_type']:>12} "
                  f"{c['drop_vs_vol20']:>9.2f} {c['sector_etf']:>5} "
                  f"{c['close_in_range']:>8.2f}")
