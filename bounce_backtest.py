#!/usr/bin/env python3
"""
bounce_backtest.py

Goal:
Given historical daily stock price data, identify large one-day drop events,
simulate forward price paths for up to 14 trading days, and compute per-event
metrics to estimate the likelihood of achieving a +1% return.

Today's locked assumptions:
- Universe: S&P 100
- Time period: last 3 years (configured below with explicit dates)
- Entry price: event-day close
- Event definition: one-day drop buckets
- Profit target: +1%
- Max holding period: 14 trading days
- Minimum history before an event: 20 trading days
- No stop loss for now
- Grouped analysis by drop-size bucket and by ticker

Outputs:
- outputs/events.csv
- outputs/summary_by_bucket.csv
- outputs/summary_by_ticker.csv
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ============================================================
# 1) CONFIG
# ============================================================

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

UNIVERSE = "sp100"

TARGET_RETURN = 0.01
MAX_HOLD_DAYS = 14
MIN_HISTORY_DAYS = 20

DROP_BUCKETS: List[Tuple[float, Optional[float], str]] = [
    (0.03, 0.05, "3-5%"),
    (0.05, 0.07, "5-7%"),
    (0.07, 0.10, "7-10%"),
    (0.10, None, "10%+"),
]

OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_by_bucket.csv")
TICKER_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_by_ticker.csv")

DOWNLOAD_AUTO_ADJUST = False
YF_GROUP_BY = "ticker"

MIN_TICKER_EVENTS_FOR_RANKING = 15
TOP_N_TICKERS_TO_PRINT = 10

TICKER_BUCKET_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_by_ticker_and_bucket.csv")
MIN_TICKER_BUCKET_EVENTS_FOR_RANKING = 8
TOP_N_TICKER_BUCKETS_TO_PRINT = 15


# ============================================================
# 2) UNIVERSE / DATA LOADING
# ============================================================

def get_universe_tickers(universe: str) -> List[str]:
    """
    Return tickers for the requested universe.
    Currently supports only S&P 100 via Wikipedia scrape.
    """
    normalized = universe.strip().lower()

    if normalized != "sp100":
        raise ValueError(f"Unsupported universe: {universe}")

    url = "https://en.wikipedia.org/wiki/S%26P_100"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    tables = pd.read_html(url, storage_options=headers)

    for table in tables:
        if "Symbol" in table.columns:
            tickers = (
                table["Symbol"]
                .astype(str)
                .str.replace(".", "-", regex=False)
                .dropna()
                .unique()
                .tolist()
            )
            if tickers:
                return sorted(tickers)

    raise RuntimeError("Could not find S&P 100 table with 'Symbol' column on Wikipedia.")


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data for all tickers.
    Returns a dict:
        ticker -> DataFrame(index=Date, columns=[Open, High, Low, Close, Volume, ...])
    """
    if not tickers:
        raise ValueError("No tickers provided.")

    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=DOWNLOAD_AUTO_ADJUST,
        progress=True,
        group_by=YF_GROUP_BY,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No data returned from yfinance.")

    out: Dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue

            df = raw[ticker].copy()
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            out[ticker] = df
    else:
        if len(tickers) != 1:
            raise RuntimeError("Unexpected flat-column format from yfinance for multiple tickers.")

        ticker = tickers[0]
        df = raw.copy().dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        out[ticker] = df

    if not out:
        raise RuntimeError("No usable price data after download/cleaning.")

    return out


# ============================================================
# 3) EVENT GENERATION
# ============================================================

def get_drop_bucket(drop_pct: float) -> Optional[str]:
    """
    Map a positive drop magnitude (e.g. 0.061 for a 6.1% drop)
    to one of the configured bucket labels.
    """
    for lower, upper, label in DROP_BUCKETS:
        if upper is None:
            if drop_pct >= lower:
                return label
        else:
            if lower <= drop_pct < upper:
                return label
    return None


def generate_events(
    price_data: Dict[str, pd.DataFrame],
    min_history_days: int,
) -> pd.DataFrame:
    """
    Generate event rows for days where the one-day drop falls into one of the buckets.

    Event-day return is:
        close_t / close_{t-1} - 1
    """
    rows = []

    for ticker, df in price_data.items():
        if "Close" not in df.columns:
            continue

        closes = df["Close"].copy()
        if closes.isna().all():
            continue

        daily_returns = closes.pct_change()

        for i in range(1, len(df)):
            if i < min_history_days:
                continue

            event_date = df.index[i]
            prev_close = float(closes.iloc[i - 1])
            entry_price = float(closes.iloc[i])
            daily_return = float(daily_returns.iloc[i])

            if pd.isna(daily_return):
                continue

            if daily_return >= 0:
                continue

            drop_pct = abs(daily_return)
            bucket = get_drop_bucket(drop_pct)

            if bucket is None:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "event_date": event_date,
                    "prev_close": prev_close,
                    "entry_price": entry_price,
                    "daily_return": daily_return,
                    "drop_pct": drop_pct,
                    "drop_bucket": bucket,
                }
            )

    events = pd.DataFrame(rows)

    if events.empty:
        return events

    events = events.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return events


# ============================================================
# 4) FORWARD SIMULATION / LABELING
# ============================================================

def label_event_outcome(
    df: pd.DataFrame,
    event_idx: int,
    entry_price: float,
    target_return: float,
    max_hold_days: int,
) -> Dict[str, object]:
    """
    For a single event, simulate forward up to max_hold_days trading days.

    Rules:
    - Entry at event-day close
    - Success if any future close >= entry_price * (1 + target_return)
    - If no success by day max_hold_days, exit at final observed close in the window
    - Track max drawdown over the holding window relative to entry
    """
    target_price = entry_price * (1.0 + target_return)

    forward = df.iloc[event_idx + 1 : event_idx + 1 + max_hold_days].copy()

    if forward.empty:
        return {
            "success": False,
            "days_to_hit": None,
            "exit_date": pd.NaT,
            "exit_price": None,
            "final_return": None,
            "max_drawdown": None,
            "min_close_during_hold": None,
            "observed_holding_days": 0,
        }

    closes = forward["Close"].astype(float)

    success = False
    days_to_hit: Optional[int] = None
    exit_date = forward.index[-1]
    exit_price = float(closes.iloc[-1])

    for offset, (dt, close_val) in enumerate(closes.items(), start=1):
        if close_val >= target_price:
            success = True
            days_to_hit = offset
            exit_date = dt
            exit_price = float(close_val)
            break

    min_close = float(closes.min())
    max_drawdown = (min_close / entry_price) - 1.0
    final_return = (exit_price / entry_price) - 1.0

    return {
        "success": success,
        "days_to_hit": days_to_hit,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "final_return": final_return,
        "max_drawdown": max_drawdown,
        "min_close_during_hold": min_close,
        "observed_holding_days": len(forward),
    }


def run_event_study(
    price_data: Dict[str, pd.DataFrame],
    events: pd.DataFrame,
    target_return: float,
    max_hold_days: int,
) -> pd.DataFrame:
    """
    Label each event with forward outcome metrics.
    """
    if events.empty:
        return events.copy()

    rows = []

    for _, event in events.iterrows():
        ticker = event["ticker"]
        event_date = pd.Timestamp(event["event_date"])
        entry_price = float(event["entry_price"])

        df = price_data[ticker]
        if event_date not in df.index:
            continue

        event_idx = df.index.get_loc(event_date)
        if isinstance(event_idx, slice):
            continue
        if not isinstance(event_idx, int):
            continue

        outcome = label_event_outcome(
            df=df,
            event_idx=event_idx,
            entry_price=entry_price,
            target_return=target_return,
            max_hold_days=max_hold_days,
        )

        row = event.to_dict()
        row.update(outcome)
        rows.append(row)

    labeled = pd.DataFrame(rows)

    if labeled.empty:
        return labeled

    labeled = labeled.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return labeled


# ============================================================
# 5) AGGREGATION / REPORTING
# ============================================================

def summarize_by_bucket(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by drop bucket.
    """
    if events_df.empty:
        return pd.DataFrame()

    def _avg_days_success_only(series: pd.Series) -> float:
        valid = series.dropna()
        return float(valid.mean()) if not valid.empty else float("nan")

    grouped = (
        events_df.groupby("drop_bucket", dropna=False)
        .agg(
            total_events=("ticker", "count"),
            success_rate=("success", "mean"),
            avg_days_to_hit=("days_to_hit", _avg_days_success_only),
            median_days_to_hit=("days_to_hit", "median"),
            avg_final_return=("final_return", "mean"),
            median_final_return=("final_return", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            median_max_drawdown=("max_drawdown", "median"),
            avg_drop_pct=("drop_pct", "mean"),
        )
        .reset_index()
    )

    grouped["failure_rate"] = 1.0 - grouped["success_rate"]

    ordered_labels = [label for _, _, label in DROP_BUCKETS]
    grouped["drop_bucket"] = pd.Categorical(
        grouped["drop_bucket"],
        categories=ordered_labels,
        ordered=True,
    )
    grouped = grouped.sort_values("drop_bucket").reset_index(drop=True)

    return grouped


def summarize_by_ticker(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by ticker.
    """
    if events_df.empty:
        return pd.DataFrame()

    def _avg_days_success_only(series: pd.Series) -> float:
        valid = series.dropna()
        return float(valid.mean()) if not valid.empty else float("nan")

    grouped = (
        events_df.groupby("ticker", dropna=False)
        .agg(
            total_events=("ticker", "count"),
            success_rate=("success", "mean"),
            avg_days_to_hit=("days_to_hit", _avg_days_success_only),
            median_days_to_hit=("days_to_hit", "median"),
            avg_final_return=("final_return", "mean"),
            median_final_return=("final_return", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            median_max_drawdown=("max_drawdown", "median"),
            avg_drop_pct=("drop_pct", "mean"),
        )
        .reset_index()
    )

    grouped["failure_rate"] = 1.0 - grouped["success_rate"]

    # Simple ranking score:
    # Higher success and better returns are good.
    # More negative drawdown is bad.
    grouped["ticker_score"] = (
        grouped["success_rate"] * 100.0
        + grouped["avg_final_return"] * 100.0
        + grouped["avg_max_drawdown"] * 10.0
    )

    grouped = grouped.sort_values(
        by=["ticker_score", "success_rate", "avg_final_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grouped

def summarize_by_ticker_and_bucket(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by (ticker, drop_bucket).
    """
    if events_df.empty:
        return pd.DataFrame()

    def _avg_days_success_only(series: pd.Series) -> float:
        valid = series.dropna()
        return float(valid.mean()) if not valid.empty else float("nan")

    grouped = (
        events_df.groupby(["ticker", "drop_bucket"], dropna=False)
        .agg(
            total_events=("ticker", "count"),
            success_rate=("success", "mean"),
            avg_days_to_hit=("days_to_hit", _avg_days_success_only),
            median_days_to_hit=("days_to_hit", "median"),
            avg_final_return=("final_return", "mean"),
            median_final_return=("final_return", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            median_max_drawdown=("max_drawdown", "median"),
            avg_drop_pct=("drop_pct", "mean"),
        )
        .reset_index()
    )

    grouped["failure_rate"] = 1.0 - grouped["success_rate"]

    grouped["combo_score"] = (
        grouped["success_rate"] * 100.0
        + grouped["avg_final_return"] * 100.0
        + grouped["avg_max_drawdown"] * 10.0
    )

    ordered_labels = [label for _, _, label in DROP_BUCKETS]
    grouped["drop_bucket"] = pd.Categorical(
        grouped["drop_bucket"],
        categories=ordered_labels,
        ordered=True,
    )

    grouped = grouped.sort_values(
        by=["combo_score", "success_rate", "avg_final_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grouped


def print_ticker_bucket_rankings(ticker_bucket_df: pd.DataFrame) -> None:
    """
    Print best and worst ticker+bucket combinations using a minimum event count filter.
    """
    if ticker_bucket_df.empty:
        print("\nNo ticker+bucket summary to print.")
        return

    eligible = ticker_bucket_df[
        ticker_bucket_df["total_events"] >= MIN_TICKER_BUCKET_EVENTS_FOR_RANKING
    ].copy()

    if eligible.empty:
        print(
            f"\nNo ticker+bucket combinations have at least "
            f"{MIN_TICKER_BUCKET_EVENTS_FOR_RANKING} events, so rankings are skipped."
        )
        return

    pct_cols = [
        "success_rate",
        "failure_rate",
        "avg_final_return",
        "median_final_return",
        "avg_max_drawdown",
        "median_max_drawdown",
        "avg_drop_pct",
    ]

    display = eligible.copy()
    for col in pct_cols:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2)

    for col in ["avg_days_to_hit", "median_days_to_hit", "combo_score"]:
        if col in display.columns:
            display[col] = display[col].round(2)

    best = display.sort_values(
        by=["combo_score", "success_rate", "avg_final_return"],
        ascending=[False, False, False],
    ).head(TOP_N_TICKER_BUCKETS_TO_PRINT)

    worst = display.sort_values(
        by=["combo_score", "success_rate", "avg_final_return"],
        ascending=[True, True, True],
    ).head(TOP_N_TICKER_BUCKETS_TO_PRINT)

    cols_to_show = [
        "ticker",
        "drop_bucket",
        "total_events",
        "success_rate",
        "failure_rate",
        "avg_days_to_hit",
        "avg_final_return",
        "avg_max_drawdown",
        "avg_drop_pct",
        "combo_score",
    ]

    print(
        f"\n=== BEST {TOP_N_TICKER_BUCKETS_TO_PRINT} TICKER + BUCKET COMBOS "
        f"(min {MIN_TICKER_BUCKET_EVENTS_FOR_RANKING} events) ==="
    )
    print(best[cols_to_show].to_string(index=False))

    print(
        f"\n=== WORST {TOP_N_TICKER_BUCKETS_TO_PRINT} TICKER + BUCKET COMBOS "
        f"(min {MIN_TICKER_BUCKET_EVENTS_FOR_RANKING} events) ==="
    )
    print(worst[cols_to_show].to_string(index=False))

def print_summary(summary_df: pd.DataFrame) -> None:
    """
    Pretty-print bucket summary.
    """
    if summary_df.empty:
        print("\nNo summary to print. No events found.")
        return

    display_df = summary_df.copy()

    pct_cols = [
        "success_rate",
        "failure_rate",
        "avg_final_return",
        "median_final_return",
        "avg_max_drawdown",
        "median_max_drawdown",
        "avg_drop_pct",
    ]

    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(2)

    numeric_cols = ["avg_days_to_hit", "median_days_to_hit"]
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    print("\n=== SUMMARY BY DROP BUCKET ===")
    print(display_df.to_string(index=False))


def print_ticker_rankings(ticker_summary_df: pd.DataFrame) -> None:
    """
    Print best and worst tickers using a minimum event count filter.
    """
    if ticker_summary_df.empty:
        print("\nNo ticker summary to print.")
        return

    eligible = ticker_summary_df[
        ticker_summary_df["total_events"] >= MIN_TICKER_EVENTS_FOR_RANKING
    ].copy()

    if eligible.empty:
        print(
            f"\nNo tickers have at least {MIN_TICKER_EVENTS_FOR_RANKING} events, "
            "so ticker rankings are skipped."
        )
        return

    pct_cols = [
        "success_rate",
        "failure_rate",
        "avg_final_return",
        "median_final_return",
        "avg_max_drawdown",
        "median_max_drawdown",
        "avg_drop_pct",
    ]

    display = eligible.copy()
    for col in pct_cols:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2)

    for col in ["avg_days_to_hit", "median_days_to_hit", "ticker_score"]:
        if col in display.columns:
            display[col] = display[col].round(2)

    best = display.sort_values(
        by=["ticker_score", "success_rate", "avg_final_return"],
        ascending=[False, False, False],
    ).head(TOP_N_TICKERS_TO_PRINT)

    worst = display.sort_values(
        by=["ticker_score", "success_rate", "avg_final_return"],
        ascending=[True, True, True],
    ).head(TOP_N_TICKERS_TO_PRINT)

    cols_to_show = [
        "ticker",
        "total_events",
        "success_rate",
        "failure_rate",
        "avg_days_to_hit",
        "avg_final_return",
        "avg_max_drawdown",
        "avg_drop_pct",
        "ticker_score",
    ]

    print(
        f"\n=== BEST {TOP_N_TICKERS_TO_PRINT} TICKERS "
        f"(min {MIN_TICKER_EVENTS_FOR_RANKING} events) ==="
    )
    print(best[cols_to_show].to_string(index=False))

    print(
        f"\n=== WORST {TOP_N_TICKERS_TO_PRINT} TICKERS "
        f"(min {MIN_TICKER_EVENTS_FOR_RANKING} events) ==="
    )
    print(worst[cols_to_show].to_string(index=False))


def save_results(
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ticker_summary_df: pd.DataFrame,
    ticker_bucket_summary_df: pd.DataFrame,
) -> None:
    """
    Save detailed events and summary tables to CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events_to_save = events_df.copy()
    summary_to_save = summary_df.copy()
    ticker_summary_to_save = ticker_summary_df.copy()
    ticker_bucket_summary_to_save = ticker_bucket_summary_df.copy()

    if not events_to_save.empty:
        events_to_save["event_date"] = pd.to_datetime(events_to_save["event_date"]).dt.strftime("%Y-%m-%d")
        events_to_save["exit_date"] = pd.to_datetime(events_to_save["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    summary_to_save.to_csv(SUMMARY_CSV, index=False)
    ticker_summary_to_save.to_csv(TICKER_SUMMARY_CSV, index=False)
    ticker_bucket_summary_to_save.to_csv(TICKER_BUCKET_SUMMARY_CSV, index=False)
    events_to_save.to_csv(EVENTS_CSV, index=False)

    print(f"\nSaved detailed events to:        {EVENTS_CSV}")
    print(f"Saved bucket summary to:         {SUMMARY_CSV}")
    print(f"Saved ticker summary to:         {TICKER_SUMMARY_CSV}")
    print(f"Saved ticker+bucket summary to:  {TICKER_BUCKET_SUMMARY_CSV}")

# ============================================================
# 6) MAIN
# ============================================================

def main() -> None:
    print("Starting bounce backtest...")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {UNIVERSE}")
    print(f"Date range: {START_DATE} -> {END_DATE}")
    print(f"Target return: {TARGET_RETURN:.2%}")
    print(f"Max hold days: {MAX_HOLD_DAYS}")
    print(f"Minimum history days: {MIN_HISTORY_DAYS}")

    tickers = get_universe_tickers(UNIVERSE)
    print(f"Loaded {len(tickers)} tickers.")

    price_data = download_price_data(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    print(f"Downloaded usable data for {len(price_data)} tickers.")

    events = generate_events(
        price_data=price_data,
        min_history_days=MIN_HISTORY_DAYS,
    )
    print(f"Generated {len(events)} qualifying drop events.")

    labeled_events = run_event_study(
        price_data=price_data,
        events=events,
        target_return=TARGET_RETURN,
        max_hold_days=MAX_HOLD_DAYS,
    )
    print(f"Labeled {len(labeled_events)} events.")

    summary_by_bucket_df = summarize_by_bucket(labeled_events)
    summary_by_ticker_df = summarize_by_ticker(labeled_events)
    summary_by_ticker_bucket_df = summarize_by_ticker_and_bucket(labeled_events)

    print_summary(summary_by_bucket_df)
    print_ticker_rankings(summary_by_ticker_df)
    print_ticker_bucket_rankings(summary_by_ticker_bucket_df)

    save_results(
        labeled_events,
        summary_by_bucket_df,
        summary_by_ticker_df,
        summary_by_ticker_bucket_df,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()