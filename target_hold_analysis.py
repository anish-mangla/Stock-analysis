#!/usr/bin/env python3
"""
target_hold_analysis.py

Tests different profit targets and holding periods against historical events.

Scenarios tested:
1. +1% target, 10-day hold (current system)
2. +1% target, 14-day hold (original backtest)
3. +1% target, unlimited hold (hold until target hit or 60 days)
4. +2% target, 14-day hold
5. +2% target, unlimited hold
6. +3% target, 14-day hold
7. +3% target, unlimited hold
8. Volatility-adjusted target: target = max(1%, 1.5 * stock_vol20)

Downloads actual forward price paths for events to compute accurate results.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")

# We'll sample events rather than downloading prices for all 7,831
# Focus on a representative sample across years and drop buckets
SAMPLE_SIZE_PER_BUCKET = 200  # ~800 total events
MAX_FORWARD_DAYS = 60  # look forward up to 60 trading days


def load_and_sample_events() -> pd.DataFrame:
    """Load events and take a stratified sample."""
    events = pd.read_csv(EVENTS_CSV)
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.dropna(subset=["entry_price"])

    # Stratified sample by drop bucket
    sampled = []
    for bucket in ["3-5%", "5-7%", "7-10%", "10%+"]:
        bucket_events = events[events["drop_bucket"] == bucket]
        n = min(SAMPLE_SIZE_PER_BUCKET, len(bucket_events))
        sampled.append(bucket_events.sample(n=n, random_state=42))

    result = pd.concat(sampled, ignore_index=True)
    result = result.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return result


def download_forward_prices(events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Download price data for all tickers in the events."""
    tickers = sorted(events["ticker"].unique().tolist())
    min_date = events["event_date"].min()
    max_date = events["event_date"].max()

    start_str = (min_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end_str = (max_date + pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"Downloading prices for {len(tickers)} tickers from {start_str} to {end_str}...")

    raw = yf.download(
        tickers=tickers,
        start=start_str,
        end=end_str,
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No data returned")

    price_data: Dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            df = raw[ticker].copy()
            if "Close" not in df.columns:
                continue
            df = df.dropna(subset=["Close"])
            df.index = pd.to_datetime(df.index)
            price_data[ticker] = df.sort_index()
    else:
        if len(tickers) == 1:
            raw.index = pd.to_datetime(raw.index)
            price_data[tickers[0]] = raw.sort_index()

    return price_data


def simulate_forward(
    price_df: pd.DataFrame,
    event_idx: int,
    entry_price: float,
    targets: List[float],
    max_days: int,
) -> Dict[str, Dict]:
    """
    Simulate forward from an event for multiple targets and holding periods.

    Returns dict keyed by scenario name with results.
    """
    forward = price_df.iloc[event_idx + 1: event_idx + 1 + max_days]
    if forward.empty:
        return {}

    closes = forward["Close"].astype(float)
    highs = forward["High"].astype(float) if "High" in forward.columns else closes

    results = {}

    for target in targets:
        target_price = entry_price * (1.0 + target)
        target_label = f"+{target*100:.0f}%"

        # Check each day: did the HIGH reach the target? (more accurate than close-only)
        hit = False
        days_to_hit = None

        for offset, (dt, high_val) in enumerate(highs.items(), start=1):
            if high_val >= target_price:
                hit = True
                days_to_hit = offset
                break

        # For different max hold periods
        for max_hold in [10, 14, 30, 60]:
            if max_hold > len(closes):
                continue

            scenario_name = f"target_{target_label}_hold_{max_hold}d"

            if hit and days_to_hit <= max_hold:
                results[scenario_name] = {
                    "success": True,
                    "days_to_hit": days_to_hit,
                    "exit_return": target,
                    "target": target,
                    "max_hold": max_hold,
                }
            else:
                # Didn't hit target within max_hold days
                exit_price = float(closes.iloc[min(max_hold - 1, len(closes) - 1)])
                exit_return = (exit_price / entry_price) - 1.0

                results[scenario_name] = {
                    "success": False,
                    "days_to_hit": None,
                    "exit_return": exit_return,
                    "target": target,
                    "max_hold": max_hold,
                }

        # Unlimited hold (up to 60 days)
        scenario_name = f"target_{target_label}_hold_unlimited"
        if hit:
            results[scenario_name] = {
                "success": True,
                "days_to_hit": days_to_hit,
                "exit_return": target,
                "target": target,
                "max_hold": 60,
            }
        else:
            exit_price = float(closes.iloc[-1])
            exit_return = (exit_price / entry_price) - 1.0
            results[scenario_name] = {
                "success": False,
                "days_to_hit": None,
                "exit_return": exit_return,
                "target": target,
                "max_hold": 60,
            }

    return results


def run_analysis(events: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run the multi-scenario analysis on all events."""
    targets = [0.01, 0.02, 0.03]
    all_rows = []

    for _, event in events.iterrows():
        ticker = event["ticker"]
        event_date = pd.Timestamp(event["event_date"])
        entry_price = float(event["entry_price"])
        drop_bucket = event["drop_bucket"]

        if ticker not in price_data:
            continue

        df = price_data[ticker]
        if event_date not in df.index:
            continue

        event_idx = df.index.get_loc(event_date)
        if isinstance(event_idx, slice):
            continue

        # Compute stock volatility for vol-adjusted target
        daily_returns = df["Close"].astype(float).pct_change()
        vol_20d = daily_returns.iloc[max(0, event_idx-20):event_idx].std()
        vol_adjusted_target = max(0.01, 1.5 * vol_20d) if pd.notna(vol_20d) else 0.01

        # Add vol-adjusted target to the list
        all_targets = targets + [round(vol_adjusted_target, 4)]

        results = simulate_forward(
            price_df=df,
            event_idx=event_idx,
            entry_price=entry_price,
            targets=all_targets,
            max_days=MAX_FORWARD_DAYS,
        )

        for scenario_name, result in results.items():
            all_rows.append({
                "ticker": ticker,
                "event_date": event_date,
                "entry_price": entry_price,
                "drop_bucket": drop_bucket,
                "drop_pct": event["drop_pct"],
                "vol_20d": vol_20d,
                "vol_adjusted_target": vol_adjusted_target,
                "scenario": scenario_name,
                **result,
            })

    return pd.DataFrame(all_rows)


def summarize_scenarios(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize results by scenario."""
    summary = (
        results.groupby("scenario")
        .agg(
            total_events=("success", "count"),
            success_rate=("success", "mean"),
            avg_exit_return=("exit_return", "mean"),
            avg_days_to_hit=("days_to_hit", "mean"),
        )
        .reset_index()
    )

    # Parse scenario name for sorting
    summary["target_pct"] = summary["scenario"].str.extract(r"target_\+(\d+)%").astype(float)
    summary["hold_type"] = summary["scenario"].apply(
        lambda x: "unlimited" if "unlimited" in x else x.split("_hold_")[1] if "_hold_" in x else "?"
    )

    summary = summary.sort_values(["target_pct", "hold_type"]).reset_index(drop=True)
    return summary


def summarize_by_bucket(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize by drop bucket × scenario."""
    summary = (
        results.groupby(["drop_bucket", "scenario"])
        .agg(
            total_events=("success", "count"),
            success_rate=("success", "mean"),
            avg_exit_return=("exit_return", "mean"),
            avg_days_to_hit=("days_to_hit", "mean"),
        )
        .reset_index()
    )
    return summary


def main():
    print("Loading and sampling events...")
    events = load_and_sample_events()
    print(f"Sampled {len(events)} events")
    print(f"By bucket: {events['drop_bucket'].value_counts().to_dict()}")

    print("\nDownloading forward price data...")
    price_data = download_forward_prices(events)
    print(f"Got price data for {len(price_data)} tickers")

    print("\nRunning multi-scenario analysis...")
    results = run_analysis(events, price_data)
    print(f"Generated {len(results)} scenario results")

    # Overall summary
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON (all events)")
    print("=" * 80)

    summary = summarize_scenarios(results)

    # Filter to the key scenarios we care about
    key_scenarios = [
        "target_+1%_hold_10d",
        "target_+1%_hold_14d",
        "target_+1%_hold_30d",
        "target_+1%_hold_unlimited",
        "target_+2%_hold_14d",
        "target_+2%_hold_30d",
        "target_+2%_hold_unlimited",
        "target_+3%_hold_14d",
        "target_+3%_hold_30d",
        "target_+3%_hold_unlimited",
    ]

    display = summary[summary["scenario"].isin(key_scenarios)].copy()
    display["success_rate"] = (display["success_rate"] * 100).round(1)
    display["avg_exit_return"] = (display["avg_exit_return"] * 100).round(2)
    display["avg_days_to_hit"] = display["avg_days_to_hit"].round(1)

    print(display[["scenario", "total_events", "success_rate", "avg_exit_return", "avg_days_to_hit"]].to_string(index=False))

    # By drop bucket for key scenarios
    print("\n" + "=" * 80)
    print("BY DROP BUCKET × TARGET × HOLD PERIOD")
    print("=" * 80)

    bucket_summary = summarize_by_bucket(results)
    for scenario in ["target_+1%_hold_14d", "target_+2%_hold_14d", "target_+2%_hold_unlimited",
                      "target_+1%_hold_unlimited"]:
        subset = bucket_summary[bucket_summary["scenario"] == scenario].copy()
        if subset.empty:
            continue
        subset["success_rate"] = (subset["success_rate"] * 100).round(1)
        subset["avg_exit_return"] = (subset["avg_exit_return"] * 100).round(2)
        subset["avg_days_to_hit"] = subset["avg_days_to_hit"].round(1)
        print(f"\n  {scenario}:")
        print(subset[["drop_bucket", "total_events", "success_rate", "avg_exit_return", "avg_days_to_hit"]].to_string(index=False))

    # Vol-adjusted target analysis
    print("\n" + "=" * 80)
    print("VOLATILITY-ADJUSTED TARGET ANALYSIS")
    print("=" * 80)

    vol_results = results[results["scenario"].str.contains("target_\\+[0-9]\\.[0-9]")].copy()
    if not vol_results.empty:
        vol_summary = (
            vol_results.groupby("scenario")
            .agg(
                total_events=("success", "count"),
                success_rate=("success", "mean"),
                avg_target=("target", "mean"),
                avg_exit_return=("exit_return", "mean"),
            )
            .reset_index()
        )
        vol_summary["success_rate"] = (vol_summary["success_rate"] * 100).round(1)
        vol_summary["avg_target"] = (vol_summary["avg_target"] * 100).round(2)
        vol_summary["avg_exit_return"] = (vol_summary["avg_exit_return"] * 100).round(2)
        print(vol_summary.to_string(index=False))
    else:
        # Compute vol-adjusted stats manually
        vol_events = results[results["scenario"] == "target_+1%_hold_14d"].copy()
        vol_events["vol_bucket"] = pd.cut(vol_events["vol_20d"], bins=[0, 0.02, 0.03, 0.05, 1.0],
                                           labels=["low_vol", "med_vol", "high_vol", "very_high_vol"])
        vol_by_bucket = (
            vol_events.groupby("vol_bucket", observed=False)
            .agg(
                total=("success", "count"),
                success_rate=("success", "mean"),
                avg_vol=("vol_20d", "mean"),
            )
            .reset_index()
        )
        vol_by_bucket["success_rate"] = (vol_by_bucket["success_rate"] * 100).round(1)
        vol_by_bucket["avg_vol"] = (vol_by_bucket["avg_vol"] * 100).round(2)
        print("\n  Success rate by stock volatility (for +1% target, 14d hold):")
        print(vol_by_bucket.to_string(index=False))

    # Save
    results.to_csv(os.path.join(OUTPUT_DIR, "target_hold_analysis.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "target_hold_summary.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
