#!/usr/bin/env python3
"""
entry_optimization.py

Workstream C: Entry Price Optimization

Tests different limit order entry strategies against historical data to find
the optimal entry offset for each confidence tier.

For each event in events.csv, simulates placing a limit order at various
levels below the close price and measures:
- Fill rate: how often the limit order would have been filled
- Win rate on fills: success rate when the order fills
- Expected value: fill_rate × win_rate (the metric that matters)
- Improved return: average return from the lower entry price

Uses min_close_during_hold as a proxy for whether the limit would fill.
This is conservative — intraday lows would be even lower than daily closes,
so actual fill rates would be higher.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
STOCK_LABELS_CSV = os.path.join(OUTPUT_DIR, "stock_level_labels.csv")
EVENTS_LABELED_CSV = os.path.join(OUTPUT_DIR, "events_fully_labeled.csv")

# Limit order offsets to test (negative = below close)
OFFSETS_TO_TEST = [0.0, -0.0025, -0.005, -0.0075, -0.01, -0.015, -0.02, -0.03]

# Target return
TARGET_RETURN = 0.01  # +1%

# Max hold days for the adjusted entry
MAX_HOLD_DAYS = 14


def load_events() -> pd.DataFrame:
    """Load events with labels if available, otherwise plain events."""
    if os.path.exists(EVENTS_LABELED_CSV):
        df = pd.read_csv(EVENTS_LABELED_CSV)
    else:
        df = pd.read_csv(EVENTS_CSV)

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.dropna(subset=["entry_price", "min_close_during_hold", "exit_price"])
    return df


def simulate_limit_entry(
    events: pd.DataFrame,
    offset: float,
) -> pd.DataFrame:
    """
    For each event, simulate a limit order at entry_price * (1 + offset).

    Returns a DataFrame with columns:
    - limit_price: the limit order price
    - would_fill: True if min_close_during_hold <= limit_price
    - adjusted_target: limit_price * (1 + TARGET_RETURN)
    - adjusted_success: True if exit_price >= adjusted_target (simplified)
    - adjusted_return: (exit_price / limit_price) - 1
    """
    df = events.copy()

    df["limit_price"] = df["entry_price"] * (1.0 + offset)

    # Would the limit fill?
    # Conservative: check if min close during hold is at or below limit
    # In reality, intraday lows would be lower, so more would fill
    if offset == 0.0:
        # At close price, always fills
        df["would_fill"] = True
    else:
        df["would_fill"] = df["min_close_during_hold"] <= df["limit_price"]

    # For filled orders, compute adjusted success
    df["adjusted_target"] = df["limit_price"] * (1.0 + TARGET_RETURN)

    # Simplified: if the original exit_price >= adjusted_target, it's a win
    # This isn't perfect because the exit timing might differ, but it's a
    # reasonable approximation for the optimization
    df["adjusted_success"] = False
    filled = df["would_fill"]

    # For filled orders: check if the stock ever reached the adjusted target
    # We use exit_price as a proxy — if the stock reached +1% from original entry,
    # it almost certainly reached +1% from a lower entry too
    # But we also need to account for cases where the original was a failure
    # but the lower entry would have been a success
    df.loc[filled, "adjusted_success"] = (
        df.loc[filled, "exit_price"] >= df.loc[filled, "adjusted_target"]
    )

    # Also check: even if original exit_price < adjusted_target,
    # the stock might have hit the target during the hold period
    # We can approximate this: if original success=True (hit +1% from original entry),
    # then it definitely hit +1% from a lower entry
    df.loc[filled & (df["success"] == True), "adjusted_success"] = True

    # Adjusted return for filled orders
    df.loc[filled, "adjusted_return"] = (
        df.loc[filled, "exit_price"] / df.loc[filled, "limit_price"]
    ) - 1.0

    df["offset"] = offset

    return df


def analyze_offsets(events: pd.DataFrame) -> pd.DataFrame:
    """Test all offset levels and compute summary statistics."""
    rows = []

    for offset in OFFSETS_TO_TEST:
        result = simulate_limit_entry(events, offset)

        total = len(result)
        filled = result["would_fill"].sum()
        fill_rate = filled / total if total > 0 else 0

        filled_df = result[result["would_fill"]]
        win_rate = filled_df["adjusted_success"].mean() if len(filled_df) > 0 else 0
        avg_return = filled_df["adjusted_return"].mean() if len(filled_df) > 0 else 0

        # Expected value: probability of getting a winning trade
        # = fill_rate × win_rate
        expected_value = fill_rate * win_rate

        # Compare to baseline (offset=0)
        baseline_win_rate = events["success"].mean()

        rows.append({
            "offset_pct": offset * 100,
            "offset_label": f"{offset*100:+.2f}%" if offset != 0 else "close",
            "total_events": total,
            "filled": int(filled),
            "fill_rate": round(fill_rate, 4),
            "win_rate_on_fills": round(win_rate, 4),
            "avg_return_on_fills": round(avg_return, 6),
            "expected_value": round(expected_value, 4),
            "baseline_win_rate": round(baseline_win_rate, 4),
            "ev_vs_baseline": round(expected_value - baseline_win_rate, 4),
        })

    return pd.DataFrame(rows)


def analyze_by_drop_bucket(events: pd.DataFrame) -> pd.DataFrame:
    """Analyze offsets separately for each drop bucket."""
    all_rows = []

    for bucket in ["3-5%", "5-7%", "7-10%", "10%+"]:
        bucket_events = events[events["drop_bucket"] == bucket]
        if len(bucket_events) < 20:
            continue

        for offset in OFFSETS_TO_TEST:
            result = simulate_limit_entry(bucket_events, offset)

            total = len(result)
            filled = result["would_fill"].sum()
            fill_rate = filled / total if total > 0 else 0

            filled_df = result[result["would_fill"]]
            win_rate = filled_df["adjusted_success"].mean() if len(filled_df) > 0 else 0

            expected_value = fill_rate * win_rate

            all_rows.append({
                "drop_bucket": bucket,
                "offset_pct": offset * 100,
                "total_events": total,
                "filled": int(filled),
                "fill_rate": round(fill_rate, 4),
                "win_rate_on_fills": round(win_rate, 4),
                "expected_value": round(expected_value, 4),
            })

    return pd.DataFrame(all_rows)


def analyze_by_event_type(events: pd.DataFrame) -> pd.DataFrame:
    """Analyze offsets by stock event type (for labeled events only)."""
    if "stock_event_type" not in events.columns:
        return pd.DataFrame()

    labeled = events[events["stock_event_type"] != "unlabeled"]
    if len(labeled) < 20:
        return pd.DataFrame()

    all_rows = []

    for etype in labeled["stock_event_type"].unique():
        type_events = labeled[labeled["stock_event_type"] == etype]
        if len(type_events) < 10:
            continue

        for offset in [0.0, -0.005, -0.01, -0.02]:
            result = simulate_limit_entry(type_events, offset)

            total = len(result)
            filled = result["would_fill"].sum()
            fill_rate = filled / total if total > 0 else 0

            filled_df = result[result["would_fill"]]
            win_rate = filled_df["adjusted_success"].mean() if len(filled_df) > 0 else 0

            expected_value = fill_rate * win_rate

            all_rows.append({
                "event_type": etype,
                "offset_pct": offset * 100,
                "total_events": total,
                "filled": int(filled),
                "fill_rate": round(fill_rate, 4),
                "win_rate_on_fills": round(win_rate, 4),
                "expected_value": round(expected_value, 4),
            })

    return pd.DataFrame(all_rows)


def _fmt_pct(val):
    return f"{val*100:.2f}%"


def main():
    events = load_events()
    print(f"Loaded {len(events)} events")

    # === Overall analysis ===
    print("\n" + "=" * 80)
    print("OVERALL: LIMIT ORDER OFFSET ANALYSIS")
    print("=" * 80)

    overall = analyze_offsets(events)
    display = overall.copy()
    for col in ["fill_rate", "win_rate_on_fills", "expected_value", "baseline_win_rate", "ev_vs_baseline"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_return_on_fills" in display.columns:
        display["avg_return_on_fills"] = (display["avg_return_on_fills"] * 100).round(2)
    print(display.to_string(index=False))

    overall.to_csv(os.path.join(OUTPUT_DIR, "entry_optimization_overall.csv"), index=False)

    # === By drop bucket ===
    print("\n" + "=" * 80)
    print("BY DROP BUCKET: LIMIT ORDER OFFSET ANALYSIS")
    print("=" * 80)

    by_bucket = analyze_by_drop_bucket(events)
    display = by_bucket.copy()
    for col in ["fill_rate", "win_rate_on_fills", "expected_value"]:
        display[col] = (display[col] * 100).round(2)
    print(display.to_string(index=False))

    by_bucket.to_csv(os.path.join(OUTPUT_DIR, "entry_optimization_by_bucket.csv"), index=False)

    # === By event type ===
    print("\n" + "=" * 80)
    print("BY EVENT TYPE: LIMIT ORDER OFFSET ANALYSIS")
    print("=" * 80)

    by_type = analyze_by_event_type(events)
    if not by_type.empty:
        display = by_type.copy()
        for col in ["fill_rate", "win_rate_on_fills", "expected_value"]:
            display[col] = (display[col] * 100).round(2)
        print(display.to_string(index=False))
        by_type.to_csv(os.path.join(OUTPUT_DIR, "entry_optimization_by_event_type.csv"), index=False)
    else:
        print("No labeled events available for event-type analysis.")

    # === Optimal offsets summary ===
    print("\n" + "=" * 80)
    print("OPTIMAL OFFSET BY DROP BUCKET (highest expected value)")
    print("=" * 80)

    for bucket in ["3-5%", "5-7%", "7-10%", "10%+"]:
        bucket_data = by_bucket[by_bucket["drop_bucket"] == bucket]
        if bucket_data.empty:
            continue
        best = bucket_data.loc[bucket_data["expected_value"].idxmax()]
        print(f"  {bucket}: best offset = {best['offset_pct']:+.2f}%, "
              f"fill rate = {best['fill_rate']*100:.1f}%, "
              f"win rate = {best['win_rate_on_fills']*100:.1f}%, "
              f"EV = {best['expected_value']*100:.1f}%")

    print("\nDone. Results saved to outputs/")


if __name__ == "__main__":
    main()
