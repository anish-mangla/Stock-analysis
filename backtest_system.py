#!/usr/bin/env python3
"""
backtest_system.py

Backtests the full trading system against historical data.

Two modes:
1. LABELED EVENTS (336 events with stock-level classifications)
   - Tests the full system: quantitative + market context + news classification
   - Uses the ChatGPT-verified labels as a proxy for what Claude would have returned

2. ALL EVENTS (7,831 events)
   - Tests the quantitative-only version (no news classification)
   - Uses market context and drop bucket to score
   - Shows what the system would do without Claude

For each event, the backtest:
- Runs it through the trade scorer
- Gets the confidence tier (high/medium/low/skip)
- Checks the actual outcome (did it hit +1% within 14 days?)
- Computes portfolio-level metrics
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from trade_scorer import TradeScorer


OUTPUT_DIR = "outputs"
EVENTS_LABELED_CSV = os.path.join(OUTPUT_DIR, "events_fully_labeled.csv")

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 10_000.0
MAX_POSITIONS = 10


def load_labeled_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_LABELED_CSV)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def backtest_labeled_events(events: pd.DataFrame, scorer: TradeScorer) -> pd.DataFrame:
    """
    Backtest using the 336 labeled events (full system test).
    """
    labeled = events[events["stock_event_type"] != "unlabeled"].copy()
    print(f"Backtesting {len(labeled)} labeled events...")

    results = []
    for _, row in labeled.iterrows():
        market_stressed = row.get("liquidity_credit_stress_severity", "none") in ["medium", "high"]

        score = scorer.score_candidate(
            drop_bucket=row["drop_bucket"],
            stock_event_type=row["stock_event_type"],
            stock_event_severity=row["stock_event_severity"],
            market_stressed=market_stressed,
        )

        results.append({
            "ticker": row["ticker"],
            "event_date": row["event_date"],
            "drop_pct": row["drop_pct"],
            "drop_bucket": row["drop_bucket"],
            "stock_event_type": row["stock_event_type"],
            "stock_event_severity": row["stock_event_severity"],
            "market_stressed": market_stressed,
            "actual_success": row["success"],
            "actual_final_return": row["final_return"],
            "actual_max_drawdown": row["max_drawdown"],
            "actual_days_to_hit": row["days_to_hit"],
            "predicted_success_rate": score["success_rate"],
            "confidence": score["confidence"],
            "lookup_level": score["lookup_level"],
            "n_events_in_lookup": score["n_events"],
            "would_trade": score["confidence"] != "skip",
        })

    return pd.DataFrame(results)


def backtest_all_events(events: pd.DataFrame, scorer: TradeScorer) -> pd.DataFrame:
    """
    Backtest using all 7,831 events (quantitative-only, no news classification).
    Uses drop_bucket + market context only.
    """
    print(f"Backtesting {len(events)} events (quantitative only)...")

    results = []
    for _, row in events.iterrows():
        market_stressed = row.get("liquidity_credit_stress_severity", "none") in ["medium", "high"]

        # Without Claude, we don't know the event type — use "unlabeled"
        # The scorer will fall back to drop_bucket only
        score = scorer.score_candidate(
            drop_bucket=row["drop_bucket"],
            stock_event_type="unlabeled",
            stock_event_severity="medium",
            market_stressed=market_stressed,
        )

        results.append({
            "ticker": row["ticker"],
            "event_date": row["event_date"],
            "drop_pct": row["drop_pct"],
            "drop_bucket": row["drop_bucket"],
            "market_stressed": market_stressed,
            "actual_success": row["success"],
            "actual_final_return": row["final_return"],
            "actual_max_drawdown": row["max_drawdown"],
            "predicted_success_rate": score["success_rate"],
            "confidence": score["confidence"],
            "would_trade": score["confidence"] != "skip",
        })

    return pd.DataFrame(results)


def compute_metrics(results: pd.DataFrame, label: str) -> Dict:
    """Compute summary metrics for a backtest."""
    total = len(results)
    traded = results[results["would_trade"]]
    skipped = results[~results["would_trade"]]

    metrics = {
        "label": label,
        "total_events": total,
        "traded": len(traded),
        "skipped": len(skipped),
        "trade_rate": len(traded) / total if total > 0 else 0,
    }

    if len(traded) > 0:
        metrics["traded_win_rate"] = traded["actual_success"].mean()
        metrics["traded_avg_return"] = traded["actual_final_return"].mean()
        metrics["traded_avg_drawdown"] = traded["actual_max_drawdown"].mean()
        if "actual_days_to_hit" in traded.columns:
            valid_days = traded["actual_days_to_hit"].dropna()
            metrics["traded_avg_days_to_hit"] = valid_days.mean() if len(valid_days) > 0 else None
        else:
            metrics["traded_avg_days_to_hit"] = None
    else:
        metrics["traded_win_rate"] = None
        metrics["traded_avg_return"] = None
        metrics["traded_avg_drawdown"] = None
        metrics["traded_avg_days_to_hit"] = None

    if len(skipped) > 0:
        metrics["skipped_win_rate"] = skipped["actual_success"].mean()
        metrics["skipped_avg_return"] = skipped["actual_final_return"].mean()
    else:
        metrics["skipped_win_rate"] = None
        metrics["skipped_avg_return"] = None

    # Baseline: what if we traded everything?
    metrics["baseline_win_rate"] = results["actual_success"].mean()
    metrics["baseline_avg_return"] = results["actual_final_return"].mean()

    # Value added: traded win rate vs baseline
    if metrics["traded_win_rate"] is not None:
        metrics["win_rate_improvement"] = metrics["traded_win_rate"] - metrics["baseline_win_rate"]
    else:
        metrics["win_rate_improvement"] = None

    return metrics


def print_metrics(metrics: Dict):
    print(f"\n{'=' * 60}")
    print(f"  {metrics['label']}")
    print(f"{'=' * 60}")
    print(f"  Total events:        {metrics['total_events']}")
    print(f"  Traded:              {metrics['traded']} ({metrics['trade_rate']*100:.1f}%)")
    print(f"  Skipped:             {metrics['skipped']}")
    print()
    print(f"  BASELINE (trade everything):")
    print(f"    Win rate:          {metrics['baseline_win_rate']*100:.1f}%")
    print(f"    Avg return:        {metrics['baseline_avg_return']*100:.2f}%")
    print()
    print(f"  SYSTEM (trade only selected):")
    if metrics["traded_win_rate"] is not None:
        print(f"    Win rate:          {metrics['traded_win_rate']*100:.1f}%")
        print(f"    Avg return:        {metrics['traded_avg_return']*100:.2f}%")
        print(f"    Avg drawdown:      {metrics['traded_avg_drawdown']*100:.2f}%")
        if metrics["traded_avg_days_to_hit"]:
            print(f"    Avg days to hit:   {metrics['traded_avg_days_to_hit']:.1f}")
    print()
    if metrics["skipped_win_rate"] is not None:
        print(f"  SKIPPED events (would have avoided):")
        print(f"    Win rate:          {metrics['skipped_win_rate']*100:.1f}%")
        print(f"    Avg return:        {metrics['skipped_avg_return']*100:.2f}%")
    print()
    if metrics["win_rate_improvement"] is not None:
        direction = "↑" if metrics["win_rate_improvement"] > 0 else "↓"
        print(f"  WIN RATE IMPROVEMENT: {direction} {abs(metrics['win_rate_improvement'])*100:.1f} percentage points")


def main():
    scorer = TradeScorer()
    events = load_labeled_events()

    # === Backtest 1: Labeled events (full system) ===
    labeled_results = backtest_labeled_events(events, scorer)
    labeled_metrics = compute_metrics(labeled_results, "FULL SYSTEM (336 labeled events)")
    print_metrics(labeled_metrics)

    # Breakdown by confidence tier
    print("\n  BY CONFIDENCE TIER:")
    for conf in ["high", "medium", "low", "skip"]:
        subset = labeled_results[labeled_results["confidence"] == conf]
        if len(subset) == 0:
            continue
        wr = subset["actual_success"].mean()
        ar = subset["actual_final_return"].mean()
        print(f"    {conf:>6}: {len(subset):>4} events, win rate {wr*100:.1f}%, avg return {ar*100:.2f}%")

    # Breakdown by event type
    print("\n  BY EVENT TYPE (traded only):")
    traded = labeled_results[labeled_results["would_trade"]]
    for etype in traded["stock_event_type"].unique():
        subset = traded[traded["stock_event_type"] == etype]
        if len(subset) < 3:
            continue
        wr = subset["actual_success"].mean()
        print(f"    {etype:>25}: {len(subset):>3} events, win rate {wr*100:.1f}%")

    # === Backtest 2: All events (quantitative only) ===
    all_results = backtest_all_events(events, scorer)
    all_metrics = compute_metrics(all_results, "QUANTITATIVE ONLY (7,831 events, no Claude)")
    print_metrics(all_metrics)

    # === Backtest 3: What if we only traded "no_clear_catalyst" events? ===
    no_catalyst = labeled_results[labeled_results["stock_event_type"] == "no_clear_catalyst"]
    if len(no_catalyst) > 0:
        nc_metrics = compute_metrics(no_catalyst, "NO CATALYST ONLY (the sweet spot)")
        print_metrics(nc_metrics)

    # === Backtest 4: What if we avoided product failures and severe guidance cuts? ===
    avoid_mask = (
        (labeled_results["stock_event_type"] == "product_service_failure") |
        ((labeled_results["stock_event_type"] == "guidance_cut") &
         (labeled_results["stock_event_severity"] == "high") &
         (~labeled_results["market_stressed"]))
    )
    filtered = labeled_results[~avoid_mask]
    if len(filtered) > 0:
        filt_metrics = compute_metrics(filtered, "AVOID PRODUCT FAILURES + SEVERE GUIDANCE CUTS IN CALM MARKETS")
        print_metrics(filt_metrics)

    # Save results
    labeled_results.to_csv(os.path.join(OUTPUT_DIR, "backtest_labeled_results.csv"), index=False)
    all_results.to_csv(os.path.join(OUTPUT_DIR, "backtest_all_results.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
