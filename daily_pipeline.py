#!/usr/bin/env python3
"""
daily_pipeline.py

The unified daily pipeline for the mean-reversion trading system.

Flow:
1. Scan S&P 100 for candidates (candidate_scanner)
2. Compute market context (market_context)
3. Classify candidates via Claude (news_classifier)
4. Score and rank candidates (trade_scorer)
5. Output trade recommendations
6. Save everything for audit trail

This replaces the old main.py pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from candidate_scanner import scan_candidates
from market_context import get_market_context
from news_classifier import classify_candidates
from trade_scorer import TradeScorer


OUTPUT_DIR = "daily_runs"
MAX_POSITIONS = 10


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_run_date() -> str:
    """Get today's date string for the output directory."""
    return datetime.now(timezone.utc).date().isoformat()


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run_pipeline(
    skip_claude: bool = False,
    max_candidates: int = 20,
) -> Dict[str, Any]:
    """
    Run the full daily pipeline.

    Args:
        skip_claude: if True, skip the Claude classification step (for testing)
        max_candidates: max candidates to process

    Returns:
        Dict with all pipeline outputs
    """
    run_timestamp = _now_utc_iso()
    run_date = _get_run_date()

    print(f"{'=' * 70}")
    print(f"DAILY PIPELINE — {run_date}")
    print(f"{'=' * 70}")

    # ============================================================
    # STEP 1: Scan for candidates
    # ============================================================
    print("\n[1/5] Scanning for candidates...")
    scan_result = scan_candidates(max_candidates=max_candidates)
    candidates = scan_result["candidates"]
    print(f"  Found {len(candidates)} candidates")

    if not candidates:
        print("  No candidates found. Pipeline complete.")
        return {"run_date": run_date, "candidates": 0, "trades": []}

    # ============================================================
    # STEP 2: Compute market context
    # ============================================================
    print("\n[2/5] Computing market context...")
    market_ctx = get_market_context()

    # ============================================================
    # STEP 3: Classify via Claude
    # ============================================================
    if skip_claude:
        print("\n[3/5] Skipping Claude classification (test mode)")
        # Default all to no_clear_catalyst for testing
        classification_map = {}
        for c in candidates:
            classification_map[c["ticker"]] = {
                "ticker": c["ticker"],
                "event_type_id": 11,
                "event_type_name": "no_clear_catalyst",
                "severity": "medium",
                "description": "Classification skipped (test mode)",
            }
        classifier_result = {
            "classifications": list(classification_map.values()),
            "classification_map": classification_map,
            "metadata": {"skipped": True},
        }
    else:
        print(f"\n[3/5] Classifying {len(candidates)} candidates via Claude...")
        classifier_result = classify_candidates(
            candidates=candidates,
            market_context=market_ctx,
            include_context=True,
        )
        classification_map = classifier_result.get("classification_map", {})
        print(f"  Received {len(classification_map)} classifications")

        if classifier_result["metadata"].get("error"):
            print(f"  WARNING: {classifier_result['metadata']['error']}")

    # ============================================================
    # STEP 4: Score and rank
    # ============================================================
    print("\n[4/5] Scoring and ranking candidates...")
    scorer = TradeScorer()

    # Merge classifications into candidates
    scored_candidates = []
    for c in candidates:
        ticker = c["ticker"]
        classification = classification_map.get(ticker, {})

        c_enriched = {
            **c,
            "stock_event_type": classification.get("event_type_name", "no_clear_catalyst"),
            "stock_event_severity": classification.get("severity", "medium"),
            "stock_event_description": classification.get("description", ""),
            "market_stressed": market_ctx.get("market_stressed", False),
        }
        scored_candidates.append(c_enriched)

    ranked = scorer.score_and_rank_candidates(scored_candidates)

    # ============================================================
    # STEP 5: Generate trade recommendations
    # ============================================================
    print("\n[5/5] Generating trade recommendations...")

    trades = []
    skipped = []

    for r in ranked:
        if r["confidence"] == "skip":
            skipped.append(r)
            continue

        if len(trades) >= MAX_POSITIONS:
            break

        trades.append(r)

    print(f"\n{'=' * 70}")
    print(f"RESULTS — {run_date}")
    print(f"{'=' * 70}")
    print(f"Market stress: {market_ctx.get('stress_level', 'unknown')} (VIX: {market_ctx.get('vix_level', '?')})")
    print(f"Candidates scanned: {len(candidates)}")
    print(f"Trades recommended: {len(trades)}")
    print(f"Skipped (low probability): {len(skipped)}")

    if trades:
        print(f"\n{'Rank':>4} {'Ticker':>6} {'Drop':>7} {'Event Type':>22} {'Sev':>4} "
              f"{'Success':>8} {'Conf':>6} {'Entry$':>8} {'Target$':>8}")
        print("-" * 85)
        for i, t in enumerate(trades, 1):
            print(
                f"{i:>4} {t['ticker']:>6} {t['drop_magnitude']*100:>6.1f}% "
                f"{t['stock_event_type']:>22} {t['stock_event_severity']:>4} "
                f"{t['success_rate']*100:>7.1f}% {t['confidence']:>6} "
                f"${t['entry_price']:>7.2f} ${t['target_price']:>7.2f}"
            )

    if skipped:
        print(f"\nSkipped candidates:")
        for s in skipped:
            print(
                f"  {s['ticker']:>6} {s['drop_magnitude']*100:>6.1f}% "
                f"{s['stock_event_type']:>22} → {s['success_rate']*100:.1f}% (SKIP)"
            )

    # ============================================================
    # Save outputs
    # ============================================================
    run_dir = Path(OUTPUT_DIR) / run_date / "v2_pipeline"
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_json(run_dir / "scan_result.json", scan_result)
    _save_json(run_dir / "market_context.json", market_ctx)
    _save_json(run_dir / "classifier_result.json", {
        "classifications": classifier_result.get("classifications", []),
        "metadata": classifier_result.get("metadata", {}),
    })
    _save_json(run_dir / "trades.json", {
        "run_date": run_date,
        "run_timestamp": run_timestamp,
        "market_context": {
            "vix": market_ctx.get("vix_level"),
            "spy_return": market_ctx.get("spy_daily_return"),
            "stress_level": market_ctx.get("stress_level"),
        },
        "trades": trades,
        "skipped": skipped,
        "summary": {
            "candidates_scanned": len(candidates),
            "trades_recommended": len(trades),
            "skipped_count": len(skipped),
            "avg_success_rate": (
                sum(t["success_rate"] for t in trades) / len(trades)
                if trades else None
            ),
        },
    })

    print(f"\nOutputs saved to: {run_dir}")

    return {
        "run_date": run_date,
        "candidates": len(candidates),
        "trades": trades,
        "skipped": skipped,
        "market_context": market_ctx,
    }


if __name__ == "__main__":
    import sys

    skip_claude = "--skip-claude" in sys.argv or "--test" in sys.argv

    if skip_claude:
        print("Running in TEST MODE (Claude classification skipped)")
        print()

    result = run_pipeline(skip_claude=skip_claude)
