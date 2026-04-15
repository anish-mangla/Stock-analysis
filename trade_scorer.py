#!/usr/bin/env python3
"""
trade_scorer.py

The core decision engine. Takes a classified candidate and returns
a trade decision based on hierarchical lookup of historical success rates.

This module:
1. Loads the lookup tables built in Step 5
2. For each candidate with a news classification, performs hierarchical lookup
3. Returns a confidence tier and suggested entry price

The lookup is hierarchical — it tries the most specific combination first,
then falls back to less specific ones if there isn't enough historical data.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


OUTPUT_DIR = "outputs"

# Minimum events required to trust a lookup cell
MIN_EVENTS_FOR_LOOKUP = 10
MIN_EVENTS_FALLBACK = 5

# Confidence tiers
# Data shows entering at close is optimal across all buckets.
# Limit orders below close reduce expected value because fill rate drops
# faster than win rate improves. So: trade at close or don't trade.
CONFIDENCE_TIERS = {
    "high": {"min_success_rate": 0.80, "entry_offset": 0.000},    # enter at close
    "medium": {"min_success_rate": 0.65, "entry_offset": 0.000},  # enter at close (data says limits hurt EV)
    "low": {"min_success_rate": 0.50, "entry_offset": 0.000},     # enter at close (if trading at all)
    "skip": {"min_success_rate": 0.0, "entry_offset": None},       # don't trade
}


class TradeScorer:
    """Scores candidates using hierarchical lookup of historical success rates."""

    def __init__(self):
        self.events_labeled = None
        self._load_data()

    def _load_data(self):
        """Load the fully labeled events for custom lookups."""
        path = os.path.join(OUTPUT_DIR, "events_fully_labeled.csv")
        if os.path.exists(path):
            self.events_labeled = pd.read_csv(path)
            self.events_labeled["event_date"] = pd.to_datetime(self.events_labeled["event_date"])
            print(f"Loaded {len(self.events_labeled)} labeled events for lookup.")
        else:
            print(f"WARNING: {path} not found. Scorer will use fallback only.")

    def _compute_success_rate(self, mask: pd.Series) -> Optional[Tuple[float, int, float, float]]:
        """
        Given a boolean mask over events_labeled, compute success stats.
        Returns (success_rate, n_events, avg_days_to_hit, avg_max_drawdown) or None if not enough data.
        """
        if self.events_labeled is None:
            return None

        subset = self.events_labeled[mask]
        n = len(subset)

        if n < MIN_EVENTS_FALLBACK:
            return None

        success_rate = float(subset["success"].mean())
        valid_days = subset["days_to_hit"].dropna()
        avg_days = float(valid_days.mean()) if len(valid_days) > 0 else None
        avg_drawdown = float(subset["max_drawdown"].mean())

        return (success_rate, n, avg_days, avg_drawdown)

    def score_candidate(
        self,
        drop_bucket: str,
        stock_event_type: str,
        stock_event_severity: str,
        market_stressed: bool,
        sector_rotation_outflow: bool = False,
        sector_rotation_severity: str = "none",
    ) -> Dict[str, Any]:
        """
        Score a candidate using hierarchical lookup.

        Args:
            drop_bucket: "3-5%", "5-7%", "7-10%", "10%+"
            stock_event_type: one of the 11 event type names from taxonomy
            stock_event_severity: "low", "medium", "high"
            market_stressed: True if liquidity_credit_stress is medium or high
            sector_rotation_outflow: True if sector is experiencing outflow
            sector_rotation_severity: "low", "medium", "high", "none"

        Returns:
            Dict with: success_rate, confidence, entry_offset, lookup_level, n_events, etc.
        """
        if self.events_labeled is None:
            return self._fallback_score(drop_bucket)

        df = self.events_labeled

        # Build masks for each level of specificity
        # Level 1: event_type + severity + market_stressed + drop_bucket (most specific)
        mask_l1 = (
            (df["stock_event_type"] == stock_event_type) &
            (df["stock_event_severity"] == stock_event_severity) &
            (df["liquidity_credit_stress_severity"].isin(["medium", "high"]) == market_stressed) &
            (df["drop_bucket"] == drop_bucket)
        )

        # Level 2: event_type + severity + market_stressed
        mask_l2 = (
            (df["stock_event_type"] == stock_event_type) &
            (df["stock_event_severity"] == stock_event_severity) &
            (df["liquidity_credit_stress_severity"].isin(["medium", "high"]) == market_stressed)
        )

        # Level 3: event_type + severity
        mask_l3 = (
            (df["stock_event_type"] == stock_event_type) &
            (df["stock_event_severity"] == stock_event_severity)
        )

        # Level 4: event_type alone
        mask_l4 = (df["stock_event_type"] == stock_event_type)

        # Level 5: drop_bucket alone (always has enough data)
        mask_l5 = (df["drop_bucket"] == drop_bucket)

        # Try each level in order
        for level, mask, min_n, label in [
            (1, mask_l1, MIN_EVENTS_FOR_LOOKUP, "event_type+severity+market+bucket"),
            (2, mask_l2, MIN_EVENTS_FOR_LOOKUP, "event_type+severity+market"),
            (3, mask_l3, MIN_EVENTS_FALLBACK, "event_type+severity"),
            (4, mask_l4, MIN_EVENTS_FALLBACK, "event_type"),
            (5, mask_l5, MIN_EVENTS_FALLBACK, "drop_bucket"),
        ]:
            result = self._compute_success_rate(mask)
            if result is not None and result[1] >= min_n:
                success_rate, n_events, avg_days, avg_drawdown = result
                confidence = self._get_confidence_tier(success_rate)

                return {
                    "success_rate": round(success_rate, 4),
                    "n_events": n_events,
                    "avg_days_to_hit": round(avg_days, 2) if avg_days else None,
                    "avg_max_drawdown": round(avg_drawdown, 4) if avg_drawdown else None,
                    "confidence": confidence,
                    "entry_offset": CONFIDENCE_TIERS[confidence]["entry_offset"],
                    "lookup_level": level,
                    "lookup_description": label,
                    "inputs": {
                        "drop_bucket": drop_bucket,
                        "stock_event_type": stock_event_type,
                        "stock_event_severity": stock_event_severity,
                        "market_stressed": market_stressed,
                    },
                }

        # Should never reach here since level 5 always has data, but just in case
        return self._fallback_score(drop_bucket)

    def _get_confidence_tier(self, success_rate: float) -> str:
        """Map success rate to confidence tier."""
        if success_rate >= 0.80:
            return "high"
        elif success_rate >= 0.65:
            return "medium"
        elif success_rate >= 0.50:
            return "low"
        else:
            return "skip"

    def _fallback_score(self, drop_bucket: str) -> Dict[str, Any]:
        """Fallback when no lookup data is available."""
        # Use hardcoded baseline from our analysis
        baseline_rates = {
            "3-5%": 0.8032,
            "5-7%": 0.8088,
            "7-10%": 0.8519,
            "10%+": 0.8857,
        }
        rate = baseline_rates.get(drop_bucket, 0.80)
        confidence = self._get_confidence_tier(rate)

        return {
            "success_rate": rate,
            "n_events": None,
            "avg_days_to_hit": None,
            "avg_max_drawdown": None,
            "confidence": confidence,
            "entry_offset": CONFIDENCE_TIERS[confidence]["entry_offset"],
            "lookup_level": 0,
            "lookup_description": "fallback_baseline",
            "inputs": {"drop_bucket": drop_bucket},
        }

    def compute_entry_price(self, close_price: float, entry_offset: Optional[float]) -> Optional[float]:
        """Compute the suggested entry price based on confidence tier."""
        if entry_offset is None:
            return None  # skip
        return round(close_price * (1.0 + entry_offset), 2)

    def score_and_rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Score a list of candidates and return them ranked by success rate.

        Each candidate dict must have:
        - drop_bucket, close_price
        - stock_event_type, stock_event_severity (from Claude classification)
        - market_stressed (bool)
        """
        scored = []

        for c in candidates:
            score = self.score_candidate(
                drop_bucket=c["drop_bucket"],
                stock_event_type=c.get("stock_event_type", "unlabeled"),
                stock_event_severity=c.get("stock_event_severity", "medium"),
                market_stressed=c.get("market_stressed", False),
            )

            entry_price = self.compute_entry_price(
                close_price=c["close_price"],
                entry_offset=score["entry_offset"],
            )

            scored.append({
                **c,
                **score,
                "entry_price": entry_price,
                "target_price": round(entry_price * 1.01, 2) if entry_price else None,
            })

        # Sort by success rate descending, skip items last
        scored.sort(key=lambda x: (-1 if x["confidence"] == "skip" else x["success_rate"]), reverse=True)

        return scored


if __name__ == "__main__":
    scorer = TradeScorer()

    # Test with some example scenarios
    print("\n=== TEST SCENARIOS ===\n")

    scenarios = [
        {"name": "No catalyst + market stressed + 7-10% drop",
         "drop_bucket": "7-10%", "stock_event_type": "no_clear_catalyst",
         "stock_event_severity": "medium", "market_stressed": True},
        {"name": "Guidance cut (high) + calm market + 10%+ drop",
         "drop_bucket": "10%+", "stock_event_type": "guidance_cut",
         "stock_event_severity": "high", "market_stressed": False},
        {"name": "Product failure (high) + calm market + 10%+ drop",
         "drop_bucket": "10%+", "stock_event_type": "product_service_failure",
         "stock_event_severity": "high", "market_stressed": False},
        {"name": "Demand weakness (medium) + stressed market + 5-7% drop",
         "drop_bucket": "5-7%", "stock_event_type": "demand_weakness",
         "stock_event_severity": "medium", "market_stressed": True},
        {"name": "Earnings miss (high) + calm market + 7-10% drop",
         "drop_bucket": "7-10%", "stock_event_type": "earnings_miss",
         "stock_event_severity": "high", "market_stressed": False},
    ]

    for s in scenarios:
        result = scorer.score_candidate(
            drop_bucket=s["drop_bucket"],
            stock_event_type=s["stock_event_type"],
            stock_event_severity=s["stock_event_severity"],
            market_stressed=s["market_stressed"],
        )

        entry = scorer.compute_entry_price(100.0, result["entry_offset"])

        print(f"Scenario: {s['name']}")
        print(f"  Success rate: {result['success_rate']*100:.1f}% (n={result['n_events']})")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Lookup level: {result['lookup_level']} ({result['lookup_description']})")
        print(f"  Entry price (if close=$100): ${entry}")
        print(f"  Avg days to hit: {result['avg_days_to_hit']}")
        print(f"  Avg max drawdown: {result['avg_max_drawdown']}")
        print()
