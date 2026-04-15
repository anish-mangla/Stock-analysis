#!/usr/bin/env python3
"""
news_classifier.py

Sends candidates to Claude for classification using our event taxonomy.

Claude's job is CLASSIFICATION, not judgment. It identifies what TYPE of event
caused the drop and how SEVERE it is. The system then looks up the historical
success rate for that classification.

Uses Claude with web search to research recent news for each candidate.
Returns structured classifications matching our 11 stock-level event types.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_WEB_SEARCH_MAX_USES = 20

TAXONOMY_PROMPT = """You are classifying stock price drops for a mean-reversion trading system.

For each stock below, search the web for what caused the recent price decline, then classify it into EXACTLY ONE of these event types:

EVENT TYPES:
1. EARNINGS_MISS — reported EPS or revenue below consensus
2. GUIDANCE_CUT — company lowered forward outlook or gave weak forecast
3. ANALYST_DOWNGRADE — Wall Street firms lowered rating or price target
4. PRODUCT_SERVICE_FAILURE — something went wrong with a product, drug trial, safety issue, etc.
5. REGULATORY_LEGAL — government action, lawsuit, investigation, or regulatory ruling against the company
6. MANAGEMENT_CHANGE — key executive departure, shakeup, or distraction (e.g., CEO controversy)
7. COMPETITIVE_THREAT — new competitor, disruption narrative, or loss of competitive position
8. CAPITAL_STRUCTURE — secondary offering, dividend cut, debt issues, M&A deal backlash
9. INSIDER_SELLING — large insider sales or institutional exits
10. DEMAND_WEAKNESS — company-specific signs of weakening demand, soft orders, customer pullback
11. NO_CLEAR_CATALYST — no obvious stock-specific news found; drop appears sentiment/market-driven

SEVERITY (for each classification):
- LOW: minor or speculative impact
- MEDIUM: meaningful but not existential
- HIGH: major material impact on the business

RULES:
- Search the web for each stock to find what actually happened
- Use the MOST RECENT news (last 3-5 days) that explains the drop
- If the drop was caused by broad market/sector selling with no stock-specific news, use NO_CLEAR_CATALYST
- Be honest — if you can't find a clear cause, say so rather than guessing
- Return ONLY valid JSON matching the schema below
"""


def _build_output_schema() -> Dict[str, Any]:
    """JSON schema for Claude's structured output."""
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "event_type_id": {"type": "integer"},
                        "event_type_name": {
                            "type": "string",
                            "enum": [
                                "earnings_miss", "guidance_cut", "analyst_downgrade",
                                "product_service_failure", "regulatory_legal",
                                "management_change", "competitive_threat",
                                "capital_structure", "insider_selling",
                                "demand_weakness", "no_clear_catalyst",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["ticker", "event_type_id", "event_type_name", "severity", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }


def _build_candidate_block(candidates: List[Dict[str, Any]], include_context: bool = True) -> str:
    """Format candidates into a readable block for the prompt."""
    lines = []
    for c in candidates:
        line = (
            f"- {c['ticker']}: dropped {c['drop_magnitude']*100:.1f}% "
            f"(bucket: {c['drop_bucket']}, trigger: {c['trigger_type']})"
        )
        if include_context:
            line += (
                f"\n  Sector: {c['sector_etf']}, "
                f"Drop vs vol: {c['drop_vs_vol20']:.1f}x, "
                f"5d return: {c['stock_return_5d']*100:.1f}%, "
                f"20d return: {c['stock_return_20d']*100:.1f}%, "
                f"Close in range: {c['close_in_range']:.2f}"
            )
        lines.append(line)
    return "\n".join(lines)


def _build_market_context_block(market_context: Dict[str, Any]) -> str:
    """Format market context into a readable block."""
    parts = []
    if market_context.get("vix_level"):
        parts.append(f"VIX: {market_context['vix_level']:.1f}")
    if market_context.get("spy_daily_return") is not None:
        parts.append(f"SPY today: {market_context['spy_daily_return']*100:+.2f}%")
    if market_context.get("stress_level"):
        parts.append(f"Market stress: {market_context['stress_level']}")
    return ", ".join(parts) if parts else "No market context available"


def _classify_batch(
    candidates: List[Dict[str, Any]],
    market_context: Optional[Dict[str, Any]],
    include_context: bool,
    model_name: str,
    max_tokens: int,
    web_search_max_uses: int,
) -> Dict[str, Any]:
    """Classify a single batch of candidates via Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    candidate_block = _build_candidate_block(candidates, include_context)
    prompt = TAXONOMY_PROMPT + "\n\nSTOCKS TO CLASSIFY:\n" + candidate_block

    if market_context and include_context:
        context_block = _build_market_context_block(market_context)
        prompt += f"\n\nMARKET CONTEXT TODAY: {context_block}"

    prompt += "\n\nClassify each stock. Return ONLY the JSON."

    start_time = time.time()

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": web_search_max_uses,
            }
        ],
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _build_output_schema(),
            }
        },
    )

    elapsed = round(time.time() - start_time, 3)

    raw_text = ""
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            raw_text += block.text

    parsed = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw_text[start:end+1])
            except json.JSONDecodeError:
                pass

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    return {
        "classifications": parsed.get("classifications", []) if parsed else [],
        "raw_text": raw_text,
        "latency_seconds": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": None if parsed else "Failed to parse JSON response",
    }


# Max candidates per Claude call to stay under token limits
# With web search enabled, each candidate generates ~10K tokens of search results
BATCH_SIZE = 8


def classify_candidates(
    candidates: List[Dict[str, Any]],
    market_context: Optional[Dict[str, Any]] = None,
    include_context: bool = True,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    web_search_max_uses: int = DEFAULT_WEB_SEARCH_MAX_USES,
) -> Dict[str, Any]:
    """
    Send candidates to Claude for classification, batching to stay under token limits.

    Args:
        candidates: list of candidate dicts from candidate_scanner
        market_context: optional dict with VIX, SPY return, stress level
        include_context: whether to include quantitative context in the prompt
        model_name: Claude model to use
        max_tokens: max output tokens
        web_search_max_uses: max web searches Claude can perform per batch

    Returns:
        Dict with classifications, metadata, and any errors
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "classifications": [],
            "classification_map": {},
            "metadata": {"error": "Missing ANTHROPIC_API_KEY"},
        }

    if not candidates:
        return {
            "classifications": [],
            "classification_map": {},
            "metadata": {"error": "No candidates to classify"},
        }

    # Split into batches
    batches = [candidates[i:i+BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]

    all_classifications = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0
    errors = []

    for batch_idx, batch in enumerate(batches):
        # Rate limit: wait between batches to avoid hitting token/minute limits
        if batch_idx > 0:
            wait_seconds = 65  # Claude rate limits are per-minute
            print(f"    Waiting {wait_seconds}s for rate limit cooldown...")
            time.sleep(wait_seconds)

        print(f"    Batch {batch_idx+1}/{len(batches)}: {len(batch)} candidates "
              f"({', '.join(c['ticker'] for c in batch)})")

        try:
            result = _classify_batch(
                candidates=batch,
                market_context=market_context,
                include_context=include_context,
                model_name=model_name,
                max_tokens=max_tokens,
                web_search_max_uses=web_search_max_uses,
            )

            all_classifications.extend(result["classifications"])
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            total_latency += result["latency_seconds"]

            if result["error"]:
                errors.append(f"Batch {batch_idx+1}: {result['error']}")

            print(f"      → {len(result['classifications'])} classified, "
                  f"{result['latency_seconds']:.1f}s, "
                  f"{result['input_tokens']} in / {result['output_tokens']} out tokens")

        except Exception as exc:
            errors.append(f"Batch {batch_idx+1}: {exc}")
            print(f"      → ERROR: {exc}")

    # Build ticker -> classification map
    classification_map = {}
    for c in all_classifications:
        classification_map[c["ticker"]] = c

    return {
        "classifications": all_classifications,
        "classification_map": classification_map,
        "metadata": {
            "model": model_name,
            "total_latency_seconds": round(total_latency, 3),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "web_search_max_uses_per_batch": web_search_max_uses,
            "candidates_sent": len(candidates),
            "batches": len(batches),
            "batch_size": BATCH_SIZE,
            "classifications_received": len(all_classifications),
            "include_context": include_context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": errors if errors else None,
        },
    }


if __name__ == "__main__":
    # Test with mock candidates (doesn't call Claude)
    mock_candidates = [
        {
            "ticker": "AAPL",
            "drop_magnitude": 0.05,
            "drop_bucket": "3-5%",
            "trigger_type": "drawdown",
            "sector_etf": "XLK",
            "drop_vs_vol20": 2.1,
            "stock_return_5d": -0.06,
            "stock_return_20d": -0.08,
            "close_in_range": 0.3,
        },
    ]

    print("News Classifier Module")
    print("=" * 50)
    print(f"Taxonomy: 11 event types")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Web search: up to {DEFAULT_WEB_SEARCH_MAX_USES} searches per call")
    print()
    print("Sample prompt block:")
    print(_build_candidate_block(mock_candidates))
    print()
    print("To run live classification, call classify_candidates() with real candidates.")
