"""
prompt_composition.py

Builds the research prompt for AI models from the selected ticker payload
returned by ticker_selection.py.

This module is model-agnostic. It does not call any model APIs directly.
Its only job is to transform structured ticker-selection output into a
prompt bundle that downstream model execution can use.

The prompt is designed for a stock-research workflow where the model must
assess whether recent price weakness is logically explained by fresh,
stock-specific negative information, or whether the move appears weakly
explained and therefore more likely sentiment-driven / temporarily oversold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PROMPT_VERSION = "v2"


def _now_utc_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_ticker_block(tickers: List[Dict[str, Any]]) -> str:
    """
    Format selected tickers into a readable block for the prompt.
    """
    lines: List[str] = []

    for item in tickers:
        ticker = item["ticker"]
        rank = item.get("rank", "?")
        tp_change_pct = item.get("tp_change_pct")
        tp_change_abs = item.get("tp_change_abs")
        last_date = item.get("last_date")

        pct_display = (
            f"{tp_change_pct:.2%}"
            if isinstance(tp_change_pct, (int, float))
            else str(tp_change_pct)
        )
        abs_display = (
            f"{tp_change_abs:.2f}"
            if isinstance(tp_change_abs, (int, float))
            else str(tp_change_abs)
        )

        lines.append(
            f"- Rank {rank}: {ticker} | 5-day TP change: {pct_display} "
            f"(abs: {abs_display}) | last date: {last_date}"
        )

    return "\n".join(lines)


def _build_research_prompt(
    tickers: List[Dict[str, Any]],
    selection_rule: Dict[str, Any],
    extra_instructions: Optional[str] = None,
) -> str:
    """
    Build the main natural-language prompt for the research task.
    """
    tickers_block = _format_ticker_block(tickers)
    threshold_display = selection_rule.get("threshold_pct_display", "N/A")

    prompt = f"""You are evaluating whether recent stock price weakness is logically explained by fresh, stock-specific negative information.

Use web search extensively.

The stocks below were pre-filtered because their 5-day Typical Price (TP) change was at or below {threshold_display}.

Typical Price formula:
TP = (High + Low + Close) / 3

Selected tickers:
{tickers_block}

Task:
For each ticker, search for the most important company-specific news from the last 3 to 5 calendar days and determine whether the recent price drop is logically explained by fresh negative fundamentals.

Rules:
1. Prioritize stock-specific news over broad market, index, macro, or sector-wide explanations.
2. Focus on materially meaningful developments such as:
   - earnings/guidance cuts
   - analyst downgrades tied to fundamentals
   - regulatory actions
   - litigation
   - product problems
   - management departures
   - financing stress
   - demand weakness
   - major contract loss
   - margin pressure
   - any clearly negative business update
3. Distinguish between:
   - strongly explained by fresh negative fundamentals
   - partially explained / mixed
   - weakly explained or not clearly explained
4. If there is weak or no meaningful stock-specific news, treat the move as more likely sentiment-driven, temporary, or a short-term mean-reversion candidate.
5. Be decisive and direct.
6. Analyze every ticker exactly once.
7. Every ticker must appear in exactly one of these category arrays:
   - clear_negative_recent_news
   - mixed_medium_negatives
   - mostly_neutral_or_weak_news
8. Also include one ticker_notes object for every ticker.
9. Return only the structured result matching the schema. Do not include markdown. Do not include prose before or after the result.

Field definitions:
- explanation_strength must be one of:
  - "strongly_explained"
  - "partially_explained"
  - "weakly_explained"
- likely_mean_reversion_candidate must be one of:
  - "yes"
  - "maybe"
  - "no"

When classifying:
- clear_negative_recent_news = strong, recent, stock-specific negative developments that plausibly explain the move
- mixed_medium_negatives = some negative information exists, but the causal link is incomplete, mixed, or only moderately convincing
- mostly_neutral_or_weak_news = no strong stock-specific new negative information found; move appears more sentiment-driven, temporary, or oversold
"""

    if extra_instructions:
        prompt += f"\nAdditional instructions:\n{extra_instructions.strip()}\n"

    return prompt.strip()


def _build_output_schema() -> Dict[str, Any]:
    """
    Define the target structured output schema in a model-agnostic way.
    """
    return {
        "type": "object",
        "properties": {
            "clear_negative_recent_news": {
                "type": "array",
                "items": {"type": "string"},
            },
            "mixed_medium_negatives": {
                "type": "array",
                "items": {"type": "string"},
            },
            "mostly_neutral_or_weak_news": {
                "type": "array",
                "items": {"type": "string"},
            },
            "ticker_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "recent_price_drop_pct": {"type": "number"},
                        "summary": {"type": "string"},
                        "strongest_negative_news": {"type": "string"},
                        "explanation_strength": {
                            "type": "string",
                            "enum": [
                                "strongly_explained",
                                "partially_explained",
                                "weakly_explained",
                            ],
                        },
                        "likely_mean_reversion_candidate": {
                            "type": "string",
                            "enum": ["yes", "maybe", "no"],
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "ticker",
                        "recent_price_drop_pct",
                        "summary",
                        "strongest_negative_news",
                        "explanation_strength",
                        "likely_mean_reversion_candidate",
                        "citations",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "clear_negative_recent_news",
            "mixed_medium_negatives",
            "mostly_neutral_or_weak_news",
            "ticker_notes",
        ],
        "additionalProperties": False,
    }


def compose_prompt(
    ticker_data: Dict[str, Any],
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compose the prompt bundle from ticker selection output.

    Args:
        ticker_data:
            Structured payload returned by ticker_selection.get_selected_tickers()
        extra_instructions:
            Optional additional instructions appended to the prompt.

    Returns:
        A structured prompt bundle for downstream model execution.
    """
    if not isinstance(ticker_data, dict):
        raise TypeError("ticker_data must be a dictionary.")

    tickers = ticker_data.get("tickers", [])
    selection_rule = ticker_data.get("selection_rule", {})

    if not tickers:
        raise ValueError("ticker_data contains no selected tickers.")

    prompt_text = _build_research_prompt(
        tickers=tickers,
        selection_rule=selection_rule,
        extra_instructions=extra_instructions,
    )

    prompt_bundle: Dict[str, Any] = {
        "component": "prompt_composition",
        "created_at_utc": _now_utc_iso(),
        "prompt_version": PROMPT_VERSION,
        "prompt_text": prompt_text,
        "output_schema": _build_output_schema(),
        "input_summary": {
            "ticker_count": len(tickers),
            "tickers": [item["ticker"] for item in tickers],
            "selection_threshold": selection_rule.get("threshold"),
            "selection_threshold_display": selection_rule.get(
                "threshold_pct_display"
            ),
        },
    }

    return prompt_bundle


if __name__ == "__main__":
    from ticker_selection import get_selected_tickers

    ticker_data = get_selected_tickers()
    if not ticker_data["tickers"]:
        print("No selected tickers found. Prompt not generated.")
    else:
        prompt_bundle = compose_prompt(ticker_data)

        print("Prompt Composition Result")
        print("-" * 80)
        print(f"Created at (UTC): {prompt_bundle['created_at_utc']}")
        print(f"Prompt version:   {prompt_bundle['prompt_version']}")
        print(f"Ticker count:     {prompt_bundle['input_summary']['ticker_count']}")
        print()
        print(prompt_bundle["prompt_text"])