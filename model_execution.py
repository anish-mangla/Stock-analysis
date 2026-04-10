"""
model_execution.py

Executes AI model calls for the stock-news analysis workflow.

Version 2 design:
- Fully implements Claude via Anthropic SDK
- Uses Structured Outputs for schema-valid JSON
- Keeps web search enabled
- Returns:
    - raw response text (JSON string when structured output succeeds)
    - parsed structured JSON
    - metadata
    - error payload if anything fails

Environment variables expected:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY (optional for future use)

This module is designed to be called by main.py.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 6500
DEFAULT_WEB_SEARCH_MAX_USES = 40


def _now_utc_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _extract_text_from_claude_response(response: Any) -> str:
    """
    Extract readable text from Anthropic response content blocks.

    With structured outputs enabled, the response content should contain
    a text block holding the JSON result as a string.
    """
    text_parts = []

    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    return "\n".join(text_parts).strip()


def _best_effort_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON from returned text.
    """
    if not text:
        return None

    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _estimate_claude_cost_usd(
    input_tokens: int,
    output_tokens: int,
    search_uses: int = 0,
) -> float:
    """
    Rough cost estimate.

    Assumptions:
    - Claude Sonnet input:  $3 / 1M tokens
    - Claude Sonnet output: $15 / 1M tokens
    - Web search:           $10 / 1000 searches = $0.01 / search

    This is an estimate only.
    """
    input_cost = (input_tokens / 1_000_000) * 3.0
    output_cost = (output_tokens / 1_000_000) * 15.0
    search_cost = search_uses * 0.01
    return round(input_cost + output_cost + search_cost, 6)


def run_claude(
    prompt_bundle: Dict[str, Any],
    ticker_data: Dict[str, Any],
    model_name: str = DEFAULT_CLAUDE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    web_search_max_uses: int = DEFAULT_WEB_SEARCH_MAX_USES,
) -> Dict[str, Any]:
    """
    Run Claude with web search enabled and Structured Outputs enforced.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "model_name": model_name,
            "provider": "anthropic",
            "raw_text": "",
            "parsed_output": None,
            "metadata": {
                "created_at_utc": _now_utc_iso(),
            },
            "error": "Missing ANTHROPIC_API_KEY in environment.",
        }

    prompt_text = prompt_bundle["prompt_text"]
    output_schema = prompt_bundle["output_schema"]

    client = Anthropic(api_key=api_key)

    start_time = time.time()

    try:
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
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": output_schema,
                }
            },
        )

        elapsed_seconds = round(time.time() - start_time, 3)

        raw_text = _extract_text_from_claude_response(response)
        parsed_output = _best_effort_parse_json(raw_text)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return {
            "model_name": model_name,
            "provider": "anthropic",
            "raw_text": raw_text,
            "parsed_output": parsed_output,
            "metadata": {
                "created_at_utc": _now_utc_iso(),
                "latency_seconds": elapsed_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "web_search_max_uses": web_search_max_uses,
                "estimated_cost_usd": _estimate_claude_cost_usd(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    search_uses=web_search_max_uses,
                ),
                "prompt_version": prompt_bundle.get("prompt_version"),
                "ticker_count": len(ticker_data.get("tickers", [])),
                "structured_output_enabled": True,
            },
            "error": None if parsed_output is not None else (
                "Claude returned a response, but it could not be parsed as JSON."
            ),
        }

    except Exception as exc:
        return {
            "model_name": model_name,
            "provider": "anthropic",
            "raw_text": "",
            "parsed_output": None,
            "metadata": {
                "created_at_utc": _now_utc_iso(),
                "prompt_version": prompt_bundle.get("prompt_version"),
                "ticker_count": len(ticker_data.get("tickers", [])),
                "structured_output_enabled": True,
            },
            "error": f"Claude execution failed: {exc}",
        }


def run_openai(
    prompt_bundle: Dict[str, Any],
    ticker_data: Dict[str, Any],
    model_name: str = "openai-placeholder",
) -> Dict[str, Any]:
    """
    Placeholder for future OpenAI implementation.
    """
    return {
        "model_name": model_name,
        "provider": "openai",
        "raw_text": "",
        "parsed_output": None,
        "metadata": {
            "created_at_utc": _now_utc_iso(),
            "prompt_version": prompt_bundle.get("prompt_version"),
            "ticker_count": len(ticker_data.get("tickers", [])),
        },
        "error": "OpenAI execution not implemented yet.",
    }


def run_model(
    model_name: str,
    prompt_bundle: Dict[str, Any],
    ticker_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Dispatch to the correct model runner.
    """
    normalized = model_name.strip().lower()

    if normalized == "claude":
        return run_claude(prompt_bundle=prompt_bundle, ticker_data=ticker_data)

    if normalized == "openai":
        return run_openai(prompt_bundle=prompt_bundle, ticker_data=ticker_data)

    return {
        "model_name": model_name,
        "provider": "unknown",
        "raw_text": "",
        "parsed_output": None,
        "metadata": {
            "created_at_utc": _now_utc_iso(),
            "prompt_version": prompt_bundle.get("prompt_version"),
            "ticker_count": len(ticker_data.get("tickers", [])),
        },
        "error": f"Unsupported model_name: {model_name}",
    }


if __name__ == "__main__":
    from ticker_selection import get_selected_tickers
    from prompt_composition import compose_prompt

    ticker_data = get_selected_tickers()

    if not ticker_data["tickers"]:
        print("No selected tickers found. Model execution skipped.")
    else:
        prompt_bundle = compose_prompt(ticker_data)
        result = run_model("claude", prompt_bundle, ticker_data)

        print("Model Execution Result")
        print("-" * 80)
        print(f"Provider:    {result['provider']}")
        print(f"Model:       {result['model_name']}")
        print(f"Error:       {result['error']}")
        print(f"Latency:     {result['metadata'].get('latency_seconds')}")
        print(f"Input toks:  {result['metadata'].get('input_tokens')}")
        print(f"Output toks: {result['metadata'].get('output_tokens')}")
        print(f"Est. cost:   {result['metadata'].get('estimated_cost_usd')}")
        print()

        if result["parsed_output"] is not None:
            print("Parsed JSON preview:")
            print(json.dumps(result["parsed_output"], indent=2)[:3000])
        else:
            print("Raw text preview:")
            print(result["raw_text"][:3000] if result["raw_text"] else "[no text returned]")