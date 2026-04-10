"""
response_storage.py

Stores pipeline outputs to disk in a simple, auditable folder structure.

For each daily run, files are saved under:

daily_runs/
  YYYY-MM-DD/
    input_tickers.json
    prompt.txt
    <model_name>/
      raw.txt
      parsed.json
      metadata.json
      snapshot.json

This module is intentionally file-based and simple so that:
- raw model outputs are preserved exactly
- parsed outputs are easy to inspect later
- future evaluation/backtesting can enrich these records
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


BASE_OUTPUT_DIR = Path("daily_runs")


def _now_utc_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_run_date_str(ticker_data: Dict[str, Any]) -> str:
    """
    Determine the run date folder name.

    Priority:
    1. Date portion of ticker_data['run_timestamp_utc'] if present
    2. Current UTC date
    """
    run_timestamp = ticker_data.get("run_timestamp_utc")
    if isinstance(run_timestamp, str) and len(run_timestamp) >= 10:
        return run_timestamp[:10]
    return datetime.now(timezone.utc).date().isoformat()


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    """Write plain text file using UTF-8."""
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON file using UTF-8 with stable pretty formatting."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )


def save_run_outputs(
    model_name: str,
    ticker_data: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
    model_result: Dict[str, Any],
) -> Dict[str, str]:
    """
    Save all outputs for a single model execution.

    Args:
        model_name:
            Logical model label used by main.py, e.g. "claude"
        ticker_data:
            Structured output from ticker_selection.get_selected_tickers()
        prompt_bundle:
            Structured output from prompt_composition.compose_prompt()
        model_result:
            Structured output from model_execution.run_model()

    Returns:
        Dictionary of saved file paths as strings.
    """
    run_date = _get_run_date_str(ticker_data)

    run_dir = BASE_OUTPUT_DIR / run_date
    model_dir = run_dir / model_name

    _ensure_dir(run_dir)
    _ensure_dir(model_dir)

    input_tickers_path = run_dir / "input_tickers.json"
    prompt_txt_path = run_dir / "prompt.txt"
    raw_txt_path = model_dir / "raw.txt"
    parsed_json_path = model_dir / "parsed.json"
    metadata_json_path = model_dir / "metadata.json"
    snapshot_json_path = model_dir / "snapshot.json"

    # Save shared run-level files once per run. Overwriting is fine because
    # they should be identical for all models in the same run.
    _write_json(input_tickers_path, ticker_data)
    _write_text(prompt_txt_path, prompt_bundle.get("prompt_text", ""))

    # Save raw response exactly as returned.
    _write_text(raw_txt_path, model_result.get("raw_text", ""))

    # Save parsed output if available, else save a minimal placeholder object.
    parsed_output = model_result.get("parsed_output")
    if isinstance(parsed_output, dict):
        _write_json(parsed_json_path, parsed_output)
    else:
        _write_json(
            parsed_json_path,
            {
                "parsed_output": None,
                "note": "No structured JSON could be parsed from raw_text.",
            },
        )

    # Save metadata.
    metadata_payload = {
        "saved_at_utc": _now_utc_iso(),
        "model_name": model_result.get("model_name", model_name),
        "provider": model_result.get("provider"),
        "error": model_result.get("error"),
        "metadata": model_result.get("metadata", {}),
        "prompt_version": prompt_bundle.get("prompt_version"),
        "ticker_count": len(ticker_data.get("tickers", [])),
    }
    _write_json(metadata_json_path, metadata_payload)

    # Save a full snapshot for convenience and future debugging.
    snapshot_payload = {
        "saved_at_utc": _now_utc_iso(),
        "model_name": model_name,
        "ticker_data": ticker_data,
        "prompt_bundle": {
            "prompt_version": prompt_bundle.get("prompt_version"),
            "created_at_utc": prompt_bundle.get("created_at_utc"),
            "input_summary": prompt_bundle.get("input_summary"),
            "prompt_text": prompt_bundle.get("prompt_text"),
            "output_schema": prompt_bundle.get("output_schema"),
        },
        "model_result": model_result,
    }
    _write_json(snapshot_json_path, snapshot_payload)

    return {
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "input_tickers_path": str(input_tickers_path),
        "prompt_txt_path": str(prompt_txt_path),
        "raw_txt_path": str(raw_txt_path),
        "parsed_json_path": str(parsed_json_path),
        "metadata_json_path": str(metadata_json_path),
        "snapshot_json_path": str(snapshot_json_path),
    }


if __name__ == "__main__":
    # Lightweight self-test using mock data so this file can be checked
    # independently without calling external APIs.
    mock_ticker_data = {
        "run_timestamp_utc": "2026-04-09T20:00:00+00:00",
        "tickers": [
            {
                "ticker": "AAPL",
                "tp_change_abs": -5.12,
                "tp_change_pct": -0.031,
                "last_date": "2026-04-09",
                "rank": 1,
            }
        ],
    }

    mock_prompt_bundle = {
        "prompt_version": "v1",
        "created_at_utc": "2026-04-09T20:01:00+00:00",
        "input_summary": {"ticker_count": 1, "tickers": ["AAPL"]},
        "prompt_text": "Mock prompt text",
        "output_schema": {"type": "object"},
    }

    mock_model_result = {
        "model_name": "claude-sonnet-test",
        "provider": "anthropic",
        "raw_text": "Mock raw output from model",
        "parsed_output": {
            "grouped_summary": {
                "clear_negative_recent_news": [],
                "mixed_medium_negatives": [],
                "mostly_neutral_or_weak_news": ["AAPL"],
            },
            "per_ticker": [
                {
                    "ticker": "AAPL",
                    "recent_price_drop_pct": -0.031,
                    "summary": "No strong stock-specific negative news found.",
                    "strongest_negative_news": "",
                    "explanation_strength": "weakly_explained",
                    "likely_mean_reversion_candidate": "yes",
                    "confidence": "medium",
                    "citations": [],
                }
            ],
        },
        "metadata": {
            "created_at_utc": "2026-04-09T20:02:00+00:00",
            "latency_seconds": 12.4,
            "input_tokens": 1234,
            "output_tokens": 567,
            "estimated_cost_usd": 0.42,
        },
        "error": None,
    }

    saved = save_run_outputs(
        model_name="claude",
        ticker_data=mock_ticker_data,
        prompt_bundle=mock_prompt_bundle,
        model_result=mock_model_result,
    )

    print("Response Storage Test Complete")
    print("-" * 80)
    for key, value in saved.items():
        print(f"{key}: {value}")