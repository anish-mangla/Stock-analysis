#!/usr/bin/env python3

import ast
import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")
INPUT_CSV = OUTPUT_DIR / "events_with_market_context.csv"

FILTERED_BASE_CSV = OUTPUT_DIR / "events_filtered_market_rules.csv"
FILTERED_STRICT_CSV = OUTPUT_DIR / "events_filtered_market_rules_plus_good_tag.csv"
SUMMARY_CSV = OUTPUT_DIR / "final_market_filter_summary.csv"

BAD_TAGS = {"rates_up", "valuation_reset", "growth_fear"}
GOOD_TAGS = {"oversold_rebound", "liquidity_stress", "ai_leadership"}


def parse_tags(value) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    s = str(value).strip()
    if not s:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    s = s.strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


def summarize(df: pd.DataFrame, label: str) -> dict:
    out = {
        "variant": label,
        "total_events": len(df),
        "success_rate": df["success"].mean() if len(df) else None,
        "avg_final_return": df["final_return"].mean() if len(df) else None,
        "median_final_return": df["final_return"].median() if len(df) else None,
        "avg_max_drawdown": df["max_drawdown"].mean() if len(df) else None,
        "avg_days_to_hit": df["days_to_hit"].mean() if len(df) else None,
    }
    return out


def print_summary_table(summary_df: pd.DataFrame) -> None:
    display = summary_df.copy()
    pct_cols = ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]
    for col in pct_cols:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2)

    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)

    print("\n=== FINAL MARKET FILTER SUMMARY ===")
    print(display.to_string(index=False))


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("Input CSV is empty")

    df["primary_tags_list"] = df["primary_tags_list"].apply(parse_tags)

    df["has_bad_tag"] = df["primary_tags_list"].apply(lambda tags: any(tag in BAD_TAGS for tag in tags))
    df["has_good_tag"] = df["primary_tags_list"].apply(lambda tags: any(tag in GOOD_TAGS for tag in tags))

    # Variant 0: all matched events
    baseline = df.copy()

    # Variant 1: learned first-pass rule
    filtered_base = df[
        (df["risk_regime"] != "rotation") &
        (~df["has_bad_tag"])
    ].copy()

    # Variant 2: stricter version
    filtered_strict = df[
        (df["risk_regime"] != "rotation") &
        (~df["has_bad_tag"]) &
        (df["has_good_tag"])
    ].copy()

    summary_rows = [
        summarize(baseline, "baseline_all"),
        summarize(filtered_base, "no_rotation_no_bad_tags"),
        summarize(filtered_strict, "no_rotation_no_bad_tags_plus_good_tag"),
    ]
    summary_df = pd.DataFrame(summary_rows)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered_base.to_csv(FILTERED_BASE_CSV, index=False)
    filtered_strict.to_csv(FILTERED_STRICT_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # Print high-level counts
    print(f"Baseline events: {len(baseline)}")
    print(f"Filtered base events: {len(filtered_base)}")
    print(f"Filtered strict events: {len(filtered_strict)}")

    print(f"\nBad tag count: {int(df['has_bad_tag'].sum())}")
    print(f"Good tag count: {int(df['has_good_tag'].sum())}")
    print(f"Rotation count: {int((df['risk_regime'] == 'rotation').sum())}")

    print_summary_table(summary_df)

    # Optional extra: by drop bucket for the base filtered version
    if "drop_bucket_simple" not in filtered_base.columns and "drop_pct" in filtered_base.columns:
        filtered_base["drop_bucket_simple"] = pd.cut(
            filtered_base["drop_pct"],
            bins=[0, 0.05, 0.07, 0.10, 1],
            labels=["3-5%", "5-7%", "7-10%", "10%+"]
        )

    if "drop_bucket_simple" in filtered_base.columns:
        by_bucket = (
            filtered_base.groupby("drop_bucket_simple", observed=False)
            .agg(
                total_events=("ticker", "count"),
                success_rate=("success", "mean"),
                avg_final_return=("final_return", "mean"),
                avg_max_drawdown=("max_drawdown", "mean"),
            )
            .reset_index()
        )

        display = by_bucket.copy()
        for col in ["success_rate", "avg_final_return", "avg_max_drawdown"]:
            display[col] = (display[col] * 100).round(2)

        print("\n=== BASE FILTER BY DROP BUCKET ===")
        print(display.to_string(index=False))

    print(f"\nSaved base filtered data to:   {FILTERED_BASE_CSV}")
    print(f"Saved strict filtered data to: {FILTERED_STRICT_CSV}")
    print(f"Saved summary to:              {SUMMARY_CSV}")


if __name__ == "__main__":
    main()