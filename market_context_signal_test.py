#!/usr/bin/env python3

import json
import os
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")
EVENTS_CSV = OUTPUT_DIR / "events.csv"
MARKET_CONTEXT_CSV = OUTPUT_DIR / "market_weekly_context.csv"

JOINED_CSV = OUTPUT_DIR / "events_with_market_context.csv"
SUMMARY_BY_REGIME_CSV = OUTPUT_DIR / "market_context_summary_by_regime.csv"
SUMMARY_BY_BOUNCE_SCORE_CSV = OUTPUT_DIR / "market_context_summary_by_bounce_score.csv"
SUMMARY_BY_TAG_CSV = OUTPUT_DIR / "market_context_summary_by_tag.csv"


REQUIRED_MARKET_COLUMNS = [
    "week_start",
    "week_end",
    "risk_regime",
    "risk_regime_strength",
    "bounce_friendly_regime_score",
    "primary_tags",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"Missing {EVENTS_CSV}")
    if not MARKET_CONTEXT_CSV.exists():
        raise FileNotFoundError(f"Missing {MARKET_CONTEXT_CSV}")

    events = pd.read_csv(EVENTS_CSV)
    market = pd.read_csv(MARKET_CONTEXT_CSV)

    if events.empty:
        raise ValueError("events.csv is empty")
    if market.empty:
        raise ValueError("market_weekly_context.csv is empty")

    missing = [c for c in REQUIRED_MARKET_COLUMNS if c not in market.columns]
    if missing:
        raise ValueError(f"market_weekly_context.csv missing columns: {missing}")

    events["event_date"] = pd.to_datetime(events["event_date"])
    market["week_start"] = pd.to_datetime(market["week_start"])
    market["week_end"] = pd.to_datetime(market["week_end"])

    return events, market


def normalize_tags(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    s = str(value).strip()
    if not s:
        return []

    # Try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Fallback: split on comma
    s = s.strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


def attach_market_context(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market_sorted = market.sort_values(["week_start", "week_end"]).reset_index(drop=True)

    joined_rows = []

    for _, event_row in events.iterrows():
        event_date = event_row["event_date"]

        matches = market_sorted[
            (market_sorted["week_start"] <= event_date)
            & (market_sorted["week_end"] >= event_date)
        ]

        if matches.empty:
            continue

        market_row = matches.iloc[0]
        combined = {**event_row.to_dict(), **market_row.to_dict()}
        joined_rows.append(combined)

    if not joined_rows:
        raise ValueError("No event rows matched any market week range")

    joined = pd.DataFrame(joined_rows)
    joined["primary_tags_list"] = joined["primary_tags"].apply(normalize_tags)

    return joined


def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("risk_regime", dropna=False)
        .agg(
            total_events=("ticker", "count"),
            success_rate=("success", "mean"),
            avg_final_return=("final_return", "mean"),
            median_final_return=("final_return", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_days_to_hit=("days_to_hit", "mean"),
            avg_bounce_score=("bounce_friendly_regime_score", "mean"),
        )
        .reset_index()
        .sort_values("success_rate", ascending=False)
    )
    return out


def summarize_by_bounce_score(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-5.1, -3, -1, 1, 3, 5.1]
    labels = ["very_bad", "bad", "neutral", "good", "very_good"]

    temp = df.copy()
    temp["bounce_score_bucket"] = pd.cut(
        temp["bounce_friendly_regime_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    out = (
        temp.groupby("bounce_score_bucket", dropna=False, observed=False)
        .agg(
            total_events=("ticker", "count"),
            success_rate=("success", "mean"),
            avg_final_return=("final_return", "mean"),
            median_final_return=("final_return", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_days_to_hit=("days_to_hit", "mean"),
        )
        .reset_index()
    )
    return out


def summarize_by_tag(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        tags = row["primary_tags_list"]
        for tag in tags:
            rows.append(
                {
                    "tag": tag,
                    "success": row["success"],
                    "final_return": row["final_return"],
                    "max_drawdown": row["max_drawdown"],
                    "days_to_hit": row["days_to_hit"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=[
            "tag", "total_events", "success_rate",
            "avg_final_return", "median_final_return",
            "avg_max_drawdown", "avg_days_to_hit"
        ])

    tag_df = pd.DataFrame(rows)

    out = (
        tag_df.groupby("tag", dropna=False)
        .agg(
            total_events=("tag", "count"),
            success_rate=("success", "mean"),
            avg_final_return=("final_return", "mean"),
            median_final_return=("final_return", "median"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_days_to_hit=("days_to_hit", "mean"),
        )
        .reset_index()
        .sort_values(["total_events", "success_rate"], ascending=[False, False])
    )

    return out


def print_summary(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("No rows")
        return

    display = df.copy()
    pct_cols = ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]
    for col in pct_cols:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2)

    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    if "avg_bounce_score" in display.columns:
        display["avg_bounce_score"] = display["avg_bounce_score"].round(2)

    print(display.to_string(index=False))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events, market = load_data()
    joined = attach_market_context(events, market)

    by_regime = summarize_by_regime(joined)
    by_bounce = summarize_by_bounce_score(joined)
    by_tag = summarize_by_tag(joined)

    joined.to_csv(JOINED_CSV, index=False)
    by_regime.to_csv(SUMMARY_BY_REGIME_CSV, index=False)
    by_bounce.to_csv(SUMMARY_BY_BOUNCE_SCORE_CSV, index=False)
    by_tag.to_csv(SUMMARY_BY_TAG_CSV, index=False)

    print(f"Matched events: {len(joined)}")
    print_summary("SUMMARY BY RISK REGIME", by_regime)
    print_summary("SUMMARY BY BOUNCE SCORE", by_bounce)
    print_summary("TOP TAGS BY EVENT COUNT", by_tag.head(20))

    print(f"\nSaved joined data to: {JOINED_CSV}")
    print(f"Saved regime summary to: {SUMMARY_BY_REGIME_CSV}")
    print(f"Saved bounce score summary to: {SUMMARY_BY_BOUNCE_SCORE_CSV}")
    print(f"Saved tag summary to: {SUMMARY_BY_TAG_CSV}")


if __name__ == "__main__":
    main()