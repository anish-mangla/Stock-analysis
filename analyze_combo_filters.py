#!/usr/bin/env python3
"""
analyze_combo_filters.py

Reads saved CSV outputs from bounce_backtest.py and learns:
- GOOD ticker + drop-bucket combos
- BAD ticker + drop-bucket combos

Then compares filtered strategy views:
- all events
- good combos only
- everything except bad combos
- neutral combos only

Expected input files:
- outputs/events.csv
- outputs/summary_by_ticker_and_bucket.csv

Outputs:
- outputs/learned_combo_labels.csv
- outputs/summary_filtered_views.csv
"""

from __future__ import annotations

import os
from typing import Set, Tuple

import pandas as pd


# ============================================================
# 1) CONFIG
# ============================================================

OUTPUT_DIR = "outputs"

EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
TICKER_BUCKET_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_by_ticker_and_bucket.csv")

LEARNED_COMBO_LABELS_CSV = os.path.join(OUTPUT_DIR, "learned_combo_labels.csv")
FILTERED_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_filtered_views.csv")
FILTERED_EVENTS_CSV = os.path.join(OUTPUT_DIR, "events_with_combo_labels.csv")

GOOD_COMBO_MIN_EVENTS = 10
GOOD_COMBO_MIN_SUCCESS_RATE = 0.85
GOOD_COMBO_MIN_AVG_FINAL_RETURN = 0.0

BAD_COMBO_MIN_EVENTS = 8
BAD_COMBO_MAX_SUCCESS_RATE = 0.65
BAD_COMBO_MAX_AVG_FINAL_RETURN = 0.0

TOP_N_TO_PRINT = 25


# ============================================================
# 2) LOADING
# ============================================================

def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(EVENTS_CSV):
        raise FileNotFoundError(f"Missing required file: {EVENTS_CSV}")

    if not os.path.exists(TICKER_BUCKET_SUMMARY_CSV):
        raise FileNotFoundError(f"Missing required file: {TICKER_BUCKET_SUMMARY_CSV}")

    events_df = pd.read_csv(EVENTS_CSV)
    combo_summary_df = pd.read_csv(TICKER_BUCKET_SUMMARY_CSV)

    if events_df.empty:
        raise ValueError("events.csv is empty.")

    if combo_summary_df.empty:
        raise ValueError("summary_by_ticker_and_bucket.csv is empty.")

    return events_df, combo_summary_df


# ============================================================
# 3) CLASSIFICATION
# ============================================================

def classify_combos(
    combo_summary_df: pd.DataFrame,
) -> tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], pd.DataFrame]:
    """
    Learn GOOD and BAD combos from ticker+bucket summary.

    GOOD combo:
    - enough events
    - high success rate
    - positive average final return

    BAD combo:
    - enough events
    - low success rate
    - negative average final return
    """
    df = combo_summary_df.copy()

    df["drop_bucket"] = df["drop_bucket"].astype(str)

    good_mask = (
        (df["total_events"] >= GOOD_COMBO_MIN_EVENTS)
        & (df["success_rate"] >= GOOD_COMBO_MIN_SUCCESS_RATE)
        & (df["avg_final_return"] > GOOD_COMBO_MIN_AVG_FINAL_RETURN)
    )

    bad_mask = (
        (df["total_events"] >= BAD_COMBO_MIN_EVENTS)
        & (df["success_rate"] <= BAD_COMBO_MAX_SUCCESS_RATE)
        & (df["avg_final_return"] < BAD_COMBO_MAX_AVG_FINAL_RETURN)
    )

    good_combos = set(zip(df.loc[good_mask, "ticker"], df.loc[good_mask, "drop_bucket"]))
    bad_combos = set(zip(df.loc[bad_mask, "ticker"], df.loc[bad_mask, "drop_bucket"]))

    labeled = df.copy()
    labeled["combo_label"] = "neutral"
    labeled.loc[good_mask, "combo_label"] = "good"
    labeled.loc[bad_mask, "combo_label"] = "bad"

    # Be conservative if anything overlaps
    overlap_mask = good_mask & bad_mask
    labeled.loc[overlap_mask, "combo_label"] = "bad"

    return good_combos, bad_combos, labeled


# ============================================================
# 4) APPLY LABELS TO EVENT-LEVEL DATA
# ============================================================

def apply_combo_labels(
    events_df: pd.DataFrame,
    good_combos: Set[Tuple[str, str]],
    bad_combos: Set[Tuple[str, str]],
) -> pd.DataFrame:
    df = events_df.copy()
    df["drop_bucket"] = df["drop_bucket"].astype(str)

    combo_keys = list(zip(df["ticker"], df["drop_bucket"]))

    df["is_good_combo"] = [key in good_combos for key in combo_keys]
    df["is_bad_combo"] = [key in bad_combos for key in combo_keys]

    def label_row(is_good: bool, is_bad: bool) -> str:
        if is_bad:
            return "bad"
        if is_good:
            return "good"
        return "neutral"

    df["combo_label"] = [
        label_row(good, bad)
        for good, bad in zip(df["is_good_combo"], df["is_bad_combo"])
    ]

    return df


# ============================================================
# 5) FILTERED VIEW SUMMARY
# ============================================================

def summarize_filtered_views(events_df: pd.DataFrame) -> pd.DataFrame:
    views = {
        "all_events": events_df,
        "good_only": events_df[events_df["combo_label"] == "good"].copy(),
        "exclude_bad": events_df[events_df["combo_label"] != "bad"].copy(),
        "neutral_only": events_df[events_df["combo_label"] == "neutral"].copy(),
        "bad_only": events_df[events_df["combo_label"] == "bad"].copy(),
    }

    rows = []

    for view_name, df_view in views.items():
        if df_view.empty:
            rows.append(
                {
                    "view": view_name,
                    "total_events": 0,
                    "success_rate": None,
                    "avg_days_to_hit": None,
                    "avg_final_return": None,
                    "avg_max_drawdown": None,
                    "failure_rate": None,
                }
            )
            continue

        valid_days = df_view["days_to_hit"].dropna()

        success_rate = float(df_view["success"].mean())

        rows.append(
            {
                "view": view_name,
                "total_events": int(len(df_view)),
                "success_rate": success_rate,
                "avg_days_to_hit": float(valid_days.mean()) if not valid_days.empty else None,
                "avg_final_return": float(df_view["final_return"].mean()),
                "avg_max_drawdown": float(df_view["max_drawdown"].mean()),
                "failure_rate": 1.0 - success_rate,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 6) PRINTING
# ============================================================

def _format_pct_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = (out[col] * 100).round(2)
    return out


def print_combo_lists(labeled_combos_df: pd.DataFrame) -> None:
    print("\n=== LEARNED COMBO COUNTS ===")
    print(f"Good combos: {(labeled_combos_df['combo_label'] == 'good').sum()}")
    print(f"Bad combos:  {(labeled_combos_df['combo_label'] == 'bad').sum()}")

    cols = [
        "ticker",
        "drop_bucket",
        "total_events",
        "success_rate",
        "failure_rate",
        "avg_final_return",
        "avg_max_drawdown",
        "combo_label",
    ]

    display = _format_pct_columns(
        labeled_combos_df,
        ["success_rate", "failure_rate", "avg_final_return", "avg_max_drawdown"],
    )

    good_df = display[display["combo_label"] == "good"].sort_values(
        by=["success_rate", "avg_final_return"],
        ascending=[False, False],
    )

    bad_df = display[display["combo_label"] == "bad"].sort_values(
        by=["success_rate", "avg_final_return"],
        ascending=[True, True],
    )

    if not good_df.empty:
        print(f"\n=== GOOD COMBOS (top {TOP_N_TO_PRINT}) ===")
        print(good_df[cols].head(TOP_N_TO_PRINT).to_string(index=False))

    if not bad_df.empty:
        print(f"\n=== BAD COMBOS (top {TOP_N_TO_PRINT}) ===")
        print(bad_df[cols].head(TOP_N_TO_PRINT).to_string(index=False))


def print_filtered_view_summary(filtered_summary_df: pd.DataFrame) -> None:
    display = _format_pct_columns(
        filtered_summary_df,
        ["success_rate", "failure_rate", "avg_final_return", "avg_max_drawdown"],
    )

    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)

    print("\n=== FILTERED VIEW SUMMARY ===")
    print(display.to_string(index=False))


# ============================================================
# 7) SAVE
# ============================================================

def save_outputs(
    labeled_events_df: pd.DataFrame,
    labeled_combos_df: pd.DataFrame,
    filtered_summary_df: pd.DataFrame,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labeled_events_df.to_csv(FILTERED_EVENTS_CSV, index=False)
    labeled_combos_df.to_csv(LEARNED_COMBO_LABELS_CSV, index=False)
    filtered_summary_df.to_csv(FILTERED_SUMMARY_CSV, index=False)

    print(f"\nSaved event-level combo labels to: {FILTERED_EVENTS_CSV}")
    print(f"Saved learned combo labels to:     {LEARNED_COMBO_LABELS_CSV}")
    print(f"Saved filtered summary to:         {FILTERED_SUMMARY_CSV}")


# ============================================================
# 8) MAIN
# ============================================================

def main() -> None:
    events_df, combo_summary_df = load_csvs()

    good_combos, bad_combos, labeled_combos_df = classify_combos(combo_summary_df)
    labeled_events_df = apply_combo_labels(events_df, good_combos, bad_combos)
    filtered_summary_df = summarize_filtered_views(labeled_events_df)

    print_combo_lists(labeled_combos_df)
    print_filtered_view_summary(filtered_summary_df)

    save_outputs(labeled_events_df, labeled_combos_df, filtered_summary_df)


if __name__ == "__main__":
    main()