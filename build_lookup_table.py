#!/usr/bin/env python3
"""
build_lookup_table.py

Step 5 of the historical labeling process — THE PAYOFF.

Joins all labels (market, sector, stock) to event outcomes and computes
historical success rates by event type, severity, and combinations.

This produces the objective lookup table that converts any classification
into a historical bounce probability.

Inputs:
- outputs/events.csv (7,831 events with outcomes)
- outputs/market_level_labels.csv (879 market labels)
- outputs/sector_level_labels.csv (7,181 sector labels)
- outputs/stock_level_labels.csv (336 stock labels)

Outputs:
- outputs/lookup_by_stock_event_type.csv
- outputs/lookup_by_market_context.csv
- outputs/lookup_by_sector_context.csv
- outputs/lookup_combined.csv
- outputs/events_fully_labeled.csv (master file with all labels joined)
"""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd


OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
MARKET_LABELS_CSV = os.path.join(OUTPUT_DIR, "market_level_labels.csv")
SECTOR_LABELS_CSV = os.path.join(OUTPUT_DIR, "sector_level_labels.csv")
STOCK_LABELS_CSV = os.path.join(OUTPUT_DIR, "stock_level_labels.csv")

# Ticker to sector ETF mapping
TICKER_TO_SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "AMAT": "SMH", "LRCX": "SMH", "QCOM": "SMH", "TXN": "SMH", "MU": "SMH",
    "INTC": "SMH", "NOW": "XLK", "INTU": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "IBM": "XLK", "ACN": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "AMZN": "XLY", "META": "XLC", "NFLX": "XLC", "CMCSA": "XLC", "TMUS": "XLC",
    "T": "XLC", "VZ": "XLC", "DIS": "XLC", "UBER": "XLY", "TSLA": "XLY",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF",
    "MS": "XLF", "AXP": "XLF", "BLK": "XLF", "USB": "XLF", "SCHW": "XLF",
    "LLY": "XLV", "PFE": "XLV", "BMY": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "TMO": "XLV", "DHR": "XLV", "ABT": "XLV", "MDT": "XLV", "GILD": "XLV",
    "ISRG": "XLV", "UNH": "XLV", "CVS": "XLV", "AMGN": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    "CAT": "XLI", "DE": "XLI", "HON": "XLI", "GE": "XLI", "GD": "XLI",
    "LMT": "XLI", "UNP": "XLI", "UPS": "XLI", "RTX": "XLI", "BA": "XLI",
    "HD": "XLY", "LOW": "XLY", "MCD": "XLY", "NKE": "XLY", "SBUX": "XLY",
    "BKNG": "XLY", "COST": "XLP", "WMT": "XLP", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "PM": "XLP", "MDLZ": "XLP", "CL": "XLP",
    "DUK": "XLU", "NEE": "XLU", "SO": "XLU", "AMT": "XLRE",
    "SPG": "XLRE", "LIN": "XLB", "PLTR": "XLK", "GEV": "XLI",
    "MO": "XLP", "MMM": "XLI", "EMR": "XLI", "FDX": "XLI",
    "COF": "XLF", "BK": "XLF",
}


def load_data():
    """Load all datasets."""
    events = pd.read_csv(EVENTS_CSV)
    events["event_date"] = pd.to_datetime(events["event_date"])

    market_labels = pd.read_csv(MARKET_LABELS_CSV)
    market_labels["date"] = pd.to_datetime(market_labels["date"])

    sector_labels = pd.read_csv(SECTOR_LABELS_CSV)
    sector_labels["date"] = pd.to_datetime(sector_labels["date"])

    stock_labels = pd.read_csv(STOCK_LABELS_CSV)
    stock_labels["date"] = pd.to_datetime(stock_labels["date"])

    # Add sector ETF to events
    events["sector_etf"] = events["ticker"].map(TICKER_TO_SECTOR_ETF).fillna("SPY")

    return events, market_labels, sector_labels, stock_labels


def build_market_context_per_event(events: pd.DataFrame, market_labels: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, summarize the market-level context on that date.
    Creates columns like: has_tariff_event, tariff_severity, has_liquidity_stress, etc.
    """
    out = events.copy()

    # Pivot market labels: for each date, what types are active and at what max severity
    severity_rank = {"low": 1, "medium": 2, "high": 3}

    for etype_id, etype_name in [
        (16, "monetary_policy_shock"),
        (17, "trade_tariff_policy"),
        (18, "geopolitical_event"),
        (19, "macro_data_shock"),
        (20, "liquidity_credit_stress"),
    ]:
        subset = market_labels[market_labels["event_type_id"] == etype_id].copy()
        if subset.empty:
            out[f"has_{etype_name}"] = False
            out[f"{etype_name}_severity"] = "none"
            continue

        # Get max severity per date
        subset["sev_rank"] = subset["severity"].map(severity_rank)
        max_sev = subset.groupby("date")["sev_rank"].max().reset_index()
        rank_to_sev = {1: "low", 2: "medium", 3: "high"}
        max_sev["max_severity"] = max_sev["sev_rank"].map(rank_to_sev)

        date_set = set(max_sev["date"])
        out[f"has_{etype_name}"] = out["event_date"].isin(date_set)

        sev_map = dict(zip(max_sev["date"], max_sev["max_severity"]))
        out[f"{etype_name}_severity"] = out["event_date"].map(sev_map).fillna("none")

    return out


def build_sector_context_per_event(events: pd.DataFrame, sector_labels: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, summarize the sector-level context on that date for that event's sector.
    """
    out = events.copy()

    severity_rank = {"low": 1, "medium": 2, "high": 3}

    for etype_id, etype_name in [
        (12, "sector_regulation"),
        (13, "sector_demand_shift"),
        (14, "sector_rotation"),
        (15, "sector_contagion"),
    ]:
        subset = sector_labels[sector_labels["event_type_id"] == etype_id].copy()
        if subset.empty:
            out[f"has_{etype_name}"] = False
            out[f"{etype_name}_severity"] = "none"
            out[f"{etype_name}_direction"] = "none"
            continue

        subset["sev_rank"] = subset["severity"].map(severity_rank)

        # For each (date, sector_etf), get max severity and direction
        agg = subset.groupby(["date", "sector_etf"]).agg(
            max_sev_rank=("sev_rank", "max"),
            direction=("direction", "first"),
        ).reset_index()

        rank_to_sev = {1: "low", 2: "medium", 3: "high"}
        agg["max_severity"] = agg["max_sev_rank"].map(rank_to_sev)

        # Create lookup key
        agg["key"] = agg["date"].astype(str) + "_" + agg["sector_etf"]
        out["key"] = out["event_date"].astype(str) + "_" + out["sector_etf"]

        sev_map = dict(zip(agg["key"], agg["max_severity"]))
        dir_map = dict(zip(agg["key"], agg["direction"]))
        key_set = set(agg["key"])

        out[f"has_{etype_name}"] = out["key"].isin(key_set)
        out[f"{etype_name}_severity"] = out["key"].map(sev_map).fillna("none")
        out[f"{etype_name}_direction"] = out["key"].map(dir_map).fillna("none")

        out = out.drop(columns=["key"])

    return out


def build_stock_context_per_event(events: pd.DataFrame, stock_labels: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, attach the stock-level label if one exists.
    """
    out = events.copy()

    # Create lookup key
    stock_labels["key"] = stock_labels["date"].astype(str) + "_" + stock_labels["ticker"]
    out["key"] = out["event_date"].astype(str) + "_" + out["ticker"]

    label_map = dict(zip(stock_labels["key"], stock_labels["event_type_name"]))
    severity_map = dict(zip(stock_labels["key"], stock_labels["severity"]))
    desc_map = dict(zip(stock_labels["key"], stock_labels["description"]))

    out["stock_event_type"] = out["key"].map(label_map).fillna("unlabeled")
    out["stock_event_severity"] = out["key"].map(severity_map).fillna("unlabeled")
    out["stock_event_description"] = out["key"].map(desc_map).fillna("")

    out = out.drop(columns=["key"])

    return out


def compute_lookup_table(df: pd.DataFrame, group_cols: list, min_events: int = 10) -> pd.DataFrame:
    """
    Compute success rates grouped by the specified columns.
    Only include groups with at least min_events events.
    """
    def _agg(g):
        valid_days = g["days_to_hit"].dropna()
        return pd.Series({
            "total_events": len(g),
            "success_rate": g["success"].mean(),
            "avg_days_to_hit": valid_days.mean() if len(valid_days) > 0 else None,
            "avg_final_return": g["final_return"].mean(),
            "median_final_return": g["final_return"].median(),
            "avg_max_drawdown": g["max_drawdown"].mean(),
        })

    result = df.groupby(group_cols).apply(_agg).reset_index()
    result = result[result["total_events"] >= min_events]
    result = result.sort_values("success_rate", ascending=False).reset_index(drop=True)

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    events, market_labels, sector_labels, stock_labels = load_data()
    print(f"  Events: {len(events)}")
    print(f"  Market labels: {len(market_labels)}")
    print(f"  Sector labels: {len(sector_labels)}")
    print(f"  Stock labels: {len(stock_labels)}")

    print("\nBuilding market context per event...")
    events = build_market_context_per_event(events, market_labels)

    print("Building sector context per event...")
    events = build_sector_context_per_event(events, sector_labels)

    print("Building stock context per event...")
    events = build_stock_context_per_event(events, stock_labels)

    # Save the fully labeled events
    events.to_csv(os.path.join(OUTPUT_DIR, "events_fully_labeled.csv"), index=False)
    print(f"\nSaved events_fully_labeled.csv ({len(events)} events, {len(events.columns)} columns)")

    # === LOOKUP TABLE 1: By stock event type ===
    print("\n=== LOOKUP: BY STOCK EVENT TYPE ===")
    lookup_stock = compute_lookup_table(
        events[events["stock_event_type"] != "unlabeled"],
        ["stock_event_type", "stock_event_severity"],
        min_events=5,
    )
    lookup_stock.to_csv(os.path.join(OUTPUT_DIR, "lookup_by_stock_event_type.csv"), index=False)

    display = lookup_stock.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    # === LOOKUP TABLE 2: By market context ===
    print("\n=== LOOKUP: BY LIQUIDITY/CREDIT STRESS ===")
    lookup_liq = compute_lookup_table(
        events,
        ["liquidity_credit_stress_severity"],
        min_events=20,
    )
    lookup_liq.to_csv(os.path.join(OUTPUT_DIR, "lookup_by_liquidity_stress.csv"), index=False)

    display = lookup_liq.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    print("\n=== LOOKUP: BY TRADE/TARIFF CONTEXT ===")
    lookup_tariff = compute_lookup_table(
        events,
        ["has_trade_tariff_policy", "trade_tariff_policy_severity"],
        min_events=20,
    )
    display = lookup_tariff.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    # === LOOKUP TABLE 3: By sector rotation ===
    print("\n=== LOOKUP: BY SECTOR ROTATION (outflow from sector) ===")
    outflow = events[events["sector_rotation_direction"] == "outflow"].copy()
    lookup_rot = compute_lookup_table(
        outflow,
        ["sector_rotation_severity"],
        min_events=10,
    )
    display = lookup_rot.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    # === LOOKUP TABLE 4: By drop bucket ===
    print("\n=== LOOKUP: BY DROP BUCKET ===")
    lookup_bucket = compute_lookup_table(events, ["drop_bucket"], min_events=20)
    display = lookup_bucket.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    # === LOOKUP TABLE 5: Stock event type × drop bucket ===
    print("\n=== LOOKUP: STOCK EVENT TYPE × DROP BUCKET ===")
    labeled_events = events[events["stock_event_type"] != "unlabeled"]
    lookup_cross = compute_lookup_table(
        labeled_events,
        ["stock_event_type", "drop_bucket"],
        min_events=5,
    )
    lookup_cross.to_csv(os.path.join(OUTPUT_DIR, "lookup_stock_type_x_bucket.csv"), index=False)

    display = lookup_cross.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    # === LOOKUP TABLE 6: No catalyst vs has catalyst ===
    print("\n=== LOOKUP: NO CATALYST vs HAS CATALYST ===")
    labeled_events = events[events["stock_event_type"] != "unlabeled"].copy()
    labeled_events["has_catalyst"] = labeled_events["stock_event_type"] != "no_clear_catalyst"
    lookup_catalyst = compute_lookup_table(labeled_events, ["has_catalyst"], min_events=5)
    display = lookup_catalyst.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    # === COMBINED LOOKUP ===
    print("\n=== COMBINED LOOKUP: STOCK EVENT × MARKET STRESS ===")
    labeled_events = events[events["stock_event_type"] != "unlabeled"].copy()
    labeled_events["market_stressed"] = labeled_events["liquidity_credit_stress_severity"].isin(["medium", "high"])
    lookup_combined = compute_lookup_table(
        labeled_events,
        ["stock_event_type", "market_stressed"],
        min_events=5,
    )
    lookup_combined.to_csv(os.path.join(OUTPUT_DIR, "lookup_combined.csv"), index=False)

    display = lookup_combined.copy()
    for col in ["success_rate", "avg_final_return", "median_final_return", "avg_max_drawdown"]:
        display[col] = (display[col] * 100).round(2)
    if "avg_days_to_hit" in display.columns:
        display["avg_days_to_hit"] = display["avg_days_to_hit"].round(2)
    print(display.to_string(index=False))

    print("\n=== DONE ===")
    print(f"All lookup tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
