#!/usr/bin/env python3
"""
build_market_level_labels.py

Step 1 of the historical labeling process.

Builds MARKET-LEVEL event labels (types 16, 19, 20) for every trading day
in the events.csv date range, using purely quantitative data.

What this script does:
1. Downloads VIX, 2-year Treasury yield proxy (SHY), SPY, and credit spread
   proxies (HYG for high-yield, LQD for investment-grade) from yfinance
2. Pulls major macro data release dates and surprise magnitudes from a
   hardcoded reference table (since this data isn't freely available via API)
3. Computes severity for:
   - Type 16: Monetary Policy Shock (2Y yield daily move)
   - Type 19: Macro Data Shock (from reference table)
   - Type 20: Liquidity/Credit Stress (VIX level + credit spread moves)
4. Outputs a CSV with one row per (date, event_type) that can be joined
   to events.csv

Types 17 (Trade/Tariff) and 18 (Geopolitical) require manual labeling
and are NOT covered here. They will be added in a separate step.

Outputs:
- outputs/market_level_labels.csv
- outputs/market_daily_features.csv (raw features for debugging/analysis)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
MARKET_LABELS_CSV = os.path.join(OUTPUT_DIR, "market_level_labels.csv")
MARKET_FEATURES_CSV = os.path.join(OUTPUT_DIR, "market_daily_features.csv")

# Date range: go back further than events to have lookback data for moving averages
DATA_START = "2019-06-01"
DATA_END = "2026-01-15"

# Proxies:
# - ^VIX: CBOE Volatility Index
# - ^IRX: 13-week Treasury Bill yield (3-month, proxy for short rates)
# - SHY: iShares 1-3 Year Treasury Bond ETF (inverse proxy for 2Y yield moves)
# - TLT: iShares 20+ Year Treasury Bond ETF (long duration, for rate shock detection)
# - HYG: iShares iBoxx High Yield Corporate Bond ETF (credit stress proxy)
# - LQD: iShares iBoxx Investment Grade Corporate Bond ETF (IG credit proxy)
# - SPY: S&P 500 ETF
SYMBOLS = ["^VIX", "SPY", "HYG", "LQD", "SHY", "TLT"]

DOWNLOAD_AUTO_ADJUST = False


# ============================================================
# TYPE 16: MONETARY POLICY SHOCK
# ============================================================

# Fed meeting dates 2020-2025 (announcement days)
# Source: Federal Reserve calendar
# We measure the daily move in short-duration bond ETF (SHY) as a proxy
# for 2-year yield moves. SHY moves inversely to yields, so a big DROP
# in SHY = yields spiked = hawkish surprise, big RISE = dovish surprise.
#
# We also look at TLT for broader rate shock detection.

FED_MEETING_DATES = [
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
]


def label_monetary_policy_shock(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label Type 16: Monetary Policy Shock.

    Uses SHY daily return as proxy for 2-year yield move.
    SHY is a short-duration bond ETF — when yields spike, SHY drops.

    We measure the absolute daily return of SHY on Fed meeting days.
    Bigger moves = bigger surprise.

    Severity thresholds (mapped from 2Y yield move to SHY return):
    - SHY has ~2 year duration, so a 10bp yield move ≈ 0.2% SHY move
    - Low: |SHY return| < 0.2% (≈ < 10bp yield move)
    - Medium: 0.2% - 0.5% (≈ 10-25bp)
    - High: > 0.5% (≈ > 25bp)

    Also checks TLT for broader rate shocks on non-Fed days.
    """
    fed_dates = set(pd.to_datetime(FED_MEETING_DATES).date)

    rows = []
    for date, row in features_df.iterrows():
        date_val = pd.Timestamp(date).date()
        shy_return = row.get("shy_daily_return")
        tlt_return = row.get("tlt_daily_return")

        if pd.isna(shy_return):
            continue

        is_fed_day = date_val in fed_dates
        abs_shy_move = abs(shy_return)

        # On Fed days, use SHY move directly
        if is_fed_day:
            if abs_shy_move >= 0.005:
                severity = "high"
            elif abs_shy_move >= 0.002:
                severity = "medium"
            else:
                severity = "low"

            direction = "hawkish" if shy_return < 0 else "dovish"

            rows.append({
                "date": date,
                "event_type_id": 16,
                "event_type_name": "monetary_policy_shock",
                "severity": severity,
                "direction": direction,
                "evidence": f"Fed day. SHY return: {shy_return:.4f} ({direction})",
            })

        # On non-Fed days, check for extreme rate moves (speeches, surprise data)
        # Only flag if TLT moves > 1.5% (very large rate move)
        elif not pd.isna(tlt_return) and abs(tlt_return) >= 0.015:
            if abs_shy_move >= 0.005:
                severity = "high"
            elif abs_shy_move >= 0.002:
                severity = "medium"
            else:
                continue  # TLT moved but short end didn't — not a policy shock

            direction = "hawkish" if shy_return < 0 else "dovish"

            rows.append({
                "date": date,
                "event_type_id": 16,
                "event_type_name": "monetary_policy_shock",
                "severity": severity,
                "direction": direction,
                "evidence": f"Non-Fed rate shock. SHY: {shy_return:.4f}, TLT: {tlt_return:.4f}",
            })

    return pd.DataFrame(rows)


# ============================================================
# TYPE 19: MACRO DATA SHOCK
# ============================================================

# Major macro data surprises 2020-2025
# Format: (date, data_type, actual, consensus, sigma_surprise, direction)
#
# sigma_surprise: how many standard deviations the actual was from consensus
# direction: "hot" (inflationary/strong) or "cold" (deflationary/weak)
#
# This is a curated list of the SIGNIFICANT surprises only (>= 1 sigma).
# Minor surprises are not worth labeling.
#
# Sources: BLS, BEA, ISM releases; consensus from Bloomberg/Reuters archives
# Note: this list covers the most impactful releases. It is not exhaustive
# for every single data print, but captures the ones that moved markets.

MACRO_DATA_SURPRISES: List[Dict] = [
    # 2020 — COVID shock dominates
    {"date": "2020-03-06", "data_type": "nonfarm_payrolls", "sigma": 0.5, "direction": "cold",
     "note": "Feb jobs strong but COVID fear overshadowed"},
    {"date": "2020-04-03", "data_type": "nonfarm_payrolls", "sigma": 3.0, "direction": "cold",
     "note": "701K jobs lost, first COVID impact"},
    {"date": "2020-05-08", "data_type": "nonfarm_payrolls", "sigma": 4.0, "direction": "cold",
     "note": "20.5M jobs lost, worst ever"},
    {"date": "2020-06-05", "data_type": "nonfarm_payrolls", "sigma": 3.0, "direction": "hot",
     "note": "2.5M jobs added vs -7.5M expected, massive upside surprise"},
    {"date": "2020-07-30", "data_type": "gdp", "sigma": 3.0, "direction": "cold",
     "note": "Q2 GDP -32.9% annualized"},

    # 2021 — inflation starts
    {"date": "2021-05-07", "data_type": "nonfarm_payrolls", "sigma": 2.5, "direction": "cold",
     "note": "266K vs 1M expected, huge miss"},
    {"date": "2021-05-12", "data_type": "cpi", "sigma": 2.0, "direction": "hot",
     "note": "CPI 4.2% YoY vs 3.6% expected, inflation shock"},
    {"date": "2021-06-10", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "CPI 5.0% YoY, above expectations"},
    {"date": "2021-10-13", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "CPI 5.4% YoY, persistent inflation"},
    {"date": "2021-11-10", "data_type": "cpi", "sigma": 2.0, "direction": "hot",
     "note": "CPI 6.2% YoY, 30-year high"},

    # 2022 — aggressive tightening
    {"date": "2022-01-12", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "CPI 7.0% YoY"},
    {"date": "2022-02-10", "data_type": "cpi", "sigma": 2.0, "direction": "hot",
     "note": "CPI 7.5% YoY, above expectations"},
    {"date": "2022-03-10", "data_type": "cpi", "sigma": 1.0, "direction": "hot",
     "note": "CPI 7.9% YoY"},
    {"date": "2022-04-12", "data_type": "cpi", "sigma": 1.0, "direction": "hot",
     "note": "CPI 8.5% YoY, peak"},
    {"date": "2022-06-10", "data_type": "cpi", "sigma": 2.5, "direction": "hot",
     "note": "CPI 8.6% YoY vs 8.3% expected, triggered 75bp hike"},
    {"date": "2022-09-13", "data_type": "cpi", "sigma": 2.0, "direction": "hot",
     "note": "Core CPI reaccelerated, market crashed 4%+"},
    {"date": "2022-10-13", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "Core CPI still hot, but market reversed and rallied"},

    # 2023 — disinflation + soft landing debate
    {"date": "2023-01-06", "data_type": "nonfarm_payrolls", "sigma": 1.5, "direction": "hot",
     "note": "223K jobs, strong labor market"},
    {"date": "2023-02-03", "data_type": "nonfarm_payrolls", "sigma": 3.0, "direction": "hot",
     "note": "517K jobs vs 185K expected, massive beat"},
    {"date": "2023-02-14", "data_type": "cpi", "sigma": 1.0, "direction": "hot",
     "note": "CPI 6.4% slightly above expectations"},
    {"date": "2023-07-12", "data_type": "cpi", "sigma": 1.5, "direction": "cold",
     "note": "CPI 3.0% YoY, big drop, dovish signal"},
    {"date": "2023-11-14", "data_type": "cpi", "sigma": 1.5, "direction": "cold",
     "note": "CPI cooler than expected, rate cut hopes surged"},

    # 2024 — rate cut expectations
    {"date": "2024-01-05", "data_type": "nonfarm_payrolls", "sigma": 1.5, "direction": "hot",
     "note": "216K vs 170K expected"},
    {"date": "2024-02-13", "data_type": "cpi", "sigma": 2.0, "direction": "hot",
     "note": "CPI 3.1% vs 2.9% expected, rate cut hopes faded"},
    {"date": "2024-03-12", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "Core CPI sticky, 3 months of hot prints"},
    {"date": "2024-04-10", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "CPI 3.5% vs 3.4%, rate cuts pushed further out"},
    {"date": "2024-07-11", "data_type": "cpi", "sigma": 1.5, "direction": "cold",
     "note": "CPI 3.0%, cooler, rate cut back on table"},
    {"date": "2024-08-02", "data_type": "nonfarm_payrolls", "sigma": 2.0, "direction": "cold",
     "note": "114K vs 175K expected, unemployment rose to 4.3%, recession fears"},
    {"date": "2024-09-06", "data_type": "nonfarm_payrolls", "sigma": 1.0, "direction": "cold",
     "note": "142K vs 160K, soft but not terrible"},

    # 2025 — tariff era + mixed data
    {"date": "2025-01-10", "data_type": "nonfarm_payrolls", "sigma": 2.0, "direction": "hot",
     "note": "256K vs 160K expected, strong jobs pushed back rate cuts"},
    {"date": "2025-02-12", "data_type": "cpi", "sigma": 1.5, "direction": "hot",
     "note": "CPI hotter than expected, tariff inflation fears"},
    {"date": "2025-03-07", "data_type": "nonfarm_payrolls", "sigma": 1.0, "direction": "cold",
     "note": "151K vs 160K, slight miss but unemployment ticked up"},
    {"date": "2025-05-02", "data_type": "nonfarm_payrolls", "sigma": 1.0, "direction": "cold",
     "note": "Weak jobs, but tariff uncertainty dominated"},
    {"date": "2025-05-13", "data_type": "cpi", "sigma": 1.5, "direction": "cold",
     "note": "CPI cooled, helped risk-on rally"},
    {"date": "2025-08-01", "data_type": "nonfarm_payrolls", "sigma": 2.0, "direction": "cold",
     "note": "Weak July jobs, recession fears spiked"},
]


def label_macro_data_shock(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label Type 19: Macro Data Shock.

    Uses the curated reference table of significant macro surprises.
    """
    surprise_map = {}
    for entry in MACRO_DATA_SURPRISES:
        dt = pd.Timestamp(entry["date"])
        surprise_map[dt] = entry

    rows = []
    for date in features_df.index:
        ts = pd.Timestamp(date)
        if ts in surprise_map:
            entry = surprise_map[ts]
            sigma = entry["sigma"]

            if sigma >= 2.0:
                severity = "high"
            elif sigma >= 1.0:
                severity = "medium"
            else:
                severity = "low"

            rows.append({
                "date": date,
                "event_type_id": 19,
                "event_type_name": "macro_data_shock",
                "severity": severity,
                "direction": entry["direction"],
                "evidence": f"{entry['data_type']}: {entry['note']} (sigma={sigma})",
            })

    return pd.DataFrame(rows)


# ============================================================
# TYPE 20: LIQUIDITY / CREDIT STRESS
# ============================================================

def label_liquidity_credit_stress(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label Type 20: Liquidity / Credit Stress.

    Uses VIX level and credit spread proxy (HYG weekly return as inverse
    proxy for high-yield spread moves — when HYG drops, spreads are widening).

    VIX severity:
    - < 20: calm (no label)
    - 20-25: elevated → low
    - 25-35: stressed → medium
    - > 35: crisis → high

    Credit stress (HYG 5-day return):
    - > -1%: normal (no additional stress)
    - -1% to -3%: notable → bumps severity up one level
    - < -3%: severe → bumps severity up two levels (capped at high)

    We only emit a label when VIX >= 20 (no point labeling calm days).
    """
    rows = []

    for date, row in features_df.iterrows():
        vix = row.get("vix_close")
        hyg_5d = row.get("hyg_5d_return")

        if pd.isna(vix):
            continue

        # Base severity from VIX
        if vix >= 35:
            base_severity = 3  # high
        elif vix >= 25:
            base_severity = 2  # medium
        elif vix >= 20:
            base_severity = 1  # low
        else:
            continue  # calm, no label

        # Credit stress bump
        credit_bump = 0
        if not pd.isna(hyg_5d):
            if hyg_5d <= -0.03:
                credit_bump = 2
            elif hyg_5d <= -0.01:
                credit_bump = 1

        final_severity_num = min(base_severity + credit_bump, 3)
        severity_map = {1: "low", 2: "medium", 3: "high"}
        severity = severity_map[final_severity_num]

        evidence_parts = [f"VIX={vix:.1f}"]
        if not pd.isna(hyg_5d):
            evidence_parts.append(f"HYG 5d return={hyg_5d:.4f}")

        rows.append({
            "date": date,
            "event_type_id": 20,
            "event_type_name": "liquidity_credit_stress",
            "severity": severity,
            "direction": "stress",
            "evidence": ". ".join(evidence_parts),
        })

    return pd.DataFrame(rows)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_market_data() -> pd.DataFrame:
    """
    Download all required market data and build a daily features DataFrame.
    """
    print(f"Downloading market data from {DATA_START} to {DATA_END}...")
    print(f"Symbols: {SYMBOLS}")

    raw = yf.download(
        tickers=SYMBOLS,
        start=DATA_START,
        end=DATA_END,
        auto_adjust=DOWNLOAD_AUTO_ADJUST,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No data returned from yfinance.")

    # Extract close prices for each symbol
    close_map: Dict[str, pd.Series] = {}

    for symbol in SYMBOLS:
        clean_name = symbol.replace("^", "")

        if isinstance(raw.columns, pd.MultiIndex):
            if symbol in raw.columns.get_level_values(0):
                df = raw[symbol].copy()
            else:
                print(f"Warning: {symbol} not found in downloaded data, skipping.")
                continue

            if "Close" not in df.columns:
                print(f"Warning: {symbol} missing Close column, skipping.")
                continue

            s = df["Close"].copy().dropna()
        else:
            # Single symbol fallback
            s = raw["Close"].copy().dropna()

        s.index = pd.to_datetime(s.index)
        close_map[clean_name.lower()] = s.sort_index()

    # Build features DataFrame
    all_dates = sorted(set().union(*[set(s.index) for s in close_map.values()]))
    features = pd.DataFrame(index=pd.DatetimeIndex(all_dates))

    for name, s in close_map.items():
        features[f"{name}_close"] = s

    features = features.sort_index().ffill()

    # Compute derived features
    if "vix_close" in features.columns:
        features["vix_5d_avg"] = features["vix_close"].rolling(5).mean()
        features["vix_20d_avg"] = features["vix_close"].rolling(20).mean()

    if "spy_close" in features.columns:
        features["spy_daily_return"] = features["spy_close"].pct_change()
        features["spy_5d_return"] = features["spy_close"].pct_change(5)
        features["spy_20d_return"] = features["spy_close"].pct_change(20)

    if "shy_close" in features.columns:
        features["shy_daily_return"] = features["shy_close"].pct_change()

    if "tlt_close" in features.columns:
        features["tlt_daily_return"] = features["tlt_close"].pct_change()

    if "hyg_close" in features.columns:
        features["hyg_daily_return"] = features["hyg_close"].pct_change()
        features["hyg_5d_return"] = features["hyg_close"].pct_change(5)

    if "lqd_close" in features.columns:
        features["lqd_daily_return"] = features["lqd_close"].pct_change()
        features["lqd_5d_return"] = features["lqd_close"].pct_change(5)

    return features


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load events to know which dates we care about
    events_df = pd.read_csv(EVENTS_CSV)
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])
    event_dates = set(events_df["event_date"].dt.date)
    print(f"Loaded {len(events_df)} events across {len(event_dates)} unique dates.")

    # Download market data
    features_df = download_market_data()
    print(f"Built features DataFrame: {len(features_df)} trading days, {len(features_df.columns)} columns.")

    # Save raw features for debugging
    features_df.to_csv(MARKET_FEATURES_CSV)
    print(f"Saved market features to: {MARKET_FEATURES_CSV}")

    # Generate labels for each type
    print("\nLabeling Type 16: Monetary Policy Shock...")
    type16_df = label_monetary_policy_shock(features_df)
    print(f"  Generated {len(type16_df)} labels.")

    print("Labeling Type 19: Macro Data Shock...")
    type19_df = label_macro_data_shock(features_df)
    print(f"  Generated {len(type19_df)} labels.")

    print("Labeling Type 20: Liquidity / Credit Stress...")
    type20_df = label_liquidity_credit_stress(features_df)
    print(f"  Generated {len(type20_df)} labels.")

    # Combine all labels
    all_labels = pd.concat([type16_df, type19_df, type20_df], ignore_index=True)

    if all_labels.empty:
        print("\nNo labels generated. Something may be wrong.")
        return

    all_labels["date"] = pd.to_datetime(all_labels["date"])
    all_labels = all_labels.sort_values(["date", "event_type_id"]).reset_index(drop=True)

    # Save
    all_labels.to_csv(MARKET_LABELS_CSV, index=False)
    print(f"\nSaved {len(all_labels)} market-level labels to: {MARKET_LABELS_CSV}")

    # Print summary
    print("\n=== LABEL SUMMARY ===")
    summary = (
        all_labels.groupby(["event_type_id", "event_type_name", "severity"])
        .size()
        .reset_index(name="count")
        .sort_values(["event_type_id", "severity"])
    )
    print(summary.to_string(index=False))

    # Show how many event dates have at least one market label
    label_dates = set(all_labels["date"].dt.date)
    overlap = event_dates & label_dates
    print(f"\nEvent dates with at least one market label: {len(overlap)} / {len(event_dates)}")

    # Show some examples
    print("\n=== SAMPLE LABELS ===")
    for etype in [16, 19, 20]:
        subset = all_labels[all_labels["event_type_id"] == etype].head(5)
        if not subset.empty:
            print(f"\nType {etype}:")
            for _, row in subset.iterrows():
                print(f"  {row['date'].date()} | {row['severity']:6s} | {row['evidence']}")


if __name__ == "__main__":
    main()
