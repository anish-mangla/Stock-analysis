#!/usr/bin/env python3
"""
build_sector_level_labels.py

Step 3 of the historical labeling process.

Builds SECTOR-LEVEL event labels (types 12-15) for every trading day
in the events.csv date range.

Type 14 (Sector Rotation) and Type 13 (Sector Demand Shift) are computed
programmatically from price data. Types 12 and 15 require manual research
and will be added in a separate step.

What this script does:
1. Downloads sector ETF data and underlying demand drivers
2. Computes sector ETF performance vs SPY for rotation detection (Type 14)
3. Computes demand driver moves for sector demand shifts (Type 13)
4. Outputs labels per (date, sector) pair

Sector ETFs used:
- XLK: Technology
- SMH: Semiconductors
- XLC: Communication Services
- XLY: Consumer Discretionary
- XLP: Consumer Staples
- XLF: Financials
- XLV: Healthcare
- XLE: Energy
- XLI: Industrials
- XLU: Utilities
- XLRE: Real Estate
- XLB: Materials

Demand drivers:
- Energy (XLE): WTI crude oil (CL=F)
- Financials (XLF): 10-year Treasury yield proxy (TLT inverse)
- REITs (XLRE): 10-year Treasury yield proxy (TLT inverse)
- Utilities (XLU): 10-year Treasury yield proxy (TLT inverse)
- Materials (XLB): Copper (HG=F or CPER ETF)
- Industrials (XLI): ISM PMI (manual, not in this script)

Outputs:
- outputs/sector_level_labels.csv
- outputs/sector_daily_features.csv
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import yfinance as yf


OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")
SECTOR_LABELS_CSV = os.path.join(OUTPUT_DIR, "sector_level_labels.csv")
SECTOR_FEATURES_CSV = os.path.join(OUTPUT_DIR, "sector_daily_features.csv")

DATA_START = "2019-06-01"
DATA_END = "2026-01-15"

DOWNLOAD_AUTO_ADJUST = False

# All sector ETFs we track
SECTOR_ETFS = ["XLK", "SMH", "XLC", "XLY", "XLP", "XLF", "XLV", "XLE", "XLI", "XLU", "XLRE", "XLB"]

# Demand driver symbols
# CL=F = WTI crude futures (energy driver)
# TLT = 20+ year Treasury bond ETF (inverse rate proxy — when TLT drops, yields rise)
# CPER = copper ETF (materials/industrials driver)
DEMAND_DRIVER_SYMBOLS = ["CL=F", "TLT", "CPER"]

# Mapping: which demand driver affects which sector ETFs
SECTOR_DEMAND_DRIVERS = {
    "XLE": {"driver": "CL=F", "driver_name": "oil", "relationship": "positive"},
    "XLF": {"driver": "TLT", "driver_name": "rates", "relationship": "negative"},  # TLT down = yields up = good for banks
    "XLRE": {"driver": "TLT", "driver_name": "rates", "relationship": "positive"},  # TLT down = yields up = bad for REITs
    "XLU": {"driver": "TLT", "driver_name": "rates", "relationship": "positive"},   # TLT down = yields up = bad for utilities
    "XLB": {"driver": "CPER", "driver_name": "copper", "relationship": "positive"},
}

# Ticker to sector ETF mapping (same as in existing backtest files)
TICKER_TO_SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "AMAT": "SMH", "LRCX": "SMH", "QCOM": "SMH", "TXN": "SMH", "MU": "SMH",
    "INTC": "SMH", "NOW": "XLK", "INTU": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "IBM": "XLK", "ACN": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "AMZN": "XLY", "META": "XLC", "NFLX": "XLC", "CMCSA": "XLC", "TMUS": "XLC",
    "T": "XLC", "VZ": "XLC", "DIS": "XLC", "UBER": "XLY", "TSLA": "XLY",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF",
    "MS": "XLF", "AXP": "XLF", "BLK": "XLF", "USB": "XLF", "SCHW": "XLF",
    "SPGI": "XLF", "MCO": "XLF", "CB": "XLF", "PGR": "XLF", "BRK-B": "XLF",
    "LLY": "XLV", "PFE": "XLV", "BMY": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "TMO": "XLV", "DHR": "XLV", "ABT": "XLV", "MDT": "XLV", "GILD": "XLV",
    "ISRG": "XLV", "UNH": "XLV", "CI": "XLV", "CVS": "XLV", "AMGN": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    "CAT": "XLI", "DE": "XLI", "HON": "XLI", "GE": "XLI", "GD": "XLI",
    "LMT": "XLI", "UNP": "XLI", "UPS": "XLI", "RTX": "XLI", "BA": "XLI",
    "HD": "XLY", "LOW": "XLY", "MCD": "XLY", "NKE": "XLY", "SBUX": "XLY",
    "BKNG": "XLY", "TJX": "XLY", "COST": "XLP", "WMT": "XLP", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "PM": "XLP", "MDLZ": "XLP", "CL": "XLP",
    "DUK": "XLU", "NEE": "XLU", "SO": "XLU", "AMT": "XLRE", "PLD": "XLRE",
    "SPG": "XLRE", "LIN": "XLB", "SHW": "XLB", "APD": "XLB",
    "GEV": "XLI", "PLTR": "XLK",
}


def download_symbol(symbol: str) -> pd.Series:
    """Download close prices for a single symbol."""
    print(f"  Downloading {symbol}...")
    df = yf.download(
        tickers=symbol,
        start=DATA_START,
        end=DATA_END,
        auto_adjust=DOWNLOAD_AUTO_ADJUST,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol].copy()
        elif symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1).copy()
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float).dropna()
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def download_all_data() -> tuple[Dict[str, pd.Series], pd.Series, Dict[str, pd.Series]]:
    """Download SPY, all sector ETFs, and demand driver symbols."""
    print("Downloading SPY...")
    spy_close = download_symbol("SPY")

    print("Downloading sector ETFs...")
    sector_closes: Dict[str, pd.Series] = {}
    for etf in SECTOR_ETFS:
        try:
            sector_closes[etf] = download_symbol(etf)
        except Exception as e:
            print(f"  WARNING: Failed to download {etf}: {e}")

    print("Downloading demand drivers...")
    driver_closes: Dict[str, pd.Series] = {}
    for symbol in DEMAND_DRIVER_SYMBOLS:
        try:
            driver_closes[symbol] = download_symbol(symbol)
        except Exception as e:
            print(f"  WARNING: Failed to download {symbol}: {e}")

    return sector_closes, spy_close, driver_closes


def build_sector_features(
    sector_closes: Dict[str, pd.Series],
    spy_close: pd.Series,
    driver_closes: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Build a daily features DataFrame with sector returns, SPY-relative performance, and driver moves."""

    # Build a common date index
    all_dates = sorted(set(spy_close.index))
    features = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    features["spy_close"] = spy_close
    features["spy_daily_return"] = features["spy_close"].pct_change()

    # Sector ETF features
    for etf, close in sector_closes.items():
        features[f"{etf}_close"] = close
        features[f"{etf}_daily_return"] = features[f"{etf}_close"].pct_change()
        features[f"{etf}_5d_return"] = features[f"{etf}_close"].pct_change(5)
        features[f"{etf}_vs_spy_daily"] = features[f"{etf}_daily_return"] - features["spy_daily_return"]
        features[f"{etf}_vs_spy_5d"] = features[f"{etf}_5d_return"] - features["spy_close"].pct_change(5)

    # Demand driver features
    for symbol, close in driver_closes.items():
        clean_name = symbol.replace("=", "").replace("^", "").lower()
        features[f"{clean_name}_close"] = close
        features[f"{clean_name}_daily_return"] = features[f"{clean_name}_close"].pct_change()
        features[f"{clean_name}_5d_return"] = features[f"{clean_name}_close"].pct_change(5)

        # Compute rolling std for sigma calculation
        daily_ret = features[f"{clean_name}_daily_return"]
        features[f"{clean_name}_20d_std"] = daily_ret.rolling(20).std()

    features = features.sort_index().ffill()
    return features


# ============================================================
# TYPE 14: SECTOR ROTATION
# ============================================================

def label_sector_rotation(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Type 14: Sector Rotation.

    For each trading day and each sector ETF, measure the sector's
    daily return vs SPY. If the sector significantly underperformed
    or outperformed SPY, label it as rotation.

    Severity (from taxonomy):
    - Low: sector ETF underperformed SPY by < 2% on the day
    - Medium: sector ETF underperformed SPY by 2-5%
    - High: sector ETF underperformed SPY by > 5%

    We only emit labels for days where the underperformance is >= 1%
    (otherwise it's just normal noise).
    """
    rows = []

    for etf in SECTOR_ETFS:
        col = f"{etf}_vs_spy_daily"
        if col not in features_df.columns:
            continue

        for date, row in features_df.iterrows():
            vs_spy = row[col]
            if pd.isna(vs_spy):
                continue

            abs_vs_spy = abs(vs_spy)

            # Only label meaningful rotation (>= 1% divergence)
            if abs_vs_spy < 0.01:
                continue

            if abs_vs_spy >= 0.05:
                severity = "high"
            elif abs_vs_spy >= 0.02:
                severity = "medium"
            else:
                severity = "low"

            direction = "outflow" if vs_spy < 0 else "inflow"

            rows.append({
                "date": date,
                "event_type_id": 14,
                "event_type_name": "sector_rotation",
                "severity": severity,
                "sector_etf": etf,
                "direction": direction,
                "evidence": f"{etf} vs SPY: {vs_spy:+.4f} ({vs_spy*100:+.2f}%)",
            })

    return pd.DataFrame(rows)


# ============================================================
# TYPE 13: SECTOR DEMAND SHIFT
# ============================================================

def label_sector_demand_shift(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Type 13: Sector Demand Shift.

    For sectors with identifiable demand drivers, measure the driver's
    daily move in standard deviations. If the driver moved significantly,
    label the sector as experiencing a demand shift.

    Severity (from taxonomy):
    - Low: driver moved < 1 sigma
    - Medium: 1-2 sigma
    - High: > 2 sigma

    We only emit labels for moves >= 1 sigma.
    """
    rows = []

    for sector_etf, config in SECTOR_DEMAND_DRIVERS.items():
        driver_symbol = config["driver"]
        driver_name = config["driver_name"]
        relationship = config["relationship"]

        clean_name = driver_symbol.replace("=", "").replace("^", "").lower()
        ret_col = f"{clean_name}_daily_return"
        std_col = f"{clean_name}_20d_std"

        if ret_col not in features_df.columns or std_col not in features_df.columns:
            continue

        for date, row in features_df.iterrows():
            driver_ret = row[ret_col]
            driver_std = row[std_col]

            if pd.isna(driver_ret) or pd.isna(driver_std) or driver_std == 0:
                continue

            sigma = abs(driver_ret) / driver_std

            if sigma < 1.0:
                continue

            if sigma >= 2.0:
                severity = "high"
            elif sigma >= 1.0:
                severity = "medium"
            else:
                severity = "low"

            # Determine if this is positive or negative for the sector
            if relationship == "positive":
                # Driver up = good for sector, driver down = bad
                direction = "tailwind" if driver_ret > 0 else "headwind"
            else:
                # Driver up = bad for sector (e.g., TLT up = yields down = bad for banks)
                direction = "headwind" if driver_ret > 0 else "tailwind"

            rows.append({
                "date": date,
                "event_type_id": 13,
                "event_type_name": "sector_demand_shift",
                "severity": severity,
                "sector_etf": sector_etf,
                "direction": direction,
                "evidence": f"{driver_name} ({driver_symbol}): {driver_ret:+.4f} ({driver_ret*100:+.2f}%), {sigma:.1f} sigma. {direction} for {sector_etf}",
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load events to know which dates and sectors we care about
    events_df = pd.read_csv(EVENTS_CSV)
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])
    event_dates = set(events_df["event_date"].dt.date)

    # Map tickers to sectors for reference
    events_df["sector_etf"] = events_df["ticker"].map(TICKER_TO_SECTOR_ETF).fillna("SPY")
    sectors_in_events = set(events_df["sector_etf"].unique())
    print(f"Loaded {len(events_df)} events across {len(event_dates)} dates, {len(sectors_in_events)} sectors.")

    # Download data
    sector_closes, spy_close, driver_closes = download_all_data()
    print(f"Downloaded {len(sector_closes)} sector ETFs, {len(driver_closes)} demand drivers.")

    # Build features
    features_df = build_sector_features(sector_closes, spy_close, driver_closes)
    features_df.to_csv(SECTOR_FEATURES_CSV)
    print(f"Built sector features: {len(features_df)} days, {len(features_df.columns)} columns.")
    print(f"Saved to {SECTOR_FEATURES_CSV}")

    # Generate labels
    print("\nLabeling Type 14: Sector Rotation...")
    type14_df = label_sector_rotation(features_df)
    print(f"  Generated {len(type14_df)} labels.")

    print("Labeling Type 13: Sector Demand Shift...")
    type13_df = label_sector_demand_shift(features_df)
    print(f"  Generated {len(type13_df)} labels.")

    # Combine
    all_labels = pd.concat([type13_df, type14_df], ignore_index=True)
    all_labels["date"] = pd.to_datetime(all_labels["date"])
    all_labels = all_labels.sort_values(["date", "event_type_id", "sector_etf"]).reset_index(drop=True)

    all_labels.to_csv(SECTOR_LABELS_CSV, index=False)
    print(f"\nSaved {len(all_labels)} sector-level labels to: {SECTOR_LABELS_CSV}")

    # Summary
    print("\n=== LABEL SUMMARY BY TYPE AND SEVERITY ===")
    summary = (
        all_labels.groupby(["event_type_id", "event_type_name", "severity"])
        .size()
        .reset_index(name="count")
        .sort_values(["event_type_id", "severity"])
    )
    print(summary.to_string(index=False))

    print("\n=== TYPE 14 ROTATION BY SECTOR ===")
    if not type14_df.empty:
        rot_summary = (
            type14_df.groupby(["sector_etf", "severity"])
            .size()
            .reset_index(name="count")
            .sort_values(["sector_etf", "severity"])
        )
        print(rot_summary.to_string(index=False))

    print("\n=== TYPE 13 DEMAND SHIFT BY SECTOR ===")
    if not type13_df.empty:
        dem_summary = (
            type13_df.groupby(["sector_etf", "severity"])
            .size()
            .reset_index(name="count")
            .sort_values(["sector_etf", "severity"])
        )
        print(dem_summary.to_string(index=False))

    # Show how many event dates have sector labels
    label_dates = set(all_labels["date"].dt.date)
    overlap = event_dates & label_dates
    print(f"\nEvent dates with at least one sector label: {len(overlap)} / {len(event_dates)}")

    # Sample labels
    print("\n=== SAMPLE TYPE 14 (ROTATION) ===")
    for etf in ["XLK", "XLE", "XLF"]:
        subset = type14_df[type14_df["sector_etf"] == etf].head(3)
        for _, r in subset.iterrows():
            print(f"  {r['date'].date()} | {etf} | {r['severity']:6s} | {r['direction']:7s} | {r['evidence']}")

    print("\n=== SAMPLE TYPE 13 (DEMAND SHIFT) ===")
    for etf in ["XLE", "XLF", "XLRE"]:
        subset = type13_df[type13_df["sector_etf"] == etf].head(3)
        for _, r in subset.iterrows():
            print(f"  {r['date'].date()} | {etf} | {r['severity']:6s} | {r['direction']:9s} | {r['evidence'][:80]}")


if __name__ == "__main__":
    main()
