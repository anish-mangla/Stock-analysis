#!/usr/bin/env python3
"""
build_manual_market_labels.py

Step 2 of the historical labeling process.

Builds MANUAL market-level labels for:
- Type 17: Trade / Tariff Policy
- Type 18: Geopolitical Events

These events cannot be detected programmatically from price data alone.
They are compiled from public records and news archives, then validated
against market data (SPY reaction, oil price, VIX) using the severity
anchors defined in the taxonomy.

The script:
1. Defines reference tables of major trade/tariff and geopolitical events
2. Loads SPY, VIX, and oil data to compute secondary severity anchors
3. Assigns severity using both primary criteria (from taxonomy) and
   secondary market-reaction anchors
4. Outputs labels in the same format as build_market_level_labels.py
5. Merges with existing market_level_labels.csv

Outputs:
- outputs/manual_market_labels.csv (just types 17 & 18)
- outputs/market_level_labels.csv (updated with all types 16-20)
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


OUTPUT_DIR = "outputs"
EXISTING_LABELS_CSV = os.path.join(OUTPUT_DIR, "market_level_labels.csv")
MANUAL_LABELS_CSV = os.path.join(OUTPUT_DIR, "manual_market_labels.csv")
MARKET_FEATURES_CSV = os.path.join(OUTPUT_DIR, "market_daily_features.csv")
OIL_DATA_CSV = os.path.join(OUTPUT_DIR, "oil_daily_data.csv")


# ============================================================
# TYPE 17: TRADE / TARIFF POLICY EVENTS
# ============================================================
# Each entry has:
#   date: the trading day the market reacted (not necessarily announcement day)
#   primary_severity: based on taxonomy (dollar value of trade affected)
#   description: what happened
#   direction: escalation or de_escalation
#
# Primary severity from taxonomy:
#   low: < $10B trade affected, or threatened but not implemented
#   medium: $10-100B trade affected, or meaningful increase
#   high: > $100B trade affected, blanket tariffs, or retaliation
#
# Secondary anchor: SPY daily return (computed from data)
#   SPY drop > 2% on the day → confirms high severity
#   SPY barely moves → suggests priced in or low impact

TRADE_TARIFF_EVENTS: List[Dict] = [
    # ================================================================
    # ALL EVENTS BELOW VERIFIED BY CHATGPT WITH WEB SEARCH
    # Dates are the US trading day the market reacted (not announcement day)
    # SPY returns cross-referenced against our market_daily_features.csv
    # ================================================================

    # === 2020 ===
    {"date": "2020-01-15", "primary_severity": "medium", "direction": "de_escalation",
     "description": "US-China Phase 1 deal signed. US cut List 4A tariff from 15% to 7.5%, canceled List 4B. ~$120B tariff relief + $160B canceled"},
    {"date": "2020-02-06", "primary_severity": "medium", "direction": "de_escalation",
     "description": "China halved retaliatory tariffs on 1,717 US products (~$75B of US exports), reinforcing Phase 1 de-escalation"},
    {"date": "2020-05-15", "primary_severity": "medium", "direction": "escalation",
     "description": "US blocked foreign chipmakers from supplying Huawei using US software/technology (FDPR tightening)"},
    {"date": "2020-08-17", "primary_severity": "low", "direction": "escalation",
     "description": "US extended Huawei chip crackdown, added 38 Huawei affiliates in 21 countries to Entity List (152 total)"},
    {"date": "2020-09-15", "primary_severity": "low", "direction": "de_escalation",
     "description": "WTO ruled US Section 301 tariffs on ~$234B of Chinese goods violated WTO rules (no immediate rollback)"},

    # === 2021 ===
    {"date": "2021-06-15", "primary_severity": "low", "direction": "de_escalation",
     "description": "US-EU agreed 5-year truce in Airbus-Boeing dispute, suspending ~$11.5B of transatlantic tariffs"},
    {"date": "2021-11-01", "primary_severity": "low", "direction": "de_escalation",
     "description": "US-EU ended Trump-era steel/aluminum tariff fight with tariff-rate quota, scrapped planned EU retaliation"},

    # === 2022 ===
    {"date": "2022-03-08", "primary_severity": "medium", "direction": "escalation",
     "description": "US banned imports of Russian oil/energy products (~20.4M barrels/month, ~8% of US liquid fuel imports)"},
    {"date": "2022-03-11", "primary_severity": "medium", "direction": "escalation",
     "description": "US revoked Russia MFN status, banned Russian seafood/spirits/diamonds imports (~$1B+ revenue), restricted luxury exports"},
    {"date": "2022-10-10", "primary_severity": "medium", "direction": "escalation",
     "description": "Market reacted to Oct 7 sweeping semiconductor export controls on China (advanced chips, chipmaking equipment, AI firms)"},

    # === 2023 ===
    {"date": "2023-10-18", "primary_severity": "medium", "direction": "escalation",
     "description": "Market reacted to Oct 17 tightened China chip export controls (more AI chips blocked, broader country coverage, 13 entities added)"},

    # === 2024 ===
    {"date": "2024-05-14", "primary_severity": "low", "direction": "escalation",
     "description": "Biden announced Section 301 tariff hikes on $18B of Chinese goods (EVs 100%, solar 50%, steel/aluminum 25%). SPY +0.46%, not trade-driven"},
    {"date": "2024-09-13", "primary_severity": "low", "direction": "escalation",
     "description": "Section 301 tariff increases on China finalized (EVs, batteries, minerals). SPY +0.52%, not trade-driven"},
    {"date": "2024-11-06", "primary_severity": "medium", "direction": "escalation",
     "description": "Trump election victory triggered repricing of expected higher future tariffs alongside tax cuts/deregulation. SPY +2.49%"},

    # === 2025 H1 ===
    {"date": "2025-02-03", "primary_severity": "high", "direction": "escalation",
     "description": "25% tariffs on Canada/Mexico + 10% on China took effect (same-day pauses trimmed selloff). ~45% of US goods imports affected"},
    {"date": "2025-02-04", "primary_severity": "medium", "direction": "escalation",
     "description": "China 10% tariff took effect, China retaliated with duties on US goods. Markets rose on negotiation hopes"},
    {"date": "2025-03-03", "primary_severity": "high", "direction": "escalation",
     "description": "25% tariffs on Canada/Mexico reimposed, China fentanyl tariff raised to 20%. ~$2.2T in North American trade affected"},
    {"date": "2025-03-05", "primary_severity": "medium", "direction": "de_escalation",
     "description": "One-month USMCA auto exemption from Canada/Mexico tariffs (~$474B auto import market)"},
    {"date": "2025-03-06", "primary_severity": "medium", "direction": "de_escalation",
     "description": "Broader USMCA exemption until April 2, but markets sold off on policy whiplash"},
    {"date": "2025-03-12", "primary_severity": "high", "direction": "escalation",
     "description": "Section 232 steel/aluminum tariffs restored at 25% on all countries globally. Canada/EU retaliated"},
    {"date": "2025-03-27", "primary_severity": "high", "direction": "escalation",
     "description": "Formal 25% tariff on imported automobiles announced, effective April 3 (~$474B auto import market)"},
    {"date": "2025-04-03", "primary_severity": "high", "direction": "escalation",
     "description": "Liberation Day reaction: 10% universal baseline + country-specific rates up to 50% on nearly all US imports. SPY -4.9%"},
    {"date": "2025-04-04", "primary_severity": "high", "direction": "escalation",
     "description": "China retaliated with 34% tariff on all US goods. SPY -5.9%"},
    {"date": "2025-04-08", "primary_severity": "high", "direction": "escalation",
     "description": "104% tariffs on Chinese imports confirmed for April 9, broader country tariffs proceeding. SPY -1.6%"},
    {"date": "2025-04-09", "primary_severity": "high", "direction": "de_escalation",
     "description": "90-day pause on most reciprocal tariffs at 10% rate. China excluded, raised to 125%. SPY +10.5%"},
    {"date": "2025-04-10", "primary_severity": "high", "direction": "escalation",
     "description": "Reality check: US-China tariffs now 145%, 10% blanket on all other imports still in place. SPY -4.4%"},
    {"date": "2025-04-14", "primary_severity": "medium", "direction": "de_escalation",
     "description": "Electronics (smartphones, computers, semiconductors) exempted from reciprocal tariffs"},
    {"date": "2025-05-12", "primary_severity": "high", "direction": "de_escalation",
     "description": "US-China Geneva deal: tariffs cut from 145% to 30% for 90 days. ~$600B bilateral trade. SPY +3.3%"},
    {"date": "2025-06-02", "primary_severity": "medium", "direction": "escalation",
     "description": "Steel/aluminum tariffs doubled to 50% on all imports globally (May 30 announcement, June 4 effective)"},

    # === 2025 H2 ===
    {"date": "2025-07-07", "primary_severity": "medium", "direction": "escalation",
     "description": "Tariff letters to 14 countries + reciprocal deadline pushed to Aug 1. Treated as renewed escalation"},
    {"date": "2025-07-08", "primary_severity": "medium", "direction": "escalation",
     "description": "50% tariff on imported copper announced + semiconductor/pharma tariffs signaled (Section 232)"},
    {"date": "2025-07-23", "primary_severity": "medium", "direction": "de_escalation",
     "description": "US-Japan trade deal: Japan auto tariff cut to 15% from 27.5%, $500-550B US investment commitment"},
    {"date": "2025-07-28", "primary_severity": "medium", "direction": "de_escalation",
     "description": "US-EU framework deal: 15% tariff on most EU goods, avoiding threatened 30% rate"},
    {"date": "2025-08-01", "primary_severity": "high", "direction": "escalation",
     "description": "New tariff rates 10-41% on 69 countries signed, effective one week later. Effective US tariff rate ~18-20%. SPY -1.6%"},
    {"date": "2025-08-12", "primary_severity": "high", "direction": "de_escalation",
     "description": "US-China tariff truce extended 90 days to Nov 10 (30% US, 10% China maintained). SPY +1.1%"},
    {"date": "2025-10-10", "primary_severity": "high", "direction": "escalation",
     "description": "Additional 100% tariff on Chinese imports + critical software export controls after China rare-earth restrictions. SPY -2.7%"},
]


# ============================================================
# TYPE 18: GEOPOLITICAL EVENTS
# ============================================================
# Primary severity from taxonomy:
#   low: regional conflict with limited US economic linkage
#   medium: conflict affecting commodities/trade routes, sanctions on mid-sized economy,
#           domestic political crisis
#   high: major economy conflict, top-10 economy sanctions, global supply chain disruption
#
# Secondary anchor: oil + VIX reaction within 24 hours
#   Oil spike > 5% and/or VIX spike > 20% → High regardless

GEOPOLITICAL_EVENTS: List[Dict] = [
    # ================================================================
    # ALL EVENTS BELOW VERIFIED BY CHATGPT WITH WEB SEARCH
    # Dates are the US trading day the market reacted (not event day)
    # SPY returns cross-referenced against our market_daily_features.csv
    # ================================================================

    # === 2020 ===
    {"date": "2020-01-27", "primary_severity": "high", "direction": "escalation",
     "description": "COVID outbreak spread beyond China, investors priced in global growth shock. SPY -1.60%"},
    {"date": "2020-02-24", "primary_severity": "high", "direction": "escalation",
     "description": "COVID outbreaks in Italy, South Korea, Iran convinced markets virus was global. SPY -3.32%"},
    {"date": "2020-03-09", "primary_severity": "high", "direction": "escalation",
     "description": "Saudi-Russia oil price war + pandemic panic. Oil -24.6%, biggest SPY drop since 2008. SPY -7.81%"},
    {"date": "2020-03-11", "primary_severity": "high", "direction": "escalation",
     "description": "WHO declared COVID a pandemic. SPY -4.87%"},
    {"date": "2020-03-12", "primary_severity": "high", "direction": "escalation",
     "description": "Trump Europe travel ban + global shutdown measures amplified pandemic panic. SPY -9.57%"},
    {"date": "2020-03-16", "primary_severity": "high", "direction": "escalation",
     "description": "US social-distancing guidance, city shutdowns, recession fears. VIX record 82.7. SPY -10.94%"},
    {"date": "2020-04-06", "primary_severity": "high", "direction": "de_escalation",
     "description": "Pandemic nearing first peak, NY deaths slowed, stabilization hopes. SPY +6.72%"},
    {"date": "2020-10-28", "primary_severity": "medium", "direction": "escalation",
     "description": "Renewed COVID wave in US/Europe + election uncertainty. Oil -5.5%, VIX 40.3. SPY -3.42%"},

    # === 2021 ===
    {"date": "2021-07-19", "primary_severity": "medium", "direction": "escalation",
     "description": "Delta variant COVID surge triggered risk-off on fears of renewed restrictions. Oil -7.5%. SPY -1.48%"},
    {"date": "2021-09-20", "primary_severity": "medium", "direction": "escalation",
     "description": "China Evergrande debt crisis contagion scare hit global risk sentiment. SPY -1.67%"},
    {"date": "2021-10-04", "primary_severity": "medium", "direction": "escalation",
     "description": "US debt ceiling crisis, Biden warned could not guarantee avoiding breach. SPY -1.29%"},
    {"date": "2021-10-07", "primary_severity": "low", "direction": "de_escalation",
     "description": "Senate temporary debt ceiling truce eased default fears. SPY +0.86%"},
    {"date": "2021-11-26", "primary_severity": "high", "direction": "escalation",
     "description": "Omicron COVID variant discovered, WHO 'variant of concern'. Oil -13%, VIX ~28.6. SPY -2.23%"},

    # === 2022 ===
    {"date": "2022-02-11", "primary_severity": "medium", "direction": "escalation",
     "description": "Washington warned Russia could invade Ukraine 'any day', war fears escalated. SPY -1.97%"},
    {"date": "2022-02-22", "primary_severity": "medium", "direction": "escalation",
     "description": "Putin recognized breakaway Ukraine regions, ordered troops in. S&P entered correction. SPY -1.07%"},
    {"date": "2022-02-24", "primary_severity": "high", "direction": "escalation",
     "description": "Russia full-scale invasion of Ukraine. Stocks reversed intraday, closed higher. Oil >$100 intraday. SPY +1.50%"},
    {"date": "2022-03-01", "primary_severity": "high", "direction": "escalation",
     "description": "War deepened, Western sanctions intensified, broader spillover fears. SPY -1.52%"},
    {"date": "2022-03-07", "primary_severity": "high", "direction": "escalation",
     "description": "US/Europe considered Russian oil ban, oil hit $130.50 intraday (2008-era highs). SPY -2.95%"},
    {"date": "2022-11-28", "primary_severity": "medium", "direction": "escalation",
     "description": "Rare protests across Chinese cities against zero-COVID raised China growth/supply chain fears. SPY -1.60%"},

    # === 2023 ===
    {"date": "2023-03-10", "primary_severity": "high", "direction": "escalation",
     "description": "Silicon Valley Bank shut by regulators, biggest US banking-fear selloff of the year. SPY -1.44%"},
    {"date": "2023-03-20", "primary_severity": "medium", "direction": "de_escalation",
     "description": "UBS emergency takeover of Credit Suisse eased contagion fears. SPY +0.96%"},
    {"date": "2023-05-02", "primary_severity": "medium", "direction": "escalation",
     "description": "First Republic failure/seizure rekindled systemic contagion fears. SPY -1.12%"},
    {"date": "2023-05-23", "primary_severity": "medium", "direction": "escalation",
     "description": "Deadlocked US debt ceiling talks, cleanest debt-limit risk-off session of year. SPY -1.12%"},
    {"date": "2023-06-01", "primary_severity": "medium", "direction": "de_escalation",
     "description": "House passed debt ceiling suspension bill, S&P hit 9-month high. SPY +0.95%"},
    {"date": "2023-10-18", "primary_severity": "medium", "direction": "escalation",
     "description": "Israel-Gaza war broadening fears drove sharp risk-off. Oil $88.32 (+1.9%). SPY -1.33%"},

    # === 2024 ===
    {"date": "2024-04-15", "primary_severity": "medium", "direction": "escalation",
     "description": "Iran drone/missile attack on Israel, first direct strike in decades. SPY -1.25%"},
    {"date": "2024-10-01", "primary_severity": "medium", "direction": "escalation",
     "description": "Iran fired ~200 ballistic missiles at Israel. Oil +2.4%. SPY -0.90% (borderline)"},

    # === 2025 ===
    {"date": "2025-04-21", "primary_severity": "high", "direction": "escalation",
     "description": "Trump attacked Fed Chair Powell, central-bank independence crisis. VIX 33.8, oil -2.5%. SPY -2.38%"},
    {"date": "2025-06-13", "primary_severity": "high", "direction": "escalation",
     "description": "Israel struck Iranian nuclear facilities, Iran retaliated with missiles. Oil +7.3%. SPY -1.12%"},
    {"date": "2025-06-24", "primary_severity": "medium", "direction": "de_escalation",
     "description": "Trump announced Israel-Iran ceasefire, war oil premium crushed. Oil -6.0%. SPY +1.10%"},
    {"date": "2025-11-10", "primary_severity": "medium", "direction": "de_escalation",
     "description": "Progress toward ending record US government shutdown sparked relief rally. SPY +1.56%"},
]


# ============================================================
# SEVERITY VALIDATION USING MARKET DATA
# ============================================================

def load_market_data() -> pd.DataFrame:
    """Load SPY, VIX, and oil data for severity validation."""
    features = pd.read_csv(MARKET_FEATURES_CSV, index_col=0, parse_dates=True)

    oil = pd.read_csv(OIL_DATA_CSV, index_col=0, parse_dates=True)

    # Merge oil into features
    features = features.join(oil, how="left")

    return features


def validate_and_adjust_severity(
    events: List[Dict],
    event_type_id: int,
    event_type_name: str,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each event, check the market reaction and potentially adjust severity.

    Type 17 secondary anchor: SPY daily return
      - SPY drop > 2% → confirms/upgrades to high
      - SPY drop > 1% → confirms medium or above
      - SPY barely moves → may downgrade if primary was high

    Type 18 secondary anchor: oil daily return + VIX level
      - Oil spike > 5% and/or VIX > 30 → confirms/upgrades to high
      - Oil spike > 3% or VIX > 25 → confirms medium
    """
    rows = []

    for event in events:
        date = pd.Timestamp(event["date"])
        primary_sev = event["primary_severity"]
        direction = event["direction"]
        description = event["description"]

        # Get market data for this date
        spy_ret = None
        vix_level = None
        oil_ret = None

        if date in features_df.index:
            spy_ret = features_df.at[date, "spy_daily_return"] if "spy_daily_return" in features_df.columns else None
            vix_level = features_df.at[date, "vix_close"] if "vix_close" in features_df.columns else None
            oil_ret = features_df.at[date, "oil_daily_return"] if "oil_daily_return" in features_df.columns else None

        # Convert to float safely
        spy_ret = float(spy_ret) if pd.notna(spy_ret) else None
        vix_level = float(vix_level) if pd.notna(vix_level) else None
        oil_ret = float(oil_ret) if pd.notna(oil_ret) else None

        # Start with primary severity
        final_severity = primary_sev

        # Apply secondary anchor adjustments
        evidence_parts = [description]

        if event_type_id == 17:
            # Trade/Tariff: use SPY reaction
            if spy_ret is not None:
                evidence_parts.append(f"SPY return: {spy_ret:.4f} ({spy_ret*100:.1f}%)")
                if spy_ret <= -0.02 and primary_sev != "high":
                    final_severity = "high"
                    evidence_parts.append("(upgraded: SPY drop > 2%)")
                elif spy_ret >= 0.02 and direction == "de_escalation":
                    evidence_parts.append("(confirmed: strong positive reaction)")

        elif event_type_id == 18:
            # Geopolitical: use oil + VIX
            if oil_ret is not None:
                evidence_parts.append(f"Oil return: {oil_ret:.4f} ({oil_ret*100:.1f}%)")
            if vix_level is not None:
                evidence_parts.append(f"VIX: {vix_level:.1f}")

            # Check for upgrade conditions
            oil_spike = oil_ret is not None and abs(oil_ret) >= 0.05
            vix_high = vix_level is not None and vix_level >= 30

            if (oil_spike or vix_high) and primary_sev != "high":
                final_severity = "high"
                evidence_parts.append("(upgraded: extreme oil/VIX reaction)")

        rows.append({
            "date": date,
            "event_type_id": event_type_id,
            "event_type_name": event_type_name,
            "severity": final_severity,
            "direction": direction,
            "evidence": ". ".join(evidence_parts),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load market data for validation
    print("Loading market data for severity validation...")
    features_df = load_market_data()
    print(f"  Loaded {len(features_df)} trading days of market data.")

    # Build Type 17 labels
    print("\nLabeling Type 17: Trade / Tariff Policy...")
    type17_df = validate_and_adjust_severity(
        events=TRADE_TARIFF_EVENTS,
        event_type_id=17,
        event_type_name="trade_tariff_policy",
        features_df=features_df,
    )
    print(f"  Generated {len(type17_df)} labels.")

    # Build Type 18 labels
    print("Labeling Type 18: Geopolitical Events...")
    type18_df = validate_and_adjust_severity(
        events=GEOPOLITICAL_EVENTS,
        event_type_id=18,
        event_type_name="geopolitical_event",
        features_df=features_df,
    )
    print(f"  Generated {len(type18_df)} labels.")

    # Combine manual labels
    manual_labels = pd.concat([type17_df, type18_df], ignore_index=True)
    manual_labels["date"] = pd.to_datetime(manual_labels["date"])
    manual_labels = manual_labels.sort_values(["date", "event_type_id"]).reset_index(drop=True)

    # Save manual labels separately
    manual_labels.to_csv(MANUAL_LABELS_CSV, index=False)
    print(f"\nSaved {len(manual_labels)} manual labels to: {MANUAL_LABELS_CSV}")

    # Merge with existing programmatic labels
    print("\nMerging with existing market-level labels...")
    existing_labels = pd.read_csv(EXISTING_LABELS_CSV)
    existing_labels["date"] = pd.to_datetime(existing_labels["date"])

    all_labels = pd.concat([existing_labels, manual_labels], ignore_index=True)
    all_labels = all_labels.sort_values(["date", "event_type_id"]).reset_index(drop=True)

    all_labels.to_csv(EXISTING_LABELS_CSV, index=False)
    print(f"Updated {EXISTING_LABELS_CSV} with {len(all_labels)} total labels.")

    # Print summary
    print("\n=== FULL LABEL SUMMARY (ALL TYPES 16-20) ===")
    summary = (
        all_labels.groupby(["event_type_id", "event_type_name", "severity"])
        .size()
        .reset_index(name="count")
        .sort_values(["event_type_id", "severity"])
    )
    print(summary.to_string(index=False))

    # Print Type 17 details
    print("\n=== TYPE 17: TRADE/TARIFF EVENTS ===")
    for _, row in type17_df.iterrows():
        print(f"  {row['date'].date()} | {row['severity']:6s} | {row['direction']:15s} | {row['evidence'][:100]}")

    # Print Type 18 details
    print("\n=== TYPE 18: GEOPOLITICAL EVENTS ===")
    for _, row in type18_df.iterrows():
        print(f"  {row['date'].date()} | {row['severity']:6s} | {row['direction']:15s} | {row['evidence'][:100]}")


if __name__ == "__main__":
    main()
