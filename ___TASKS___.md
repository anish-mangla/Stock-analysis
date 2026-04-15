# WORKSTREAM B — News Intelligence Layer: Task List

## Step 1: Market-Level Labels (Types 16, 19, 20) — PROGRAMMATIC
- [x] Write `build_market_level_labels.py`
  - [x] Type 16: Monetary Policy Shock (SHY/TLT yield proxy on Fed days + extreme rate days)
  - [x] Type 19: Macro Data Shock (curated reference table of significant CPI/NFP/GDP surprises)
  - [x] Type 20: Liquidity/Credit Stress (VIX level + HYG credit spread proxy)
- [x] Run the script and validate output
  - 812 labels generated (101 T16, 35 T19, 676 T20)
  - 580 of 1121 event dates have at least one market label
- [x] Spot-check labels against known events
  - March 2020 COVID: ✓ high liquidity stress, emergency Fed cut detected
  - June 2022 75bp hike: ✓ hot CPI + rate shock + credit stress all detected
  - Sep 2022 hot CPI: ✓ high macro shock + credit stress escalation
  - April 2025 tariff crash: ✓ VIX 21→52 correctly escalated (tariff label pending Step 2)

## Step 2: Market-Level Labels (Types 17, 18) — MANUAL
- [x] Type 17: Trade/Tariff Policy — 36 events labeled (ChatGPT-verified with web search)
  - 2020: 5 events (Phase 1 deal, Huawei restrictions, WTO ruling)
  - 2021: 2 events (Boeing-Airbus truce, US-EU steel deal)
  - 2022: 3 events (Russian oil ban, MFN revocation, China chip controls)
  - 2023: 1 event (China chip controls tightened)
  - 2024: 3 events (Biden China tariffs, Section 301 finalized, Trump election repricing)
  - 2025 H1: 15 events (full Liberation Day cycle, Geneva deal, steel/aluminum)
  - 2025 H2: 7 events (69-country tariffs, Japan/EU deals, China 100% tariff)
  - All dates verified as market reaction days (not announcement days)
  - All SPY returns cross-referenced against our market_daily_features.csv
- [x] Type 18: Geopolitical Events — 31 events labeled (ChatGPT-verified with web search)
  - 2020: 8 events (COVID pandemic progression, Saudi-Russia oil war, second wave)
  - 2021: 5 events (Delta variant, Evergrande, debt ceiling, Omicron)
  - 2022: 6 events (Russia-Ukraine war buildup/invasion/escalation, China zero-COVID protests)
  - 2023: 6 events (SVB/banking crisis, First Republic, debt ceiling, Israel-Gaza)
  - 2024: 2 events (Iran-Israel strikes)
  - 2025: 4 events (Trump vs Fed Chair, Israel-Iran war/ceasefire, govt shutdown resolution)
  - All dates verified as market reaction days
  - All SPY/VIX/oil returns cross-referenced against our data
  - Removed false positives from original list (Soleimani, Capitol riot, Hamas Oct 9, Nord Stream)
- [x] Merged into unified market_level_labels.csv
  - 874 total labels across all 5 market-level types
  - Type 17 severity breakdown: 13 high, 17 medium, 6 low
  - Verified multi-label stacking (e.g., April 2025 = tariff + liquidity stress)

## Step 3: Sector-Level Labels (Types 12-15) — SEMI-PROGRAMMATIC
- [x] Type 14: Sector Rotation — 4,564 labels computed from sector ETF vs SPY
  - 12 sector ETFs tracked, daily divergence measured
  - Severity: low (1-2%), medium (2-5%), high (>5% divergence from SPY)
  - Sanity checked: March 2020 oil crash, April 2025 Liberation Day both correct
- [x] Type 13: Sector Demand Shift — 2,606 labels computed from underlying drivers
  - Energy: oil (CL=F), Financials/REITs/Utilities: rates (TLT), Materials: copper (CPER)
  - Severity based on sigma of driver move (medium: 1-2σ, high: >2σ)
- [x] Type 12: Sector Regulation/Policy — 7 events (ChatGPT-verified)
  - XLE: OPEC+ breakdown (2020), Biden leasing pause (2021)
  - SMH: China chip export controls (2022, 2023, 2025)
  - XLV: Medicare Advantage rate cut (2024)
  - XLI: Infrastructure bill (2021)
  - All sector ETF returns verified against our data
- [x] Type 15: Sector Contagion — 4 events (ChatGPT-verified)
  - XLF: SVB collapse (2023-03-13), First Republic (2023-05-02)
  - SMH: ASML demand warning (2024-10-15), DeepSeek AI repricing (2025-01-27)
  - All clearly shared problems, not company-specific
- [x] Output: sector_level_labels.csv — 7,181 total labels

## Step 4: Stock-Level Labels (Types 1-11) — AI-ASSISTED
- [x] Define the exact Claude/ChatGPT prompt for event classification using our taxonomy
- [x] Select the top ~336 isolated events (7%+ drops on non-crash days) for labeling
  - Identified 28 market-wide crash dates (560 events) → labeled as market/sector driven
  - Identified 281 isolated dates (336 events) → needed individual research
- [x] Run ChatGPT classification in batches (2 passes per year for quality)
  - 2020: 39 events (64% identified catalyst)
  - 2021: 45 events (64% identified catalyst)
  - 2022: 74 events (70% identified catalyst)
  - 2023: 54 events (72% identified catalyst)
  - 2024: 57 events (82% identified catalyst)
  - 2025: 67 events (75% identified catalyst)
- [x] All SPY returns cross-checked, all dates verified as market reaction days
- [x] Output: stock_level_labels.csv — 336 total labels
  - 242 events with identified catalyst (72%)
  - 94 events with no clear catalyst (28%)
  - Top event types: guidance_cut (86), demand_weakness (55), no_clear_catalyst (94),
    earnings_miss (22), regulatory_legal (19), product_service_failure (17)

## Step 5: Build the Lookup Table
- [x] Join all labels (market + sector + stock) to events.csv outcomes
  - Created events_fully_labeled.csv: 7,831 events × 41 columns
- [x] Compute historical success rates by (event_type, severity)
- [x] Compute success rates by common multi-label combinations
- [x] Output files:
  - lookup_by_stock_event_type.csv — success rates by news category + severity
  - lookup_by_liquidity_stress.csv — success rates by market stress level
  - lookup_stock_type_x_bucket.csv — cross-tabulation of event type × drop size
  - lookup_combined.csv — stock event type × market stress
  - events_fully_labeled.csv — master file with all labels joined
- [x] KEY FINDINGS:
  - No-catalyst drops bounce MORE (78.7%) than catalyst-driven drops (75.2%)
  - Product/service failures are the worst category (46.2% success at high severity)
  - Tariff-driven drops bounce strongly (90.6%)
  - Panic selloffs (high liquidity stress) bounce well (84.0%)
  - Sweet spot: no catalyst + market stressed = 85.7% success rate

## Step 6: Integrate Into Live Pipeline
- [x] candidate_scanner.py — new scanner using drawdown from 20-day high + single-day drops
  - Replaces old ticker_selection.py (5-day Typical Price)
  - Computes all quantitative features during scan
- [x] market_context.py — computes live market stress level (VIX, credit spreads, SPY)
  - Same logic as build_market_level_labels.py but for today's data
- [x] news_classifier.py — sends candidates to Claude for taxonomy classification
  - Uses structured JSON output with our 11 event types
  - Web search enabled for real-time news research
  - Returns event_type + severity for each candidate
- [x] trade_scorer.py — hierarchical lookup + confidence tiering + entry price
  - 5-level fallback: event+severity+market+bucket → event+severity+market → event+severity → event → bucket
  - Confidence tiers: high (≥80%) → medium (65-80%) → low (50-65%) → skip (<50%)
  - Entry price: high=close, medium=close-0.75%, low=close-1.5%, skip=don't trade
- [x] daily_pipeline.py — unified orchestrator tying everything together
  - Scan → Context → Classify → Score → Rank → Output
  - Saves full audit trail (scan, context, classifications, trades)
  - Test mode (--test) skips Claude for cost-free testing
- [x] End-to-end test successful: found 20 candidates, scored and ranked them, saved outputs

---

## Step 7: Adaptive Position Sizing (backtest_iterations/adaptive_sizing/)
- [x] V01 baseline: fixed 1/8 = +150.9%
- [x] V02 bull boost: 1/5 bull, 1/10 bear (fixed capital) = +234.2%
- [x] V03 aggressive: 1/4 strong bull, 1/6 moderate, 1/10 bear = +248.9%
- [x] V04 equity-based: 1/5 bull, 1/10 bear (CURRENT equity) = +339.1% ← compounding breakthrough
- [x] V05 confidence-sized: +319.0%
- [x] V06 max positions test: all same at +339.1%
- [x] V07 fraction sweep: 1/6 bull, 1/12 bear = +349.1%
- [x] V08 bear fraction sweep: 1/12 bear confirmed best with 50MA
- [x] V09 confidence/VIX regime: all worse than simple bull/bear
- [x] V10 final: 20MA regime + 1/14 bear + max 12 = **+401.5%** ← BEST SIZING

## Step 8: DCA / Adding to Positions (backtest_iterations/dca_adding/)
- [x] V01: Tested 6 DCA configs (3%/5% threshold, 50%/100% add, 1-2 adds)
- [x] CONCLUSION: DCA HURTS returns. Best DCA (+341%) < no DCA (+401.5%)
  - Positions that drop further are disproportionately losers (58-69% WR vs 87% overall)

## Step 9: Intraday Entry Optimization (backtest_iterations/intraday_entry/)
- [x] V01-V05: INVALIDATED — data bug (open_matrix had different dividend adjustments)
- [x] V06: Sanity check confirmed compounding was amplifying the fake advantage
- [x] V07: CORRECTED — re-cached open_matrix with auto_adjust=False
  - Open entry ≈ close entry (real gap is +0.14%, essentially zero)
  - +5%/60d adaptive, close: +325.4% vs open: +312.6%
  - Intraday optimization needs real Polygon.io data, not daily OHLC

## Current Status
✅ Steps 1-6 COMPLETE — Data pipeline, labeling, lookup tables, live pipeline
✅ Step 7 COMPLETE — Adaptive sizing: +150.9% → +401.5% (2.66x improvement)
✅ Step 8 COMPLETE — DCA exploration: confirmed DCA hurts mean reversion
✅ Step 9 DONE — Intraday entry: V01-V05 INVALIDATED (data bug), V07 corrected. Open ≈ close entry.

BEST CONFIGURATION (VALIDATED):
  Close entry, 2-day delay, +5% target, 60d hold, 1/6 bull 1/14 bear (20MA), max 12
  Return: +401.5% over 2020-2025 (with equity-based compounding)
  Win rate: 87% | 431 trades | 55 losers | avg hold 19 days
  
  NOTE: This includes look-ahead biases (crisis periods, bad/best ticker lists)
  that would not be available in live trading. Realistic live performance will be lower.

Next steps:
- [ ] Quantify the impact of each look-ahead bias (run without crisis avoidance, without ticker lists)
- [ ] Update daily_pipeline.py with best config (adaptive sizing)
- [ ] Polygon.io integration for real intraday data
- [ ] Paper-trading validation
