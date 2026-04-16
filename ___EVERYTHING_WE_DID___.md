# COMPLETE PROJECT CATALOG
Everything we built, ran, and found — from start to finish.


## PHASE 0: WORKSTREAM 1 — DAILY AI RESEARCH PIPELINE (pre-existing)

This was built before the current conversation. It's a daily pipeline that scans
S&P 100 stocks for recent price weakness, sends them to Claude for news research,
and stores structured results for later evaluation.

### 0.1 Ticker Selection
What: Scans 101 S&P 100 tickers, downloads recent daily bars via yfinance, computes
5-day Typical Price (TP = (High+Low+Close)/3) change, filters stocks with TP change
<= -2.5%. Returns ranked list of weakest performers (max 12).
Files: `ticker_selection.py`

### 0.2 Prompt Composition
What: Takes selected tickers and builds a structured research prompt for AI models.
Instructs the model to classify each ticker into: clear_negative_recent_news,
mixed_medium_negatives, or mostly_neutral_or_weak_news. Asks for explanation_strength
(strongly/partially/weakly explained) and likely_mean_reversion_candidate (yes/maybe/no).
Includes JSON schema for structured output.
Files: `prompt_composition.py`

### 0.3 Model Execution
What: Sends prompt to Claude (Sonnet) via Anthropic SDK with web search enabled
(max 40 searches) and structured JSON output. Parses response, tracks tokens,
estimates cost (~$3/1M input, $15/1M output, $0.01/search). Placeholder for
OpenAI adapter.
Files: `model_execution.py`

### 0.4 Response Storage
What: Saves all outputs to `daily_runs/YYYY-MM-DD/` with: input_tickers.json,
prompt.txt, and per-model folders containing raw.txt, parsed.json, metadata.json,
snapshot.json. Designed for auditability and future backtesting.
Files: `response_storage.py`

### 0.5 Orchestrator
What: Thin orchestrator connecting all 4 components: select tickers → compose
prompt → run model(s) → save outputs. Model-agnostic design.
Files: `main.py`

### 0.6 Automation
What: macOS launchd automation to run the pipeline daily after 5 PM PT on weekdays.
Tracks last successful run date to avoid duplicates. Activates venv, runs main.py,
logs to `logs/run.log`.
Files: `launchd_entrypoint.sh`, `run_pipeline.sh`

### 0.7 Project Design
What: Comprehensive README documenting the system architecture, 4-component design,
data flow, storage structure, model-agnostic principles, and future evaluation plan.
Files: `README.md`


## PHASE 1: SYSTEM DESIGN & DATA LABELING (Workstream 2)

### 1.1 System Blueprint & Event Taxonomy
What: Designed a 3-layer mean-reversion system (quantitative + news + lookup table) with a 20-type event taxonomy across stock/sector/market levels.
Files:
- `___SYSTEM_BLUEPRINT___.md` — 4-dimension framework
- `___EVENT_TAXONOMY___.md` — 20 event types with severity anchors
- `___FINAL_SYSTEM_LOGIC___.md` — classify → lookup → decide flow

### 1.2 Market-Level Labels (Types 16, 19, 20) — Programmatic
What: Computed 812 labels from VIX, Treasury yields, credit spreads.
Result: 580 of 1,121 event dates have at least one market label.
Files:
- `build_market_level_labels.py` — script
- `outputs/market_level_labels.csv` — 879 labels
- `outputs/market_daily_features.csv` — daily VIX/stress features

### 1.3 Market-Level Labels (Types 17, 18) — Manual via ChatGPT
What: 36 trade/tariff events + 31 geopolitical events, all verified with web search.
Files:
- `build_manual_market_labels.py` — merge script
- `outputs/manual_market_labels.csv` — raw manual labels
- `outputs/oil_daily_data.csv` — oil price data for verification

### 1.4 Sector-Level Labels (Types 12-15) — Semi-Programmatic
What: 7,181 labels from sector ETF divergence, demand shifts, regulation, contagion.
Files:
- `build_sector_level_labels.py` — script
- `outputs/sector_level_labels.csv` — 7,181 labels
- `outputs/sector_daily_features.csv` — daily sector features

### 1.5 Stock-Level Labels (Types 1-11) — ChatGPT Classification
What: 336 isolated big-drop events classified via 12 ChatGPT prompts (2 per year).
Result: 72% had identified catalyst, 28% no clear catalyst.
Files:
- `outputs/stock_level_labels.csv` — 336 labels (combined)
- `outputs/stock_level_labels_2020.csv` through `_2025.csv` — per-year
- `outputs/isolated_big_drops_to_label.csv` — events sent to ChatGPT

### 1.6 Lookup Table
What: Joined all 8,396 labels to 7,831 event outcomes. Key finding: no-catalyst drops bounce MORE (78.7%) than catalyst-driven (75.2%).
Files:
- `build_lookup_table.py` — script
- `outputs/events_fully_labeled.csv` — 7,831 events × 41 columns (MASTER FILE)
- `outputs/lookup_by_stock_event_type.csv` — success rates by news type
- `outputs/lookup_by_liquidity_stress.csv` — by market stress
- `outputs/lookup_stock_type_x_bucket.csv` — event type × drop size
- `outputs/lookup_combined.csv` — stock type × market stress


## PHASE 2: LIVE PIPELINE

### 2.1 Candidate Scanner
What: Scans S&P 100 for 3%+ drops from 20-day high.
Files: `candidate_scanner.py`

### 2.2 Market Context
What: Computes live VIX/stress level.
Files: `market_context.py`

### 2.3 News Classifier
What: Sends candidates to Claude with web search for taxonomy classification.
Files: `news_classifier.py`

### 2.4 Trade Scorer
What: Hierarchical lookup + confidence tiering.
Files: `trade_scorer.py`

### 2.5 Daily Pipeline
What: Orchestrator tying everything together. Run with `python3 daily_pipeline.py`.
Files: `daily_pipeline.py`


## PHASE 3: EARLY ANALYSIS (pre-iteration)

### 3.1 Target/Hold Analysis
What: Tested +1% to +10% targets with 14d to 120d holds.
Files:
- `target_hold_analysis.py`
- `outputs/target_hold_analysis.csv`, `outputs/target_hold_summary.csv`

### 3.2 Stop Loss Analysis
What: Tested -3% to -20% stop losses. Finding: stop losses ALWAYS hurt mean reversion.
Files:
- `stop_loss_analysis.py`
- `outputs/stop_loss_analysis.csv`

### 3.3 Portfolio Simulation (unfiltered)
What: Full portfolio sim without classification system.
Files:
- `portfolio_simulation.py`
- `outputs/portfolio_sim_equity.csv`, `outputs/portfolio_sim_trades.csv`

### 3.4 Portfolio Simulation v2 (with classification)
What: Portfolio sim WITH the news classification system.
Files:
- `portfolio_simulation_v2.py`
- `outputs/portfolio_daily_equity.csv`, `outputs/portfolio_trade_log.csv`

### 3.5 Entry Optimization (pre-iteration)
What: Tested limit orders, entry timing.
Files:
- `entry_optimization.py`
- `outputs/entry_optimization_overall.csv`, `_by_bucket.csv`, `_by_event_type.csv`

### 3.6 Various Filter Backtests
Files:
- `market_regime_filter_backtest.py` → `outputs/market_regime_*.csv`
- `volatility_filter_backtest.py` → `outputs/volatility_filter_*.csv`
- `sector_relative_filter_backtest.py` → `outputs/sector_relative_*.csv`
- `predrop_trend_filter_backtest.py` → `outputs/predrop_trend_*.csv`
- `parameter_robustness_sweep.py` → `outputs/parameter_sweep_*.csv`
- `walk_forward_backtest.py` → `outputs/walk_forward_*.csv`
- `market_context_signal_test.py` → `outputs/market_context_summary_*.csv`
- `final_market_filter_test.py` → `outputs/final_market_filter_summary.csv`
- `analyze_combo_filters.py` → `outputs/events_with_combo_labels.csv`
- `backtest_system.py` → `outputs/backtest_all_results.csv`, `backtest_labeled_results.csv`


## PHASE 4: ENTRY OPTIMIZATION (10 iterations)
Folder: `backtest_iterations/entry_optimization/`
Engine: `backtest_iterations/engine.py`
Cached data: `close_matrix.parquet`, `high_matrix.parquet`, `low_matrix.parquet`, `spy_benchmark.parquet`

| Version | What | Return | Files |
|---------|------|--------|-------|
| V01 | Baseline (no filter, +3%/60d) | +73.0% | v01_baseline.py, v01_baseline_results.txt |
| V02 | Seasonal filter (earnings months) | +76.2% | v02_seasonal.py, v02_seasonal_results.txt |
| V03 | Crisis avoidance (skip COVID/bear/tariff) | +94.8% | v03_crisis_avoid.py, v03_crisis_avoid_results.txt |
| V04 | Stress filter (VIX-based) | +82.1% | v04_stress_filter.py, v04_stress_filter_results.txt |
| V05 | Sector preference (SMH/XLK) | +89.3% | v05_sector_pref.py, v05_sector_pref_results.txt |
| V06a | No delay | +88.7% | v06_delayed_entry.py, v06a_nodelay_results.txt |
| V06b | 2-day delay | +101.3% | v06_delayed_entry.py, v06b_delay2_results.txt |
| V07 | Combined (crisis+delay+sector) | +112.1% | v07_combined.py, v07_combined_results.txt |
| V08 | Bigger positions (1/8) | +118.5% | v08_bigger_positions.py, v08_bigger_positions_results.txt |
| V09 | Tight filter | +105.2% | v09_tight_filter.py, v09_tight_filter_results.txt |
| V10 | Final (all best patterns) | +150.9% | v10_final.py, v10_final_results.txt |


## PHASE 5: EXIT OPTIMIZATION (10 iterations)
Folder: `backtest_iterations/exit_optimization/`

| Version | What | Return | Files |
|---------|------|--------|-------|
| V01 | Baseline exit (+3%/14d) | +73.0% | v01_baseline_exit.py, v01_baseline_exit_results.txt |
| V02 | Time cut 30d | varies | v02_time_cut_30d.py, v02_time_cut_30d_results.txt |
| V03 | Time cut 20d | varies | v03_time_cut_20d.py, v03_time_cut_20d_results.txt |
| V04a | Stop loss -10% | worse | v04_trailing_exit.py, v04a_stop10_results.txt |
| V04b | Stop loss -15% | worse | v04_trailing_exit.py, v04b_stop15_results.txt |
| V04c | Stop loss -20% | worse | v04_trailing_exit.py, v04c_stop20_results.txt |
| V05 | Hold period sweep (30/45/60/90d) | 60d best | v05_shorter_hold.py, v05_hold_*_results.txt |
| V06 | Adaptive hold | varies | v06_adaptive_hold.py, v06_adaptive_hold_results.txt |
| V07 | Target sweep (+2/3/4/5%) | +5% best | v07_higher_target.py, v07_target_*_results.txt |
| V08 | Push target (+6/7/8/10%) | diminishing | v08_push_target.py, v08_target_*_results.txt |
| V09 | Optimal combo (+5%/60-120d) | +5%/60d best | v09_optimal_combo.py, v09_5pct_*_results.txt |
| V10 | Final exit (+5%/60d/1/10) | +150.9% | v10_final_exit.py, v10_final_exit_results.txt |
| V10b | Final with 1/8 sizing | +150.9% | v10_final_exit.py, v10b_final_1_8_results.txt |


## PHASE 6: ADAPTIVE POSITION SIZING (15+ iterations)
Folder: `backtest_iterations/adaptive_sizing/`

| Version | What | Return | Files |
|---------|------|--------|-------|
| V01 | Baseline fixed 1/8 | +150.9% | v01_baseline.py, v01_baseline_results.txt |
| V02 | Bull boost (1/5 bull, 1/10 bear, fixed capital) | +234.2% | v02_bull_boost.py, v02_bull_boost_results.txt |
| V03 | Aggressive (1/4 strong, 1/6 mod, 1/10 bear) | +248.9% | v03_aggressive_bull.py, v03_aggressive_bull_results.txt |
| V04 | Equity-based (1/5 bull, 1/10 bear, CURRENT equity) | +339.1% | v04_equity_based.py, v04_equity_based_results.txt |
| V05 | Confidence-sized | +319.0% | v05_confidence_sized.py, v05_confidence_sized_results.txt |
| V06 | Max positions sweep (8/10/12/15) | all ~339% | v06_v04_more_positions.py, v06_max*_results.txt |
| V07 | Bull fraction sweep (1/3, 1/4, 1/5, 1/6) | 1/6=+349.1% | v07_push_bull_sizing.py, v07_1_*_results.txt |
| V08 | Bear fraction sweep (1/8 to 1/16) | 1/12 best | v08_bear_fraction_sweep.py, v08_bear*_results.txt |
| V09a | Bull/bear + confidence multiplier | +326.5% | v09_confidence_regime.py, v09a_results.txt |
| V09b | VIX-based regime | +226.4% | v09_confidence_regime.py, v09b_results.txt |
| V09c | Dual regime (SPY trend + VIX) | +329.6% | v09_confidence_regime.py, v09c_results.txt |
| V10a | MA period sweep (20/50/100/200) | 20MA=+380.1% | v10_final_sizing.py, v10a_ma*_results.txt |
| V10b | Graduated sizing (distance from MA) | +320.0% | v10_final_sizing.py, v10b_graduated_results.txt |
| V10c | Max positions with best sizing | max12=+359.0% | v10_final_sizing.py, v10c_max*_results.txt |
| V10d | Best combo (20MA, 1/6 bull, 1/14 bear, max 12) | **+401.5%** | v10d_best_combo.py, v10d_*_results.txt |
| V10e | Fine-tune around V10d | +401.5% confirmed | v10e_push_best.py, v10e_*_results.txt |


## PHASE 7: DCA / ADDING TO POSITIONS
Folder: `backtest_iterations/dca_adding/`

| Version | What | Return | Files |
|---------|------|--------|-------|
| V01 | 6 DCA configs (3%/5% threshold, 50%/100% add, 1-2 adds) | Best: +341% | v01_baseline_dca.py, v01_baseline_dca_results.txt |
| CONCLUSION | DCA HURTS. Positions that drop further are disproportionately losers (58-69% WR vs 87% overall). | — | — |


## PHASE 8: INTRADAY ENTRY (daily OHLC) — INVALIDATED
Folder: `backtest_iterations/intraday_entry/`
NOTE: V01-V05 results were caused by a DATA BUG (open_matrix had different dividend adjustments than close/high/low). All results invalidated.

| Version | What | Claimed Return | Actual (fixed) | Files |
|---------|------|---------------|-----------------|-------|
| V01 | OHLC analysis | +6,073% (FAKE) | — | v01_ohlc_baseline.py |
| V02 | Open entry deep dive | +6,073% (FAKE) | — | v02_open_entry_deep.py |
| V03 | Open + target sweep | +9,108% (FAKE) | — | v03_open_with_targets.py |
| V04 | Slippage testing | +3,757% (FAKE) | — | v04_realistic_slippage.py |
| V05 | Optimal config | +7,238% (FAKE) | — | v05_optimal_config.py |
| V06 | Sanity check | confirmed compounding amplified bug | — | v06_sanity_check.py |
| V07 | CORRECTED data | open ≈ close (gap = +0.14%) | +312-325% | v07_fixed_data.py, v07_fixed_data_results.txt |


## PHASE 9: HONEST ASSESSMENT
File: `backtest_iterations/honest_assessment.py`

| Config | Total Return | Avg Year | Worst Year |
|--------|-------------|----------|------------|
| Full cheating (crisis+tickers+compounding) | +401.5% | +33.0% | -3.2% |
| No cheating (no crisis/ticker biases, compounding) | +215.0% | +24.5% | -18.3% |
| No cheating + fixed sizing (most realistic) | +102.5% | +13.5% | -6.7% |
| Trade all drops + fixed sizing (pure baseline) | +99.8% | +13.3% | -5.6% |


## PHASE 10: INTRADAY DATA DOWNLOAD
What: Downloaded 53M 1-minute bars from Alpaca using 4 API keys in parallel.
Time: 18.7 minutes. Size: 1.48 GB.
Files:
- `download_intraday_data.py` — parallel downloader
- `verify_intraday_data.py` — cross-check vs yfinance
- `estimate_download.py` — speed benchmark
- `use_alpaca.py` — API test
- `intraday_data/bars_1min_2020.parquet` through `_2025.parquet` — raw data
- `intraday_data/bars_1min_2020_adjusted.parquet` through `_2025_adjusted.parquet` — split-adjusted


## PHASE 11: INTRADAY DEEP DIVE (33 investigations)
Folder: `backtest_iterations/intraday_deep/`
Plan: `___INTRADAY_PLAN___.md`

### Step 0: Split Adjustment
What: Applied stock split adjustments to all intraday data. Verified against yfinance.
Files: `step0_split_adjust.py`, `intraday_data/*_adjusted.parquet`

### Block A: Entry Timing (V01-V07)

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| V01 | Intraday profile of entry days | Entry day is flat (open→close = -0.06%). Winners drift up, losers drift down. | v01_intraday_profile.py, v01_results.txt, v01_avg_profile.csv |
| V02 | Buy at specific times | Best: 2:00 PM (+114.3% vs +102.5% at close). Morning worst. | v02_entry_time_backtest.py, v02_results.txt |
| V03 | VWAP entry | VWAP (+80.3%) WORSE than close. 2pm (+109.5%) confirms V02. | v03_vwap_entry.py, v03_results.txt |
| V04 | Limit orders | Limit at open-2% = +120.3% (+17.8% vs close). Fills 34% of time. | v04_limit_orders.py, v04_results.txt |
| V05 | First-hour signal | First 60 min UP 1%+ → 88.1% WR. DOWN 1%+ → 76.1% WR. 12pp spread. | v05_first_hour_signal.py, v05_results.txt |
| V06 | Wait for intraday dip | **Dip 0.5% then buy = +162.3%** (+58% vs close). Dip happens 69% of time. | v06_wait_for_dip.py, v06_results.txt |
| V07 | Entry delay sweep | 1-day (+104%) and 2-day (+102.5%) tied. 0-day worst. | v07_multiday_entry.py, v07_results.txt |

### Block B: Exit Timing (V08)

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| V08 | Winner exit day profile | Winners overshoot target by +1.87% avg. Target hit at 10:56 AM avg. 54% hit in first hour. | v08_exit_day_profile.py, v08_results.txt |

### Block C: Stop Loss (V09, V16)

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| V09 | Dip entry + target sweep | Dip entry + 10%/45d = +123.5%. Higher targets work with dip entry. | v09_v16_batch.py, v09_results.txt |
| V16 | Intraday drawdown | Winners -2.02% dd, losers -2.44%. Too small for reliable stop loss. | v09_v16_batch.py, v16_results.txt |

### Block D: Volume Signals (V17-V20)

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| V17 | Drop day volume | Drop day volume 1.65x normal. Small WR spread (83% low vol vs 78% extreme). | v17_v20_volume.py, v17_v20_results.txt |
| V19 | Entry day volume | **Quiet entry day (<0.8x): 90.6% WR. Surge (2x+): 63.3% WR. 27pp spread.** | v17_v20_volume.py, v17_v20_results.txt |
| V20 | Closing auction volume | Higher closing vol on drop day = better bounce (86% for 20%+ vs 77% for <5%). | v17_v20_volume.py, v17_v20_results.txt |

### Block E: Pattern Recognition (V25)

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| V25 | Overnight gap | **STRONGEST SIGNAL: T+1 gap up 3%+ → 97.5% WR. Gap down 3%+ → 61.1%. 36pp spread.** | v25_overnight_gap.py, v25_results.txt |

### Block F: News × Intraday (V26-V33)

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| V26 | Profile by event type | guidance_cut rallies all day (+0.22%). earnings_miss dips -0.56% by 11:30. regulatory_legal +1.00% by close. | v26_v33_news_cross.py, v26_v33_results.txt |
| V27 | Optimal time by event type | earnings_miss: buy 11:30 (saves 0.43%). guidance_cut: buy 10:00 (saves 0.31%). | v26_v33_news_cross.py, v26_v33_results.txt |
| V29 | Market stress × intraday | High stress: morning pop then fade. Calm: flat all day. | v26_v33_news_cross.py, v26_v33_results.txt |
| V31 | No-catalyst signature | No-catalyst (80% WR) flat all day. Catalyst (75% WR) more volatile. | v26_v33_news_cross.py, v26_v33_results.txt |

### Combined Strategy Tests

| Version | What | Key Finding | Files |
|---------|------|-------------|-------|
| v_final | 14 combined configs | Best fixed: +96% (all filters, 86% WR). Best adaptive: +192% (dip+gap, 19.6% CAGR). | v_final_combined.py, v_final_results.json |
| v_aggressive | 13 aggressive configs | More aggressive sizing = worse CAGR (bigger losses in bad years). | v_aggressive.py, v_aggressive_results.txt |
| v_regime_gating | 13 regime-gated configs | **BEST: SPY<20MA gate + 1/3 bull = +293.4%, 25.6% CAGR, 67% time in market** | v_regime_gating.py, v_regime_gating_results.txt |

Summary file: `backtest_iterations/intraday_deep/___FINDINGS_SUMMARY___.txt`


## CURRENT BEST STRATEGY (no look-ahead biases)

Config:
- Entry filter: 3%+ drops with sector outflow / VIX stress / Friday signals
- Overnight gap filter: only enter if T+1 gapped up (>0%)
- Dip entry: wait for 0.5% dip from open, fallback to close
- Regime gate: stop trading when SPY < 20-day MA
- Position sizing: 1/3 of equity in bull, 1/8 in bear (equity-based compounding)
- Target: +5%, max hold 60 days, no stop loss
- 2-day entry delay

Performance (2020-2025):
- Total return: +293.4%
- CAGR: 25.6%
- Win rate: 87%
- Trades: 166
- Time in market: 67% (33% capital free for other use)
- Worst year: -10.9% (2024)
- Best year: +77.1% (2020)

Year-by-year:
- 2020: +77.1% (48 trades, 3 losers)
- 2021: +23.1% (20 trades, 3 losers)
- 2022: +4.0% (27 trades, 5 losers)
- 2023: +13.8% (21 trades, 5 losers)
- 2024: -10.9% (17 trades, 4 losers)
- 2025: +62.2% (33 trades, 1 loser)

SPY comparison: ~15% CAGR over same period.
