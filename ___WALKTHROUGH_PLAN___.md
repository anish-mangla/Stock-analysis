# Guided Walkthrough Plan

A step-by-step plan for you to go through the entire codebase and understand
what everything does, what the results mean, and how it all connects.

Estimated total time: 3-4 hours if you go through everything.
You can do it in chunks — each session below is independent.


## Session 1: The Core Idea (30 min)
Read these in order. No code — just understanding the system design.

1. Read `README.md` — the original project vision
2. Read `___SYSTEM_BLUEPRINT___.md` — the 4-dimension framework we designed
3. Read `___EVENT_TAXONOMY___.md` — the 20 event types (skim the severity tables)
4. Read `___FINAL_SYSTEM_LOGIC___.md` — how classify → lookup → decide works

After this you should understand: what problem we're solving, how we think about
stock drops, and the 3-layer system (quantitative + news + lookup).


## Session 2: Workstream 1 — The Daily Pipeline (30 min)
This is the original code you built. Walk through the pipeline end-to-end.

1. Open `ticker_selection.py` — read the `get_5day_tp_change()` function.
   This is where stocks get scanned. Key: Typical Price formula, -2.5% threshold.

2. Open `prompt_composition.py` — read `_build_research_prompt()`.
   This is the exact prompt Claude sees. Key: the 3 categories, the rules.

3. Open `model_execution.py` — read `run_claude()`.
   Key: web search enabled, structured JSON output, cost estimation.

4. Open `main.py` — see how the 4 components connect.
   It's intentionally thin — just wiring.

5. Try running it: `python3 daily_pipeline.py --test` (no API cost)
   Look at the output in `daily_runs/` to see what gets saved.


## Session 3: The Data We Built (45 min)
This is the labeling work — the foundation everything else builds on.

1. Open `build_market_level_labels.py` — skim the logic for VIX/stress detection.
   Then look at `outputs/market_level_labels.csv` (open in a spreadsheet).
   Key: 879 market-level labels across 5 types.

2. Open `build_sector_level_labels.py` — skim sector rotation logic.
   Look at `outputs/sector_level_labels.csv`.
   Key: 7,181 sector labels, sector ETF vs SPY divergence.

3. Look at `outputs/stock_level_labels.csv` — these are the ChatGPT-classified events.
   Key columns: stock_event_type, stock_event_severity, stock_event_description.
   Sort by stock_event_type to see the distribution.

4. Open `build_lookup_table.py` — this joins everything together.
   Look at `outputs/events_fully_labeled.csv` — this is the MASTER FILE (7,831 rows).
   Look at `outputs/lookup_by_stock_event_type.csv` — success rates by news type.

After this you should understand: what data we have, how it was created, and
what the lookup table tells us about which drops bounce.


## Session 4: The Backtest Engine (20 min)
Understanding the engine is critical — every iteration uses it.

1. Open `backtest_iterations/engine.py`
2. Read `run_backtest()` — follow the main loop:
   - For each trading day: process exits → process entries → track equity
   - Key parameters: target_return, max_hold_days, entry_delay, sizing_fn
3. Read `_compile_results()` — what metrics we compute
4. Key concept: the `entry_filter` function is passed in by each iteration script.
   The engine doesn't decide WHAT to trade — the iteration script does.


## Session 5: Entry Optimization Journey (30 min)
Go through the 10 iterations that took us from +73% to +151%.

Open each file, read the comment at the top, then look at the results file:

1. `backtest_iterations/entry_optimization/v01_baseline.py` → `v01_baseline_results.txt`
   Baseline: trade everything, +3%/60d. Result: +73.0%

2. `v03_crisis_avoid.py` → `v03_crisis_avoid_results.txt`
   Skip COVID/bear/tariff. Result: +94.8%. Key learning: crisis avoidance is huge.

3. `v06_delayed_entry.py` → `v06b_delay2_results.txt`
   2-day delayed entry. Result: +101.3%. Key: let dust settle.

4. `v10_final.py` → `v10_final_results.txt`
   All best patterns combined. Result: +150.9%. Read the entry_filter function
   carefully — this is the logic that decides which trades to take.

Skip v02, v04, v05, v07-v09 unless you want the full detail — they're
incremental steps between the ones above.


## Session 6: Exit Optimization Journey (20 min)
10 iterations testing how to sell.

Key files to read:

1. `backtest_iterations/exit_optimization/v04_trailing_exit.py` → `v04a_stop10_results.txt`
   Stop losses tested. Result: ALL hurt. Key learning: don't use stop losses.

2. `v07_higher_target.py` → `v07_target_5pct_results.txt`
   Target sweep. Result: +5% target is the sweet spot.

3. `v10_final_exit.py` → `v10_final_exit_results.txt`
   Final: +5% target, 60d hold, no stop loss. Result: +150.9%.


## Session 7: Adaptive Sizing — The Big Jump (30 min)
This is where +151% became +401%. Read carefully.

1. `backtest_iterations/adaptive_sizing/v01_baseline.py` → `v01_baseline_results.txt`
   Fixed 1/8 sizing. Result: +150.9%.

2. `v04_equity_based.py` → `v04_equity_based_results.txt`
   THE BREAKTHROUGH: size as fraction of CURRENT equity, not initial capital.
   Bull (SPY > 50MA) = 1/5, bear = 1/10. Result: +339.1%.
   Read the sizing_fn function — it's simple but powerful.

3. `v07_push_bull_sizing.py` → `v07_1_6_bull_results.txt`
   Fraction sweep. Best: 1/6 bull, 1/12 bear = +349.1%.

4. `v10_final_sizing.py` → look at v10a_ma20_results.txt
   20-day MA for regime detection. Result: +380.1%.

5. `v10d_best_combo.py` → `v10d_20MA_1_6_bull_1_14_bear_max_12_results.txt`
   Final best: +401.5%. Read the config carefully.


## Session 8: The Data Bug & Honest Assessment (20 min)
Important context for understanding the real numbers.

1. Read `backtest_iterations/intraday_entry/v07_fixed_data.py` — the corrected test.
   Key: open_matrix had wrong dividend adjustments, inflating results by 20-30%.
   After fix: open entry ≈ close entry.

2. Run `python3 backtest_iterations/honest_assessment.py`
   This shows 4 versions: full cheating → no cheating → fixed sizing → pure baseline.
   Key takeaway: without look-ahead biases, realistic return is ~13-17% per year
   with fixed sizing, or ~20% CAGR with compounding.

3. Read the "Known Biases" section in `___FULL_PROJECT_SUMMARY___.md`.


## Session 9: Intraday Deep Dive (45 min)
The 53M minute-bar analysis. This is the newest work.

1. Read `___INTRADAY_PLAN___.md` — the 33 investigation plan.

2. Read `backtest_iterations/intraday_deep/___FINDINGS_SUMMARY___.txt`
   This is the TL;DR of all findings. Read this first.

3. Key scripts to open and understand:

   `v01_intraday_profile.py` → `v01_results.txt`
   Entry day is basically flat. No magic dip.

   `v06_wait_for_dip.py` → `v06_results.txt`
   Wait for 0.5% dip = +162% (best entry strategy). Read the logic.

   `v25_overnight_gap.py` → `v25_results.txt`
   STRONGEST SIGNAL: overnight gap predicts success. 97.5% WR for gap up 3%+.

   `v17_v20_volume.py` → `v17_v20_results.txt`
   Entry day volume: quiet = 90.6% WR, surge = 63.3% WR.

   `v26_v33_news_cross.py` → `v26_v33_results.txt`
   Different event types have different intraday shapes.

4. `v_regime_gating.py` → `v_regime_gating_results.txt`
   The final best strategy: stop trading when SPY < 20MA.
   Result: +293.4% total, 25.6% CAGR, only in market 67% of time.


## Session 10: The Live Pipeline (20 min)
How the system actually runs day-to-day.

1. Open `candidate_scanner.py` — the new scanner (replaced ticker_selection.py).
   Key: drawdown from 20-day high + single-day drops.

2. Open `market_context.py` — live VIX/stress computation.

3. Open `news_classifier.py` — how Claude classifies drops using our taxonomy.
   Key: batched at 8 per call, 65s rate limit delays.

4. Open `trade_scorer.py` — hierarchical lookup + confidence tiering.
   Key: 5-level fallback, confidence tiers (high/medium/low/skip).

5. Open `daily_pipeline.py` — the orchestrator.
   Try: `python3 daily_pipeline.py --test`


## After All Sessions

You should now understand:
- The system design and why we built it this way
- How the data was collected and labeled
- What each backtest iteration tested and found
- Where the real edge is (and where we were fooling ourselves)
- The current best strategy and its realistic expected returns
- How the live pipeline works

Next steps to discuss:
- Should we update the live pipeline with the best config?
- Do we want to explore leverage, options, or a larger universe?
- Paper trading validation plan
