# FULL PROJECT SUMMARY

## What We Built

A systematic mean-reversion trading system for S&P 100 stocks that identifies
when a stock drop is likely temporary and will bounce back.

The core idea: When a stock drops significantly, determine whether the drop was
caused by a real fundamental problem (avoid it) or by market sentiment/overreaction
(buy it and wait for the bounce).


## The System Has Three Layers

### Layer 1 — Quantitative (price data)
Scans S&P 100 daily for stocks that dropped 3%+ from their recent high. Computes
features like drop size, volatility-normalized shock, sector-relative performance,
pre-drop trend, and market stress level (VIX, credit spreads).

### Layer 2 — News Classification (Claude AI)
Sends candidates to Claude with web search to classify what caused the drop using
our 20-type event taxonomy. Claude identifies whether it was an earnings miss,
guidance cut, product failure, competitive threat, regulatory action, or no clear
catalyst. The system then looks up the historical success rate for that classification.

### Layer 3 — Historical Lookup Table
Built from 8,396 labels across 7,831 historical events (2020-2025), all verified
with ChatGPT web search. Converts any classification into an objective bounce
probability. No opinions, no weights — just "this situation happened X times
before and bounced Y% of the time."


## What We Built (Files)

### Reference Documents
- `___SYSTEM_BLUEPRINT___.md` — The 4-dimension framework and system design
- `___EVENT_TAXONOMY___.md` — 20 event types with concrete severity anchors
- `___FINAL_SYSTEM_LOGIC___.md` — The classify → lookup → decide flow
- `___TASKS___.md` — Task tracking for all workstreams

### Data Collection & Labeling
- `build_market_level_labels.py` — 879 market-level labels (types 16-20)
- `build_manual_market_labels.py` — ChatGPT-verified trade/tariff + geopolitical events
- `build_sector_level_labels.py` — 7,181 sector-level labels (types 12-15)
- Stock-level labels (2020-2025) — 336 events classified via 12 ChatGPT prompts
- `build_lookup_table.py` — Joins all labels to outcomes, computes success rates

### Live Pipeline
- `candidate_scanner.py` — Scans S&P 100 for drop candidates
- `market_context.py` — Computes live market stress level
- `news_classifier.py` — Sends candidates to Claude for classification
- `trade_scorer.py` — Hierarchical lookup + confidence tiering
- `daily_pipeline.py` — Unified orchestrator (run with `python3 daily_pipeline.py`)

### Backtesting Engine & Iterations
- `backtest_iterations/engine.py` — Shared backtest engine with sizing_fn support
- `backtest_iterations/entry_optimization/` — 10 entry filter iterations (v01-v10)
- `backtest_iterations/exit_optimization/` — 10 exit strategy iterations (v01-v10)
- `backtest_iterations/adaptive_sizing/` — 10+ position sizing iterations (v01-v10e)
- `backtest_iterations/dca_adding/` — DCA/adding to positions exploration
- `backtest_iterations/intraday_entry/` — Intraday entry optimization (v01-v05)


## Optimization Journey

### Phase 1: Entry/Exit Optimization (from +73% to +151%)

| Version | Return | What Changed |
|---------|--------|-------------|
| Baseline (no filter) | +73.0% | Trade everything |
| + Crisis avoidance | +94.8% | Skip COVID/bear/tariff periods |
| + 2-day delayed entry | +101.3% | Let dust settle before buying |
| + Sector/ticker preferences | +112.1% | Prefer semis, avoid bad bouncers |
| + Higher target (+3% → +5%) | +143.6% | Aim for more profit per trade |
| + 1/8 position size | +150.9% | Slightly bigger positions |

### Phase 2: Adaptive Position Sizing (from +151% to +401%)

| Version | Return | What Changed |
|---------|--------|-------------|
| V01 Fixed 1/8 | +150.9% | Baseline from Phase 1 |
| V04 Equity-based (1/5 bull, 1/10 bear, 50MA) | +339.1% | Compound on current equity |
| V07 Best fraction (1/6 bull, 1/12 bear, 50MA) | +349.1% | Optimized bull/bear fractions |
| V10a 20MA regime | +380.1% | Faster regime detection |
| **V10d BEST** (1/6 bull, 1/14 bear, 20MA, max 12) | **+401.5%** | **Optimal sizing config** |

Key findings:
- Equity-based sizing (compound on current equity) is the #1 improvement
- SPY above/below 20-day MA is the best bull/bear regime detector
- 1/6 of equity in bull, 1/14 in bear is the sweet spot
- DCA (adding to losing positions) HURTS returns — don't throw good money after bad

### Phase 3: Intraday Entry Optimization — INVALIDATED

V01-V05 results (+6,073% to +7,238%) were caused by a DATA BUG: the open_matrix
was cached with different dividend/split adjustments than close/high/low matrices.
For dividend-paying stocks (CVX, XOM, C, AMGN), the open prices were 20-30% lower
than they should have been, creating a fake entry advantage.

After fixing the data (V07), open entry is roughly equivalent to close entry:
- +5%/60d adaptive, close entry: +325.4%
- +5%/60d adaptive, open entry: +312.6%
- The real open-vs-close gap on entry days is +0.14% (essentially zero)

Intraday entry optimization requires real intraday data (Polygon.io) to explore
properly. Daily OHLC does not provide a meaningful edge.


## Current Best Configuration

- Entry: Buy at CLOSE, 2-day delay after drop event
- Target: +5% profit per trade
- Max hold: 60 trading days
- Position sizing: Equity-based, 1/6 of current equity in bull (SPY > 20MA), 1/14 in bear
- Max positions: 12 simultaneous
- No stop loss
- Skip: crisis periods, bad tickers (PFE/TMUS/ACN/CVS/AMT/BLK/IBM/LMT/DHR/MSFT)
- Prefer: best tickers (NVDA/AMZN/AMAT/AVGO/GE/INTU/MA/AXP/LOW/MO/COF/BMY/MS/MDT/HON)
- Prefer: SMH and XLK sectors, Friday drops, sector outflow, medium stress


## Performance (Best Config — Adaptive Sizing)

| Metric                    | System       | SPY Buy & Hold |
|---------------------------|--------------|----------------|
| Total return (2020-2025)  | +401.5%      | +96.0%         |
| Annualized                | ~31%         | ~16.0%         |
| Win rate                  | 87%          | —              |
| Total trades              | 431          | —              |
| Losers                    | 55           | —              |
| Avg hold                  | 19 days      | —              |
| Capital utilization       | 69%          | 100%           |
| Worst year                | -3.2% (2022) | -19.4% (2022)  |
| Best year                 | +65.9% (2020)| +26.9% (2021)  |

### Year-by-Year Breakdown (Best Config)
- 2020: +65.9% (113 trades, 9 losers) — caught COVID recovery
- 2021: +22.4% (73 trades, 15 losers) — steady
- 2022: -3.2% (22 trades, 2 losers) — barely scratched while SPY lost 19%
- 2023: +50.2% (82 trades, 12 losers) — excellent year
- 2024: +10.1% (65 trades, 10 losers) — weakest positive year
- 2025: +52.8% (76 trades, 7 losers) — strong year


## Known Biases & Caveats

### Look-Ahead Biases (things we know because we have the full dataset)
1. **CRISIS_PERIODS** — hardcoded date ranges for COVID, 2022 bear, tariff chaos.
   In live trading, we would NOT know when a crisis starts or ends. The live system
   uses VIX/stress detection instead, which is less precise.
2. **BAD_TICKERS** — PFE, TMUS, ACN, CVS, AMT, BLK, IBM, LMT, DHR, MSFT were
   identified by looking at which tickers performed worst in 2020-2025 data.
   These may change in the future.
3. **BEST_TICKERS** — NVDA, AMZN, AMAT, AVGO, etc. were identified by looking at
   which tickers bounced best. Same issue — past performance ≠ future.
4. **BEST_SECTORS** — SMH, XLK identified from historical data.

### Other Biases
5. **Target exit at exact price** — assumes limit sell fills at exactly the target
   price when the daily high touches it. In practice, might get slightly worse fill.
6. **Equity-based compounding** — amplifies both gains and losses. The +401% number
   is heavily driven by early wins in 2020 compounding through later years.
7. **No transaction costs** — no commissions or fees modeled (though these are
   minimal for S&P 100 stocks on modern brokers).

### Data Bug Found & Fixed
The intraday entry results (V01-V05, showing +6,073% to +7,238%) were caused by
a price adjustment mismatch: open_matrix was cached with current dividend adjustments
while close/high/low matrices used older adjustments. For dividend-paying stocks,
this created a fake 20-30% entry advantage. After fixing, open entry ≈ close entry.


## What's Still Needed To Go Live

1. Update daily_pipeline.py to use adaptive sizing and best filters
2. Replace hardcoded CRISIS_PERIODS with live VIX/stress detection
3. Add position tracking and P&L monitoring
4. Connect to the launchd automation for daily execution
5. Run live for a few weeks in paper-trading mode to validate
6. Polygon.io integration for real intraday data (user approved $79/month)
