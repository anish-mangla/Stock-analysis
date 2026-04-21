---
inclusion: manual
---

# Discussion Plan: Walk User Through Everything We Built

## How to use this
The user wants a verbal walkthrough of the entire project. Follow this checklist
topic by topic. After each topic, ask if they have questions before moving on.
If we go down a rabbit hole, come back to the next unchecked item.

## Checklist

- [x] TOPIC 1: The Big Picture (what the system does in 2 sentences)
  - Covered: 3% = single-day drop, +5% target from buy price (partial bounce), events file location
  - User wants to look at outputs/events_fully_labeled.csv themselves
- [x] TOPIC 2: How we detect drops (the scanner — what triggers a trade candidate)
  - Covered: single-day 3%+ drop, sector ETF static map, sector rotation (ETF vs SPY same day),
    market stress (VIX base + HYG 5d bump), exact thresholds for all
- [x] TOPIC 3: The event taxonomy (20 types of news, how we classify drops)
  - Covered: 11 stock-level, 4 sector-level, 5 market-level types, severity levels,
    336 ChatGPT-classified vs 7500 programmatic
- [x] TOPIC 4: The lookup table (historical success rates — the core edge)
  - Covered: how lookup works, key findings (no-catalyst bounces more, product failures worst),
    user noted: news labels only exist for 336 big drops, not the 7500 smaller ones.
    Live pipeline classifies all drops via Claude, but backtest is blind on news for small drops.
- [x] TOPIC 5: The backtest engine (how every experiment works — the loop)
  - Covered: 3-step loop (exits → entries → record equity), entry_filter is pluggable,
    engine is dumb on purpose, 87% WR means 87% of trades had positive PnL
- [x] TOPIC 6: Entry optimization journey (+73% → +151%, what each change did)
  - Covered: V01 baseline +73% (underperforms SPY +96%), crisis avoidance biggest jump,
    2-day delay is real signal, ticker/sector lists are look-ahead bias.
  - User agrees: crisis periods should use real-time rule (SPY<20MA), not hardcoded dates.
  - User agrees: sector preferences are cheating. Ticker lists might be analyzable.
  - Honest filter: drop + delay + outflow/stress/Friday + regime gate + overnight gap.
- [x] TOPIC 7: Exit optimization (why +5% target, why 60 days, why no stop loss)
  - Covered: target sweep (+5% sweet spot), hold sweep (60d best), stop losses always hurt
    (32% of winners dip below -5% before recovering)
- [x] TOPIC 8: Adaptive sizing (the compounding breakthrough, +151% → +401%)
  - Covered: equity-based sizing, bull/bear regime (SPY vs 20MA), compounding effect,
    concrete equity curve, built interactive Plotly dashboard (equity_dashboard.html).
  - User liked the dashboard.
- [x] TOPIC 9: The data bug we caught (fake 7000% returns, what went wrong)
  - Covered: open_matrix had different dividend adjustments than close/high/low.
    After fix, open entry ≈ close entry. User noted: "just your fumble"
- [x] TOPIC 10: Honest assessment (what the real returns are without cheating)
  - Covered: 4 versions, fair comparison is compounding vs SPY (24.5% CAGR vs 16% CAGR).
  - User correctly noted: fixed sizing baseline is unfair, should always compound.
- [x] TOPIC 11: Intraday data (what we downloaded, what we found)
  - Covered: 53M bars from Alpaca, split-adjusted, cross-checked vs yfinance.
- [x] TOPIC 12: Top intraday signals (overnight gap, dip entry, volume filter)
  - Covered: overnight gap (36pp spread, strongest signal), dip entry (+58% improvement),
    entry day volume (27pp spread). All files in backtest_iterations/intraday_deep/.
- [x] TOPIC 13: News × intraday crossover (different events = different patterns)
  - Covered: guidance_cut rallies all day, earnings_miss dips then recovers,
    regulatory rallies hard. Small sample sizes though (76/16 events).
- [x] TOPIC 14: Regime gating (stop trading in bad markets — the final strategy)
  - Covered: SPY<20MA gate + 1/3 bull = 25.6% CAGR, 67% time in market,
    turned 2022 from -19% into +4%. 13 gates tested.
- [x] TOPIC 15: Current best strategy (the full config, 25.6% CAGR)
  - Covered: full config end-to-end, 25.6% CAGR, 87% WR, 67% time in market.
- [x] TOPIC 16: What's real vs what's hindsight (the biases we identified)
  - Covered: all signals are real-time observable, but parameters (20MA, +5%, 1/3) were
    optimized on same data. Fundamental assumption: mean reversion continues to work.
- [ ] TOPIC 17: Ideas for improvement (what we haven't tried yet)

## Rules for the discussion
- Keep each topic to 1-2 paragraphs max unless user asks for more
- Use concrete numbers, not vague descriptions
- If user asks "how does X work", show the actual logic/formula
- If user wants to see code, read and paste the relevant snippet
- After each topic ask: "questions on this, or move on?"
- If we go off-track, note where we were and come back
