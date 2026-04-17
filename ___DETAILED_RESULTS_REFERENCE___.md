# Detailed Results Reference
For every experiment: what we configured, how the calculation works, what came out.


## How The Backtest Engine Works (applies to everything below)

Every backtest follows the same loop. Understanding this once means you understand all results.

```
Start with $100,000 cash.
For each trading day from Jan 2020 to Dec 2025:

  1. CHECK EXITS for open positions:
     - If today's HIGH price >= position's target price → sell at target price (winner)
     - If today >= max hold date → sell at today's CLOSE price (loser or small winner)
     - PnL = (sell_price - buy_price) × number_of_shares
     - Cash increases by: original_investment + PnL

  2. BUY new positions:
     - Look at events that happened `entry_delay` days ago
     - Run them through the entry_filter function (each version has different logic)
     - If filter says yes AND we have cash AND under max positions:
       - Buy at CLOSE price (or dip price in intraday versions)
       - Position size = cash / fraction (e.g., 1/10 of $100K = $10K)
       - Target price = buy_price × (1 + target_return)
       - Max exit date = entry_delay + max_hold_days trading days later
       - Cash decreases by position size

  3. RECORD daily equity = cash + market_value_of_all_open_positions

At the end: total_return = (final_equity / $100,000 - 1) × 100
Win rate = % of closed trades with positive PnL
```

Key terms:
- "target_return" = the % profit we're aiming for per trade (e.g., 0.05 = +5%)
- "max_hold_days" = if target not hit in this many trading days, sell at market
- "position_fraction" = what fraction of capital per trade (10 = 1/10 = $10K)
- "entry_delay" = days to wait after the drop before buying (2 = buy 2 days later)
- "sizing_fn" = custom function that overrides position_fraction (used in adaptive sizing)


---
## ENTRY OPTIMIZATION (10 versions)
All use: +3% target, 60d hold, 1/10 position ($10K), max 10 positions, no stop loss.
The ONLY thing that changes is the entry_filter function.

### V01 — Baseline
Filter: accept ALL events with drop >= 3%
```
Result: +73.0% | 1,121 trades | 82% WR | 198 losers
```
This is "trade everything" — no intelligence at all.

### V02 — Seasonal (earnings months)
Filter: baseline + prefer months 1,2,4,5,7,8,10,11 (earnings season)
```
Result: +76.2% | slight improvement
```
Small effect. Earnings months slightly better but not dramatic.

### V03 — Crisis Avoidance
Filter: baseline + SKIP these hardcoded date ranges:
- 2020-02-20 to 2020-04-15 (COVID crash)
- 2022-01-01 to 2022-10-31 (bear market)
- 2025-03-01 to 2025-04-30 (tariff chaos)
```
Result: +94.8% | fewer trades, fewer losers
```
BIGGEST single improvement. Not trading during known crises saves ~22%.
⚠️ LOOK-AHEAD BIAS: we know these dates because we lived through them.

### V04 — Stress Filter
Filter: baseline + use VIX/stress level to filter
```
Result: +82.1%
```
Moderate improvement. Medium stress is actually the sweet spot for mean reversion.

### V05 — Sector Preference
Filter: baseline + prefer SMH (semiconductors) and XLK (tech)
```
Result: +89.3%
```
Semis bounce better than other sectors.
⚠️ LOOK-AHEAD BIAS: we identified best sectors from the same data.

### V06 — Delayed Entry
Two versions tested:
- V06a: buy same day (delay=0) → +88.7%
- V06b: buy 2 days later (delay=2) → +101.3%
```
2-day delay is better. Lets panic selling exhaust before we buy.
```

### V07 — Combined
Filter: crisis avoidance + 2-day delay + sector preference
```
Result: +112.1%
```
Stacking the improvements.

### V08 — Bigger Positions
Same as V07 but position size = 1/8 instead of 1/10
```
Result: +118.5%
```
Slightly bigger bets = slightly more return (and more risk).

### V09 — Tight Filter
More restrictive entry rules (require multiple signals)
```
Result: +105.2%
```
Too tight — filtered out too many good trades.

### V10 — Final Entry
The full entry filter logic:
```python
# SKIP if ticker in BAD_TICKERS (PFE, TMUS, ACN, CVS, AMT, BLK, IBM, LMT, DHR, MSFT)
# SKIP if date in CRISIS_PERIODS
# ACCEPT if ticker in BEST_TICKERS (NVDA, AMZN, AMAT, AVGO, GE, INTU, MA, AXP, LOW, MO, COF, BMY, MS, MDT, HON)
# ACCEPT if drop >= 7%
# ACCEPT if drop >= 5% AND (sector outflow OR best sector OR Friday OR medium stress)
# ACCEPT if best sector AND sector outflow
# ACCEPT if Friday AND medium stress
# REJECT everything else
```
Config: +3% target, 60d hold, 1/10 position, 2-day delay
```
Result: +150.9% | 470 trades | 87% WR | 61 losers
```


---
## EXIT OPTIMIZATION (10 versions)
Uses the V10 entry filter. Changes: target, hold period, stop loss.

### V01 — Baseline Exit
+3% target, 14d hold
```
Result: +73.0%
```

### V02-V03 — Time Cuts
- 30d hold: better than 14d
- 20d hold: worse than 30d

### V04 — Stop Losses
- -10% stop: WORSE (kills winners that dip temporarily)
- -15% stop: WORSE
- -20% stop: WORSE
```
CONCLUSION: Stop losses ALWAYS hurt mean reversion.
32% of eventual winners dip below -5% before recovering.
```

### V05 — Hold Period Sweep
- 30d: OK
- 45d: better
- 60d: best
- 90d: slightly worse (ties up capital too long)
```
Sweet spot: 60 trading days
```

### V07 — Target Sweep
- +2%: +120.5%
- +3%: +143.6%
- +4%: +148.2%
- +5%: +150.9% ← best
```
Higher target = more profit per winner, but fewer winners hit it.
+5% is the sweet spot.
```

### V08 — Push Target Higher
- +6%: +145.1%
- +7%: +138.2%
- +8%: +130.4%
- +10%: +115.7%
```
Diminishing returns above +5%. Too many trades expire at max hold.
```

### V10 — Final Exit
+5% target, 60d hold, no stop loss, 1/8 position size
```
Result: +150.9% | 470 trades | 87% WR | 61 losers
Year-by-year: 2020 +52.5%, 2021 +13.6%, 2022 -3.0%, 2023 +22.5%, 2024 +1.3%, 2025 +20.0%
```


---
## ADAPTIVE POSITION SIZING (15+ versions)
Uses V10 entry filter + V10 exit config. Changes: HOW MUCH to invest per trade.

### The Key Concept: Equity-Based Sizing
Before: every trade = $100,000 / 8 = $12,500 (fixed forever)
After: every trade = CURRENT_EQUITY / fraction (grows as portfolio grows)

If portfolio is at $200K, each trade is $200K/6 = $33K instead of $12.5K.
This is compounding — winners make future positions bigger.

### V01 — Fixed 1/8
```
$12,500 per trade always. Result: +150.9%
```

### V02 — Bull/Bear (fixed capital)
Bull market (SPY > 50-day MA): 1/5 = $20K per trade
Bear market (SPY < 50-day MA): 1/10 = $10K per trade
Still using INITIAL capital, not current equity.
```
Result: +234.2%
```

### V04 — Equity-Based (THE BREAKTHROUGH)
Same bull/bear logic but sizes off CURRENT EQUITY:
```python
def sizing_fn(row, ctx):
    equity = ctx['equity']  # cash + value of open positions
    if SPY > 50-day MA:
        return equity / 5   # bull: 20% of current equity
    else:
        return equity / 10  # bear: 10% of current equity
```
```
Result: +339.1% (up from +150.9% with fixed sizing)
```
This is compounding. Early wins in 2020 make positions bigger in 2021, etc.

### V07 — Fraction Sweep
Tested 1/3, 1/4, 1/5, 1/6 for bull fraction (bear fixed at 1/12):
- 1/3 bull: +300.6% (too aggressive, big losses hurt more)
- 1/4 bull: +248.8%
- 1/5 bull: +339.1%
- 1/6 bull: +349.1% ← best
```
1/6 is the sweet spot. ~17% of equity per trade in bull markets.
```

### V08 — Bear Fraction Sweep
Bull locked at 1/6. Tested bear fractions:
- 1/8 bear: +303.5%
- 1/10 bear: +319.5%
- 1/12 bear: +349.1% ← best
- 1/14 bear: +307.9%
- 1/16 bear: +343.7%

### V10a — MA Period Sweep
Which moving average for bull/bear detection?
- 20-day MA: +380.1% ← best (faster regime switching)
- 50-day MA: +349.1%
- 100-day MA: +355.4%
- 200-day MA: +304.5%

### V10d — Best Combo
20-day MA, 1/6 bull, 1/14 bear, max 12 positions
```
Result: +401.5% | 431 trades | 87% WR | 55 losers | 19d avg hold
Year-by-year: 2020 +65.9%, 2021 +22.4%, 2022 -3.2%, 2023 +50.2%, 2024 +10.1%, 2025 +52.8%
```
⚠️ This includes look-ahead biases (crisis periods, ticker lists).


---
## DCA (Adding to Losing Positions)
Tested: if a position drops 3-5% after entry, buy more shares.

6 configs tested. ALL worse than not adding.
Best DCA: +341% vs no-DCA: +401.5%
```
CONCLUSION: Positions that drop further after entry are disproportionately
the real losers (58-69% WR vs 87% overall). Don't add to losers.
```


---
## HONEST ASSESSMENT (no look-ahead biases)
Removed: crisis periods, bad/best ticker lists, best sector lists.
Only uses signals observable in real-time.

| Config | Total | Avg/Year | Worst Year |
|--------|-------|----------|------------|
| No cheating + fixed sizing | +102.5% | +13.5% | -6.7% |
| No cheating + compounding | +215.0% | +24.5% | -18.3% |
| Trade all drops (pure baseline) | +99.8% | +13.3% | -5.6% |

The entry filter barely helps without look-ahead biases.
The real value is in the compounding (equity-based sizing).


---
## INTRADAY DEEP DIVE (using 53M 1-minute bars)
All use: no look-ahead biases, +5%/60d, 2-day delay.

### V01 — Entry Day Profile
Looked at minute-by-minute prices on the day we buy (T+2).
```
Finding: Entry day is basically flat. Open to close = -0.06%.
Winners drift up +0.12% by close. Losers drift down -0.86%.
No big consistent dip to exploit.
```

### V02 — Buy at Specific Times
Tested buying at 9:35, 9:45, 10:00, 10:30, 11:00, 12:00, 13:00, 14:00, 15:00, 15:30, close.
```
Best: 2:00 PM = +114.3% (vs close = +102.5%)
Morning (9:35-10:30) worst at ~87%.
Afternoon consistently better.
```

### V03 — VWAP Entry
Buy at the day's volume-weighted average price.
```
VWAP = +80.3% — WORSE than close (+102.5%).
```

### V04 — Limit Orders
Place limit at open minus X%. If price dips to limit, buy there. Else buy at close.
```
open-0.5%: +84.1% (fills 77% of time)
open-1.0%: +102.4% (fills 59%)
open-1.5%: +108.3% (fills 46%)
open-2.0%: +120.3% (fills 34%) ← best limit strategy
```

### V05 — First Hour Signal
Does the first 30-60 minutes predict trade success?
```
First 60 min UP 1%+: 88.1% win rate (1,479 events)
First 60 min DOWN 1%+: 76.1% win rate (1,282 events)
12 percentage point spread. YES, it predicts.
```

### V06 — Wait for Intraday Dip
Wait for stock to dip 0.5% below open, then buy. If no dip, buy at close.
```
Dip 0.5%: +162.3% ← BEST ENTRY STRATEGY (+58% improvement)
Dip 1.0%: +154.5%
Dip 1.5%: +150.5%
Dip 2.0%: +140.4%
The 0.5% dip happens 69% of the time.
```
How it works: on entry day, set limit order at open_price × 0.995.
If the stock touches that price during the day (69% chance), you buy there.
If it never dips, you buy at close as fallback.

### V07 — Entry Delay Sweep
```
0-day delay: +74.5%
1-day delay: +104.0%
2-day delay: +102.5%
3-day delay: +99.1%
4-day delay: +82.0%
5-day delay: +92.3%
```
1-2 day delay is optimal.

### V08 — Winner Exit Day Analysis
When our +5% target is hit, what happens on that day?
```
Winners overshoot target by +1.87% on average (median +0.96%).
Target hit at 10:56 AM on average.
54% of targets hit in the first hour of trading.
50% of winners close above target, 50% below.
```

### V09 — Dip Entry + Target Sweep
Since dip entry gives a cheaper buy price, can we aim for higher targets?
```
Dip + 3%/45d: +86.1%
Dip + 5%/60d: +84.1%
Dip + 7%/45d: +94.1%
Dip + 10%/45d: +123.5% ← higher targets work with dip entry
```

### V16 — Intraday Drawdown
Do winners and losers look different intraday?
```
Winners: -2.02% avg drawdown from open
Losers: -2.44% avg drawdown from open
Difference too small for a reliable stop loss signal.
```

### V17 — Drop Day Volume
Is volume on the drop day predictive?
```
Low volume drops: 83.3% WR
Normal volume: 82.1%
High volume (2-3x): 81.9%
Extreme (3x+): 78.2%
Small spread — not very useful.
```

### V19 — Entry Day Volume ⭐
Is volume on the ENTRY day predictive?
```
Quiet (<0.8x normal): 90.6% WR ← excellent
Normal (0.8-1.2x): 83.1%
Active (1.2-2x): 76.3%
Surge (2x+): 63.3% ← terrible
```
27 percentage point spread. If entry day is quiet, selling has exhausted.
If there's a volume surge, there's still active selling — avoid.

### V20 — Closing Auction Volume
Heavy closing volume on drop day = capitulation = good sign.
```
<5% of daily vol in close: 76.5% WR
5-10%: 84.3%
10-20%: 83.4%
20%+: 86.1%
```

### V25 — Overnight Gap ⭐⭐⭐
Does the gap between drop-day close and next-day open predict success?
```
T+1 gap up 3%+: 97.5% WR (527 events)
T+1 gap up 1-3%: 91.9% WR (1,357 events)
T+1 gap up 0-1%: 81.7% WR (2,144 events)
T+1 gap down 0-1%: 78.1% WR (1,803 events)
T+1 gap down 1-3%: 74.8% WR (1,165 events)
T+1 gap down 3%+: 61.1% WR (517 events)
```
36 PERCENTAGE POINT SPREAD. This is the strongest signal in the entire project.
If the stock gaps up the morning after the drop, it almost certainly bounces.

### V26 — Intraday Profile by Event Type
Different news types create different intraday shapes on entry day:
```
guidance_cut: rallies all day, +0.22% by close
earnings_miss: drops -0.56% by 11:30 then recovers
regulatory_legal: rallies all day, +1.00% by close
competitive_threat: pops early then fades to -0.25%
no_clear_catalyst: flat all day
```

### V27 — Optimal Entry Time by Event Type
```
earnings_miss: buy at 11:30 (saves 0.43% vs close)
guidance_cut: buy at 10:00 (saves 0.31% vs close)
regulatory_legal: buy at 10:30 (saves 1.01% vs close)
```

### V29 — Market Stress × Intraday
```
High stress days: morning pop (+0.27% at 10:30) then fade to -0.18%
Calm days: basically flat all day
```

### Combined Strategy — Final
14 configs tested combining all signals.
Best with compounding (no look-ahead biases):
```
Dip entry + gap>0% filter + adaptive sizing:
+192.0% total | 19.6% CAGR | 464 trades | 84% WR
```

### Regime Gating — Stop Trading in Bad Markets
Instead of just sizing down in bear markets, STOP TRADING entirely.
```
Gate: SPY < 20MA, 1/3 bull sizing when trading:
+293.4% total | 25.6% CAGR | 166 trades | 87% WR | 67% time in market
Year-by-year: 2020 +77.1%, 2021 +23.1%, 2022 +4.0%, 2023 +13.8%, 2024 -10.9%, 2025 +62.2%
```
33% of the time capital is completely free (can earn 4-5% in money market).


---
## SUMMARY: What Actually Moves The Needle

Ranked by impact on returns:

1. **Equity-based compounding** (+150% → +339%): size trades as fraction of current equity
2. **Regime gating** (stop trading when SPY < 20MA): avoids the -20% years
3. **Overnight gap filter** (97.5% WR for gap-up): strongest predictive signal
4. **Dip entry** (+102% → +162%): wait for 0.5% intraday dip before buying
5. **Entry day volume** (90.6% WR when quiet): skip high-volume entry days
6. **2-day entry delay** (+74% → +104%): let panic selling exhaust
7. **+5% target, 60d hold**: optimal target/hold combination
8. **No stop losses**: they always hurt mean reversion
9. **Crisis avoidance** (⚠️ look-ahead): biggest single improvement but uses hindsight
10. **Ticker/sector preferences** (⚠️ look-ahead): helps but uses hindsight
