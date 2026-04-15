# Intraday Data: Master Plan

## What We Have
- 53M 1-minute bars, 2020-2025, ~100 tickers, 1.48 GB
- Columns: open, high, low, close, volume, trade_count, vwap
- Regular trading hours only (9:30 AM - 4:00 PM ET)
- Loads in 3 seconds, fits in 4 GB RAM — no EMR needed

## Performance Benchmark
- Load all 6 years: 3.0s
- Filter to 1 ticker: 0.05s
- Aggregate all to daily: 8.4s
- We can run 25 iterations comfortably on the laptop


## Step 0: Data Prep (must do first)
- Build split-adjustment table for all tickers with splits in 2020-2025
- Apply adjustments so all prices are on the same scale
- Verify adjusted data against yfinance
- Save as new parquet files (adjusted versions)


## The 25 Investigations

Everything below lives in `backtest_iterations/intraday_deep/` as v01 through v25.
Each version is a self-contained script that loads data, runs analysis, saves results.
Our system holds positions for days/weeks — we're NOT day trading. We're using minute
data to find better entry points, better exit points, and new signals.

---

### BLOCK A: ENTRY TIMING (v01-v07)
When during the day should we buy?

**v01 — Intraday profile of entry days**
Pure analysis, no backtest. For every event in our dataset, pull the minute bars
on the entry day (T+2). Plot the average price path minute by minute. Answer:
what does a typical entry day look like? Is there a consistent dip? When?

**v02 — Buy at specific times vs close**
Backtest: instead of buying at close, buy at 9:45, 10:00, 10:30, 11:00, 12:00,
14:00, 15:00, 15:30. Use the actual minute-bar price at that time. Which time
gives the best average entry price? Run with fixed sizing to isolate the effect.

**v03 — VWAP entry vs close entry**
Backtest: buy at the day's VWAP instead of close. VWAP is the volume-weighted
average price — it's what institutions target. If we can consistently buy below
VWAP, that's a real edge.

**v04 — Limit order at open minus X%**
Backtest: place a limit order at open-0.5%, open-1%, open-1.5%, open-2%. If the
price dips to our limit during the day, we get filled. If not, we buy at close
as fallback. Test fill rates and average improvement.

**v05 — First-hour pattern as entry signal**
Analysis: look at what happens in the first 30/60 minutes of the entry day.
If the stock is UP in the first hour, does it keep going? If it's DOWN, does
it recover? Can we use the first-hour direction to decide whether to enter?

**v06 — Wait for intraday dip**
Backtest: instead of buying immediately, wait for the stock to dip X% below
the open during the day (X = 0.5%, 1%, 1.5%, 2%). If it never dips, buy at
close. This is a "patience" strategy — wait for a better price within the day.

**v07 — Multi-day entry: spread the buy across T+2 and T+3**
Backtest: instead of buying 100% on T+2, buy 50% on T+2 and 50% on T+3.
Or 33/33/33 across T+2/T+3/T+4. Does spreading the entry reduce risk and
improve average price?

---

### BLOCK B: EXIT TIMING (v08-v12)
When during the day should we sell?

**v08 — Intraday profile of exit days (winners)**
Analysis: for trades that hit the +5% target, what does the exit day look like
minute by minute? Does the stock overshoot the target? By how much? Is there
a better time to sell than the exact moment the target is hit?

**v09 — Trailing intraday exit**
Backtest: when the stock hits our target intraday, instead of selling immediately,
set a trailing stop at target-0.5%. Let it run. If it keeps going up, we capture
more. If it reverses, we still get close to target. Test different trail widths.

**v10 — Time-of-day exit effect**
Backtest: if the target is hit before 11am, sell immediately. If hit after 2pm,
sell immediately. If hit between 11am-2pm, wait until 3:30pm (stocks often rally
into close). Does time-of-day matter for exit quality?

**v11 — Intraday profile of max-hold exits (losers)**
Analysis: for trades that hit the 60-day max hold (our losers), what does the
last day look like? Is there a better time to sell on the final day? Morning
vs afternoon? Does the stock tend to drift down or recover intraday?

**v12 — Partial exit: sell half at target, hold rest**
Backtest: when target is hit, sell 50% and let the other 50% ride with a
trailing stop. Does this capture more upside on the big winners while still
locking in profit?

---

### BLOCK C: INTRADAY STOP LOSS (v13-v16)
Can minute data make stop losses work?

**v13 — "Confirmed breakdown" stop loss**
Backtest: only trigger stop loss if the stock drops X% AND stays below that
level for 30+ consecutive minutes. A flash dip that recovers in 5 minutes
doesn't trigger. Test X = 3%, 5%, 7% with 15/30/60 minute confirmation.

**v14 — Volume-confirmed stop loss**
Backtest: only trigger stop loss if the drop is accompanied by above-average
volume (selling pressure is real, not just a thin-market dip). Compare to
regular stop loss and no stop loss.

**v15 — Time-decay stop loss**
Backtest: tighten the stop loss as the trade ages. Day 1-5: no stop. Day 6-20:
stop at -7%. Day 21-40: stop at -5%. Day 41-60: stop at -3%. The idea: if it
hasn't bounced after 40 days, cut it tighter.

**v16 — Intraday drawdown analysis**
Analysis: for our winning trades, what's the worst intraday drawdown they
experience? How many winners dip 3%, 5%, 7% intraday before recovering?
This tells us the minimum stop loss width that doesn't kill winners.

---

### BLOCK D: VOLUME & VWAP SIGNALS (v17-v20)
New signals we couldn't see with daily data.

**v17 — Volume surge detection on drop day**
Analysis: on the day a stock drops 5%+, what does the volume profile look like?
Is there a volume spike at a specific time? Does the volume pattern predict
whether the drop will bounce or continue?

**v18 — VWAP position as entry filter**
Backtest: on the entry day, only buy if the current price is BELOW the day's
running VWAP (we're getting a "good" price relative to the day's average).
Skip if price is above VWAP. Does this filter improve win rate?

**v19 — Relative volume as confidence signal**
Backtest: compute the relative volume (today's volume / 20-day average volume)
on the drop day. High relative volume = more conviction in the drop. Does
relative volume predict bounce probability? Size positions accordingly.

**v20 — Closing auction volume**
Analysis: the last few minutes of trading (3:50-4:00 PM) often have huge volume
from the closing auction. Is there a signal in the closing auction volume on
drop days? Does heavy closing volume = capitulation = better bounce?

---

### BLOCK E: PATTERN RECOGNITION (v21-v25)
Classify intraday shapes to predict outcomes.

**v21 — Entry day shape classification**
Analysis: cluster entry days into patterns: V-shape (dip then recover),
L-shape (drop and stay), reverse-V (rally then fade), flat. Which shapes
predict successful trades? Can we detect the shape in the first 1-2 hours?

**v22 — Drop day shape as predictor**
Analysis: same as v21 but for the original drop day (T+0). Does the shape of
the drop predict the bounce? A slow steady decline vs a sudden crash — which
bounces better?

**v23 — Multi-day intraday recovery curve**
Analysis: stitch together minute bars from T+0 through T+5. Plot the average
recovery curve. Is recovery linear? Does it stall? Is there a "dead cat bounce"
pattern visible in the minute data that we can't see in daily data?

**v24 — First-hour momentum on drop day as filter**
Backtest: on the drop day, if the stock recovers 1%+ from its intraday low
in the last hour of trading, that's a sign of buying interest. Use this as
an additional entry filter. Does it improve win rate?

**v25 — Overnight gap analysis**
Analysis: compare the close on day T to the open on day T+1. How big are the
overnight gaps? Do stocks that gap DOWN on T+1 (continued selling) end up
being worse trades than stocks that gap UP (immediate recovery)? Can we use
the T+1 open as a filter before entering on T+2?


---

### BLOCK F: NEWS/EVENT TYPE × INTRADAY PATTERNS (v26-v33)
Connect our event taxonomy to minute-level behavior. This is where the news
classification work we did pays off — different event types should have
completely different intraday signatures.

**v26 — Intraday profile by event type**
Analysis: split all drop days by our stock_event_type (earnings_miss, guidance_cut,
product_failure, no_clear_catalyst, etc.) and plot the average minute-by-minute
price path for each. Do earnings misses crash at open and stabilize? Do guidance
cuts bleed all day? Does "no clear catalyst" recover faster intraday? This is
the foundation — see if different news creates different intraday shapes.

**v27 — Optimal entry time by event type**
Backtest: run v02 (buy at specific times) but separately for each event type.
Maybe earnings misses should be bought at 10:30 (after the initial panic) but
guidance cuts should be bought at close (they bleed all day). Different news =
different optimal entry time.

**v28 — Event severity × intraday volatility**
Analysis: for each severity level (low/medium/high) within each event type,
compute the intraday range, volume profile, and recovery speed. High severity
events probably have wider intraday swings — meaning more opportunity to get
a good entry price, but also more risk.

**v29 — Market stress × intraday behavior**
Analysis: cross our liquidity_credit_stress_severity (VIX-based) with intraday
patterns. During high stress (VIX > 30), do stocks have wider intraday ranges?
Is the optimal entry time different in stressed vs calm markets? In panics,
stocks might overshoot more intraday = bigger opportunity for limit orders.

**v30 — Sector rotation × intraday recovery speed**
Analysis: when sector_rotation_direction = "outflow" (money leaving the sector),
how does the intraday recovery compare to "inflow" days? Our daily backtests
showed outflow is a good entry signal — does the minute data confirm this?
Do outflow drops recover faster intraday?

**v31 — No-catalyst drops: intraday signature**
Analysis: our data shows "no clear catalyst" drops bounce 78.7% of the time
(better than catalyst-driven drops). What do these look like intraday? If
there's no news, the drop is probably driven by technical selling or fund
rebalancing — which might create a very specific intraday pattern (e.g.,
heavy selling in first 30 min then recovery). If we can identify this pattern
in real-time, we can enter with higher confidence.

**v32 — Event type × VWAP entry effectiveness**
Backtest: for each event type, test whether buying below VWAP (v03) works
better for some types than others. Hypothesis: for earnings misses (where
the news is out and priced in), VWAP entry might not help much. But for
no-catalyst drops (where the selling is technical), buying below VWAP could
be very effective because the selling is less informed.

**v33 — News-driven entry timing: full backtest**
Backtest: combine the best findings from v26-v32 into a single strategy where
the entry time depends on the event type. Earnings miss → buy at 10:30.
No catalyst → buy at VWAP. Guidance cut → buy at close. Market stress high →
use limit order at open-1%. Compare this "smart timing" strategy against our
current "always buy at close" baseline.

---

## Execution Plan

1. Run Step 0 (split adjustment) first
2. Run v01 (pure analysis) to understand the data
3. Run v02-v07 (entry timing block) — highest value
4. Run v08-v12 (exit timing)
5. Run v13-v16 (stop loss)
6. Run v17-v20 (volume/VWAP signals)
7. Run v21-v25 (pattern recognition)
8. Run v26-v33 (news × intraday) — ties everything together

Each version: self-contained script, saves results as .txt and .json,
prints summary to console. Same pattern as our previous iterations.

All scripts go in: `backtest_iterations/intraday_deep/`
