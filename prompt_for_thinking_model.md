# Task: Design an Optimal Mean-Reversion Trading System

## The Setup

I trade large-cap US stocks (S&P 100) that drop 3%+ in a single day. I buy at the close of the next trading day (D+1 close). I'm attaching a parquet file with 7,837 such events from Jan 2020 to Dec 2025.

## The Data

Each row is one drop event. The columns give you everything about the drop and what happened next:

**The drop:**
- `ticker`, `event_date`, `sector_etf`
- `drop_pct`: size of the drop (e.g. 0.05 = 5%)
- `entry_price`: the D+1 close price (our buy price)

**Full OHLC price path for 16 trading days (D+0 through D+15):**
- `d0_close`, `d0_high`, `d0_low` — the drop day itself
- `d1_close`, `d1_high`, `d1_low` — entry day (we buy at d1_close)
- `d2_close` through `d15_close` — the 14 trading days after entry
- `d2_high` through `d15_high` — intraday highs
- `d2_low` through `d15_low` — intraday lows
- `d0_spy_close` through `d15_spy_close` — SPY close for each day

**Stock characteristics:**
- `target_distance_over_atr20`: how many average daily ranges it takes to move +5%. Lower = more volatile stock.
- `atr_pct`: approximate daily ATR as % of price
- `dist_from_20dma`: how far stock is from its 20-day moving average on drop day
- `d1_return`: D+1 return (entry day behavior)
- `d1_close_vs_vwap`: D+1 close relative to VWAP
- `hyg_5d_return`: high-yield bond ETF 5-day return (credit stress indicator)
- `d2_first_hour_ret`: first hour return on D+2

**Intraday features from 1-minute bars:**
- `d0_rebound_low_to_close`: bounce from intraday low to close on drop day
- `d0_minute_of_low`: when the low occurred (0=market open, 390=close)
- `d1_pct_bars_above_vwap`: % of D+1 bars above VWAP
- `d1_longest_streak_above_vwap`: longest consecutive streak above VWAP on D+1
- `d1_last90_logprice_slope`: price trend in last 90 min of D+1
- `d1_full_day_trend_r2`: R² of D+1 price trend
- `d1_return_vs_sector`: D+1 return minus sector ETF return

**Market context:**
- `spy_d0_return`, `spy_5d_return`, `spy_20d_return`: SPY returns
- `stock_excess_vs_spy`: stock's drop minus SPY's return (how much is stock-specific)
- `num_droppers_3pct`: how many stocks in the universe dropped 3%+ on the same day
- `num_droppers_5pct`: same but 5%+

**Labels (available for ~2,500 events, rest are "unlabeled"):**
- `stock_event_type`: earnings_miss, guidance_cut, demand_weakness, macro_sensitivity, no_clear_catalyst, regulatory_legal, etc.
- `stock_event_severity`: low, medium, high
- `company_specific_factor`: "yes" = stock-specific news caused the drop, "no" = macro/market-driven

**Flags:**
- `in_crisis_window`: True for 8 identified multi-day market stress periods where behavior is different
- `success`: True if stock hit +5% from entry within 60 days (legacy metric, don't anchor on this)
- `days_to_hit`: days to hit +5% (if it did)
- `final_return`: return at 60-day mark
- `max_drawdown`: worst drawdown during 60-day hold

## What I Want

Design a complete trading system that maximizes weekly portfolio return. I need specific, implementable rules for:

1. **Which drops to buy and which to skip** — given all the features available at entry time
2. **What profit target to set for each trade** — this can be dynamic based on the stock and conditions
3. **When to sell each day** — daily hold/sell decisions based on how the position is evolving
4. **When to cut losses** — not just a fixed stop loss, but intelligent loss management
5. **How to size positions** — should all trades get equal capital or should some get more?
6. **How to prioritize** — when there are more opportunities than capital, which trades to take

## Constraints

- Entry is always at D+1 close (can't change this)
- Long only
- Max 10-15 simultaneous positions
- 20-30% of portfolio per position
- ~100 large-cap stocks in universe
- Daily decisions only (no intraday execution after entry)
- Rules must work across 2020-2025, not just one period

## What Good Looks Like

- I want rules that are specific and numeric, not vague
- I want to understand WHY each rule works (what pattern in the data supports it)
- I want year-by-year stability — rules that only work in 2020 are useless
- I want the system to be as simple as possible while capturing the edge
- Show me the expected win rate, average return per trade, average hold time, and return per day of capital deployed

## Important Notes

- The raw OHLC prices are actual dollar prices, not returns. You'll need to compute returns relative to entry_price (d1_close) yourself.
- D+0 is the drop day. D+1 is when we buy. D+2 is the first full day we hold the position.
- All the `dN_spy_close` columns let you track what the market was doing each day alongside the stock.
- The `in_crisis_window` flag marks 8 periods of extreme market stress. You may want to analyze these separately.
- About 33% of events have labeled event types from human research. The rest are "unlabeled". The labels tell you WHY the stock dropped.
