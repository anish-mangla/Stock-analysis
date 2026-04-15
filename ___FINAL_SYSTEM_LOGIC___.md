# FINAL SYSTEM LOGIC — Daily Mean Reversion Trading System

## Daily Flow

```
After market close each trading day:

1. SCAN → Identify candidates from S&P 100
2. COMPUTE → Quantitative features for each candidate
3. CONTEXT → Market-level and sector-level context
4. CLASSIFY → Send to Claude for news classification
5. LOOKUP → Historical success rate for this combination
6. DECIDE → Confidence tier → entry price → trade or skip
7. RANK → Order candidates by success rate
8. OUTPUT → List of trades with entry prices
```

## Step 1: Identify Candidates

Scan S&P 100 for stocks meeting EITHER criterion:
- Drawdown from 20-day high ≥ 3% (catches multi-day grinds)
- Single-day drop ≥ 3% (catches sharp shocks)

This replaces the old 5-day Typical Price approach with something more general.

## Step 2: Compute Quantitative Features

For each candidate, compute:
- drop_magnitude: how far from recent high (%)
- drop_bucket: 3-5%, 5-7%, 7-10%, 10%+
- drop_vs_vol20: drop normalized by stock's own 20-day volatility
- stock_minus_sector_return: stock return vs sector ETF return
- pre_drop_trend: 5-day and 20-day return before the drop
- distance_from_20dma: how stretched below moving average
- close_in_range: where the stock closed within the day's range (0=low, 1=high)
- volume_spike: today's volume vs 20-day average volume

## Step 3: Market & Sector Context

Compute from live market data (same logic as build_market_level_labels.py):
- VIX level → liquidity_stress_severity (none/low/medium/high)
- SHY/TLT moves → monetary_policy_shock (if Fed day or extreme rate move)
- Sector ETF vs SPY → sector_rotation_severity and direction
- Sector demand driver moves → sector_demand_shift_severity

Check for active trade/tariff events (maintained manually or via news feed).

## Step 4: News Classification via Claude

Send each candidate to Claude with our taxonomy. Claude's job is CLASSIFICATION, not judgment.

Prompt structure:
```
"Stock X dropped Y% today. The broad market was [up/down/flat].
The stock's sector (ETF) was [up/down Z%].

Using the event type taxonomy below, classify what caused this drop.
Return: event_type (1-11), severity (low/medium/high), one-sentence description.

[Include the 11 stock-level event types with severity definitions]"
```

Claude returns: event_type_id, event_type_name, severity, description

## Step 5: Hierarchical Lookup

Look up the historical success rate using the most specific combination
that has enough data (≥ 10 historical events):

Priority 1: event_type + severity + market_stressed + drop_bucket
Priority 2: event_type + severity + market_stressed
Priority 3: event_type + severity
Priority 4: event_type alone
Priority 5: drop_bucket alone (always has enough data)

The lookup returns:
- success_rate: probability of hitting +1% within 14 trading days
- avg_days_to_hit: expected time to reach target
- avg_max_drawdown: expected worst-case drawdown during hold

## Step 6: Decision Rules

Based on the looked-up success rate:

| Success Rate | Confidence | Entry Strategy |
|---|---|---|
| ≥ 80% | HIGH | Enter near close price (market or tight limit) |
| 65-80% | MEDIUM | Limit order at close minus 0.5-1% |
| 50-65% | LOW | Limit order at close minus 1-2%, or skip |
| < 50% | SKIP | Do not trade — this is a "don't touch" setup |

## Step 7: Portfolio Management

- Position size: 1/10th of portfolio value
- Max positions: 10 simultaneous
- No duplicate tickers
- Target: +1% from entry price
- Max hold: 10 trading days
- Stop loss: none (let the trade breathe)
- Exit: whichever comes first — target hit or max hold reached
- When more candidates than capital: rank by success rate, take the best

## Key Findings From Lookup Table

These findings drive the system's edge:

BEST setups (highest historical success rates):
- No clear catalyst + market stressed: 85.7%
- Competitive threat (medium severity): 100% (small sample)
- Guidance cut (medium severity): 94.4%
- Tariff-driven drops: 90.6%
- High sector rotation outflow: 91.7%
- 10%+ drops: 88.6%

WORST setups (avoid these):
- Product/service failure (high severity): 46.2%
- Management change (7-10% drop): 50.0%
- Guidance cut (high severity) + calm market: 68.7%

The system's alpha comes from:
1. Identifying drops with no fundamental catalyst (sentiment-driven → likely to revert)
2. Avoiding drops caused by genuine product failures or severe guidance cuts
3. Being more aggressive during market stress (panic selloffs bounce)
4. Being more cautious during calm markets (drops are more "real")
