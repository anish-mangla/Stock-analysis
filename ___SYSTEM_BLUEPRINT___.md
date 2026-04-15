# SYSTEM BLUEPRINT — Mean Reversion Trading System

## The Core Question

The system evaluates candidates across four dimensions. Each dimension produces a **score on a continuous scale**, not a binary pass/fail. The scores reflect how favorable that dimension is for a bounce trade.

The fundamental unit is a **candidate trade event**. A candidate is born when a stock in the S&P 100 has declined meaningfully over a recent window (not just a single day — could be a sharp one-day shock or a multi-day grind). The system answers two questions:

1. **What is the probability this stock reaches +1% from my entry price within 10 trading days?**
2. **What entry price should I use?**

---

## The Four Dimensions

### Dimension 1: Price Action
*"How unusual and potentially overdone is this price decline?"*

This is purely quantitative, derived from price/volume data:
- How far has the stock fallen over the recent window (not just one day)
- How does this decline compare to the stock's own normal volatility
- How did the stock get here — sharp shock vs slow grind
- Where is the stock relative to its own moving averages
- Was the stock healthy before this decline started or already weak
- Volume behavior — capitulation spike vs steady selling
- Where did the stock close within the day's range (near the low = still selling, near the high = buyers stepping in)

This dimension is the most developed in the codebase. The backtests have already identified which conditions historically produce better bounce rates.

### Dimension 2: Relative Context
*"Is this stock-specific or is everything around it also falling?"*

- Stock's decline vs its sector ETF's decline over the same window
- Stock's decline vs SPY over the same window
- Is the sector itself in a downtrend or healthy
- Is the broad market in a hostile regime or supportive

A stock that's down 5% while its sector is flat is a very different situation from a stock that's down 5% while its sector is down 4%. The first is idiosyncratic (more likely to revert), the second is systematic (less likely to revert independently).

### Dimension 3: Fundamental/News Assessment
*"What caused this decline and how severe is it?"*

This is where the AI + research comes in. The key insight is that this should be a **scored assessment, not a binary veto**. The output needs to capture:

**What type of event caused the decline:**
- Earnings miss (how bad? slight miss vs massive miss)
- Guidance cut (one quarter vs full year revision)
- Product failure (minor product vs flagship product)
- Regulatory action (investigation vs actual penalty vs structural ban)
- Analyst downgrade (one analyst vs consensus shift)
- Management change (planned succession vs sudden departure)
- Legal/litigation (nuisance suit vs existential threat)
- Competitive threat (new entrant vs fundamental disruption)
- Macro sympathy (stock got dragged down by market/sector, no stock-specific news)
- No clear catalyst found

**How material is it:**
- Does this affect the core business or a peripheral segment
- Is this a one-time event or a structural change
- What percentage of revenue/earnings is at risk
- Is this already priced in or is there more downside to come

**What's the historical bounce rate for this type + severity combination:**
Once there are enough labeled events, you can look up: "stocks that dropped 5-7% due to a minor product issue historically bounced +1% within 10 days X% of the time." That lookup table is what makes the score objective.

The same logic applies at sector and market level:
- Is there a sector-wide headwind? What type and how severe?
- Is there a macro event? What type and how severe?

### Dimension 4: Entry Optimization
*"Given my confidence level, what's the smartest entry price?"*

Based on the combined picture from dimensions 1-3:

- **High conviction** → enter near current price, don't risk missing it
- **Medium conviction** → set a limit order slightly below, improve your entry if the stock gives you the chance
- **Lower conviction** → set a deeper limit order, only take the trade if the stock comes to you at a price where the math is much more favorable

The limit order levels themselves can be informed by technical levels — recent intraday lows, support zones, round numbers — but the core idea is simple: less confidence = demand a better price.

---

## How To Score Each Dimension Without Subjectivity

For **Dimensions 1 and 2**, the answer is straightforward: everything is quantitative, and there is historical data to measure what works. Compute the features, look at historical success rates conditional on those features, and the success rate IS the score.

For **Dimension 3**, it's harder because news assessment is inherently qualitative. Here's how to make it as objective as possible:

**Step 1: Build the taxonomy of event types.** ~15-20 categories that cover the vast majority of why S&P 100 stocks drop.

**Step 2: For each event type, define severity levels.** Not subjective 1-5 ratings, but concrete criteria:
- Earnings miss: severity = magnitude of the EPS miss as a percentage of consensus
- Product failure: severity = estimated revenue impact as percentage of total revenue
- Analyst downgrade: severity = number of firms downgrading within the window

**Step 3: Build the historical reference database.** Go through events.csv, label the significant drops with event type and severity, and record the outcome. Labor-intensive but a one-time effort that creates the foundation.

**Step 4: Compute historical success rates by (event_type, severity_bucket).** This becomes the lookup table. When a new candidate comes in and Claude classifies it as "earnings miss, moderate severity," look up the historical bounce rate for that combination. That's the score. No opinion needed.

**Step 5: Use Claude as the classifier, not the judge.** Claude's job is: "read the news, tell me what type of event this is and how severe it is using these concrete criteria." The system's job is: "look up the historical bounce rate for that classification." This separates the qualitative assessment (what happened) from the quantitative decision (should I trade it).

---

## On The Multi-Day Entry Trigger

The entry trigger shouldn't be "stock dropped X% today." It should be something more like "stock is currently X% below where it was N days ago" — a rolling window that catches both sharp single-day moves and multi-day grinds.

Options to test:
- 3-day cumulative decline
- 5-day cumulative decline
- 10-day cumulative decline
- Decline from recent N-day high (drawdown from peak)

Drawdown from recent high (say 20-day high) is likely the most general — it doesn't care whether the drop happened in one day or five, it just measures "how far has this stock fallen from its recent best." But this is testable with existing data.

---

## On Correlation Between Dimensions

You can't equally weight the dimensions. Some are correlated — if the stock dropped 7% and it underperformed its sector by 5%, those aren't independent facts. The big drop IS the reason it underperformed the sector. Same with vol-normalized shock and raw drop size.

The solution: measure joint conditions directly from historical data. Don't combine individual scores with weights. Instead, look at the historical success rate when multiple conditions are true simultaneously. The data tells you the combined effect, including any correlation.

---

## The Three Workstreams To Build

### Workstream A: Quantitative Engine (mostly exists)
Consolidate the backtested insights into a single module that computes all price/volume/sector/market features for any candidate and produces the Dimension 1 and 2 scores. This is largely refactoring existing code.

### Workstream B: News Intelligence Layer (the big new build)
1. Define the event type taxonomy
2. Define severity criteria for each type
3. Build the historical reference database (labor intensive)
4. Restructure the Claude pipeline to classify rather than judge
5. Build the lookup table that converts classifications into scores

### Workstream C: Entry Optimization (new but smaller)
1. Backtest different entry strategies (close, limit below close at various levels)
2. Measure fill rate vs win rate tradeoff at each level
3. Build the confidence-to-entry-price mapping

---

## Trade Parameters (Locked)

- Universe: S&P 100
- Position size: 1/10th of portfolio
- Target: +1% from entry price
- Max hold: 10 trading days
- Stop loss: none (let the trade breathe)
- Exit: whichever comes first — target hit or max hold reached

---

## Next Step

Nail down the event type taxonomy for Workstream B. That's the structural decision everything else hangs on. Once the categories and severity definitions are agreed upon, start labeling historical events and building the reference database.
