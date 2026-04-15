# EVENT TYPE TAXONOMY — Why S&P 100 Stocks Drop

This taxonomy defines 20 event types across three levels (stock, sector, market).
Each type has concrete, measurable severity anchors so that labeling is objective and repeatable.

A candidate event can have MULTIPLE labels across levels. A stock might drop because of
an earnings miss (stock-level) during a sector rotation (sector-level) in a hawkish Fed
environment (market-level). The system captures all three.

---

## STOCK-LEVEL EVENT TYPES (1–11)

### 1. Earnings Miss
Reported EPS or revenue below consensus.

**Severity anchor: % miss vs consensus EPS**
- Low: slight miss (< 5% below consensus), beat on the other metric
- Medium: clear miss (5–15% below), both metrics soft
- High: massive miss (> 15% below), or miss + downward revision

### 2. Guidance Cut
Company lowers forward outlook.

**Severity anchor: % reduction in forward EPS/revenue guidance**
- Low: minor trim, one quarter only
- Medium: meaningful cut, multiple quarters affected
- High: full-year slashed, or guidance withdrawn entirely

### 3. Analyst Downgrade
Wall Street firms lower rating or price target.

**Severity anchor: number of firms + magnitude of target cuts**
- Low: single analyst, small target reduction
- Medium: 2–3 analysts in a short window, meaningful target cuts
- High: broad consensus shift, multiple downgrades, targets slashed 20%+

### 4. Product / Service Failure
Something went wrong with what the company sells.

**Severity anchor: revenue exposure of the affected product/service**
- Low: minor product, < 10% of revenue
- Medium: significant product, 10–30% of revenue
- High: flagship product, > 30% of revenue, or safety/recall issue

### 5. Regulatory / Legal Action
Government or legal system acts against the company.

**Severity anchor: potential financial impact as % of market cap**
- Low: investigation opened, no charges, < 1% of market cap at risk
- Medium: formal charges/fines, 1–5% of market cap
- High: structural ban, existential litigation, > 5% of market cap

### 6. Management Change
Key executive departure or shakeup.

**Severity anchor: role importance + whether planned or sudden**
- Low: planned succession, non-CEO role
- Medium: unexpected departure of key executive (CFO, division head)
- High: sudden CEO departure, or departure amid controversy

### 7. Competitive Threat
New competitor or disruption narrative.

**Severity anchor: how directly it threatens core revenue**
- Low: new entrant in adjacent market, speculative threat
- Medium: credible competitor gaining share in core market
- High: fundamental disruption to business model

### 8. Capital Structure / Dilution
Secondary offering, dividend cut, debt issues.

**Severity anchor: dilution %, dividend cut %, or credit rating change**
- Low: small secondary (< 3% dilution), minor dividend trim
- Medium: meaningful dilution (3–10%), significant dividend cut
- High: large dilution (> 10%), dividend eliminated, credit downgrade

### 9. Insider / Institutional Selling
Large insider sales or institutional exits.

**Severity anchor: size of sales relative to holdings + number of insiders**
- Low: routine scheduled sales
- Medium: unusual size or timing, multiple insiders
- High: CEO/CFO dumping large blocks, major fund liquidating position

### 10. Demand / Macro Sensitivity
Company-specific signs of weakening demand.

**Severity anchor: how leading the indicator is + breadth of weakness**
- Low: one soft data point, could be seasonal
- Medium: clear trend in orders/bookings/traffic declining
- High: major customer loss, contract cancellation, broad demand collapse

### 11. No Clear Stock-Specific Catalyst
The stock dropped but there is no obvious company news.

This is one of the most important categories because it suggests the move
is more likely sentiment-driven and therefore more likely to revert.

**Severity: N/A** — the absence of a catalyst IS the signal.

---

## SECTOR-LEVEL EVENT TYPES (12–15)

### 12. Sector Regulation / Policy Change
Government action affecting the whole sector.

Examples: pharma pricing legislation, bank capital requirements, tech antitrust,
energy policy shifts, chip export bans.

**Primary severity anchor: stage of the regulatory process**
- Low: proposal/discussion stage, no bill introduced, no formal rulemaking
- Medium: bill introduced or formal rulemaking initiated, but not yet passed/finalized.
  OR executive order signed but with delayed implementation / unclear enforcement
- High: law passed, rule finalized, or executive order with immediate enforcement.
  OR existing regulation materially tightened with clear compliance deadline

**Secondary anchor: breadth within the sector**
- Narrow: affects one sub-segment (e.g., only generic drug makers)
- Broad: affects the entire sector (e.g., all pharma pricing)

### 13. Sector Demand Shift
Macro or structural change in demand for what the sector sells.

Examples: oil price collapse → energy, rate hikes → REITs/homebuilders,
consumer spending slowdown → retail, AI spending boom → semis.

**Severity anchor: magnitude of the demand driver's move in standard deviations**
- Low: underlying driver moved < 1 sigma from its recent average
- Medium: 1–2 sigma
- High: > 2 sigma

Observable demand drivers by sector:
- Energy: crude oil price (WTI or Brent)
- Financials: 10-year Treasury yield, yield curve slope
- REITs: mortgage rates, 10-year yield
- Tech: NASDAQ relative to SPY as proxy for growth appetite
- Consumer discretionary: consumer confidence index, retail sales data
- Industrials: ISM manufacturing PMI
- Materials: commodity indices (copper, etc.)
- Utilities: interest rates (they trade as bond proxies)

### 14. Sector Rotation
Money flowing out of the sector into others — not because of sector-specific
news but because of positioning/style shifts.

Examples: growth-to-value rotation, tech selloff into defensives, risk-on to risk-off.

**Severity anchor: sector ETF performance vs SPY over the event window**
- Low: sector ETF underperformed SPY by < 2%
- Medium: sector ETF underperformed SPY by 2–5%
- High: sector ETF underperformed SPY by > 5%

How to distinguish from demand shift (#13): if there is a clear fundamental driver
(oil crashed, rates spiked), it is a demand shift. If the sector is selling off
relative to the market with no obvious fundamental catalyst, it is rotation.

### 15. Sector Contagion
One company's bad news dragging down peers that may or may not deserve it.

Examples: one bank's credit losses spooking all banks, one chipmaker's weak
guidance hitting all semis, one retailer's inventory warning dragging down peers.

**Severity anchor: how directly relevant the source company's problem is to the peer**
- Low: source company's issue is company-specific (accounting fraud, management scandal,
  unique product failure) — peers down on sympathy, not substance
- Medium: source company's issue reveals something about shared conditions (e.g., one
  chipmaker's weak China demand could mean others face it too) — plausible but
  unconfirmed read-across
- High: source company's issue is clearly shared (e.g., a supplier all peers use has a
  problem, or a customer all peers serve is cutting orders) — direct read-across confirmed

How to determine concretely: look at what the source company actually said. If their
earnings call says "our specific product had a defect," that is Low for peers. If they
say "we are seeing broad demand weakness across the industry," that is High for peers.
The source company's own words are the anchor.

---

## MARKET-LEVEL EVENT TYPES (16–20)

### 16. Monetary Policy Shock
Central bank surprise — rate decision, guidance, or communication that differs
from market expectations.

**Severity anchor: move in 2-year Treasury yield on announcement day**
- Low: < 10 basis points
- Medium: 10–25 basis points
- High: > 25 basis points

Direction matters:
- Hawkish surprise (yields up) → bad for stocks, especially growth/tech
- Dovish surprise (yields down) → generally good for stocks

### 17. Trade / Tariff Policy
Trade war escalation or de-escalation.

**Primary severity anchor: estimated dollar value of trade affected**
- Low: tariffs threatened but not implemented, or narrow product categories (< $10B affected)
- Medium: tariffs on broad categories ($10–100B affected), or existing tariffs meaningfully increased
- High: tariffs on > $100B of trade, blanket tariffs on entire countries, or retaliatory tariffs

**Secondary anchor: SPY reaction in first 2 hours after announcement**
- If SPY drops 2%+ in first 2 hours → market says it is severe
- If SPY barely moves → market says it is priced in or not that bad

### 18. Geopolitical Event
War, sanctions, political crisis, terrorism.

**Primary severity anchor: proximity to economic activity + duration uncertainty**
- Low: regional conflict with limited economic linkage to US markets
- Medium: conflict affecting commodity supply or major trade routes, OR sanctions on a
  mid-sized economy, OR domestic political crisis (shutdown, debt ceiling)
- High: direct military conflict involving major economies, sanctions on a top-10 economy,
  or event disrupting global supply chains

**Secondary anchor: oil price + VIX reaction within 24 hours**
- Oil spike > 5% and/or VIX spike > 20% → High regardless of the above
- Oil and VIX barely move → Low regardless of how scary the headlines sound

The secondary anchor lets the market tell you how severe the event actually is,
rather than trying to judge it from headlines.

### 19. Macro Data Shock
Major economic data release that surprises the market.

**Severity anchor: size of miss/beat vs consensus in standard deviations**
- Low: < 1 sigma from consensus
- Medium: 1–2 sigma
- High: > 2 sigma

Key data releases that move markets:
- Non-farm payrolls (jobs)
- CPI / PCE (inflation)
- GDP
- ISM PMI (manufacturing/services)
- Retail sales
- Jobless claims (only when trending)

Direction matters:
- Hot inflation + strong jobs → hawkish → bad for growth stocks
- Weak jobs + weak GDP → recession fear → bad for cyclicals
- Goldilocks (moderate growth, cooling inflation) → good for everything

### 20. Liquidity / Credit Stress
Market plumbing problems — funding stress, credit market seizure, forced selling.

**Severity anchor: observable market stress indicators**

VIX level:
- < 20 = calm
- 20–30 = elevated
- 30–40 = stressed
- > 40 = crisis

Credit spread move (investment-grade, weekly):
- < 20bps widening = normal
- 20–50bps = notable
- > 50bps = severe

Combined severity:
- Low: VIX 20–25, credit spreads widening modestly, no funding issues
- Medium: VIX 25–35, credit spreads widening meaningfully (> 30bps/week),
  some signs of forced selling (high cross-asset correlation)
- High: VIX > 35, credit spreads blowing out (> 50bps/week), clear forced
  liquidation (everything selling together, including safe havens)

---

## SEVERITY ANCHOR SUMMARY TABLE

| # | Category | Primary Severity Anchor |
|---|---|---|
| 1 | Earnings Miss | % miss vs consensus EPS |
| 2 | Guidance Cut | % reduction in forward guidance |
| 3 | Analyst Downgrade | Number of firms + target cut magnitude |
| 4 | Product/Service Failure | Revenue exposure of affected product |
| 5 | Regulatory/Legal Action | Financial impact as % of market cap |
| 6 | Management Change | Role importance + planned vs sudden |
| 7 | Competitive Threat | Directness of threat to core revenue |
| 8 | Capital Structure/Dilution | Dilution %, dividend cut %, credit rating |
| 9 | Insider/Institutional Selling | Sale size vs holdings + number of insiders |
| 10 | Demand/Macro Sensitivity | Leading indicator strength + breadth |
| 11 | No Clear Catalyst | N/A — absence of catalyst IS the signal |
| 12 | Sector Regulation/Policy | Stage of regulatory process |
| 13 | Sector Demand Shift | Sigma move in underlying demand driver |
| 14 | Sector Rotation | Sector ETF underperformance vs SPY |
| 15 | Sector Contagion | Source company's own description of problem |
| 16 | Monetary Policy Shock | 2-year Treasury yield move on announcement day |
| 17 | Trade/Tariff Policy | Dollar value of trade affected + SPY reaction |
| 18 | Geopolitical Event | Oil + VIX reaction within 24 hours |
| 19 | Macro Data Shock | Sigma of data surprise vs consensus |
| 20 | Liquidity/Credit Stress | VIX level + credit spread move |

---

## LABELING RULES

- Each candidate event gets labeled with ALL applicable event types (can be multiple)
- Each label includes the event type number and severity (Low / Medium / High)
- Stock-level labels are per-stock. Sector and market labels apply to all stocks on that date.
- "No Clear Stock-Specific Catalyst" (#11) is used when none of types 1–10 apply
- Sector and market labels can co-exist with stock-level labels
