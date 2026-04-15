#!/usr/bin/env python3
"""
HONEST ASSESSMENT: What would $100K actually return per year?

We run 4 versions:
1. FULL CHEATING: all look-ahead biases (crisis periods, bad/best tickers) + compounding
2. NO CHEATING: remove all look-ahead biases, keep compounding
3. NO CHEATING + FIXED SIZING: remove biases AND compounding (most conservative)
4. NO CHEATING + FIXED SIZING + NO TICKER FILTER: pure mean-reversion, no opinions

This tells us what's real signal vs what's hindsight.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import load_cached_data, run_backtest, print_results, INITIAL_CAPITAL
import pandas as pd

close, high, low, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()

# === LOOK-AHEAD BIAS COMPONENTS ===
CRISIS_PERIODS = [(pd.Timestamp('2020-02-20'),pd.Timestamp('2020-04-15')),(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-10-31')),(pd.Timestamp('2025-03-01'),pd.Timestamp('2025-04-30'))]
BAD_TICKERS = {'PFE','TMUS','ACN','CVS','AMT','BLK','IBM','LMT','DHR','MSFT'}
BEST_TICKERS = {'NVDA','AMZN','AMAT','AVGO','GE','INTU','MA','AXP','LOW','MO','COF','BMY','MS','MDT','HON'}
BEST_SECTORS = {'SMH','XLK'}

# --- VERSION 1: FULL CHEATING (our +401.5% number) ---
def filter_full_cheat(row, ctx):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for s,e in CRISIS_PERIODS:
        if s <= dt <= e: return False
    t,sec,d = row['ticker'],row.get('sector_etf',''),row['drop_pct']
    out = row.get('sector_rotation_direction','')=='outflow'
    stress = row.get('liquidity_credit_stress_severity','none')
    fri = dt.dayofweek==4
    if t in BEST_TICKERS: return True
    if d>=0.07: return True
    if d>=0.05 and (out or sec in BEST_SECTORS or fri or stress in ['medium','low']): return True
    if sec in BEST_SECTORS and out: return True
    if fri and stress=='medium': return True
    return False

def sizing_adaptive(row, ctx):
    today = ctx['today']
    equity = ctx.get('equity', INITIAL_CAPITAL)
    if today in spy_close.index and today in spy_20ma.index:
        s = spy_close.at[today]
        m = spy_20ma.at[today]
        if pd.notna(s) and pd.notna(m) and s > m:
            return equity / 6
    return equity / 14

# --- VERSION 2: NO CHEATING (remove crisis periods, bad/best tickers, best sectors) ---
# Keep only things we could actually know in real-time:
# - drop size (observable)
# - sector outflow (observable same day)
# - VIX/stress level (observable same day)
# - day of week (observable)
def filter_no_cheat(row, ctx):
    dt = row['event_date']
    d = row['drop_pct']
    out = row.get('sector_rotation_direction','')=='outflow'
    stress = row.get('liquidity_credit_stress_severity','none')
    fri = dt.dayofweek==4
    
    # Only use observable, non-look-ahead signals
    if d >= 0.07: return True  # big drops always bounce (observable)
    if d >= 0.05 and (out or fri or stress in ['medium','low']): return True
    if d >= 0.03 and out and fri: return True
    if d >= 0.03 and stress == 'medium': return True
    return False

# --- VERSION 3: SIMPLEST (just trade all 3%+ drops, no filter at all) ---
def filter_all_drops(row, ctx):
    return row['drop_pct'] >= 0.03

print("="*70)
print("  HONEST ASSESSMENT: What does $100K become after 1 year?")
print("="*70)

configs = [
    ("1. FULL SYSTEM (with look-ahead biases + compounding)", filter_full_cheat, sizing_adaptive, 12),
    ("2. NO CHEATING (no crisis/ticker biases, with compounding)", filter_no_cheat, sizing_adaptive, 12),
    ("3. NO CHEATING + FIXED SIZING (most realistic)", filter_no_cheat, None, 10),
    ("4. TRADE ALL DROPS + FIXED SIZING (pure baseline)", filter_all_drops, None, 10),
]

for label, filt, sfn, mp in configs:
    r = run_backtest(events, close, high, low, filt,
        target_return=0.05, max_hold_days=60, max_positions=mp, entry_delay=2,
        sizing_fn=sfn, position_fraction=10)
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  TOTAL (2020-2025): {r['total_return']:+.1f}%")
    print(f"  Trades: {r['trades']} | WR: {r['win_rate']:.0f}% | Losers: {r['losers']}")
    print(f"  Avg hold: {r['avg_hold_days']:.0f}d | Capital util: {r['avg_capital_utilization']:.0f}%")
    print()
    
    # Per-year returns (this is what matters for "1 year" question)
    years = sorted(r['annual'].keys())
    returns = [r['annual'][y]['return'] for y in years]
    trades_per_yr = [r['annual'][y]['trades'] for y in years]
    losers_per_yr = [r['annual'][y]['losers'] for y in years]
    
    print(f"  {'Year':>6} {'Return':>8} {'Trades':>7} {'Losers':>7} {'$100K becomes':>15}")
    print(f"  {'-'*50}")
    for y, ret, tr, lo in zip(years, returns, trades_per_yr, losers_per_yr):
        final = 100000 * (1 + ret/100)
        print(f"  {y:>6} {ret:>+7.1f}% {tr:>7} {lo:>7} {f'${final:,.0f}':>15}")
    
    avg_ret = sum(returns[:-1]) / len(returns[:-1])  # exclude 2026 partial
    median_ret = sorted(returns[:-1])[len(returns[:-1])//2]
    worst = min(returns[:-1])
    best = max(returns[:-1])
    print(f"\n  Average annual return: {avg_ret:+.1f}%")
    print(f"  Median annual return:  {median_ret:+.1f}%")
    print(f"  Worst year:            {worst:+.1f}%")
    print(f"  Best year:             {best:+.1f}%")
    print(f"  --> $100K for 1 year (avg): ${100000*(1+avg_ret/100):,.0f}")
    print(f"  --> $100K for 1 year (median): ${100000*(1+median_ret/100):,.0f}")
    print(f"  --> $100K worst case: ${100000*(1+worst/100):,.0f}")

# SPY benchmark
spy_data = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spy_benchmark.parquet'))
print(f"\n{'='*70}")
print(f"  SPY BUY & HOLD (benchmark)")
print(f"{'='*70}")
spy_s = spy_data['SPY']
spy_s.index = pd.to_datetime(spy_s.index)
for year in range(2020, 2026):
    ys = spy_s[spy_s.index.year == year]
    if len(ys) > 0:
        ret = (ys.iloc[-1] / ys.iloc[0] - 1) * 100
        final = 100000 * (1 + ret/100)
        print(f"  {year}: {ret:+.1f}% --> ${final:,.0f}")
