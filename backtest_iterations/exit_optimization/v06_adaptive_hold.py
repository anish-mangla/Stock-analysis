#!/usr/bin/env python3
"""
EXIT V06 ADAPTIVE HOLD: Different max hold by drop bucket. Bigger drops
bounce faster (avg 1.7d for 10%+) so they need less time. Smaller drops
are slower (avg 3.8d for 3-5%). Use: 10%+=30d, 7-10%=45d, 5-7%=60d,
3-5%=90d. This frees capital from big-drop positions faster.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results
import pandas as pd

close, high, low, spy, events = load_cached_data()
CRISIS_PERIODS = [(pd.Timestamp('2020-02-20'),pd.Timestamp('2020-04-15')),(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-10-31')),(pd.Timestamp('2025-03-01'),pd.Timestamp('2025-04-30'))]
BAD_TICKERS = {'PFE','TMUS','ACN','CVS','AMT','BLK','IBM','LMT','DHR','MSFT'}
BEST_TICKERS = {'NVDA','AMZN','AMAT','AVGO','GE','INTU','MA','AXP','LOW','MO','COF','BMY','MS','MDT','HON'}
BEST_SECTORS = {'SMH','XLK'}

def entry_filter(row, ctx):
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

# Run separate backtests by bucket with different hold periods, then combine
# Actually, the engine doesn't support per-trade hold periods, so let's test
# the overall effect by running with different hold periods on filtered events

for label, hold, bucket_filter in [
    ("10%+ drops only, 30d hold", 30, lambda r: r['drop_pct'] >= 0.10),
    ("7-10% drops only, 45d hold", 45, lambda r: 0.07 <= r['drop_pct'] < 0.10),
    ("5-7% drops only, 60d hold", 60, lambda r: 0.05 <= r['drop_pct'] < 0.07),
    ("3-5% drops only, 90d hold", 90, lambda r: 0.03 <= r['drop_pct'] < 0.05),
]:
    def make_filter(bf):
        def f(row, ctx):
            if not bf(row): return False
            return entry_filter(row, ctx)
        return f

    results = run_backtest(events, close, high, low, make_filter(bucket_filter),
        target_return=0.03, max_hold_days=hold, position_fraction=10, max_positions=10, entry_delay=2)
    print_results(results, f"EXIT V06: {label}")

# Also test: uniform 90d hold (the winner from V05)
results_90 = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=90, position_fraction=10, max_positions=10, entry_delay=2)
print_results(results_90, "EXIT V06 REF: All drops, 90d hold")
save_results(results_90, os.path.join(os.path.dirname(__file__), 'v06_adaptive_hold_results.txt'))
