#!/usr/bin/env python3
"""
EXIT V04 TRAILING: Instead of time-based cuts, use a trailing approach:
if a position was up at least +1% at some point but has now fallen back
below 0%, exit it. The idea: if it bounced but couldn't hold, it's done.
Also test: cut if position drops below -10% (catastrophic stop only).
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

# Catastrophic stop only: -10%
results_stop10 = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, stop_loss=-0.10,
    position_fraction=10, max_positions=10, entry_delay=2)
print_results(results_stop10, "EXIT V04a: CATASTROPHIC STOP -10% ONLY")
save_results(results_stop10, os.path.join(os.path.dirname(__file__), 'v04a_stop10_results.txt'))

# Wider stop: -15%
results_stop15 = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, stop_loss=-0.15,
    position_fraction=10, max_positions=10, entry_delay=2)
print_results(results_stop15, "EXIT V04b: CATASTROPHIC STOP -15% ONLY")
save_results(results_stop15, os.path.join(os.path.dirname(__file__), 'v04b_stop15_results.txt'))

# -20% stop (only cuts the absolute worst)
results_stop20 = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, stop_loss=-0.20,
    position_fraction=10, max_positions=10, entry_delay=2)
print_results(results_stop20, "EXIT V04c: CATASTROPHIC STOP -20% ONLY")
save_results(results_stop20, os.path.join(os.path.dirname(__file__), 'v04c_stop20_results.txt'))
