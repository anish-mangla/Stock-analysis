#!/usr/bin/env python3
"""
SIZING V06: V04 equity-based but with max 12 positions instead of 10.
V04 was the best so far. More positions = more diversification + more
capital deployed. Test if this improves or hurts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results, INITIAL_CAPITAL
import pandas as pd

close, high, low, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_50ma = spy_close.rolling(50).mean()

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

def sizing_fn(row, ctx):
    today = ctx['today']
    equity = ctx.get('equity', INITIAL_CAPITAL)
    if today in spy_close.index and today in spy_50ma.index:
        s = spy_close.at[today]
        m = spy_50ma.at[today]
        if pd.notna(s) and pd.notna(m) and s > m:
            return equity / 5
    return equity / 10

for max_pos in [8, 10, 12, 15]:
    results = run_backtest(events, close, high, low, entry_filter,
        target_return=0.05, max_hold_days=60, max_positions=max_pos, entry_delay=2,
        sizing_fn=sizing_fn)
    print_results(results, f"SIZING V06: V04 equity-based + max {max_pos} positions")
    save_results(results, os.path.join(os.path.dirname(__file__), f'v06_max{max_pos}_results.txt'))
