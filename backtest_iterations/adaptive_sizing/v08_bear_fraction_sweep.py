#!/usr/bin/env python3
"""
SIZING V08: Bull fraction locked at 1/6 (best from V07). Sweep bear fractions
to find optimal bear-market sizing. Test 1/8, 1/10, 1/12, 1/14, 1/16 bear.
All equity-based (compound on current equity, not initial capital).
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

BULL_FRAC = 6  # locked from V07

for bear_frac in [8, 10, 12, 14, 16]:
    def make_sizing(bf):
        def sizing_fn(row, ctx):
            today = ctx['today']
            equity = ctx.get('equity', INITIAL_CAPITAL)
            if today in spy_close.index and today in spy_50ma.index:
                s = spy_close.at[today]
                m = spy_50ma.at[today]
                if pd.notna(s) and pd.notna(m) and s > m:
                    return equity / BULL_FRAC
            return equity / bf
        return sizing_fn

    results = run_backtest(events, close, high, low, entry_filter,
        target_return=0.05, max_hold_days=60, max_positions=10, entry_delay=2,
        sizing_fn=make_sizing(bear_frac))
    label = f"1/6 bull, 1/{bear_frac} bear"
    print_results(results, f"SIZING V08: {label} (equity-based)")
    save_results(results, os.path.join(os.path.dirname(__file__), f'v08_bear{bear_frac}_results.txt'))
