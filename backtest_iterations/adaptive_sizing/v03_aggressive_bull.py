#!/usr/bin/env python3
"""
SIZING V03 AGGRESSIVE BULL: Push harder. SPY above 50MA AND above 200MA
(strong bull) = 1/4 position. SPY above 50MA only = 1/6. Below 50MA = 1/10.
Also increase max positions to 12 during strong bull.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results, INITIAL_CAPITAL
import pandas as pd

close, high, low, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_50ma = spy_close.rolling(50).mean()
spy_200ma = spy_close.rolling(200).mean()

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
    if today in spy_close.index and today in spy_50ma.index and today in spy_200ma.index:
        s = spy_close.at[today]
        m50 = spy_50ma.at[today]
        m200 = spy_200ma.at[today]
        if pd.notna(s) and pd.notna(m50) and pd.notna(m200):
            if s > m50 and s > m200:  # Strong bull
                return INITIAL_CAPITAL / 4
            elif s > m50:  # Moderate bull
                return INITIAL_CAPITAL / 6
    return INITIAL_CAPITAL / 10  # Bear/correction

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.05, max_hold_days=60, max_positions=12, entry_delay=2,
    sizing_fn=sizing_fn)
print_results(results, "SIZING V03: AGGRESSIVE BULL (1/4 strong, 1/6 moderate, 1/10 bear)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v03_aggressive_bull_results.txt'))
