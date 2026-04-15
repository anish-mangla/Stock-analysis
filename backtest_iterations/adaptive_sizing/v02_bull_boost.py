#!/usr/bin/env python3
"""
SIZING V02 BULL BOOST: When SPY is above its 50-day MA (bull market), use
1/5 position size instead of 1/8. When below (bear/correction), use 1/10.
This deploys more capital during good times and less during bad times.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results, INITIAL_CAPITAL
import pandas as pd

close, high, low, spy, events = load_cached_data()

# Compute SPY 50-day MA
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
    # Check if SPY is above 50-day MA (bull market)
    if today in spy_close.index and today in spy_50ma.index:
        spy_val = spy_close.at[today]
        ma_val = spy_50ma.at[today]
        if pd.notna(spy_val) and pd.notna(ma_val) and spy_val > ma_val:
            return INITIAL_CAPITAL / 5  # Bull: bigger positions
    return INITIAL_CAPITAL / 10  # Bear/correction: smaller positions

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.05, max_hold_days=60, max_positions=10, entry_delay=2,
    sizing_fn=sizing_fn)
print_results(results, "SIZING V02: BULL BOOST (1/5 in bull, 1/10 in bear)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v02_bull_boost_results.txt'))
