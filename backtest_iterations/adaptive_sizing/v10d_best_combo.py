#!/usr/bin/env python3
"""
SIZING V10d: Combine best findings. 20MA regime detection (+380.1%) with
max 12 positions (+359.0% on 50MA). Also test 20MA + max 12 + bear fraction sweep.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results, INITIAL_CAPITAL
import pandas as pd

close, high, low, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()

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

# Test combos: 20MA + different max positions + different bear fractions
configs = [
    (6, 12, 10, "20MA, 1/6 bull, 1/12 bear, max 10"),
    (6, 12, 12, "20MA, 1/6 bull, 1/12 bear, max 12"),
    (6, 12, 15, "20MA, 1/6 bull, 1/12 bear, max 15"),
    (6, 10, 12, "20MA, 1/6 bull, 1/10 bear, max 12"),
    (6, 14, 12, "20MA, 1/6 bull, 1/14 bear, max 12"),
    (5, 12, 12, "20MA, 1/5 bull, 1/12 bear, max 12"),
    (7, 12, 12, "20MA, 1/7 bull, 1/12 bear, max 12"),
]

for bull_f, bear_f, max_pos, label in configs:
    def make_sizing(bf, brf):
        def sizing_fn(row, ctx):
            today = ctx['today']
            equity = ctx.get('equity', INITIAL_CAPITAL)
            if today in spy_close.index and today in spy_20ma.index:
                s = spy_close.at[today]
                m = spy_20ma.at[today]
                if pd.notna(s) and pd.notna(m) and s > m:
                    return equity / bf
            return equity / brf
        return sizing_fn

    results = run_backtest(events, close, high, low, entry_filter,
        target_return=0.05, max_hold_days=60, max_positions=max_pos, entry_delay=2,
        sizing_fn=make_sizing(bull_f, bear_f))
    print_results(results, f"V10d: {label}")
    safe_label = label.replace('/', '_').replace(' ', '_').replace(',', '')
    save_results(results, os.path.join(os.path.dirname(__file__), f'v10d_{safe_label}_results.txt'))
