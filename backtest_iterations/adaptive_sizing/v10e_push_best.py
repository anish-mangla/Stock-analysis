#!/usr/bin/env python3
"""
SIZING V10e: Push the new best (20MA, 1/6 bull, 1/14 bear = +401.5%).
Fine-tune around this sweet spot. Also try 1/16 bear and 1/18 bear.
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

configs = [
    (6, 14, 10, "20MA 1/6 bull 1/14 bear max10"),
    (6, 14, 15, "20MA 1/6 bull 1/14 bear max15"),
    (6, 16, 10, "20MA 1/6 bull 1/16 bear max10"),
    (6, 16, 12, "20MA 1/6 bull 1/16 bear max12"),
    (6, 18, 12, "20MA 1/6 bull 1/18 bear max12"),
    (6, 20, 12, "20MA 1/6 bull 1/20 bear max12"),
    (5, 14, 12, "20MA 1/5 bull 1/14 bear max12"),
    (5, 16, 12, "20MA 1/5 bull 1/16 bear max12"),
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
    print_results(results, f"V10e: {label}")
    safe = label.replace('/', '_').replace(' ', '_')
    save_results(results, os.path.join(os.path.dirname(__file__), f'v10e_{safe}_results.txt'))
