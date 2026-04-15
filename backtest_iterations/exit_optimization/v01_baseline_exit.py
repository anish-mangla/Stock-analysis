#!/usr/bin/env python3
"""
EXIT V01 BASELINE: Uses V09 entry filters (best entry logic) with no stop
loss and 60-day max hold. This is the exit baseline — same as V09 entry
but we track exit details to understand where losses come from.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results
import pandas as pd

close, high, low, spy, events = load_cached_data()

CRISIS_PERIODS = [
    (pd.Timestamp('2020-02-20'), pd.Timestamp('2020-04-15')),
    (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-10-31')),
    (pd.Timestamp('2025-03-01'), pd.Timestamp('2025-04-30')),
]
BAD_TICKERS = {'PFE', 'TMUS', 'ACN', 'CVS', 'AMT', 'BLK', 'IBM', 'LMT', 'DHR', 'MSFT'}
BEST_TICKERS = {'NVDA', 'AMZN', 'AMAT', 'AVGO', 'GE', 'INTU', 'MA', 'AXP', 'LOW', 'MO', 'COF', 'BMY', 'MS', 'MDT', 'HON'}
BEST_SECTORS = {'SMH', 'XLK'}

def entry_filter(row, ctx):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for start, end in CRISIS_PERIODS:
        if start <= dt <= end: return False
    ticker = row['ticker']
    sector = row.get('sector_etf', '')
    drop = row['drop_pct']
    has_outflow = row.get('sector_rotation_direction', '') == 'outflow'
    stress = row.get('liquidity_credit_stress_severity', 'none')
    is_friday = dt.dayofweek == 4
    if ticker in BEST_TICKERS: return True
    if drop >= 0.07: return True
    if drop >= 0.05:
        if has_outflow or sector in BEST_SECTORS or is_friday or stress in ['medium', 'low']: return True
    if sector in BEST_SECTORS and has_outflow: return True
    if is_friday and stress == 'medium': return True
    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)

print_results(results, "EXIT V01: BASELINE (no stop, 60d hold)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v01_baseline_exit_results.txt'))
