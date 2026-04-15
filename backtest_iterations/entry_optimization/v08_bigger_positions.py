#!/usr/bin/env python3
"""
V08 BIGGER POSITIONS: Same filters as V07 but with 1/7 position size and
max 7 positions instead of 1/10 and 10. Capital utilization was only 42%
in V07, meaning 58% of capital was idle. Bigger positions should capture
more return from the same trades. +3% target, 60d hold, 2-day delay.
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
BAD_TICKERS = {'PFE', 'TMUS', 'ACN', 'CVS', 'AMT'}
BEST_SECTORS = {'SMH', 'XLK', 'XLP', 'XLI'}
EARNINGS_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}

def entry_filter(row, ctx):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for start, end in CRISIS_PERIODS:
        if start <= dt <= end: return False
    sector = row.get('sector_etf', '')
    drop = row['drop_pct']
    month = dt.month
    is_earnings = month in EARNINGS_MONTHS
    is_friday = dt.dayofweek == 4
    has_outflow = row.get('sector_rotation_direction', '') == 'outflow'
    if is_earnings:
        if sector in BEST_SECTORS: return True
        if drop >= 0.05: return True
        if has_outflow or is_friday: return True
        return False
    if drop >= 0.05: return True
    if sector in BEST_SECTORS and (has_outflow or is_friday): return True
    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=7, max_positions=7, entry_delay=2)

print_results(results, "V08: BIGGER POSITIONS (1/7, max 7)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v08_bigger_positions_results.txt'))
