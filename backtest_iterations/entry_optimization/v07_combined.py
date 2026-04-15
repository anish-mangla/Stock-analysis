#!/usr/bin/env python3
"""
V07 COMBINED BEST: Crisis avoid + sector preference + 2-day delayed entry
+ prefer earnings season months (83.4% vs 77.3% success). Skip bad tickers.
During non-earnings months, require 5%+ drop or sector outflow. +3% target,
60d hold, no stop loss, 1/10 position.
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

    # During earnings season: trade best sectors always, others need 5%+ or outflow
    if is_earnings:
        if sector in BEST_SECTORS: return True
        if drop >= 0.05: return True
        if has_outflow or is_friday: return True
        return False

    # Non-earnings months: be more selective — need 5%+ drop or best sector + outflow
    if drop >= 0.05: return True
    if sector in BEST_SECTORS and (has_outflow or is_friday): return True
    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)

print_results(results, "V07: COMBINED (crisis+sector+delay+earnings)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v07_combined_results.txt'))
