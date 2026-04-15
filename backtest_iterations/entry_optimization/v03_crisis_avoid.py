#!/usr/bin/env python3
"""
V03 CRISIS AVOID: Skip trading during known crisis periods (COVID crash,
2022 bear, tariff crash) but trade normally otherwise. Also skip bad
tickers. Tests whether avoiding crises improves risk-adjusted returns.
+3% target, 60d hold, no stop loss, 1/10 position.
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

def entry_filter(row, ctx):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for start, end in CRISIS_PERIODS:
        if start <= dt <= end: return False
    return True

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10)

print_results(results, "V03: CRISIS AVOID (skip COVID/bear/tariff + bad tickers)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v03_crisis_avoid_results.txt'))
