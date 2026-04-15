#!/usr/bin/env python3
"""
V05 SECTOR PREFERENCE: Build on V03 (crisis avoid). Prioritize semis (SMH)
which have 85.7% success and +1.58% return. Also prefer Friday drops (85.2%
success). Skip bad tickers. When multiple candidates, prefer SMH sector
and Friday events. +3% target, 60d hold, no stop loss.
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
BEST_SECTORS = {'SMH'}
GOOD_SECTORS = {'XLK', 'XLP', 'XLI'}

def entry_filter(row, ctx):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for start, end in CRISIS_PERIODS:
        if start <= dt <= end: return False

    sector = row.get('sector_etf', '')
    drop = row['drop_pct']

    # Best sectors: trade all drops
    if sector in BEST_SECTORS: return True
    # Good sectors: trade all drops
    if sector in GOOD_SECTORS: return True
    # Other sectors: only trade 5%+ drops
    if drop >= 0.05: return True
    # 3-5% drops in other sectors: only on Fridays or with sector outflow
    if drop >= 0.03:
        is_friday = dt.dayofweek == 4
        has_outflow = row.get('sector_rotation_direction', '') == 'outflow'
        if is_friday or has_outflow: return True
    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10)

print_results(results, "V05: SECTOR PREF (SMH/XLK always, others need 5%+ or Friday/outflow)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v05_sector_pref_results.txt'))
