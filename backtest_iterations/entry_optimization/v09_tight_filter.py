#!/usr/bin/env python3
"""
V09 TIGHT FILTER: Same as V07 but add: require sector outflow OR 5%+ drop
for ALL trades (not just non-earnings months). Also add best-bouncer ticker
preference (NVDA/AMZN/AMAT/AVGO/GE get priority). Skip 3-5% drops in calm
markets entirely. +3% target, 60d hold, 2-day delay, 1/10 position.
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
BAD_TICKERS = {'PFE', 'TMUS', 'ACN', 'CVS', 'AMT', 'AMT', 'BLK', 'IBM', 'LMT'}
BEST_TICKERS = {'NVDA', 'AMZN', 'AMAT', 'AVGO', 'GE', 'INTU', 'MA', 'AXP', 'LOW', 'MO'}
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

    # Best tickers: always trade if 3%+ drop
    if ticker in BEST_TICKERS: return True

    # 7%+ drops: always trade
    if drop >= 0.07: return True

    # 5-7% drops: trade if outflow or best sector or Friday
    if drop >= 0.05:
        if has_outflow or sector in BEST_SECTORS or is_friday: return True
        if stress in ['medium', 'low']: return True
        return False

    # 3-5% drops: only trade best sectors with outflow or medium stress
    if sector in BEST_SECTORS and (has_outflow or stress == 'medium'): return True
    if has_outflow and is_friday: return True

    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)

print_results(results, "V09: TIGHT FILTER (best tickers priority + stricter 3-5%)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v09_tight_filter_results.txt'))
