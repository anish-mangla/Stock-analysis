#!/usr/bin/env python3
"""
V04 STRESS FILTER: Build on V03 (crisis avoid + bad tickers). Add: only
trade when market stress is medium (VIX 20-30) or during calm with 5%+
drops. Skip calm-market 3-5% drops (highest loser rate). This targets
the medium-stress sweet spot we found (85.4% success vs 78.5% calm).
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

    stress = row.get('liquidity_credit_stress_severity', 'none')
    drop = row['drop_pct']

    # Medium stress: trade everything
    if stress == 'medium': return True
    # Low stress: trade 5%+ drops only
    if stress == 'low' and drop >= 0.05: return True
    # No stress: trade 7%+ drops only
    if stress == 'none' and drop >= 0.07: return True
    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10)

print_results(results, "V04: STRESS FILTER (medium=all, low=5%+, none=7%+)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v04_stress_filter_results.txt'))
