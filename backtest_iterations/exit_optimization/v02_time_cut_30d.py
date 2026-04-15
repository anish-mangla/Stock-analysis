#!/usr/bin/env python3
"""
EXIT V02 TIME CUT 30D: Same entry as V01 but cut positions that are still
negative after 30 days. Hypothesis: if it hasn't bounced in 30 days, it
probably won't, and the freed capital can earn more on new trades.
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

def exit_loser_30d(pos, today, days_held, current_return):
    return days_held >= 30 and current_return < 0

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10,
    entry_delay=2, exit_loser_fn=exit_loser_30d)

print_results(results, "EXIT V02: CUT LOSERS AFTER 30D IF NEGATIVE")
save_results(results, os.path.join(os.path.dirname(__file__), 'v02_time_cut_30d_results.txt'))
