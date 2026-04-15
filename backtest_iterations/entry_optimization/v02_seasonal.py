#!/usr/bin/env python3
"""
V02 SEASONAL: Same as baseline but skip trading in Feb, Sep, Dec (worst
months with 71-76% success vs 87-88% in best months). Also skip known
bad tickers PFE/TMUS/ACN/CVS/AMT. +3% target, 60d hold, no stop loss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results

close, high, low, spy, events = load_cached_data()

BAD_MONTHS = {2, 9, 12}
BAD_TICKERS = {'PFE', 'TMUS', 'ACN', 'CVS', 'AMT'}

def entry_filter(row, ctx):
    month = row['event_date'].month
    if month in BAD_MONTHS: return False
    if row['ticker'] in BAD_TICKERS: return False
    return True

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10)

print_results(results, "V02: SEASONAL (skip Feb/Sep/Dec + bad tickers)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v02_seasonal_results.txt'))
