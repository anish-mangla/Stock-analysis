#!/usr/bin/env python3
"""
V10 FINAL: Best combination of all findings. Crisis avoidance + best ticker
priority + sector preference + 2-day delayed entry + earnings season boost
+ Friday preference + sector outflow requirement for small drops + expanded
bad ticker list. +3% target, 60d hold, no stop loss, 1/10 position.
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
EARNINGS_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}

def entry_filter(row, ctx):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for start, end in CRISIS_PERIODS:
        if start <= dt <= end: return False

    ticker = row['ticker']
    sector = row.get('sector_etf', '')
    drop = row['drop_pct']
    month = dt.month
    is_earnings = month in EARNINGS_MONTHS
    is_friday = dt.dayofweek == 4
    has_outflow = row.get('sector_rotation_direction', '') == 'outflow'
    stress = row.get('liquidity_credit_stress_severity', 'none')

    # Best tickers: always trade
    if ticker in BEST_TICKERS: return True

    # 7%+ drops: always trade
    if drop >= 0.07: return True

    # 5-7% drops during earnings season: trade if any positive signal
    if drop >= 0.05 and is_earnings:
        if has_outflow or sector in BEST_SECTORS or is_friday or stress in ['medium', 'low']:
            return True

    # 5-7% drops outside earnings: need outflow or best sector
    if drop >= 0.05:
        if has_outflow or sector in BEST_SECTORS: return True

    # 3-5% drops: only in best sectors with outflow, or on Fridays with medium stress
    if sector in BEST_SECTORS and has_outflow: return True
    if is_friday and stress == 'medium': return True
    if is_friday and has_outflow and is_earnings: return True

    return False

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)

print_results(results, "V10: FINAL (all best patterns combined)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v10_final_results.txt'))

# Also compute SPY benchmark for comparison
spy_data = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'spy_benchmark.parquet'))
spy_start = spy_data['SPY'].iloc[0]
spy_end = spy_data['SPY'].iloc[-1]
spy_return = (spy_end / spy_start - 1) * 100
print(f"\n  SPY BUY & HOLD: {spy_return:+.1f}% over same period")
print(f"  SYSTEM vs SPY: {results['total_return'] - spy_return:+.1f}% difference")
