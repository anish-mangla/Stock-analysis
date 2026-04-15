#!/usr/bin/env python3
"""
V01 BASELINE: Trade all events with +3% target, 60d hold, no stop loss,
1/10 position size, max 10 positions. No filtering at all. This is the
baseline to beat. Uses event-day close as entry price.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results

close, high, low, spy, events = load_cached_data()

def entry_filter(row, ctx):
    return True

results = run_backtest(events, close, high, low, entry_filter,
    target_return=0.03, max_hold_days=60, position_fraction=10, max_positions=10)

print_results(results, "V01: BASELINE (no filter)")
save_results(results, os.path.join(os.path.dirname(__file__), 'v01_baseline_results.txt'))
