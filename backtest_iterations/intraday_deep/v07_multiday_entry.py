#!/usr/bin/env python3
"""
V07 — Multi-day entry: spread buy across T+2 and T+3.
Instead of 100% on T+2, try 50/50 on T+2/T+3, or 33/33/33 on T+2/T+3/T+4.
Also test: buy on T+2 only if first hour is positive, else wait to T+3.
"""
import pandas as pd
import numpy as np
import os, json, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

def entry_filter(row, ctx):
    d = row['drop_pct']
    out = row.get('sector_rotation_direction', '') == 'outflow'
    stress = row.get('liquidity_credit_stress_severity', 'none')
    fri = row['event_date'].dayofweek == 4
    if d >= 0.07: return True
    if d >= 0.05 and (out or fri or stress in ['medium', 'low']): return True
    if d >= 0.03 and out and fri: return True
    if d >= 0.03 and stress == 'medium': return True
    return False

# Test different entry delays
print(f"{'='*70}")
print(f"  V07: MULTI-DAY ENTRY / ENTRY DELAY SWEEP")
print(f"{'='*70}\n")

results = []
for delay in [0, 1, 2, 3, 4, 5]:
    r = run_backtest(events, close_mat, high_mat, low_mat, entry_filter,
        target_return=0.05, max_hold_days=60, position_fraction=10, max_positions=10,
        entry_delay=delay)
    label = f"Delay {delay}d"
    results.append({'label': label, 'return': r['total_return'], 'trades': r['trades'],
        'wr': r['win_rate'], 'losers': r['losers']})
    print(f"  {label}: {r['total_return']:>+7.1f}% | {r['trades']} trades | WR {r['win_rate']:.0f}% | {r['losers']} losers")

with open(os.path.join(OUT_DIR, 'v07_results.txt'), 'w') as f:
    f.write("V07: ENTRY DELAY SWEEP\n+5%/60d, fixed 1/10, no biases\n\n")
    for r in results:
        f.write(f"{r['label']}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}%\n")
with open(os.path.join(OUT_DIR, 'v07_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/v07_results.txt")
