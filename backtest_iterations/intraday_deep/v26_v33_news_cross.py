#!/usr/bin/env python3
"""
V26-V33: Cross event taxonomy with intraday patterns.
V26: Intraday profile by event type
V27: Optimal entry time by event type
V29: Market stress × intraday behavior
V31: No-catalyst drops intraday signature
"""
import pandas as pd
import numpy as np
import os, json, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

print("Loading intraday data...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'close', 'volume']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['minute'] = (bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute) - (13*60+30)
bars_index = {k: v for k, v in bars.groupby(['symbol', 'date'])}
del bars
print(f"Indexed {len(bars_index):,} groups")

# Build profiles for each event, tagged with event type and market stress
print("Building event profiles...")
profiles = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    entry_day = future[1].date()
    
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    key = (alpaca_t, entry_day)
    if key not in bars_index: continue
    
    day_bars = bars_index[key].sort_values('timestamp')
    if len(day_bars) < 100: continue
    
    open_price = float(day_bars.iloc[0]['open'])
    if open_price <= 0: continue
    
    event_type = str(row.get('stock_event_type', 'unlabeled'))
    stress = str(row.get('liquidity_credit_stress_severity', 'none'))
    success = bool(row.get('success', False))
    
    # Sample key time points instead of all bars (faster)
    for target_min in [0, 15, 30, 60, 120, 180, 240, 300, 360, 389]:
        nearby = day_bars[(day_bars['minute'] >= target_min-2) & (day_bars['minute'] <= target_min+2)]
        if len(nearby) == 0: continue
        price = float(nearby.iloc[0]['close'])
        profiles.append({
            'minute': target_min,
            'norm_price': price / open_price - 1,
            'event_type': event_type,
            'stress': stress,
            'success': success,
            'drop_pct': float(row['drop_pct']),
        })

pdf = pd.DataFrame(profiles)
print(f"Collected {len(pdf):,} observations")

# ============================================================
# V26: Intraday profile by event type
# ============================================================
print(f"\n{'='*70}")
print(f"  V26: INTRADAY PROFILE BY EVENT TYPE")
print(f"{'='*70}")

top_types = pdf['event_type'].value_counts().head(8).index.tolist()
time_labels = {0:'9:30', 15:'9:45', 30:'10:00', 60:'10:30', 120:'11:30', 
               180:'12:30', 240:'13:30', 300:'14:30', 360:'15:30', 389:'15:59'}

print(f"\n{'Type':>25}", end="")
for m in sorted(time_labels.keys()):
    print(f" {time_labels[m]:>7}", end="")
print()
print('-' * 100)

for etype in top_types:
    sub = pdf[pdf['event_type'] == etype]
    print(f"{etype:>25}", end="")
    for m in sorted(time_labels.keys()):
        val = sub[sub['minute'] == m]['norm_price'].mean() * 100
        print(f" {val:>+6.2f}%", end="")
    n = len(sub[sub['minute'] == 0])
    print(f"  (n={n})")

# ============================================================
# V29: Market stress × intraday behavior
# ============================================================
print(f"\n{'='*70}")
print(f"  V29: MARKET STRESS × INTRADAY BEHAVIOR")
print(f"{'='*70}")

print(f"\n{'Stress':>15}", end="")
for m in sorted(time_labels.keys()):
    print(f" {time_labels[m]:>7}", end="")
print()
print('-' * 90)

for stress in ['none', 'low', 'medium', 'high']:
    sub = pdf[pdf['stress'] == stress]
    if len(sub) < 100: continue
    print(f"{stress:>15}", end="")
    for m in sorted(time_labels.keys()):
        val = sub[sub['minute'] == m]['norm_price'].mean() * 100
        print(f" {val:>+6.2f}%", end="")
    n = len(sub[sub['minute'] == 0])
    print(f"  (n={n})")

# ============================================================
# V31: No-catalyst drops intraday signature
# ============================================================
print(f"\n{'='*70}")
print(f"  V31: NO-CATALYST vs CATALYST INTRADAY SIGNATURE")
print(f"{'='*70}")

for label, mask in [
    ('no_clear_catalyst', pdf['event_type'] == 'no_clear_catalyst'),
    ('earnings_miss', pdf['event_type'] == 'earnings_miss'),
    ('guidance_cut', pdf['event_type'] == 'guidance_cut'),
    ('demand_weakness', pdf['event_type'] == 'demand_weakness'),
    ('all_with_catalyst', ~pdf['event_type'].isin(['no_clear_catalyst', 'unlabeled'])),
    ('unlabeled', pdf['event_type'] == 'unlabeled'),
]:
    sub = pdf[mask]
    if len(sub) < 50: continue
    
    # Get open-to-close return
    open_vals = sub[sub['minute'] == 0]['norm_price'].mean() * 100
    close_vals = sub[sub['minute'] == 389]['norm_price'].mean() * 100
    n = len(sub[sub['minute'] == 0])
    wr = sub[sub['minute'] == 0]['success'].mean() * 100
    
    # Best entry time (lowest avg price)
    by_min = sub.groupby('minute')['norm_price'].mean()
    best_min = by_min.idxmin()
    best_val = by_min.min() * 100
    
    print(f"\n  {label} (n={n}, WR={wr:.0f}%):")
    print(f"    Open: {open_vals:+.2f}% | Close: {close_vals:+.2f}% | Best: min {best_min} ({best_val:+.2f}%)")

# ============================================================
# V27: Optimal entry time by event type (summary)
# ============================================================
print(f"\n{'='*70}")
print(f"  V27: OPTIMAL ENTRY TIME BY EVENT TYPE")
print(f"{'='*70}")

print(f"\n{'Type':>25} {'Best Min':>10} {'Best Price':>12} {'Close Price':>12} {'Savings':>8}")
print('-'*70)

for etype in top_types:
    sub = pdf[pdf['event_type'] == etype]
    by_min = sub.groupby('minute')['norm_price'].mean()
    best_min = by_min.idxmin()
    best_val = by_min.min() * 100
    close_val = by_min.get(389, 0) * 100
    savings = close_val - best_val
    
    best_time = time_labels.get(best_min, f"min{best_min}")
    print(f"{etype:>25} {best_time:>10} {best_val:>+11.2f}% {close_val:>+11.2f}% {savings:>+7.2f}%")

# Save
results = {
    'v26_event_types_analyzed': len(top_types),
    'v27_optimal_times': {},
    'v29_stress_levels': ['none', 'low', 'medium', 'high'],
    'v31_no_catalyst_wr': round(pdf[(pdf['event_type'] == 'no_clear_catalyst') & (pdf['minute'] == 0)]['success'].mean() * 100, 1),
}
for etype in top_types:
    sub = pdf[pdf['event_type'] == etype]
    by_min = sub.groupby('minute')['norm_price'].mean()
    best_min = by_min.idxmin()
    results['v27_optimal_times'][etype] = time_labels.get(best_min, f"min{best_min}")

with open(os.path.join(OUT_DIR, 'v26_v33_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, 'v26_v33_results.txt'), 'w') as f:
    f.write("V26-V33: NEWS × INTRADAY ANALYSIS\n\n")
    f.write("V27: Optimal entry time by event type:\n")
    for etype, time in results['v27_optimal_times'].items():
        f.write(f"  {etype}: {time}\n")

print(f"\nSaved to {OUT_DIR}/v26_v33_results.txt")
