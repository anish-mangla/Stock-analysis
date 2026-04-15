#!/usr/bin/env python3
"""
V25 — Overnight gap analysis.
Compare close on drop day (T+0) to open on T+1. Do stocks that gap DOWN
on T+1 end up being worse trades? Can we use T+1 open as a filter?
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

# Load open prices from intraday data
print("Loading intraday data for overnight gaps...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date

# Get first bar (open) for each (symbol, date)
opens = bars.sort_values('timestamp').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'open']]
open_lookup = {(r['symbol'], r['date']): float(r['open']) for _, r in opens.iterrows()}
del bars, opens
print("Done")

# For each event, compute overnight gaps
print("Computing overnight gaps...")
gap_data = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    
    future = trading_dates[trading_dates > dt]
    if len(future) < 3: continue
    
    t1 = future[0].date()  # T+1
    t2 = future[1].date()  # T+2 (entry day)
    
    # Close on drop day
    if ticker not in close_mat.columns or dt not in close_mat.index: continue
    close_t0 = close_mat.at[dt, ticker]
    if pd.isna(close_t0): continue
    close_t0 = float(close_t0)
    
    # Open on T+1
    open_t1 = open_lookup.get((alpaca_t, t1))
    if open_t1 is None: continue
    
    # Open on T+2
    open_t2 = open_lookup.get((alpaca_t, t2))
    
    gap_t1 = (open_t1 / close_t0 - 1) * 100
    gap_t2 = (open_t2 / open_t1 - 1) * 100 if open_t2 else None
    
    gap_data.append({
        'ticker': ticker, 'event_date': dt,
        'success': bool(row.get('success', False)),
        'drop_pct': float(row['drop_pct']),
        'gap_t1': gap_t1,
        'gap_t2': gap_t2,
    })

gdf = pd.DataFrame(gap_data)
print(f"Events with gap data: {len(gdf)}")

print(f"\n{'='*70}")
print(f"  V25: OVERNIGHT GAP ANALYSIS")
print(f"{'='*70}")

print(f"\n  T+0 close → T+1 open gap:")
print(f"    Mean: {gdf['gap_t1'].mean():+.2f}%")
print(f"    Median: {gdf['gap_t1'].median():+.2f}%")
print(f"    % gap down: {(gdf['gap_t1'] < 0).mean()*100:.1f}%")
print(f"    % gap up: {(gdf['gap_t1'] > 0).mean()*100:.1f}%")

print(f"\n  Win rate by T+1 gap direction:")
for lo, hi, label in [(-100, -3, 'Gap down 3%+'), (-3, -1, 'Gap down 1-3%'), 
                       (-1, 0, 'Gap down 0-1%'), (0, 1, 'Gap up 0-1%'),
                       (1, 3, 'Gap up 1-3%'), (3, 100, 'Gap up 3%+')]:
    sub = gdf[(gdf['gap_t1'] >= lo) & (gdf['gap_t1'] < hi)]
    if len(sub) > 20:
        wr = sub['success'].mean() * 100
        print(f"    {label:>20}: {len(sub):>5} events | WR {wr:.1f}%")

# T+1 → T+2 gap
gdf2 = gdf.dropna(subset=['gap_t2'])
print(f"\n  T+1 close → T+2 open gap:")
print(f"    Mean: {gdf2['gap_t2'].mean():+.2f}%")

print(f"\n  Win rate by T+2 gap direction:")
for lo, hi, label in [(-100, -1, 'Gap down 1%+'), (-1, 0, 'Gap down 0-1%'),
                       (0, 1, 'Gap up 0-1%'), (1, 100, 'Gap up 1%+')]:
    sub = gdf2[(gdf2['gap_t2'] >= lo) & (gdf2['gap_t2'] < hi)]
    if len(sub) > 20:
        wr = sub['success'].mean() * 100
        print(f"    {label:>20}: {len(sub):>5} events | WR {wr:.1f}%")

# Save
results = {
    'events': len(gdf),
    'gap_t1_mean': round(gdf['gap_t1'].mean(), 2),
    'gap_t1_median': round(gdf['gap_t1'].median(), 2),
    'gap_down_wr': round(gdf[gdf['gap_t1'] < 0]['success'].mean() * 100, 1),
    'gap_up_wr': round(gdf[gdf['gap_t1'] > 0]['success'].mean() * 100, 1),
}
with open(os.path.join(OUT_DIR, 'v25_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, 'v25_results.txt'), 'w') as f:
    f.write("V25: OVERNIGHT GAP ANALYSIS\n\n")
    for k, v in results.items():
        f.write(f"{k}: {v}\n")
print(f"\nSaved to {OUT_DIR}/v25_results.txt")
