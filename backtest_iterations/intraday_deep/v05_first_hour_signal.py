#!/usr/bin/env python3
"""
V05 — First-hour pattern as entry signal.
If stock is UP in first 30/60 min, does it keep going? If DOWN, does it recover?
Can we use first-hour direction to decide whether to enter?
"""
import pandas as pd
import numpy as np
import os, json, sys, time as timer

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
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'close']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute

# Compute first-hour return for each (symbol, date)
# 9:30-10:00 = first 30 min, 9:30-10:30 = first 60 min
print("Computing first-hour returns...")
open_bars = bars[bars['utc_min'] == 810].groupby(['symbol', 'date'])['open'].first()  # 13:30 UTC = 9:30 ET... wait
# Actually 9:30 ET = 13:30 UTC = 810 UTC minutes. Let me check.
# 13*60+30 = 810. Yes.

# Get price at open (first bar), at 10:00 (30 min), at 10:30 (60 min)
first_bar = bars.sort_values('timestamp').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'open']]
first_bar.columns = ['symbol', 'date', 'price_open']

for target_min, label in [(840, 'price_30m'), (870, 'price_60m')]:
    nearby = bars[(bars['utc_min'] >= target_min-3) & (bars['utc_min'] <= target_min+3)]
    best = nearby.sort_values('utc_min').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'close']]
    best.columns = ['symbol', 'date', label]
    first_bar = first_bar.merge(best, on=['symbol', 'date'], how='left')

first_bar['ret_30m'] = (first_bar['price_30m'] / first_bar['price_open'] - 1) * 100
first_bar['ret_60m'] = (first_bar['price_60m'] / first_bar['price_open'] - 1) * 100

# Pivot to matrices
for col in ['ret_30m', 'ret_60m']:
    mat = first_bar.pivot(index='date', columns='symbol', values=col)
    mat.index = pd.to_datetime(mat.index)
    if 'BRK.B' in mat.columns:
        mat.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)
    if col == 'ret_30m':
        ret30_mat = mat
    else:
        ret60_mat = mat

del bars
print("Done")

# Now: for each event, check if first-hour return predicts trade success
print("\nAnalyzing first-hour return vs trade outcome...")
events_with_fh = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    entry_day = future[1]
    
    r30 = ret30_mat.at[entry_day, ticker] if entry_day in ret30_mat.index and ticker in ret30_mat.columns else None
    r60 = ret60_mat.at[entry_day, ticker] if entry_day in ret60_mat.index and ticker in ret60_mat.columns else None
    
    if r30 is None or pd.isna(r30): continue
    
    events_with_fh.append({
        'ticker': ticker, 'event_date': dt, 'entry_day': entry_day,
        'ret_30m': float(r30), 'ret_60m': float(r60) if r60 is not None and pd.notna(r60) else None,
        'success': bool(row.get('success', False)),
        'drop_pct': float(row['drop_pct']),
    })

edf = pd.DataFrame(events_with_fh)
print(f"Events with first-hour data: {len(edf)}")

# Analysis
print(f"\n{'='*70}")
print(f"  V05: FIRST-HOUR SIGNAL ANALYSIS")
print(f"{'='*70}")

# Split by first-30-min direction
for threshold in [0, 0.5, 1.0, -0.5, -1.0]:
    if threshold >= 0:
        mask = edf['ret_30m'] >= threshold
        label = f"30m return >= {threshold:+.1f}%"
    else:
        mask = edf['ret_30m'] <= threshold
        label = f"30m return <= {threshold:+.1f}%"
    sub = edf[mask]
    if len(sub) == 0: continue
    wr = sub['success'].mean() * 100
    print(f"  {label:>25}: {len(sub):>5} events | WR {wr:.1f}%")

print()
for threshold in [0, 0.5, 1.0, -0.5, -1.0]:
    if threshold >= 0:
        mask = edf['ret_60m'] >= threshold
        label = f"60m return >= {threshold:+.1f}%"
    else:
        mask = edf['ret_60m'] <= threshold
        label = f"60m return <= {threshold:+.1f}%"
    sub = edf[mask.fillna(False)]
    if len(sub) == 0: continue
    wr = sub['success'].mean() * 100
    print(f"  {label:>25}: {len(sub):>5} events | WR {wr:.1f}%")

# Bucket analysis
print(f"\n  First 30-min return buckets:")
edf['bucket_30m'] = pd.cut(edf['ret_30m'], bins=[-20, -2, -1, -0.5, 0, 0.5, 1, 2, 20])
for bucket, group in edf.groupby('bucket_30m', observed=True):
    wr = group['success'].mean() * 100
    print(f"    {str(bucket):>15}: {len(group):>5} events | WR {wr:.1f}%")

# Save
results = {
    'events_analyzed': len(edf),
    'overall_wr': round(edf['success'].mean() * 100, 1),
    'first_30m_up_wr': round(edf[edf['ret_30m'] > 0]['success'].mean() * 100, 1),
    'first_30m_down_wr': round(edf[edf['ret_30m'] < 0]['success'].mean() * 100, 1),
    'first_30m_up_count': int((edf['ret_30m'] > 0).sum()),
    'first_30m_down_count': int((edf['ret_30m'] < 0).sum()),
}
with open(os.path.join(OUT_DIR, 'v05_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, 'v05_results.txt'), 'w') as f:
    f.write("V05: FIRST-HOUR SIGNAL ANALYSIS\n\n")
    f.write(f"Events: {results['events_analyzed']}\n")
    f.write(f"Overall WR: {results['overall_wr']:.1f}%\n")
    f.write(f"First 30m UP: {results['first_30m_up_count']} events, WR {results['first_30m_up_wr']:.1f}%\n")
    f.write(f"First 30m DOWN: {results['first_30m_down_count']} events, WR {results['first_30m_down_wr']:.1f}%\n")
print(f"\nSaved to {OUT_DIR}/v05_results.txt")
