#!/usr/bin/env python3
"""
V01 — Intraday profile of entry days (T+2 after drop).
Pure analysis: for every event, pull minute bars on entry day and compute
the average price path minute by minute.
"""
import pandas as pd
import numpy as np
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

events = pd.read_csv(os.path.join(BASE, '..', 'outputs', 'events_fully_labeled.csv'))
events['event_date'] = pd.to_datetime(events['event_date'])
events = events.dropna(subset=['entry_price'])

# Load adjusted intraday data and pre-index by (symbol, date)
print("Loading and indexing intraday data...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
# Pre-compute minute offset from 9:30 ET (13:30 UTC)
bars['minute'] = (bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute) - (13 * 60 + 30)
# Build lookup: (symbol, date) -> DataFrame
bars_index = {k: v for k, v in bars.groupby(['symbol', 'date'])}
print(f"Indexed {len(bars_index):,} (symbol, date) groups")

# Get trading dates
close = pd.read_parquet(os.path.join(BASE, 'close_matrix.parquet'))
trading_dates = pd.DatetimeIndex(sorted(close.index))

# Collect profiles
print("Building entry day profiles...")
profiles = []
event_count = 0

for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    
    future = trading_dates[trading_dates > dt]
    if len(future) < 2:
        continue
    entry_day = future[1].date()
    
    alpaca_ticker = ticker.replace('BRK-B', 'BRK.B')
    key = (alpaca_ticker, entry_day)
    if key not in bars_index:
        continue
    
    day_bars = bars_index[key].sort_values('timestamp')
    if len(day_bars) < 100:
        continue
    
    open_price = float(day_bars.iloc[0]['open'])
    if open_price <= 0:
        continue
    
    event_count += 1
    close_price = float(day_bars.iloc[-1]['close'])
    
    for _, bar in day_bars.iterrows():
        profiles.append({
            'minute': int(bar['minute']),
            'norm_price': float(bar['close']) / open_price - 1,
            'volume': float(bar['volume']),
            'drop_pct': float(row['drop_pct']),
            'success': bool(row.get('success', False)),
            'event_type': str(row.get('stock_event_type', 'unknown')),
        })

pdf = pd.DataFrame(profiles)
print(f"Collected {len(pdf):,} observations from {event_count} events")

# Average profile
avg = pdf.groupby('minute').agg(
    avg_price=('norm_price', 'mean'),
    median_price=('norm_price', 'median'),
    p25=('norm_price', lambda x: x.quantile(0.25)),
    p75=('norm_price', lambda x: x.quantile(0.75)),
    avg_vol=('volume', 'mean'),
    count=('norm_price', 'count'),
).reset_index()

# Winners vs losers
for label, val in [('winners', True), ('losers', False)]:
    sub = pdf[pdf['success'] == val].groupby('minute')['norm_price'].mean().reset_index()
    sub.columns = ['minute', f'avg_{label}']
    avg = avg.merge(sub, on='minute', how='left')

# By drop size
for label, lo, hi in [('small', 0.03, 0.05), ('medium', 0.05, 0.07), ('big', 0.07, 1.0)]:
    sub = pdf[(pdf['drop_pct'] >= lo) & (pdf['drop_pct'] < hi)].groupby('minute')['norm_price'].mean().reset_index()
    sub.columns = ['minute', f'avg_{label}']
    avg = avg.merge(sub, on='minute', how='left')

# Print
print(f"\n{'='*70}")
print(f"  V01: INTRADAY PROFILE OF ENTRY DAYS (T+2)")
print(f"{'='*70}")

key_mins = [0, 15, 30, 60, 90, 120, 180, 240, 300, 360, 389]
key_labels = ['9:30','9:45','10:00','10:30','11:00','11:30','12:30','13:30','14:30','15:30','15:59']

print(f"\n{'Time':>8} {'Avg%':>8} {'Med%':>8} {'P25%':>8} {'P75%':>8} {'Winners':>9} {'Losers':>9}")
print('-'*65)
for m, t in zip(key_mins, key_labels):
    r = avg[avg['minute'] == m]
    if len(r) == 0: continue
    r = r.iloc[0]
    print(f"{t:>8} {r['avg_price']*100:>+7.2f}% {r['median_price']*100:>+7.2f}% "
          f"{r['p25']*100:>+7.2f}% {r['p75']*100:>+7.2f}% "
          f"{r.get('avg_winners',0)*100:>+8.2f}% {r.get('avg_losers',0)*100:>+8.2f}%")

best = avg.loc[avg['avg_price'].idxmin()]
close_avg = avg[avg['minute'] >= 385]
close_val = close_avg.iloc[-1]['avg_price'] if len(close_avg) > 0 else avg.iloc[-1]['avg_price']
savings = (close_val - best['avg_price']) * 100

print(f"\nLowest avg price: minute {int(best['minute'])} ({best['avg_price']*100:+.3f}% from open)")
print(f"Close avg: {close_val*100:+.3f}% from open")
print(f"Potential savings (buy at low point vs close): {savings:.3f}%")

print(f"\nBy drop size:")
print(f"{'Time':>8} {'3-5%':>8} {'5-7%':>8} {'7%+':>8}")
print('-'*35)
for m, t in zip(key_mins, key_labels):
    r = avg[avg['minute'] == m]
    if len(r) == 0: continue
    r = r.iloc[0]
    print(f"{t:>8} {r.get('avg_small',0)*100:>+7.2f}% {r.get('avg_medium',0)*100:>+7.2f}% {r.get('avg_big',0)*100:>+7.2f}%")

# Save
results = {
    'events_analyzed': event_count,
    'observations': len(pdf),
    'best_minute': int(best['minute']),
    'best_minute_pct': round(float(best['avg_price']) * 100, 3),
    'close_pct': round(float(close_val) * 100, 3),
    'potential_savings_pct': round(float(savings), 3),
}
with open(os.path.join(OUT_DIR, 'v01_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
avg.to_csv(os.path.join(OUT_DIR, 'v01_avg_profile.csv'), index=False)

with open(os.path.join(OUT_DIR, 'v01_results.txt'), 'w') as f:
    f.write("V01: INTRADAY PROFILE OF ENTRY DAYS (T+2)\n\n")
    f.write(f"Events analyzed: {event_count}\n")
    f.write(f"Observations: {len(pdf):,}\n")
    f.write(f"Best minute: {results['best_minute']} ({results['best_minute_pct']:+.3f}%)\n")
    f.write(f"Close: {results['close_pct']:+.3f}%\n")
    f.write(f"Potential savings: {results['potential_savings_pct']:.3f}%\n\n")
    f.write("Full profile:\n")
    for _, r in avg.iterrows():
        m = int(r['minute'])
        h = (m + 9*60+30) // 60
        mn = (m + 9*60+30) % 60
        f.write(f"  {m:>4} ({h:>2}:{mn:02d}) avg={r['avg_price']*100:>+7.3f}% med={r['median_price']*100:>+7.3f}%\n")

print(f"\nSaved to {OUT_DIR}/v01_results.txt")
