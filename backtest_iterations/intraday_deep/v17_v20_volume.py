#!/usr/bin/env python3
"""
V17-V20: Volume and VWAP signal analysis.
V17: Volume surge on drop day
V18: VWAP position as entry filter  
V19: Relative volume as confidence signal
V20: Closing auction volume analysis
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
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'close', 'volume', 'vwap']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute

# Daily volume per (symbol, date)
print("Computing daily volume stats...")
daily_vol = bars.groupby(['symbol', 'date']).agg(
    total_vol=('volume', 'sum'),
    close=('close', 'last'),
    vwap=('vwap', 'mean'),
    bar_count=('close', 'count'),
).reset_index()

# 20-day rolling average volume
daily_vol = daily_vol.sort_values(['symbol', 'date'])
daily_vol['avg_vol_20d'] = daily_vol.groupby('symbol')['total_vol'].transform(
    lambda x: x.rolling(20, min_periods=10).mean().shift(1)
)
daily_vol['rel_volume'] = daily_vol['total_vol'] / daily_vol['avg_vol_20d']

# Closing auction: last 10 minutes (15:50-16:00 ET = 19:50-20:00 UTC = 1190-1200)
closing_bars = bars[(bars['utc_min'] >= 1190) & (bars['utc_min'] <= 1200)]
closing_vol = closing_bars.groupby(['symbol', 'date'])['volume'].sum().reset_index()
closing_vol.columns = ['symbol', 'date', 'closing_vol']
daily_vol = daily_vol.merge(closing_vol, on=['symbol', 'date'], how='left')
daily_vol['closing_vol_pct'] = daily_vol['closing_vol'] / daily_vol['total_vol'] * 100

del bars, closing_bars
print("Done")

# Match events to volume data
print("Matching events to volume data...")
event_vol = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    
    # Drop day volume
    drop_day = daily_vol[(daily_vol['symbol'] == alpaca_t) & (daily_vol['date'] == dt.date())]
    if len(drop_day) == 0: continue
    
    # Entry day volume (T+2)
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    entry_day = future[1].date()
    entry_vol = daily_vol[(daily_vol['symbol'] == alpaca_t) & (daily_vol['date'] == entry_day)]
    
    event_vol.append({
        'ticker': ticker, 'event_date': dt,
        'success': bool(row.get('success', False)),
        'drop_pct': float(row['drop_pct']),
        'drop_day_rel_vol': float(drop_day.iloc[0]['rel_volume']) if pd.notna(drop_day.iloc[0]['rel_volume']) else None,
        'drop_day_closing_vol_pct': float(drop_day.iloc[0]['closing_vol_pct']) if pd.notna(drop_day.iloc[0].get('closing_vol_pct')) else None,
        'entry_day_rel_vol': float(entry_vol.iloc[0]['rel_volume']) if len(entry_vol) > 0 and pd.notna(entry_vol.iloc[0]['rel_volume']) else None,
    })

evdf = pd.DataFrame(event_vol)
evdf = evdf.dropna(subset=['drop_day_rel_vol'])
print(f"Events with volume data: {len(evdf)}")

# ============================================================
# V17: Volume surge on drop day
# ============================================================
print(f"\n{'='*70}")
print(f"  V17: VOLUME SURGE ON DROP DAY")
print(f"{'='*70}")

print(f"\n  Relative volume on drop day (vs 20-day avg):")
print(f"    Mean: {evdf['drop_day_rel_vol'].mean():.2f}x")
print(f"    Median: {evdf['drop_day_rel_vol'].median():.2f}x")

print(f"\n  Win rate by drop-day relative volume:")
for lo, hi, label in [(0, 1, '<1x (low vol)'), (1, 1.5, '1-1.5x'), (1.5, 2, '1.5-2x'), 
                       (2, 3, '2-3x (high)'), (3, 100, '3x+ (extreme)')]:
    sub = evdf[(evdf['drop_day_rel_vol'] >= lo) & (evdf['drop_day_rel_vol'] < hi)]
    if len(sub) > 10:
        wr = sub['success'].mean() * 100
        print(f"    {label:>20}: {len(sub):>5} events | WR {wr:.1f}%")

# ============================================================
# V19: Relative volume as confidence signal
# ============================================================
print(f"\n{'='*70}")
print(f"  V19: ENTRY DAY RELATIVE VOLUME")
print(f"{'='*70}")

evdf2 = evdf.dropna(subset=['entry_day_rel_vol'])
print(f"\n  Win rate by entry-day relative volume:")
for lo, hi, label in [(0, 0.8, '<0.8x (quiet)'), (0.8, 1.2, '0.8-1.2x (normal)'), 
                       (1.2, 2, '1.2-2x (active)'), (2, 100, '2x+ (surge)')]:
    sub = evdf2[(evdf2['entry_day_rel_vol'] >= lo) & (evdf2['entry_day_rel_vol'] < hi)]
    if len(sub) > 10:
        wr = sub['success'].mean() * 100
        print(f"    {label:>20}: {len(sub):>5} events | WR {wr:.1f}%")

# ============================================================
# V20: Closing auction volume
# ============================================================
print(f"\n{'='*70}")
print(f"  V20: CLOSING AUCTION VOLUME ON DROP DAY")
print(f"{'='*70}")

evdf3 = evdf.dropna(subset=['drop_day_closing_vol_pct'])
print(f"\n  Closing volume as % of daily volume:")
print(f"    Mean: {evdf3['drop_day_closing_vol_pct'].mean():.1f}%")
print(f"    Median: {evdf3['drop_day_closing_vol_pct'].median():.1f}%")

print(f"\n  Win rate by closing auction volume %:")
for lo, hi, label in [(0, 5, '<5%'), (5, 10, '5-10%'), (10, 20, '10-20%'), (20, 100, '20%+')]:
    sub = evdf3[(evdf3['drop_day_closing_vol_pct'] >= lo) & (evdf3['drop_day_closing_vol_pct'] < hi)]
    if len(sub) > 10:
        wr = sub['success'].mean() * 100
        print(f"    {label:>10}: {len(sub):>5} events | WR {wr:.1f}%")

# Save
results = {
    'v17_drop_day_rel_vol_mean': round(evdf['drop_day_rel_vol'].mean(), 2),
    'v19_entry_day_events': len(evdf2),
    'v20_closing_vol_mean': round(evdf3['drop_day_closing_vol_pct'].mean(), 1) if len(evdf3) > 0 else None,
}
with open(os.path.join(OUT_DIR, 'v17_v20_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, 'v17_v20_results.txt'), 'w') as f:
    f.write("V17-V20: VOLUME AND VWAP ANALYSIS\n\n")
    f.write("V17: Drop day relative volume\n")
    f.write(f"  Mean: {evdf['drop_day_rel_vol'].mean():.2f}x\n\n")
    f.write("V19: Entry day relative volume\n")
    f.write(f"  Events: {len(evdf2)}\n\n")
    f.write("V20: Closing auction volume\n")
    f.write(f"  Mean: {evdf3['drop_day_closing_vol_pct'].mean():.1f}%\n" if len(evdf3) > 0 else "  No data\n")

print(f"\nSaved to {OUT_DIR}/v17_v20_results.txt")
