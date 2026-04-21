#!/usr/bin/env python3
"""
Recovery curve analysis: for every 3%+ drop, track what happens
day by day for the next 5 trading days. Slice by drop size, year,
month, winners vs losers, and look for patterns.
"""
import pandas as pd
import numpy as np
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data

close, high, low, spy, events = load_cached_data()
trading_dates = pd.DatetimeIndex(sorted(close.index))

print("Computing day-by-day recovery for all events...")

rows = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    entry_price = row['entry_price']  # close on drop day
    if pd.isna(entry_price) or entry_price <= 0:
        continue
    if ticker not in close.columns:
        continue
    
    # Find the next 5 trading days after the drop
    future = trading_dates[trading_dates > dt]
    if len(future) < 5:
        continue
    
    day_returns = {}
    for d in range(1, 6):
        fdate = future[d - 1]
        if fdate in close.index:
            c = close.at[fdate, ticker]
            h = high.at[fdate, ticker] if ticker in high.columns else None
            l = low.at[fdate, ticker] if ticker in low.columns else None
            if pd.notna(c):
                day_returns[f'close_d{d}'] = (float(c) / entry_price - 1) * 100
            if h is not None and pd.notna(h):
                day_returns[f'high_d{d}'] = (float(h) / entry_price - 1) * 100
            if l is not None and pd.notna(l):
                day_returns[f'low_d{d}'] = (float(l) / entry_price - 1) * 100
    
    if not day_returns:
        continue
    
    rows.append({
        'ticker': ticker,
        'event_date': dt,
        'drop_pct': row['drop_pct'],
        'drop_bucket': row['drop_bucket'],
        'success': row.get('success', False),
        'year': dt.year,
        'month': dt.month,
        'day_of_week': dt.dayofweek,
        'stress': row.get('liquidity_credit_stress_severity', 'none'),
        'sector_etf': row.get('sector_etf', ''),
        **day_returns,
    })

df = pd.DataFrame(rows)
print(f"Events with recovery data: {len(df)}")

# ============================================================
# ANALYSIS
# ============================================================

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"\n{'='*80}")
print(f"  RECOVERY CURVE ANALYSIS: What happens after a drop?")
print(f"{'='*80}")

# 1. Overall average recovery by day
print(f"\n--- OVERALL (all {len(df)} events) ---")
print(f"{'':>20} {'D+1':>8} {'D+2':>8} {'D+3':>8} {'D+4':>8} {'D+5':>8}")
print('-'*60)
for metric, label in [('close', 'Avg close return'), ('high', 'Avg high return'), ('low', 'Avg low return')]:
    vals = []
    for d in range(1, 6):
        col = f'{metric}_d{d}'
        vals.append(df[col].mean() if col in df.columns else 0)
    print(f"{label:>20} {vals[0]:>+7.2f}% {vals[1]:>+7.2f}% {vals[2]:>+7.2f}% {vals[3]:>+7.2f}% {vals[4]:>+7.2f}%")

# % positive by day
print(f"\n{'':>20} {'D+1':>8} {'D+2':>8} {'D+3':>8} {'D+4':>8} {'D+5':>8}")
print('-'*60)
vals = []
for d in range(1, 6):
    col = f'close_d{d}'
    pct_pos = (df[col] > 0).mean() * 100 if col in df.columns else 0
    vals.append(pct_pos)
print(f"{'% positive close':>20} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}% {vals[3]:>7.1f}% {vals[4]:>7.1f}%")

# 2. By drop bucket
print(f"\n--- BY DROP SIZE ---")
for bucket in ['3-5%', '5-7%', '7-10%', '10%+']:
    sub = df[df['drop_bucket'] == bucket]
    if len(sub) < 20: continue
    print(f"\n  {bucket} drops (n={len(sub)}):")
    print(f"  {'':>18} {'D+1':>8} {'D+2':>8} {'D+3':>8} {'D+4':>8} {'D+5':>8}")
    vals = [sub[f'close_d{d}'].mean() for d in range(1, 6)]
    print(f"  {'Avg close':>18} {vals[0]:>+7.2f}% {vals[1]:>+7.2f}% {vals[2]:>+7.2f}% {vals[3]:>+7.2f}% {vals[4]:>+7.2f}%")
    vals = [(sub[f'close_d{d}'] > 0).mean() * 100 for d in range(1, 6)]
    print(f"  {'% positive':>18} {vals[0]:>7.1f}% {vals[1]:>7.1f}% {vals[2]:>7.1f}% {vals[3]:>7.1f}% {vals[4]:>7.1f}%")

# 3. By year
print(f"\n--- BY YEAR ---")
for year in sorted(df['year'].unique()):
    sub = df[df['year'] == year]
    if len(sub) < 20: continue
    vals = [sub[f'close_d{d}'].mean() for d in range(1, 6)]
    pct_pos_d1 = (sub['close_d1'] > 0).mean() * 100
    print(f"  {year} (n={len(sub):>4}): D+1={vals[0]:>+5.2f}% D+2={vals[1]:>+5.2f}% D+3={vals[2]:>+5.2f}% D+4={vals[3]:>+5.2f}% D+5={vals[4]:>+5.2f}% | D+1 pos: {pct_pos_d1:.0f}%")

# 4. By month
print(f"\n--- BY MONTH ---")
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
for month in range(1, 13):
    sub = df[df['month'] == month]
    if len(sub) < 20: continue
    vals = [sub[f'close_d{d}'].mean() for d in range(1, 6)]
    wr = sub['success'].mean() * 100
    print(f"  {month_names[month]} (n={len(sub):>4}): D+1={vals[0]:>+5.2f}% D+2={vals[1]:>+5.2f}% D+3={vals[2]:>+5.2f}% D+4={vals[3]:>+5.2f}% D+5={vals[4]:>+5.2f}% | WR: {wr:.0f}%")

# 5. By day of week (drop day)
print(f"\n--- BY DAY OF WEEK (drop day) ---")
dow_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
for dow in range(5):
    sub = df[df['day_of_week'] == dow]
    if len(sub) < 50: continue
    vals = [sub[f'close_d{d}'].mean() for d in range(1, 6)]
    wr = sub['success'].mean() * 100
    print(f"  {dow_names[dow]} (n={len(sub):>4}): D+1={vals[0]:>+5.2f}% D+2={vals[1]:>+5.2f}% D+3={vals[2]:>+5.2f}% D+4={vals[3]:>+5.2f}% D+5={vals[4]:>+5.2f}% | WR: {wr:.0f}%")

# 6. Winners vs losers — what's different in the first 2 days?
print(f"\n--- WINNERS vs LOSERS (first 5 days) ---")
for label, mask in [('Winners', df['success'] == True), ('Losers', df['success'] == False)]:
    sub = df[mask]
    vals = [sub[f'close_d{d}'].mean() for d in range(1, 6)]
    pct_pos = [(sub[f'close_d{d}'] > 0).mean() * 100 for d in range(1, 6)]
    print(f"  {label} (n={len(sub)}):")
    print(f"    Avg close:  D+1={vals[0]:>+5.2f}% D+2={vals[1]:>+5.2f}% D+3={vals[2]:>+5.2f}% D+4={vals[3]:>+5.2f}% D+5={vals[4]:>+5.2f}%")
    print(f"    % positive: D+1={pct_pos[0]:>5.1f}% D+2={pct_pos[1]:>5.1f}% D+3={pct_pos[2]:>5.1f}% D+4={pct_pos[3]:>5.1f}% D+5={pct_pos[4]:>5.1f}%")

# 7. By market stress
print(f"\n--- BY MARKET STRESS ---")
for stress in ['none', 'low', 'medium', 'high']:
    sub = df[df['stress'] == stress]
    if len(sub) < 30: continue
    vals = [sub[f'close_d{d}'].mean() for d in range(1, 6)]
    wr = sub['success'].mean() * 100
    print(f"  {stress:>6} (n={len(sub):>4}): D+1={vals[0]:>+5.2f}% D+2={vals[1]:>+5.2f}% D+3={vals[2]:>+5.2f}% D+4={vals[3]:>+5.2f}% D+5={vals[4]:>+5.2f}% | WR: {wr:.0f}%")

# 8. Can D+1 return predict success?
print(f"\n--- D+1 RETURN AS PREDICTOR ---")
for lo, hi, label in [(-100, -3, 'D+1 < -3%'), (-3, -1, 'D+1 -3% to -1%'), (-1, 0, 'D+1 -1% to 0%'),
                        (0, 1, 'D+1 0% to +1%'), (1, 3, 'D+1 +1% to +3%'), (3, 100, 'D+1 > +3%')]:
    sub = df[(df['close_d1'] >= lo) & (df['close_d1'] < hi)]
    if len(sub) < 20: continue
    wr = sub['success'].mean() * 100
    avg_d2_5 = sub[[f'close_d{d}' for d in range(2, 6)]].mean(axis=1).mean()
    print(f"  {label:>18} (n={len(sub):>4}): WR={wr:.1f}% | avg D+2 to D+5: {avg_d2_5:>+5.2f}%")

# Save
with open(os.path.join(OUT_DIR, 'recovery_curve_results.txt'), 'w') as f:
    f.write("RECOVERY CURVE ANALYSIS\n")
    f.write(f"Events: {len(df)}\n\n")
    f.write("Overall avg close return from drop-day close:\n")
    for d in range(1, 6):
        f.write(f"  D+{d}: {df[f'close_d{d}'].mean():+.3f}%\n")

df.to_csv(os.path.join(OUT_DIR, 'recovery_curve_data.csv'), index=False)
print(f"\nRaw data saved to {OUT_DIR}/recovery_curve_data.csv")
print(f"Summary saved to {OUT_DIR}/recovery_curve_results.txt")
