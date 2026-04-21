"""
Target Distribution Analysis
=============================
For each event, compute the max return achievable within 5 trading days
(using intraday highs), then bucket it to see the distribution.
Do this for D+0 entry and D+1 entry.
"""

import pandas as pd
import numpy as np

close = pd.read_parquet('backtest_iterations/close_matrix.parquet')
high = pd.read_parquet('backtest_iterations/high_matrix.parquet')
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])

trading_dates = sorted(close.index)
date_to_idx = {d: i for i, d in enumerate(trading_dates)}

rows = []

for _, row in events.iterrows():
    ticker = row['ticker']
    event_date = row['event_date']
    if ticker not in close.columns or event_date not in date_to_idx:
        continue
    idx = date_to_idx[event_date]

    for delay in [0, 1]:
        entry_idx = idx + delay
        if entry_idx >= len(trading_dates):
            continue
        entry_price = close.loc[trading_dates[entry_idx], ticker]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        max_ret = 0.0
        for d in range(1, 6):
            look_idx = entry_idx + d
            if look_idx >= len(trading_dates):
                break
            h = high.loc[trading_dates[look_idx], ticker]
            if not pd.isna(h):
                ret = (h - entry_price) / entry_price
                if ret > max_ret:
                    max_ret = ret

        rows.append({
            'ticker': ticker,
            'drop_pct': row['drop_pct'],
            'delay': delay,
            'max_5d_ret': max_ret,
        })

df = pd.DataFrame(rows)

# Bucket the max return
bins = [-999, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 999]
labels = ['<0%', '0-0.5%', '0.5-1%', '1-2%', '2-3%', '3-5%', '5-10%', '10%+']
df['bucket'] = pd.cut(df['max_5d_ret'], bins=bins, labels=labels)

def print_table(sub, title):
    print(f"\n{'=' * 75}")
    print(title)
    print('=' * 75)
    
    for delay in [0, 1]:
        s = sub[sub['delay'] == delay]
        n = len(s)
        entry = f"D+{delay} entry"
        counts = s['bucket'].value_counts().reindex(labels, fill_value=0)
        
        print(f"\n  {entry} (n={n:,})")
        print(f"  {'Bucket':>10}  {'Count':>7}  {'%':>7}  {'Cumul%':>7}  Bar")
        print(f"  {'-'*60}")
        
        cumul = 0
        for b in labels:
            c = counts[b]
            pct = c / n * 100
            cumul += pct
            bar = '█' * int(pct / 2)
            print(f"  {b:>10}  {c:>7,}  {pct:>6.1f}%  {cumul:>6.1f}%  {bar}")
        
        # Summary stats
        print(f"\n  Mean max return: {s['max_5d_ret'].mean()*100:.2f}%")
        print(f"  Median max return: {s['max_5d_ret'].median()*100:.2f}%")

# All events
print_table(df, 'ALL EVENTS — Max return within 5 days (intraday highs)')

# By drop size
for bucket_name, lo, hi in [('3-5% drops', 0.03, 0.05), ('5-7% drops', 0.05, 0.07), ('7%+ drops', 0.07, 1.0)]:
    sub = df[(df['drop_pct'] >= lo) & (df['drop_pct'] < hi)]
    print_table(sub, f'{bucket_name.upper()} — Max return within 5 days')

print()
