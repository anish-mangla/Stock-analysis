"""
Build Forward Path Columns
============================
For each event, compute daily close/high/low returns relative to entry price
(D+1 close) for days D+2 through D+60. This is what the screener backtest needs.

Output: outputs/events_with_forward_path.parquet
"""

import pandas as pd
import numpy as np

print("Loading price matrices...")
close = pd.read_parquet('backtest_iterations/close_matrix.parquet')
high = pd.read_parquet('backtest_iterations/high_matrix.parquet')
low = pd.read_parquet('backtest_iterations/low_matrix.parquet')

print("Loading events...")
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])

# Load indicators and market data for the screener columns
indicators = pd.read_csv('backtest_iterations/intraday_deep/intraday_indicators_features.csv')
indicators['event_date'] = pd.to_datetime(indicators['event_date'])

market = pd.read_csv('outputs/market_daily_features.csv')
if 'Unnamed: 0' in market.columns:
    market = market.rename(columns={'Unnamed: 0': 'date'})
market['date'] = pd.to_datetime(market['date'])

# Load enriched labels for company_specific
import glob
v2_files = sorted(glob.glob('outputs/stock_level_labels_v2/batch_*_classified.csv'))
v2_dfs = []
for f in v2_files:
    try:
        v2_dfs.append(pd.read_csv(f, on_bad_lines='skip'))
    except:
        pass
v2 = pd.concat(v2_dfs, ignore_index=True) if v2_dfs else pd.DataFrame()
if len(v2) > 0:
    v2['event_date'] = pd.to_datetime(v2['event_date'])

# Merge
print("Merging datasets...")
df = events.merge(indicators, on=['ticker', 'event_date'], how='left', suffixes=('', '_ind'))
df = df.merge(market, left_on='event_date', right_on='date', how='left')
if len(v2) > 0:
    df = df.merge(v2[['ticker', 'event_date', 'company_specific_factor']].drop_duplicates(),
                  on=['ticker', 'event_date'], how='left')

# Build forward path
trading_dates = sorted(close.index)
date_to_idx = {d: i for i, d in enumerate(trading_dates)}

print("Computing forward path columns (D+2 to D+60 relative to D+1 close)...")
# Entry is at D+1 close. So entry_price = close on D+1.
# Forward returns are relative to that entry price.

max_days = 60
# Pre-allocate arrays
n = len(df)
close_ret = np.full((n, max_days + 1), np.nan)  # index 0 unused, 1=d1, 2=d2, ...60=d60
high_ret = np.full((n, max_days + 1), np.nan)
low_ret = np.full((n, max_days + 1), np.nan)

for i, row in df.iterrows():
    ticker = row['ticker']
    event_date = row['event_date']
    
    if ticker not in close.columns or event_date not in date_to_idx:
        continue
    
    idx = date_to_idx[event_date]
    
    # Entry is at D+1 close
    entry_idx = idx + 1
    if entry_idx >= len(trading_dates):
        continue
    
    entry_price = close.loc[trading_dates[entry_idx], ticker]
    if pd.isna(entry_price) or entry_price <= 0:
        continue
    
    # Compute returns for D+2 through D+60 relative to entry
    for day in range(2, max_days + 1):
        look_idx = idx + day  # D+0 is idx, so D+day is idx+day
        if look_idx >= len(trading_dates):
            break
        
        dt = trading_dates[look_idx]
        c = close.loc[dt, ticker]
        h = high.loc[dt, ticker]
        l = low.loc[dt, ticker]
        
        if not pd.isna(c):
            close_ret[i, day] = (c - entry_price) / entry_price
        if not pd.isna(h):
            high_ret[i, day] = (h - entry_price) / entry_price
        if not pd.isna(l):
            low_ret[i, day] = (l - entry_price) / entry_price
    
    if i % 1000 == 0:
        print(f"  {i}/{n} events processed...")

# Add columns to dataframe
print("Adding path columns to dataframe...")
for day in range(2, max_days + 1):
    df[f'd{day}_close_ret'] = close_ret[:, day]
    df[f'd{day}_high_ret'] = high_ret[:, day]
    df[f'd{day}_low_ret'] = low_ret[:, day]

# Save
output_path = 'outputs/events_with_forward_path.parquet'
print(f"Saving to {output_path}...")
df.to_parquet(output_path, index=False)
print(f"Done! {len(df)} events × {len(df.columns)} columns saved.")
print(f"Path columns: d2_close_ret through d60_close_ret, d2_high_ret through d60_high_ret, d2_low_ret through d60_low_ret")
