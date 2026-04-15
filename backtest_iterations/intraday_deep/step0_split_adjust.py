#!/usr/bin/env python3
"""
STEP 0: Apply split adjustments to intraday data.
All prices before a split date get divided by the split ratio.
This makes all prices comparable across time (same scale as most recent).
Saves adjusted parquet files alongside originals.
"""
import pandas as pd
import os

INTRADAY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'intraday_data')

# Split table: ticker -> list of (date, ratio)
# Ratio means: 1 old share becomes `ratio` new shares
# To adjust pre-split prices: divide by ratio
SPLITS = {
    'AAPL':  [('2020-08-31', 4.0)],
    'AMZN':  [('2022-06-06', 20.0)],
    'AVGO':  [('2024-07-15', 10.0)],
    'BKNG':  [('2025-06-17', 25.0)],  # announced, effective June 2025
    'DHR':   [('2023-10-02', 1.128)],  # Veralto spinoff adjustment
    'GE':    [('2021-08-02', 0.125), ('2023-01-04', 1.281), ('2024-04-02', 1.253)],
    'GOOG':  [('2022-07-18', 20.0)],
    'GOOGL': [('2022-07-18', 20.0)],
    'HON':   [('2025-10-30', 1.061)],  # spinoff adjustment
    'IBM':   [('2021-11-04', 1.046)],  # Kyndryl spinoff
    'ISRG':  [('2021-10-05', 3.0)],
    'LRCX':  [('2024-10-03', 10.0)],
    'MMM':   [('2024-04-01', 1.196)],  # Solventum spinoff
    'MRK':   [('2021-06-03', 1.048)],  # Organon spinoff
    'NEE':   [('2020-10-27', 4.0)],
    'NFLX':  [('2025-11-17', 10.0)],
    'NOW':   [('2025-12-18', 5.0)],
    'NVDA':  [('2021-07-20', 4.0), ('2024-06-10', 10.0)],
    'PFE':   [('2020-11-17', 1.054)],  # Upjohn/Viatris spinoff
    'RTX':   [('2020-04-03', 1.589)],  # Raytheon merger adjustment
    'T':     [('2022-04-11', 1.324)],  # Warner Bros spinoff
    'TSLA':  [('2020-08-31', 5.0), ('2022-08-25', 3.0)],
    'WMT':   [('2024-02-26', 3.0)],
}

# Compute cumulative adjustment factor for each ticker
# For a ticker with splits at dates D1 (ratio R1) and D2 (ratio R2) where D1 < D2:
# - Bars before D1: divide by R1 * R2
# - Bars between D1 and D2: divide by R2
# - Bars after D2: no adjustment
def get_adjustment_factor(ticker, timestamp):
    """Returns the factor to DIVIDE the raw price by to get adjusted price."""
    if ticker not in SPLITS:
        return 1.0
    factor = 1.0
    for split_date_str, ratio in SPLITS[ticker]:
        split_date = pd.Timestamp(split_date_str, tz='UTC')
        if timestamp < split_date:
            factor *= ratio
    return factor

print("="*60)
print("  STEP 0: Split-adjusting intraday data")
print("="*60)

for year in range(2020, 2026):
    raw_path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}.parquet')
    adj_path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    
    if os.path.exists(adj_path):
        print(f"  {year}: adjusted file already exists, skipping")
        continue
    
    if not os.path.exists(raw_path):
        print(f"  {year}: raw file missing, skipping")
        continue
    
    print(f"  {year}: loading...", end=" ", flush=True)
    df = pd.read_parquet(raw_path)
    print(f"{len(df):,} bars", end=" ", flush=True)
    
    # For each ticker that has splits, compute and apply adjustment
    price_cols = ['open', 'high', 'low', 'close', 'vwap']
    adjusted_count = 0
    
    for ticker, split_list in SPLITS.items():
        mask = df['symbol'] == ticker
        if not mask.any():
            continue
        
        ticker_rows = df.loc[mask].copy()
        
        for split_date_str, ratio in split_list:
            split_ts = pd.Timestamp(split_date_str, tz='UTC')
            pre_split = mask & (df['timestamp'] < split_ts)
            n_adjusted = pre_split.sum()
            if n_adjusted > 0:
                for col in price_cols:
                    df.loc[pre_split, col] = df.loc[pre_split, col] / ratio
                # Volume goes the other way (more shares after split)
                df.loc[pre_split, 'volume'] = df.loc[pre_split, 'volume'] * ratio
                adjusted_count += n_adjusted
    
    print(f"| adjusted {adjusted_count:,} bars", end=" ", flush=True)
    
    df.to_parquet(adj_path)
    sz = os.path.getsize(adj_path) / 1e6
    print(f"| saved {sz:.0f} MB")

# Verify: check a few known splits
print(f"\n{'='*60}")
print(f"  VERIFICATION")
print(f"{'='*60}")

import yfinance as yf

for year, ticker, date_str, expected_close in [
    (2020, 'AAPL', '2020-08-28', None),   # day before AAPL 4:1 split
    (2020, 'TSLA', '2020-08-28', None),   # day before TSLA 5:1 split
    (2024, 'NVDA', '2024-06-07', None),   # day before NVDA 10:1 split
    (2024, 'WMT', '2024-02-23', None),    # day before WMT 3:1 split
]:
    adj_path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if not os.path.exists(adj_path):
        continue
    df = pd.read_parquet(adj_path)
    
    dt = pd.Timestamp(date_str, tz='UTC')
    rows = df[(df['symbol'] == ticker) & (df['timestamp'].dt.date == dt.date())]
    if len(rows) == 0:
        print(f"  {ticker} {date_str}: no data")
        continue
    
    adj_close = rows.iloc[-1]['close']
    
    # Compare to yfinance adjusted
    yf_t = ticker.replace('BRK.B', 'BRK-B')
    yf_data = yf.download(yf_t, start=date_str, end=pd.Timestamp(date_str) + pd.Timedelta(days=3), progress=False)
    if len(yf_data) > 0:
        yf_close = float(yf_data.iloc[0]['Close'])
        diff = abs(adj_close - yf_close) / yf_close * 100
        status = "OK" if diff < 2 else "MISMATCH"
        print(f"  {ticker} {date_str}: adjusted={adj_close:.2f} yfinance={yf_close:.2f} diff={diff:.1f}% [{status}]")
