#!/usr/bin/env python3
"""Cache price data so all backtest iterations can reuse it."""
import pandas as pd
import yfinance as yf
import os

events = pd.read_csv('outputs/events.csv')
events['event_date'] = pd.to_datetime(events['event_date'])
tickers = sorted(events['ticker'].unique().tolist())
start = (events['event_date'].min() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
end = (events['event_date'].max() + pd.Timedelta(days=90)).strftime('%Y-%m-%d')

print(f'Downloading {len(tickers)} tickers from {start} to {end}...')
raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=True, group_by='ticker', threads=True)

close_data, high_data, low_data = {}, {}, {}
if isinstance(raw.columns, pd.MultiIndex):
    for t in tickers:
        if t not in raw.columns.get_level_values(0): continue
        tdf = raw[t].copy()
        for col, store in [('Close', close_data), ('High', high_data), ('Low', low_data)]:
            if col in tdf.columns:
                s = tdf[col].dropna().astype(float)
                s.index = pd.to_datetime(s.index)
                store[t] = s

idx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in close_data.values()])))
for name, data in [('close', close_data), ('high', high_data), ('low', low_data)]:
    df = pd.DataFrame({t: s for t, s in data.items()}, index=idx).sort_index().ffill()
    path = f'backtest_iterations/{name}_matrix.parquet'
    df.to_parquet(path)
    print(f'Saved {path}: {df.shape}')

# Also save SPY for benchmark
spy = yf.download('SPY', start=start, end=end, auto_adjust=False, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = [c[0] for c in spy.columns]
spy['Close'].to_frame('SPY').to_parquet('backtest_iterations/spy_benchmark.parquet')
print('Saved SPY benchmark')
print('Done.')
