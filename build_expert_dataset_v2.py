"""
Build raw OHLCV dataset for thinking model.
Include actual prices for D+0 through D+15, not just returns.
Let the model figure out the strategy from raw data.
"""
import pandas as pd, numpy as np

# Load price matrices
close_mat = pd.read_parquet('backtest_iterations/close_matrix.parquet')
high_mat = pd.read_parquet('backtest_iterations/high_matrix.parquet')
low_mat = pd.read_parquet('backtest_iterations/low_matrix.parquet')

# Load events
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

# Load speed features for ATR and intraday stuff
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])

# Load labels
master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])

# SPY benchmark
spy_df = pd.read_parquet('backtest_iterations/spy_benchmark.parquet')
spy_df.index = pd.to_datetime(spy_df.index)
spy_close = spy_df.iloc[:, 0]

# Build trading day list
trading_days = sorted(close_mat.index)
td_idx = {d: i for i, d in enumerate(trading_days)}

print("Building OHLCV paths for each event...")
rows = []
for idx, row in fp.iterrows():
    ticker = row['ticker']
    event_date = row['event_date']
    
    if ticker not in close_mat.columns:
        continue
    if event_date not in td_idx:
        continue
    
    base_idx = td_idx[event_date]
    
    r = {
        'ticker': ticker,
        'event_date': event_date,
        'drop_pct': row['drop_pct'],
        'entry_price': row['entry_price'],
    }
    
    # D+0 through D+15 OHLC (relative to event date)
    for offset in range(0, 16):
        day_idx = base_idx + offset
        if day_idx >= len(trading_days):
            break
        date = trading_days[day_idx]
        
        c = close_mat.loc[date, ticker] if date in close_mat.index else np.nan
        h = high_mat.loc[date, ticker] if date in high_mat.index else np.nan
        l = low_mat.loc[date, ticker] if date in low_mat.index else np.nan
        
        # Also get SPY close for this day
        spy_c = spy_close.loc[date] if date in spy_close.index else np.nan
        
        r[f'd{offset}_close'] = c
        r[f'd{offset}_high'] = h
        r[f'd{offset}_low'] = l
        r[f'd{offset}_spy_close'] = spy_c
    
    rows.append(r)
    if idx % 1000 == 0:
        print(f"  {idx}/{len(fp)}...")

ohlcv = pd.DataFrame(rows)
print(f"OHLCV base: {len(ohlcv)} events")

# Merge features from speed_features
feat_cols = ['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct',
             'num_droppers_5pct', 'd0_rebound_low_to_close', 'd0_minute_of_low',
             'd1_pct_bars_above_vwap', 'd1_longest_streak_above_vwap',
             'd1_last90_logprice_slope', 'd1_full_day_trend_r2', 'd1_return_vs_sector']
sf_sub = sf[[c for c in feat_cols if c in sf.columns]].copy()
sf_sub['event_date'] = pd.to_datetime(sf_sub['event_date'])
ohlcv = ohlcv.merge(sf_sub, on=['ticker', 'event_date'], how='left')

# Merge labels
label_cols = ['ticker', 'event_date', 'stock_event_type', 'stock_event_severity',
              'company_specific_factor', 'sector_etf']
lab = master[[c for c in label_cols if c in master.columns]].drop_duplicates()
ohlcv = ohlcv.merge(lab, on=['ticker', 'event_date'], how='left')

# Merge some forward path outcome columns
outcome_cols = ['ticker', 'event_date', 'success', 'days_to_hit', 'final_return',
                'max_drawdown', 'd1_return', 'd1_close_vs_vwap', 'hyg_5d_return', 'd2_first_hour_ret']
fp_sub = fp[[c for c in outcome_cols if c in fp.columns]].copy()
ohlcv = ohlcv.merge(fp_sub, on=['ticker', 'event_date'], how='left')

# Crisis flag
clusters = [('2020-02-13', '2020-03-18'), ('2022-04-20', '2022-05-18'),
    ('2022-09-13', '2022-10-07'), ('2025-02-18', '2025-03-13'),
    ('2022-06-01', '2022-06-17'), ('2022-03-29', '2022-04-13'),
    ('2022-01-05', '2022-01-21'), ('2025-03-21', '2025-04-08')]
crisis = pd.Series(False, index=ohlcv.index)
for s, e in clusters:
    crisis |= (ohlcv['event_date'] >= s) & (ohlcv['event_date'] <= e)
ohlcv['in_crisis_window'] = crisis

# Compute volume proxy: ATR as pct
ohlcv['atr_pct'] = 0.05 / ohlcv['target_distance_over_atr20']

# SPY trailing returns
spy_daily = spy_close.pct_change()
spy_5d = spy_close.pct_change(5)
spy_20d = spy_close.pct_change(20)
ohlcv['spy_d0_return'] = ohlcv['event_date'].map(spy_daily.to_dict())
ohlcv['spy_5d_return'] = ohlcv['event_date'].map(spy_5d.to_dict())
ohlcv['spy_20d_return'] = ohlcv['event_date'].map(spy_20d.to_dict())

# 20DMA distance
ma20 = close_mat.rolling(20).mean()
dist_20dma = (close_mat - ma20) / ma20
dma_dict = {}
for col in dist_20dma.columns:
    for date in dist_20dma.index:
        val = dist_20dma.loc[date, col]
        if not pd.isna(val):
            dma_dict[(col, date)] = val
ohlcv['dist_from_20dma'] = [dma_dict.get((r['ticker'], r['event_date']), np.nan) for _, r in ohlcv.iterrows()]

# Stock excess vs SPY on drop day
ohlcv['stock_excess_vs_spy'] = ohlcv['d1_return'] - ohlcv['spy_d0_return'] if 'd1_return' in ohlcv.columns else np.nan

print(f"\nFinal dataset: {len(ohlcv)} events x {len(ohlcv.columns)} columns")
print(f"Crisis: {ohlcv['in_crisis_window'].sum()}, Clean: {(~ohlcv['in_crisis_window']).sum()}")

ohlcv.to_parquet('outputs/expert_ohlcv_dataset.parquet', index=False)
print(f"Saved to outputs/expert_ohlcv_dataset.parquet")

# Print column list
print(f"\nColumns ({len(ohlcv.columns)}):")
for c in sorted(ohlcv.columns):
    print(f"  {c}: {ohlcv[c].dtype}, {ohlcv[c].notna().sum()} non-null")
