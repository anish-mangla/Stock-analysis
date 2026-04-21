"""
Build a rich dataset for the thinking model to analyze.
Include: all tradable events (clean set, no D+1 veto) with full price paths,
features, and context.
"""
import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct',
                   'num_droppers_5pct', 'd0_rebound_low_to_close', 'd0_minute_of_low',
                   'd1_pct_bars_above_vwap', 'd1_longest_streak_above_vwap',
                   'd1_last90_logprice_slope', 'd1_full_day_trend_r2', 'd1_return_vs_sector']],
              on=['ticker', 'event_date'], how='left')

# Labels
master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])
label_map = dict(zip(zip(master['ticker'], master['event_date']),
                      zip(master['stock_event_type'], master['stock_event_severity'],
                          master.get('company_specific_factor', pd.Series()))))
fp['event_type'] = [label_map.get((r['ticker'], r['event_date']), ('unlabeled', 'unlabeled', np.nan))[0] for _, r in fp.iterrows()]
fp['event_severity'] = [label_map.get((r['ticker'], r['event_date']), ('unlabeled', 'unlabeled', np.nan))[1] for _, r in fp.iterrows()]
fp['company_specific'] = [label_map.get((r['ticker'], r['event_date']), ('unlabeled', 'unlabeled', np.nan))[2] for _, r in fp.iterrows()]

# SPY
spy_df = pd.read_parquet('backtest_iterations/spy_benchmark.parquet')
spy_df.index = pd.to_datetime(spy_df.index)
spy_daily = spy_df.iloc[:, 0].pct_change()
spy_5d = spy_df.iloc[:, 0].pct_change(5)
spy_20d = spy_df.iloc[:, 0].pct_change(20)
fp['spy_d0_return'] = fp['event_date'].map(spy_daily.to_dict())
fp['spy_5d_return'] = fp['event_date'].map(spy_5d.to_dict())
fp['spy_20d_return'] = fp['event_date'].map(spy_20d.to_dict())
fp['stock_excess_vs_spy'] = fp['daily_return'] - fp['spy_d0_return']

# 20DMA distance
close_mat = pd.read_parquet('backtest_iterations/close_matrix.parquet')
ma20 = close_mat.rolling(20).mean()
dist_20dma = (close_mat - ma20) / ma20
dma_dict = {}
for col in dist_20dma.columns:
    for date in dist_20dma.index:
        val = dist_20dma.loc[date, col]
        if not pd.isna(val):
            dma_dict[(col, date)] = val
fp['dist_from_20dma'] = [dma_dict.get((r['ticker'], r['event_date']), np.nan) for _, r in fp.iterrows()]

# ATR-based target
fp['atr_pct'] = 0.05 / fp['target_distance_over_atr20']

# Crisis window flag
clusters = [('2020-02-13', '2020-03-18'), ('2022-04-20', '2022-05-18'),
    ('2022-09-13', '2022-10-07'), ('2025-02-18', '2025-03-13'),
    ('2022-06-01', '2022-06-17'), ('2022-03-29', '2022-04-13'),
    ('2022-01-05', '2022-01-21'), ('2025-03-21', '2025-04-08')]
crisis = pd.Series(False, index=fp.index)
for s, e in clusters: crisis |= (fp['event_date'] >= s) & (fp['event_date'] <= e)
fp['in_crisis_window'] = crisis

# Compute some outcome labels
# Max high within N days
for w in [3, 5, 7, 10]:
    cols = [f'd{d}_high_ret' for d in range(2, w+2) if f'd{d}_high_ret' in fp.columns]
    fp[f'max_high_{w}d'] = fp[cols].max(axis=1)
for w in [3, 5, 7, 10]:
    cols = [f'd{d}_low_ret' for d in range(2, w+2) if f'd{d}_low_ret' in fp.columns]
    fp[f'max_low_{w}d'] = fp[cols].min(axis=1)
for w in [3, 5, 7, 10]:
    cols = [f'd{d}_close_ret' for d in range(2, w+2) if f'd{d}_close_ret' in fp.columns]
    fp[f'final_close_{w}d'] = fp[cols].iloc[:, -1] if cols else np.nan

# Day the max high was reached
for w in [5, 10]:
    high_cols = [f'd{d}_high_ret' for d in range(2, w+2) if f'd{d}_high_ret' in fp.columns]
    fp[f'day_of_max_high_{w}d'] = fp[high_cols].idxmax(axis=1).str.extract(r'd(\d+)').astype(float)

# Select columns to export
keep_cols = [
    # Identity
    'ticker', 'event_date', 'sector_etf',
    # Drop characteristics
    'drop_pct', 'daily_return', 'prev_close', 'entry_price',
    # Market context
    'spy_d0_return', 'spy_5d_return', 'spy_20d_return', 'stock_excess_vs_spy',
    'num_droppers_3pct', 'num_droppers_5pct',
    # Stock context
    'target_distance_over_atr20', 'atr_pct', 'dist_from_20dma',
    'd1_return', 'd1_close_vs_vwap', 'hyg_5d_return', 'd2_first_hour_ret',
    # Intraday features
    'd0_rebound_low_to_close', 'd0_minute_of_low',
    'd1_pct_bars_above_vwap', 'd1_longest_streak_above_vwap',
    'd1_last90_logprice_slope', 'd1_full_day_trend_r2', 'd1_return_vs_sector',
    # Labels
    'event_type', 'event_severity', 'company_specific',
    # Crisis flag
    'in_crisis_window',
    # Outcomes
    'success', 'days_to_hit', 'final_return', 'max_drawdown',
    'max_high_3d', 'max_high_5d', 'max_high_7d', 'max_high_10d',
    'max_low_3d', 'max_low_5d', 'max_low_7d', 'max_low_10d',
    'day_of_max_high_5d', 'day_of_max_high_10d',
]

# Add all daily price path columns
for day in range(2, 21):
    for suffix in ['close_ret', 'high_ret', 'low_ret']:
        col = f'd{day}_{suffix}'
        if col in fp.columns:
            keep_cols.append(col)

existing = [c for c in keep_cols if c in fp.columns]
export = fp[existing].copy()

print(f"Exporting {len(export)} events x {len(existing)} columns")
print(f"  Crisis window events: {export['in_crisis_window'].sum()}")
print(f"  Clean events: {(~export['in_crisis_window']).sum()}")

export.to_parquet('outputs/expert_analysis_dataset.parquet', index=False)
print(f"\nSaved to outputs/expert_analysis_dataset.parquet")

# Also save as CSV for easy inspection
export.head(50).to_csv('outputs/expert_analysis_sample.csv', index=False)
print(f"Saved sample to outputs/expert_analysis_sample.csv")
print(f"\nColumn list:")
for c in existing:
    print(f"  {c}")
