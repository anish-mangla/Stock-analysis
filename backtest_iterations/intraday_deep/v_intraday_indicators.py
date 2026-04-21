#!/usr/bin/env python3
"""
Compute intraday indicators for every event and build a correlation matrix
with trade success. No threshold guessing — just measure what correlates.
"""
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data

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
        all_bars.append(pd.read_parquet(path))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute

# Pre-compute per (symbol, date) aggregates
print("Computing daily intraday aggregates...")
daily = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'),
    day_high=('high', 'max'),
    day_low=('low', 'min'),
    day_close=('close', 'last'),
    total_vol=('volume', 'sum'),
    total_trades=('trade_count', 'sum'),
    bar_count=('close', 'count'),
    vwap_avg=('vwap', 'mean'),
).reset_index()

# Close in range: (close - low) / (high - low)
daily['close_in_range'] = np.where(
    daily['day_high'] > daily['day_low'],
    (daily['day_close'] - daily['day_low']) / (daily['day_high'] - daily['day_low']),
    0.5
)

# Intraday range as % of open
daily['intraday_range_pct'] = (daily['day_high'] - daily['day_low']) / daily['day_open'] * 100

# Close vs VWAP
daily['close_vs_vwap'] = np.where(
    daily['vwap_avg'] > 0,
    (daily['day_close'] / daily['vwap_avg'] - 1) * 100,
    0
)

# Trade size proxy: volume / trade_count (avg shares per trade)
daily['avg_trade_size'] = np.where(
    daily['total_trades'] > 0,
    daily['total_vol'] / daily['total_trades'],
    0
)

# 20-day rolling averages for relative metrics
daily = daily.sort_values(['symbol', 'date'])
daily['avg_vol_20d'] = daily.groupby('symbol')['total_vol'].transform(
    lambda x: x.rolling(20, min_periods=10).mean().shift(1))
daily['avg_range_20d'] = daily.groupby('symbol')['intraday_range_pct'].transform(
    lambda x: x.rolling(20, min_periods=10).mean().shift(1))
daily['avg_trade_size_20d'] = daily.groupby('symbol')['avg_trade_size'].transform(
    lambda x: x.rolling(20, min_periods=10).mean().shift(1))

# Relative metrics
daily['rvol'] = daily['total_vol'] / daily['avg_vol_20d']
daily['rel_range'] = daily['intraday_range_pct'] / daily['avg_range_20d']
daily['rel_trade_size'] = daily['avg_trade_size'] / daily['avg_trade_size_20d']

# Tail ratio: lower wick / total range
daily['lower_wick'] = np.where(
    daily['day_high'] > daily['day_low'],
    (np.minimum(daily['day_open'], daily['day_close']) - daily['day_low']) / (daily['day_high'] - daily['day_low']),
    0
)
daily['upper_wick'] = np.where(
    daily['day_high'] > daily['day_low'],
    (daily['day_high'] - np.maximum(daily['day_open'], daily['day_close'])) / (daily['day_high'] - daily['day_low']),
    0
)

# OBV-like: compute from minute bars
# Up bars volume vs down bars volume
print("Computing OBV ratio...")
bars['price_change'] = bars.groupby(['symbol', 'date'])['close'].diff()
obv_data = bars.groupby(['symbol', 'date']).apply(
    lambda g: pd.Series({
        'up_vol': g.loc[g['price_change'] > 0, 'volume'].sum(),
        'down_vol': g.loc[g['price_change'] < 0, 'volume'].sum(),
    })
).reset_index()
obv_data['obv_ratio'] = np.where(
    (obv_data['up_vol'] + obv_data['down_vol']) > 0,
    obv_data['up_vol'] / (obv_data['up_vol'] + obv_data['down_vol']),
    0.5
)
daily = daily.merge(obv_data[['symbol', 'date', 'obv_ratio']], on=['symbol', 'date'], how='left')

# First 30 min volume as % of total
print("Computing first-30-min metrics...")
first30 = bars[bars['utc_min'] <= 840]  # 9:30-10:00 ET
first30_vol = first30.groupby(['symbol', 'date'])['volume'].sum().reset_index()
first30_vol.columns = ['symbol', 'date', 'first30_vol']
daily = daily.merge(first30_vol, on=['symbol', 'date'], how='left')
daily['first30_vol_pct'] = daily['first30_vol'] / daily['total_vol'] * 100

# Last 30 min volume as % of total
last30 = bars[bars['utc_min'] >= 1170]  # 3:30-4:00 ET
last30_vol = last30.groupby(['symbol', 'date'])['volume'].sum().reset_index()
last30_vol.columns = ['symbol', 'date', 'last30_vol']
daily = daily.merge(last30_vol, on=['symbol', 'date'], how='left')
daily['last30_vol_pct'] = daily['last30_vol'] / daily['total_vol'] * 100

# First hour return
first_bar = bars.sort_values('timestamp').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'open']]
first_bar.columns = ['symbol', 'date', 'first_open']
hr1 = bars[(bars['utc_min'] >= 867) & (bars['utc_min'] <= 873)]
hr1_price = hr1.sort_values('utc_min').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'close']]
hr1_price.columns = ['symbol', 'date', 'price_1hr']
first_bar = first_bar.merge(hr1_price, on=['symbol', 'date'], how='left')
first_bar['first_hour_ret'] = (first_bar['price_1hr'] / first_bar['first_open'] - 1) * 100
daily = daily.merge(first_bar[['symbol', 'date', 'first_hour_ret']], on=['symbol', 'date'], how='left')

del bars, first30, last30, obv_data, first_bar, hr1, hr1_price
print("Done computing indicators")

# Build daily lookup
daily_lookup = daily.set_index(['symbol', 'date'])

# Now match events to indicators for BOTH drop day (T+0) and entry day (T+2)
print("Matching events to indicators...")
rows = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    t1 = future[0].date()
    t2 = future[1].date()
    drop_date = dt.date()
    
    r = {
        'ticker': ticker,
        'event_date': dt,
        'drop_pct': row['drop_pct'],
        'success': 1 if row.get('success', False) else 0,
        'days_to_hit': row.get('days_to_hit', np.nan),
        'final_return': row.get('final_return', np.nan),
    }
    
    # Drop day (T+0) indicators
    if (alpaca_t, drop_date) in daily_lookup.index:
        d0 = daily_lookup.loc[(alpaca_t, drop_date)]
        r['d0_close_in_range'] = d0.get('close_in_range', np.nan)
        r['d0_intraday_range_pct'] = d0.get('intraday_range_pct', np.nan)
        r['d0_close_vs_vwap'] = d0.get('close_vs_vwap', np.nan)
        r['d0_rvol'] = d0.get('rvol', np.nan)
        r['d0_rel_range'] = d0.get('rel_range', np.nan)
        r['d0_rel_trade_size'] = d0.get('rel_trade_size', np.nan)
        r['d0_lower_wick'] = d0.get('lower_wick', np.nan)
        r['d0_upper_wick'] = d0.get('upper_wick', np.nan)
        r['d0_obv_ratio'] = d0.get('obv_ratio', np.nan)
        r['d0_first30_vol_pct'] = d0.get('first30_vol_pct', np.nan)
        r['d0_last30_vol_pct'] = d0.get('last30_vol_pct', np.nan)
        r['d0_first_hour_ret'] = d0.get('first_hour_ret', np.nan)
    
    # T+1 indicators
    if (alpaca_t, t1) in daily_lookup.index:
        d1 = daily_lookup.loc[(alpaca_t, t1)]
        r['d1_close_in_range'] = d1.get('close_in_range', np.nan)
        r['d1_rvol'] = d1.get('rvol', np.nan)
        r['d1_obv_ratio'] = d1.get('obv_ratio', np.nan)
        r['d1_first_hour_ret'] = d1.get('first_hour_ret', np.nan)
        r['d1_close_vs_vwap'] = d1.get('close_vs_vwap', np.nan)
        r['d1_lower_wick'] = d1.get('lower_wick', np.nan)
        # D+1 return (close to close)
        if ticker in close_mat.columns and dt in close_mat.index and future[0] in close_mat.index:
            c0 = close_mat.at[dt, ticker]
            c1 = close_mat.at[future[0], ticker]
            if pd.notna(c0) and pd.notna(c1) and float(c0) > 0:
                r['d1_return'] = (float(c1) / float(c0) - 1) * 100
    
    # T+2 (entry day) indicators
    if (alpaca_t, t2) in daily_lookup.index:
        d2 = daily_lookup.loc[(alpaca_t, t2)]
        r['d2_rvol'] = d2.get('rvol', np.nan)
        r['d2_obv_ratio'] = d2.get('obv_ratio', np.nan)
        r['d2_first_hour_ret'] = d2.get('first_hour_ret', np.nan)
        r['d2_close_vs_vwap'] = d2.get('close_vs_vwap', np.nan)
    
    rows.append(r)

feature_df = pd.DataFrame(rows)
print(f"Feature matrix: {len(feature_df)} events × {len(feature_df.columns)} columns")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print(f"  CORRELATION WITH SUCCESS (point-biserial)")
print(f"{'='*80}\n")

# Compute correlation of each feature with success
indicator_cols = [c for c in feature_df.columns if c.startswith('d0_') or c.startswith('d1_') or c.startswith('d2_') or c == 'drop_pct']
correlations = {}
for col in indicator_cols:
    valid = feature_df[[col, 'success']].dropna()
    if len(valid) > 100:
        corr = valid[col].corr(valid['success'])
        correlations[col] = round(corr, 4)

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"{'Indicator':>30} {'Corr with success':>18} {'Direction':>10}")
print('-'*62)
for col, corr in sorted_corr:
    direction = '↑ = better' if corr > 0 else '↓ = better'
    strength = '***' if abs(corr) > 0.1 else '**' if abs(corr) > 0.05 else '*' if abs(corr) > 0.03 else ''
    print(f"{col:>30} {corr:>+17.4f} {direction:>10} {strength}")

# Also correlate with days_to_hit (speed of recovery)
print(f"\n{'='*80}")
print(f"  CORRELATION WITH SPEED (days_to_hit, lower = faster recovery)")
print(f"{'='*80}\n")

speed_corr = {}
for col in indicator_cols:
    valid = feature_df[[col, 'days_to_hit']].dropna()
    if len(valid) > 100:
        corr = valid[col].corr(valid['days_to_hit'])
        speed_corr[col] = round(corr, 4)

sorted_speed = sorted(speed_corr.items(), key=lambda x: abs(x[1]), reverse=True)
print(f"{'Indicator':>30} {'Corr with days_to_hit':>22} {'Direction':>12}")
print('-'*68)
for col, corr in sorted_speed:
    direction = '↑ = slower' if corr > 0 else '↑ = faster'
    strength = '***' if abs(corr) > 0.1 else '**' if abs(corr) > 0.05 else '*' if abs(corr) > 0.03 else ''
    print(f"{col:>30} {corr:>+21.4f} {direction:>12} {strength}")

# Cross-correlation matrix between indicators
print(f"\n{'='*80}")
print(f"  INTER-INDICATOR CORRELATION (are they redundant?)")
print(f"{'='*80}\n")

# Top indicators only
top_indicators = [col for col, _ in sorted_corr[:10]]
if top_indicators:
    cross_corr = feature_df[top_indicators].corr()
    print("Top 10 indicators cross-correlation:")
    print(cross_corr.round(2).to_string())

# Save feature matrix
feature_df.to_csv(os.path.join(OUT_DIR, 'intraday_indicators_features.csv'), index=False)

# Save correlation results
with open(os.path.join(OUT_DIR, 'v_indicators_correlation.txt'), 'w') as f:
    f.write("INTRADAY INDICATORS — CORRELATION WITH SUCCESS\n\n")
    f.write(f"{'Indicator':>30} {'Corr':>8}\n")
    f.write('-'*42 + '\n')
    for col, corr in sorted_corr:
        f.write(f"{col:>30} {corr:>+7.4f}\n")
    f.write(f"\n\nCORRELATION WITH SPEED (days_to_hit)\n\n")
    for col, corr in sorted_speed:
        f.write(f"{col:>30} {corr:>+7.4f}\n")

print(f"\nFeature matrix saved to {OUT_DIR}/intraday_indicators_features.csv")
print(f"Correlations saved to {OUT_DIR}/v_indicators_correlation.txt")
