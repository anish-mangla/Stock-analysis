"""
Speed Feature Engineering Pipeline
====================================
Computes the 12 intraday features from 1-min bars for the speed predictor.
Processes year-by-year to manage memory on 106M bars.

Output: outputs/speed_features.parquet
"""

import pandas as pd
import numpy as np
import glob
import os

# ─── Helper functions ───

def longest_true_streak(x):
    best = cur = 0
    for v in x:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def count_near_low_retests(low_arr, session_low, first_low_idx, tol=0.0025):
    if first_low_idx >= len(low_arr) - 1:
        return 0
    near = (low_arr[first_low_idx + 1:] / session_low - 1.0) <= tol
    count = 0
    prev = False
    for x in near:
        if x and not prev:
            count += 1
        prev = x
    return count


def regression_slope_r2(y):
    if len(y) < 2:
        return np.nan, np.nan
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan
    x = np.arange(len(y), dtype=float)
    xm, ym = x.mean(), y.mean()
    cov = ((x - xm) * (y - ym)).sum()
    var_x = ((x - xm) ** 2).sum()
    if var_x == 0:
        return np.nan, np.nan
    slope = cov / var_x
    y_hat = ym + slope * (x - xm)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, r2


def summarize_symbol_day(g):
    """Compute all intraday summary features for one symbol-day."""
    g = g.sort_values('minute_of_day')
    close = g['close'].values.astype(float)
    low = g['low'].values.astype(float)
    vol = g['volume'].values.astype(float)
    
    if len(close) < 10:
        return None
    
    open_px = float(g['open'].iloc[0])
    close_px = float(close[-1])
    low_px = float(low.min())
    total_vol = float(vol.sum())
    
    first_low_idx = int(np.argmin(low))
    minute_of_low = int(g['minute_of_day'].iloc[first_low_idx])
    
    # Rebound low→close
    rebound = close_px / low_px - 1.0 if low_px > 0 else np.nan
    
    # % volume after low
    pct_vol_after = float(vol[first_low_idx:].sum() / total_vol) if total_vol > 0 else np.nan
    
    # Near-low retests
    retests = count_near_low_retests(low, low_px, first_low_idx)
    
    # Running VWAP
    cum_dollar = np.cumsum(close * vol)
    cum_vol = np.cumsum(vol)
    running_vwap = np.where(cum_vol > 0, cum_dollar / cum_vol, np.nan)
    above_vwap = close > running_vwap
    pct_above = float(np.nanmean(above_vwap))
    streak = longest_true_streak(above_vwap)
    
    # Last 90 min slope
    last90 = np.log(close[-90:]) if len(close) >= 90 else np.log(close[-max(10, len(close)//2):])
    l90_slope, l90_r2 = regression_slope_r2(last90)
    
    # Full day trend
    full_slope, full_r2 = regression_slope_r2(np.log(close))
    
    # Day return
    day_ret = close_px / open_px - 1.0 if open_px > 0 else np.nan
    
    # Last 60 return
    last60_ret = float(close[-1] / close[-60] - 1.0) if len(close) >= 61 else np.nan
    
    return {
        'session_open': open_px,
        'session_close': close_px,
        'session_low': low_px,
        'day_return_oc': day_ret,
        'last60_return': last60_ret,
        'minute_of_low': minute_of_low,
        'rebound_low_to_close': rebound,
        'pct_volume_after_low': pct_vol_after,
        'near_low_retest_count': retests,
        'pct_bars_above_vwap': pct_above,
        'longest_streak_above_vwap': streak,
        'last90_logprice_slope': l90_slope,
        'last90_trend_r2': l90_r2,
        'full_day_trend_r2': full_r2,
        'bar_count': len(g),
    }


# ─── Process all years ───
print("Building symbol-day intraday features from 1-min bars...")

all_summaries = []
files = sorted(glob.glob('intraday_data/bars_1min_*_adjusted.parquet'))

for fpath in files:
    year = fpath.split('_')[-2]
    print(f"  Processing {fpath} ({year})...")
    
    df = pd.read_parquet(fpath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
    df['session_date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # Regular session only (9:30 - 16:00)
    mask = ((df['hour'] > 9) | ((df['hour'] == 9) & (df['minute'] >= 30))) & (df['hour'] < 16)
    df = df[mask].copy()
    df['minute_of_day'] = (df['hour'] * 60 + df['minute']) - (9 * 60 + 30)
    
    # Group and summarize
    rows = []
    for (sym, sd), g in df.groupby(['symbol', 'session_date']):
        result = summarize_symbol_day(g)
        if result is not None:
            result['symbol'] = sym
            result['session_date'] = sd
            rows.append(result)
    
    year_df = pd.DataFrame(rows)
    all_summaries.append(year_df)
    print(f"    {len(year_df):,} symbol-days computed")
    del df

summary_df = pd.concat(all_summaries, ignore_index=True)
print(f"\nTotal symbol-day summaries: {len(summary_df):,}")

# ─── Cross-sectional features ───
print("\nComputing cross-sectional features...")
close_mat = pd.read_parquet('backtest_iterations/close_matrix.parquet')
daily_rets = close_mat.pct_change()

cross_rows = []
for date in daily_rets.index:
    rets = daily_rets.loc[date].dropna()
    cross_rows.append({
        'session_date': date.date(),
        'num_droppers_3pct': int((rets <= -0.03).sum()),
        'num_droppers_5pct': int((rets <= -0.05).sum()),
        'frac_droppers_3pct': float((rets <= -0.03).mean()),
    })
cross_df = pd.DataFrame(cross_rows)

# ─── ATR20 ───
print("Computing 20-day ATR...")
high_mat = pd.read_parquet('backtest_iterations/high_matrix.parquet')
low_mat = pd.read_parquet('backtest_iterations/low_matrix.parquet')

tr = pd.DataFrame(index=close_mat.index, columns=close_mat.columns, dtype=float)
for col in close_mat.columns:
    h = high_mat[col]
    l = low_mat[col]
    pc = close_mat[col].shift(1)
    tr[col] = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

atr20 = tr.rolling(20).mean()

atr_rows = []
for date in atr20.index:
    for sym in atr20.columns:
        val = atr20.loc[date, sym]
        if not pd.isna(val):
            atr_rows.append({'symbol': sym, 'session_date': date.date(), 'atr20': val})
atr_df = pd.DataFrame(atr_rows)

del high_mat, low_mat, tr, atr20, close_mat

# ─── Sector ETF mapping ───
print("Loading sector mapping...")
events = pd.read_csv('outputs/events_fully_labeled.csv')
sector_map = events[['ticker', 'sector_etf']].drop_duplicates().rename(columns={'ticker': 'symbol'})

# ─── Build sector-relative features ───
print("Building sector-relative features...")
summary_df = summary_df.merge(sector_map, on='symbol', how='left')

# Get sector ETF summaries
sector_etfs = sector_map['sector_etf'].dropna().unique()
sector_summaries = summary_df[summary_df['symbol'].isin(sector_etfs)][
    ['symbol', 'session_date', 'day_return_oc', 'last60_return']
].rename(columns={
    'symbol': 'sector_etf',
    'day_return_oc': 'sector_day_return',
    'last60_return': 'sector_last60_return',
})

summary_df = summary_df.merge(sector_summaries, on=['sector_etf', 'session_date'], how='left')
summary_df['return_vs_sector'] = summary_df['day_return_oc'] - summary_df['sector_day_return']
summary_df['last60_return_vs_sector'] = summary_df['last60_return'] - summary_df['sector_last60_return']

# ─── Build trading day map ───
print("Building next-trading-day map...")
events['event_date'] = pd.to_datetime(events['event_date'])
trading_dates = sorted(pd.read_parquet('backtest_iterations/close_matrix.parquet').index)
td_dates = [d.date() for d in trading_dates]
next_day = {td_dates[i]: td_dates[i+1] for i in range(len(td_dates)-1)}

# ─── Join to events ───
print("Joining features to events...")
ev = events[['ticker', 'event_date']].copy()
ev = ev.rename(columns={'ticker': 'symbol'})
ev['d0_date'] = ev['event_date'].dt.date
ev['d1_date'] = ev['d0_date'].map(next_day)

# D+0 features
d0_cols = ['symbol', 'session_date', 'minute_of_low', 'rebound_low_to_close',
           'pct_volume_after_low', 'near_low_retest_count']
d0 = summary_df[d0_cols].copy()
d0 = d0.rename(columns={
    'session_date': 'd0_date',
    'minute_of_low': 'd0_minute_of_low',
    'rebound_low_to_close': 'd0_rebound_low_to_close',
    'pct_volume_after_low': 'd0_pct_volume_after_low',
    'near_low_retest_count': 'd0_near_low_retest_count',
})
ev = ev.merge(d0, on=['symbol', 'd0_date'], how='left')

# D+1 features
d1_cols = ['symbol', 'session_date', 'pct_bars_above_vwap', 'longest_streak_above_vwap',
           'last90_logprice_slope', 'full_day_trend_r2', 'return_vs_sector',
           'last60_return_vs_sector', 'session_close']
d1 = summary_df[d1_cols].copy()
d1 = d1.rename(columns={
    'session_date': 'd1_date',
    'pct_bars_above_vwap': 'd1_pct_bars_above_vwap',
    'longest_streak_above_vwap': 'd1_longest_streak_above_vwap',
    'last90_logprice_slope': 'd1_last90_logprice_slope',
    'full_day_trend_r2': 'd1_full_day_trend_r2',
    'return_vs_sector': 'd1_return_vs_sector',
    'last60_return_vs_sector': 'd1_last60_return_vs_sector',
    'session_close': 'd1_close',
})
ev = ev.merge(d1, on=['symbol', 'd1_date'], how='left')

# ATR
atr_join = atr_df.rename(columns={'session_date': 'd1_date'})
ev = ev.merge(atr_join, on=['symbol', 'd1_date'], how='left')
ev['target_distance_over_atr20'] = (0.05 * ev['d1_close']) / ev['atr20']

# Cross-sectional
cs = cross_df.rename(columns={'session_date': 'd0_date'})
ev = ev.merge(cs, on='d0_date', how='left')

# Rename back
ev = ev.rename(columns={'symbol': 'ticker'})

# Save
output_path = 'outputs/speed_features.parquet'
ev.to_parquet(output_path, index=False)
print(f"\nSaved {len(ev)} events × {len(ev.columns)} columns to {output_path}")
print(f"New feature columns:")
for c in ['d0_minute_of_low', 'd0_rebound_low_to_close', 'd0_pct_volume_after_low',
          'd0_near_low_retest_count', 'd1_pct_bars_above_vwap', 'd1_longest_streak_above_vwap',
          'd1_last90_logprice_slope', 'd1_full_day_trend_r2', 'd1_return_vs_sector',
          'd1_last60_return_vs_sector', 'target_distance_over_atr20',
          'num_droppers_3pct', 'num_droppers_5pct']:
    vals = ev[c].dropna()
    print(f"  {c}: {len(vals)} non-null, mean={vals.mean():.4f}, median={vals.median():.4f}")
