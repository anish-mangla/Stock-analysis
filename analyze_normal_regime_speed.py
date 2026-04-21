"""
Normal-Regime Speed Analysis
==============================
Expert said: re-run speed analysis EXCLUDING crisis dates (num_droppers_3pct in [61,93])
to find normal-regime speed alpha.

Goal: find features that predict FAST_WIN in non-crisis conditions.
"""
import pandas as pd
import numpy as np

# Load
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
sl = pd.read_csv('phase1_artifacts/event_speed_labels_full.csv')
sl['event_date'] = pd.to_datetime(sl['event_date'])

df = sf.merge(sl[['ticker', 'event_date', 'speed_label', 'days_to_hit', 'ret_per_day_5d',
                   'score', 'tier', 'success', 'max_5d_high_ret']],
              on=['ticker', 'event_date'], how='inner')

# Exclude crisis regime
normal = df[~((df['num_droppers_3pct'] >= 61) & (df['num_droppers_3pct'] <= 93))].copy()
print(f"Total events: {len(df)}")
print(f"Normal regime (excl crisis): {len(normal)}")
print(f"Speed label distribution in normal regime:")
print(normal['speed_label'].value_counts())
print(f"FAST_WIN rate: {(normal['speed_label'] == 'FAST_WIN').mean():.1%}")
print()

# ─── Feature analysis for normal regime ───
features = [
    'target_distance_over_atr20',
    'd0_rebound_low_to_close',
    'd0_minute_of_low',
    'd0_pct_volume_after_low',
    'd0_near_low_retest_count',
    'd1_pct_bars_above_vwap',
    'd1_longest_streak_above_vwap',
    'd1_last90_logprice_slope',
    'd1_full_day_trend_r2',
    'd1_return_vs_sector',
    'num_droppers_3pct',
    'num_droppers_5pct',
]

# Only look at tradable events (pass screener)
tradable = normal[normal['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])].copy()
print(f"Tradable events in normal regime: {len(tradable)}")
print(f"FAST_WIN rate in tradable: {(tradable['speed_label'] == 'FAST_WIN').mean():.1%}")
print()

# ─── Heatmap: target_distance_over_atr20 vs key features ───
print(f"{'='*70}")
print("HEATMAP: ATR bins vs speed label (normal regime, tradable only)")
print(f"{'='*70}")

atr_bins = [0, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 999]
atr_labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.3', '1.3-1.6', '1.6-2.0', '2.0-3.0', '3.0+']
tradable['atr_bin'] = pd.cut(tradable['target_distance_over_atr20'], bins=atr_bins, labels=atr_labels)

print(f"\n{'ATR bin':<12} {'n':>6} {'FAST%':>8} {'SLOW%':>8} {'LOSE%':>8} {'avg_ret':>10}")
print("-" * 55)
for atr_bin in atr_labels:
    g = tradable[tradable['atr_bin'] == atr_bin]
    if len(g) == 0:
        continue
    n = len(g)
    fast = (g['speed_label'] == 'FAST_WIN').mean()
    slow = (g['speed_label'] == 'SLOW_WIN').mean()
    lose = (g['speed_label'] == 'LOSE').mean()
    avg = g['ret_per_day_5d'].mean()
    print(f"{atr_bin:<12} {n:>6} {fast:>8.1%} {slow:>8.1%} {lose:>8.1%} {avg:>10.3%}")

# ─── Heatmap: ATR vs num_droppers (normal regime) ───
print(f"\n{'='*70}")
print("HEATMAP: ATR bins vs num_droppers_3pct bins (normal regime, tradable)")
print(f"{'='*70}")

dropper_bins = [0, 5, 10, 15, 23, 40, 60, 999]
dropper_labels = ['0-5', '6-10', '11-15', '16-23', '24-40', '41-60', '60+']
tradable['dropper_bin'] = pd.cut(tradable['num_droppers_3pct'], bins=dropper_bins, labels=dropper_labels)

print(f"\n{'ATR':<12} {'Droppers':<10} {'n':>5} {'FAST%':>8} {'LOSE%':>8} {'ret/day':>10}")
print("-" * 55)
for atr_bin in atr_labels:
    for d_bin in dropper_labels:
        g = tradable[(tradable['atr_bin'] == atr_bin) & (tradable['dropper_bin'] == d_bin)]
        if len(g) < 10:
            continue
        n = len(g)
        fast = (g['speed_label'] == 'FAST_WIN').mean()
        lose = (g['speed_label'] == 'LOSE').mean()
        rpd = g['ret_per_day_5d'].mean()
        marker = " ***" if fast > 0.6 and lose < 0.1 else ""
        print(f"{atr_bin:<12} {d_bin:<10} {n:>5} {fast:>8.1%} {lose:>8.1%} {rpd:>10.3%}{marker}")

# ─── D+1 features as speed predictors ───
print(f"\n{'='*70}")
print("D+1 INTRADAY FEATURES vs SPEED (normal regime, tradable)")
print(f"{'='*70}")

# d1_pct_bars_above_vwap
vwap_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
vwap_labels = ['<30%', '30-50%', '50-70%', '70-90%', '90%+']
tradable['vwap_bin'] = pd.cut(tradable['d1_pct_bars_above_vwap'], bins=vwap_bins, labels=vwap_labels)

print(f"\nD+1 % bars above VWAP:")
print(f"{'Bin':<12} {'n':>6} {'FAST%':>8} {'LOSE%':>8}")
print("-" * 40)
for b in vwap_labels:
    g = tradable[tradable['vwap_bin'] == b]
    if len(g) < 10:
        continue
    print(f"{b:<12} {len(g):>6} {(g['speed_label']=='FAST_WIN').mean():>8.1%} {(g['speed_label']=='LOSE').mean():>8.1%}")

# d0_rebound_low_to_close
reb_bins = [0, 0.01, 0.02, 0.03, 0.05, 0.10, 999]
reb_labels = ['<1%', '1-2%', '2-3%', '3-5%', '5-10%', '10%+']
tradable['reb_bin'] = pd.cut(tradable['d0_rebound_low_to_close'], bins=reb_bins, labels=reb_labels)

print(f"\nD+0 rebound (low to close):")
print(f"{'Bin':<12} {'n':>6} {'FAST%':>8} {'LOSE%':>8}")
print("-" * 40)
for b in reb_labels:
    g = tradable[tradable['reb_bin'] == b]
    if len(g) < 10:
        continue
    print(f"{b:<12} {len(g):>6} {(g['speed_label']=='FAST_WIN').mean():>8.1%} {(g['speed_label']=='LOSE').mean():>8.1%}")

# d1_return_vs_sector
sec_bins = [-999, -0.03, -0.01, 0.01, 0.03, 999]
sec_labels = ['<-3%', '-3 to -1%', '-1 to +1%', '+1 to +3%', '+3%+']
tradable['sec_bin'] = pd.cut(tradable['d1_return_vs_sector'], bins=sec_bins, labels=sec_labels)

print(f"\nD+1 return vs sector:")
print(f"{'Bin':<12} {'n':>6} {'FAST%':>8} {'LOSE%':>8}")
print("-" * 40)
for b in sec_labels:
    g = tradable[tradable['sec_bin'] == b]
    if len(g) < 10:
        continue
    print(f"{b:<12} {len(g):>6} {(g['speed_label']=='FAST_WIN').mean():>8.1%} {(g['speed_label']=='LOSE').mean():>8.1%}")

# ─── Score vs speed ───
print(f"\n{'='*70}")
print("SCREENER SCORE vs SPEED (normal regime, tradable)")
print(f"{'='*70}")

print(f"\n{'Score':<8} {'n':>6} {'FAST%':>8} {'LOSE%':>8} {'avg_days':>10}")
print("-" * 45)
for score_val in sorted(tradable['score'].unique()):
    g = tradable[tradable['score'] == score_val]
    if len(g) < 10:
        continue
    fast = (g['speed_label'] == 'FAST_WIN').mean()
    lose = (g['speed_label'] == 'LOSE').mean()
    avg_d = g['days_to_hit'].dropna().mean()
    print(f"{score_val:<8} {len(g):>6} {fast:>8.1%} {lose:>8.1%} {avg_d:>10.1f}")

# ─── Combined best normal-regime speed filter ───
print(f"\n{'='*70}")
print("CANDIDATE NORMAL-REGIME SPEED FILTERS (tradable only)")
print(f"{'='*70}")

filters = {
    'ATR<0.8': tradable[tradable['target_distance_over_atr20'] < 0.8],
    'ATR<1.0': tradable[tradable['target_distance_over_atr20'] < 1.0],
    'ATR<0.8 + VWAP>50%': tradable[(tradable['target_distance_over_atr20'] < 0.8) & (tradable['d1_pct_bars_above_vwap'] > 0.5)],
    'ATR<1.0 + VWAP>50%': tradable[(tradable['target_distance_over_atr20'] < 1.0) & (tradable['d1_pct_bars_above_vwap'] > 0.5)],
    'ATR<0.8 + score>=8': tradable[(tradable['target_distance_over_atr20'] < 0.8) & (tradable['score'] >= 8)],
    'ATR<1.0 + score>=8': tradable[(tradable['target_distance_over_atr20'] < 1.0) & (tradable['score'] >= 8)],
    'VWAP>70% + score>=8': tradable[(tradable['d1_pct_bars_above_vwap'] > 0.7) & (tradable['score'] >= 8)],
    'ATR<1.0 + VWAP>50% + score>=8': tradable[(tradable['target_distance_over_atr20'] < 1.0) & (tradable['d1_pct_bars_above_vwap'] > 0.5) & (tradable['score'] >= 8)],
    'ATR<0.8 + rebound>3%': tradable[(tradable['target_distance_over_atr20'] < 0.8) & (tradable['d0_rebound_low_to_close'] > 0.03)],
    'score>=10': tradable[tradable['score'] >= 10],
    'CONF_YES only': tradable[tradable['tier'] == 'CONFIDENT_YES'],
    'CONF_YES + ATR<1.0': tradable[(tradable['tier'] == 'CONFIDENT_YES') & (tradable['target_distance_over_atr20'] < 1.0)],
}

print(f"\n{'Filter':<35} {'n':>5} {'FAST%':>8} {'LOSE%':>8} {'avg_days':>10} {'ret/day':>10}")
print("-" * 80)
for name, g in filters.items():
    if len(g) < 10:
        continue
    n = len(g)
    fast = (g['speed_label'] == 'FAST_WIN').mean()
    lose = (g['speed_label'] == 'LOSE').mean()
    avg_d = g['days_to_hit'].dropna().mean()
    rpd = g['ret_per_day_5d'].mean()
    marker = " ***" if fast > 0.5 and lose < 0.15 and n >= 30 else ""
    print(f"{name:<35} {n:>5} {fast:>8.1%} {lose:>8.1%} {avg_d:>10.1f} {rpd:>10.3%}{marker}")

# ─── Year stability of best candidates ───
print(f"\n{'='*70}")
print("YEAR STABILITY OF TOP CANDIDATES (normal regime, tradable)")
print(f"{'='*70}")

best_filters = {
    'ATR<1.0 + VWAP>50% + score>=8': tradable[(tradable['target_distance_over_atr20'] < 1.0) & (tradable['d1_pct_bars_above_vwap'] > 0.5) & (tradable['score'] >= 8)],
    'CONF_YES + ATR<1.0': tradable[(tradable['tier'] == 'CONFIDENT_YES') & (tradable['target_distance_over_atr20'] < 1.0)],
    'ATR<0.8': tradable[tradable['target_distance_over_atr20'] < 0.8],
}

for name, g in best_filters.items():
    if len(g) < 10:
        continue
    print(f"\n{name} (n={len(g)}):")
    g = g.copy()
    g['year'] = g['event_date'].dt.year
    print(f"  {'Year':<8} {'n':>5} {'FAST%':>8} {'LOSE%':>8}")
    print(f"  {'-'*35}")
    for year, yg in g.groupby('year'):
        if len(yg) < 3:
            continue
        print(f"  {year:<8} {len(yg):>5} {(yg['speed_label']=='FAST_WIN').mean():>8.1%} {(yg['speed_label']=='LOSE').mean():>8.1%}")
