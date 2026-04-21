import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

# SPY
spy_df = pd.read_parquet('backtest_iterations/spy_benchmark.parquet')
spy_df.index = pd.to_datetime(spy_df.index)
spy_daily = spy_df.iloc[:, 0].pct_change()
fp['spy_d0'] = fp['event_date'].map(spy_daily.to_dict())

# SPY 5-day trailing return (is the market in a selloff?)
spy_5d = spy_df.iloc[:, 0].pct_change(5)
fp['spy_5d'] = fp['event_date'].map(spy_5d.to_dict())

# SPY 3-day trailing
spy_3d = spy_df.iloc[:, 0].pct_change(3)
fp['spy_3d'] = fp['event_date'].map(spy_3d.to_dict())

# Stock excess vs SPY
fp['excess_vs_spy'] = fp['daily_return'] - fp['spy_d0']

# Speed features
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct']],
              on=['ticker', 'event_date'], how='left')

# 20DMA
close_mat = pd.read_parquet('backtest_iterations/close_matrix.parquet')
ma20 = close_mat.rolling(20).mean()
dist_20dma = (close_mat - ma20) / ma20
dma_dict = {}
for col in dist_20dma.columns:
    for date in dist_20dma.index:
        val = dist_20dma.loc[date, col]
        if not pd.isna(val):
            dma_dict[(col, date)] = val
fp['d0_dist_20dma'] = [dma_dict.get((r['ticker'], r['event_date']), np.nan) for _, r in fp.iterrows()]

def hit_1pct_5d(row):
    for day in range(2, 7):
        h = row.get(f'd{day}_high_ret', np.nan)
        if not pd.isna(h) and float(h) >= 0.01:
            return True
    return False

fp['hit'] = fp.apply(hit_1pct_5d, axis=1)
print(f"Total: {len(fp)}, Bounced: {fp['hit'].sum()} ({fp['hit'].mean():.1%})")


# ═══════════════════════════════════════════════════════════
# First: how do different "broad selloff" definitions affect stuck rate?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BROAD SELLOFF INDICATORS vs STUCK RATE")
print("=" * 70)

# A) SPY same-day return
print("\nA) SPY same-day return:")
print(f"{'SPY D0':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for lo, hi, label in [(-999, -0.03, '<-3% (crash)'), (-0.03, -0.01, '-3 to -1%'),
                       (-0.01, 0, '-1 to 0%'), (0, 999, '>0% (green)')]:
    sub = fp[(fp['spy_d0'] >= lo) & (fp['spy_d0'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    print(f"{label:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}")

# B) SPY 5-day trailing
print("\nB) SPY 5-day trailing return (is market in a selloff?):")
print(f"{'SPY 5d':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for lo, hi, label in [(-999, -0.05, '<-5%'), (-0.05, -0.03, '-5 to -3%'),
                       (-0.03, -0.01, '-3 to -1%'), (-0.01, 0, '-1 to 0%'),
                       (0, 0.02, '0 to +2%'), (0.02, 999, '+2%+')]:
    sub = fp[(fp['spy_5d'] >= lo) & (fp['spy_5d'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    print(f"{label:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}")

# C) Number of stocks dropping 3%+ same day
print("\nC) Number of stocks dropping 3%+ same day:")
print(f"{'Droppers':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for lo, hi, label in [(0, 5, '0-4 (isolated)'), (5, 15, '5-14'),
                       (15, 30, '15-29'), (30, 50, '30-49'),
                       (50, 999, '50+ (broad)')]:
    sub = fp[(fp['num_droppers_3pct'] >= lo) & (fp['num_droppers_3pct'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    print(f"{label:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}")

# D) Stock-specific excess vs SPY
print("\nD) Stock excess vs SPY (is this a stock-specific drop?):")
print(f"{'Excess':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for lo, hi, label in [(-999, -0.05, '<-5% (stock)'), (-0.05, -0.02, '-5 to -2%'),
                       (-0.02, 0, '-2 to 0%'), (0, 999, '>0% (market)')]:
    sub = fp[(fp['excess_vs_spy'] >= lo) & (fp['excess_vs_spy'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    print(f"{label:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}")

# ═══════════════════════════════════════════════════════════
# Now: layer selloff filters on top of ATR < 0.8 base
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYERING SELLOFF FILTERS ON TOP OF ATR < 0.8")
print("=" * 70)

base = fp[fp['target_distance_over_atr20'] < 0.8]
b0 = base['hit'].sum(); s0 = len(base) - b0
print(f"\n{'Filter':>50} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 78)
print(f"{'ATR < 0.8 (base)':>50} {len(base):>6} {b0/len(base):>9.1%} {s0/len(base):>9.1%}")

selloff_filters = {
    '+ SPY D0 > -1%': base[base['spy_d0'] > -0.01],
    '+ SPY D0 > -2%': base[base['spy_d0'] > -0.02],
    '+ SPY D0 > -3%': base[base['spy_d0'] > -0.03],
    '+ SPY 5d > -3%': base[base['spy_5d'] > -0.03],
    '+ SPY 5d > -5%': base[base['spy_5d'] > -0.05],
    '+ droppers < 15': base[base['num_droppers_3pct'] < 15],
    '+ droppers < 25': base[base['num_droppers_3pct'] < 25],
    '+ droppers < 40': base[base['num_droppers_3pct'] < 40],
    '+ excess < -2% (stock-specific)': base[base['excess_vs_spy'] < -0.02],
    '+ excess < -3% (stock-specific)': base[base['excess_vs_spy'] < -0.03],
    '+ SPY>-2% AND excess<-2%': base[(base['spy_d0'] > -0.02) & (base['excess_vs_spy'] < -0.02)],
    '+ SPY>-1% AND excess<-2%': base[(base['spy_d0'] > -0.01) & (base['excess_vs_spy'] < -0.02)],
    '+ droppers<25 AND excess<-2%': base[(base['num_droppers_3pct'] < 25) & (base['excess_vs_spy'] < -0.02)],
    '+ droppers<15 AND excess<-2%': base[(base['num_droppers_3pct'] < 15) & (base['excess_vs_spy'] < -0.02)],
}
for name, sub in selloff_filters.items():
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    m = ' <<<' if s/len(sub) < 0.10 and len(sub) > 30 else ''
    print(f"{name:>50} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")

# ═══════════════════════════════════════════════════════════
# Same but on ATR < 1.0 base (larger universe)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYERING SELLOFF FILTERS ON TOP OF ATR < 1.0")
print("=" * 70)

base1 = fp[fp['target_distance_over_atr20'] < 1.0]
b1 = base1['hit'].sum(); s1 = len(base1) - b1
print(f"\n{'Filter':>50} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 78)
print(f"{'ATR < 1.0 (base)':>50} {len(base1):>6} {b1/len(base1):>9.1%} {s1/len(base1):>9.1%}")

for name, sub in {
    '+ SPY D0 > -1%': base1[base1['spy_d0'] > -0.01],
    '+ SPY D0 > -2%': base1[base1['spy_d0'] > -0.02],
    '+ SPY 5d > -3%': base1[base1['spy_5d'] > -0.03],
    '+ droppers < 15': base1[base1['num_droppers_3pct'] < 15],
    '+ droppers < 25': base1[base1['num_droppers_3pct'] < 25],
    '+ excess < -2%': base1[base1['excess_vs_spy'] < -0.02],
    '+ excess < -3%': base1[base1['excess_vs_spy'] < -0.03],
    '+ SPY>-2% AND excess<-2%': base1[(base1['spy_d0'] > -0.02) & (base1['excess_vs_spy'] < -0.02)],
    '+ droppers<25 AND excess<-2%': base1[(base1['num_droppers_3pct'] < 25) & (base1['excess_vs_spy'] < -0.02)],
    '+ droppers<15 AND excess<-3%': base1[(base1['num_droppers_3pct'] < 15) & (base1['excess_vs_spy'] < -0.03)],
}.items():
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    m = ' <<<' if s/len(sub) < 0.10 and len(sub) > 30 else ''
    print(f"{name:>50} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")

# ═══════════════════════════════════════════════════════════
# Add 20DMA to the best combos
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BEST COMBOS WITH 20DMA ADDED")
print("=" * 70)

combos = {
    'ATR<0.8 + 20DMA(-5,0%) (prev best)':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0)],
    'ATR<0.8 + 20DMA(-5,0%) + SPY>-2%':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0) & (fp['spy_d0'] > -0.02)],
    'ATR<0.8 + 20DMA(-5,0%) + droppers<25':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0) & (fp['num_droppers_3pct'] < 25)],
    'ATR<0.8 + 20DMA(-5,0%) + excess<-2%':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0) & (fp['excess_vs_spy'] < -0.02)],
    'ATR<0.8 + 20DMA(-10,0%) + droppers<25':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.10) & (fp['d0_dist_20dma'] < 0) & (fp['num_droppers_3pct'] < 25)],
    'ATR<0.8 + 20DMA(-10,0%) + excess<-2%':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.10) & (fp['d0_dist_20dma'] < 0) & (fp['excess_vs_spy'] < -0.02)],
    'ATR<1.0 + 20DMA(-5,+5%) + droppers<25':
        fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0.05) & (fp['num_droppers_3pct'] < 25)],
    'ATR<1.0 + 20DMA(-5,+5%) + excess<-2%':
        fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0.05) & (fp['excess_vs_spy'] < -0.02)],
}

print(f"\n{'Filter':>50} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 78)
for name, sub in combos.items():
    sub = sub.dropna(subset=['hit'])
    if len(sub) < 10: continue
    b = sub['hit'].sum(); s = len(sub) - b
    m = ' <<<' if s/len(sub) < 0.08 and len(sub) > 20 else (' <<' if s/len(sub) < 0.10 and len(sub) > 30 else '')
    print(f"{name:>50} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")
