import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

# SPY benchmark
spy_df = pd.read_parquet('backtest_iterations/spy_benchmark.parquet')
spy_df.index = pd.to_datetime(spy_df.index)
spy_daily = spy_df.iloc[:, 0].pct_change()
fp['spy_d0'] = fp['event_date'].map(spy_daily.to_dict())

# Stock excess vs SPY
fp['excess_vs_spy'] = fp['daily_return'] - fp['spy_d0']

# Speed features
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20']],
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
print(f"Total: {len(fp)}, Bounced: {fp['hit'].sum()} ({fp['hit'].mean():.1%}), Stuck: {(~fp['hit']).sum()}")


# ═══ TABLE 3: Stock excess vs SPY ═══
print("\n" + "=" * 65)
print("TABLE 3: STOCK EXCESS vs SPY on drop day")
print("More negative = stock dropped much more than market = stock-specific")
print("=" * 65)
exc_bins = [(-999, -0.08), (-0.08, -0.05), (-0.05, -0.03), (-0.03, -0.01),
            (-0.01, 0), (0, 0.01), (0.01, 999)]
exc_labels = ['<-8% (very stock-specific)', '-8 to -5%', '-5 to -3%', '-3 to -1%',
              '-1 to 0% (moved with market)', '0 to +1%', '+1%+ (beat market)']
print(f"\n{'Excess vs SPY':>30} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 58)
for (lo, hi), label in zip(exc_bins, exc_labels):
    sub = fp[(fp['excess_vs_spy'] >= lo) & (fp['excess_vs_spy'] < hi)]
    if len(sub) < 10:
        continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '#' * int(s / len(sub) * 40)
    print(f"{label:>30} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")

# ═══ TABLE 4: SPY on drop day ═══
print("\n" + "=" * 65)
print("TABLE 4: SPY RETURN ON DROP DAY")
print("=" * 65)
spy_bins = [(-999, -0.03), (-0.03, -0.02), (-0.02, -0.01), (-0.01, 0),
            (0, 0.005), (0.005, 0.01), (0.01, 999)]
spy_labels = ['SPY <-3%', 'SPY -3 to -2%', 'SPY -2 to -1%', 'SPY -1 to 0%',
              'SPY 0 to +0.5%', 'SPY +0.5 to +1%', 'SPY +1%+']
print(f"\n{'SPY D0':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for (lo, hi), label in zip(spy_bins, spy_labels):
    sub = fp[(fp['spy_d0'] >= lo) & (fp['spy_d0'] < hi)]
    if len(sub) < 10:
        continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '#' * int(s / len(sub) * 40)
    print(f"{label:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")

# ═══ TABLE 7: COMBO ATR x excess ═══
print("\n" + "=" * 65)
print("TABLE 7: COMBO -- ATR x Stock excess vs SPY")
print("=" * 65)
atr_c = [(0, 0.8, '<0.8'), (0.8, 1.3, '0.8-1.3'), (1.3, 2.0, '1.3-2.0'), (2.0, 999, '2.0+')]
exc_c = [(-999, -0.05, 'very stock-spec'), (-0.05, -0.02, 'mostly stock'),
         (-0.02, 0, 'mixed'), (0, 999, 'market move')]
print(f"\n{'ATR':>10} {'Drop type':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 56)
for alo, ahi, al in atr_c:
    for elo, ehi, el in exc_c:
        sub = fp[(fp['target_distance_over_atr20'] >= alo) & (fp['target_distance_over_atr20'] < ahi) &
                 (fp['excess_vs_spy'] >= elo) & (fp['excess_vs_spy'] < ehi)]
        if len(sub) < 20:
            continue
        b = sub['hit'].sum()
        s = len(sub) - b
        m = ' <<<' if s / len(sub) < 0.10 else (' !!!' if s / len(sub) > 0.25 else '')
        print(f"{al:>10} {el:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")

# ═══ FULL SUMMARY ═══
print("\n" + "=" * 65)
print("FULL SUMMARY: BEST FILTERS TO AVOID STUCK EVENTS")
print("=" * 65)
filters = {
    'ALL': fp,
    'ATR < 1.0': fp[fp['target_distance_over_atr20'] < 1.0],
    'ATR < 0.8': fp[fp['target_distance_over_atr20'] < 0.8],
    'Stock-spec (excess<-3% vs SPY)': fp[fp['excess_vs_spy'] < -0.03],
    'Stock-spec (excess<-5% vs SPY)': fp[fp['excess_vs_spy'] < -0.05],
    'Below 20DMA (<-5%)': fp[fp['d0_dist_20dma'] < -0.05],
    'ATR<1.0 + stock-spec(<-3%)': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_spy'] < -0.03)],
    'ATR<1.0 + stock-spec(<-5%)': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_spy'] < -0.05)],
    'ATR<0.8 + stock-spec(<-3%)': fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['excess_vs_spy'] < -0.03)],
    'ATR<1.0 + below20DMA(<-5%)': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['d0_dist_20dma'] < -0.05)],
    'ATR<0.8 + below20DMA(-5 to 0%)': fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['d0_dist_20dma'] >= -0.05) & (fp['d0_dist_20dma'] < 0)],
    'ATR<1 + spec<-3% + below20DMA': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_spy'] < -0.03) & (fp['d0_dist_20dma'] < -0.05)],
}
print(f"\n{'Filter':>42} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 70)
for name, sub in filters.items():
    sub = sub.dropna(subset=['hit'])
    if len(sub) < 10:
        continue
    b = sub['hit'].sum()
    s = len(sub) - b
    m = ' <-- BEST' if s / len(sub) < 0.10 and len(sub) > 50 else (' <--' if s / len(sub) < 0.12 and len(sub) > 100 else '')
    print(f"{name:>42} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")
