import pandas as pd, numpy as np

close_mat = pd.read_parquet('backtest_iterations/close_matrix.parquet')
daily_rets = close_mat.pct_change()

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])
sec_map = dict(zip(zip(master['ticker'], master['event_date']), master['sector_etf']))
fp['sector_etf'] = [sec_map.get((r['ticker'], r['event_date']), 'SPY') for _, r in fp.iterrows()]

# SPY return on D0 — SPY not in close_matrix, use benchmark file
import os
spy_path = 'backtest_iterations/spy_benchmark.parquet'
if os.path.exists(spy_path):
    spy_df = pd.read_parquet(spy_path)
    spy_df.index = pd.to_datetime(spy_df.index)
    if 'close' in spy_df.columns:
        spy_daily = spy_df['close'].pct_change().to_dict()
    elif 'SPY' in spy_df.columns:
        spy_daily = spy_df['SPY'].pct_change().to_dict()
    else:
        spy_daily = spy_df.iloc[:, 0].pct_change().to_dict()
    fp['spy_d0'] = fp['event_date'].map(spy_daily)
else:
    fp['spy_d0'] = np.nan

# Stock and sector return on D0
stock_d0 = []
sector_d0 = []
for _, row in fp.iterrows():
    d = row['event_date']
    t = row['ticker']
    s = row['sector_etf']
    sr = daily_rets.loc[d, t] if d in daily_rets.index and t in daily_rets.columns else np.nan
    sec = daily_rets.loc[d, s] if d in daily_rets.index and s in daily_rets.columns else np.nan
    stock_d0.append(sr)
    sector_d0.append(sec)

fp['stock_d0'] = stock_d0
fp['sector_d0'] = sector_d0
fp['excess_vs_sector'] = fp['stock_d0'] - fp['sector_d0']

# Speed features
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20']], on=['ticker', 'event_date'], how='left')

# 20DMA
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
print(f"Total: {len(fp)}, Bounced: {fp['hit'].sum()}, Stuck: {(~fp['hit']).sum()}")
print(f"spy_d0 non-null: {fp['spy_d0'].notna().sum()}")
print(f"excess_vs_sector non-null: {fp['excess_vs_sector'].notna().sum()}")
print(f"d0_dist_20dma non-null: {fp['d0_dist_20dma'].notna().sum()}")


# ═══════════════════════════════════════════════════════════
# TABLE 3: Stock vs sector
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 3: STOCK DROP vs SECTOR (excess return on D+0)")
print("More negative = stock dropped much more than sector = stock-specific")
print(f"{'='*65}")
exc_bins = [(-999, -0.08), (-0.08, -0.05), (-0.05, -0.03), (-0.03, -0.01),
            (-0.01, 0), (0, 0.01), (0.01, 999)]
exc_labels = ['<-8% (very stock-specific)', '-8 to -5%', '-5 to -3%', '-3 to -1%',
              '-1 to 0% (moved with sector)', '0 to +1%', '+1%+ (beat sector)']
print(f"\n{'Excess vs Sector':>30} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 58)
for (lo, hi), label in zip(exc_bins, exc_labels):
    sub = fp[(fp['excess_vs_sector'] >= lo) & (fp['excess_vs_sector'] < hi)]
    if len(sub) < 10:
        continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '#' * int(s / len(sub) * 40)
    print(f"{label:>30} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")

# ═══════════════════════════════════════════════════════════
# TABLE 4: SPY on drop day
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 4: SPY RETURN ON DROP DAY")
print(f"{'='*65}")
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

# ═══════════════════════════════════════════════════════════
# TABLE 7: COMBO ATR x excess
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 7: COMBO -- ATR x Stock-specific excess")
print(f"{'='*65}")
atr_c = [(0, 0.8, '<0.8'), (0.8, 1.3, '0.8-1.3'), (1.3, 2.0, '1.3-2.0'), (2.0, 999, '2.0+')]
exc_c = [(-999, -0.05, 'very stock-spec'), (-0.05, -0.02, 'mostly stock'),
         (-0.02, 0, 'mixed'), (0, 999, 'sector/market')]
print(f"\n{'ATR':>10} {'Drop type':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 56)
for alo, ahi, al in atr_c:
    for elo, ehi, el in exc_c:
        sub = fp[(fp['target_distance_over_atr20'] >= alo) & (fp['target_distance_over_atr20'] < ahi) &
                 (fp['excess_vs_sector'] >= elo) & (fp['excess_vs_sector'] < ehi)]
        if len(sub) < 20:
            continue
        b = sub['hit'].sum()
        s = len(sub) - b
        m = ' <<<' if s / len(sub) < 0.10 else (' !!!' if s / len(sub) > 0.25 else '')
        print(f"{al:>10} {el:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")

# ═══════════════════════════════════════════════════════════
# TABLE 8: COMBO ATR x 20DMA
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 8: COMBO -- ATR x Distance from 20DMA")
print(f"{'='*65}")
dma_c = [(-999, -0.10, '<-10%'), (-0.10, -0.05, '-10 to -5%'),
         (-0.05, 0, '-5 to 0%'), (0, 0.05, '0 to +5%'), (0.05, 999, '+5%+')]
print(f"\n{'ATR':>10} {'vs 20DMA':>14} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 52)
for alo, ahi, al in atr_c:
    for dlo, dhi, dl in dma_c:
        sub = fp[(fp['target_distance_over_atr20'] >= alo) & (fp['target_distance_over_atr20'] < ahi) &
                 (fp['d0_dist_20dma'] >= dlo) & (fp['d0_dist_20dma'] < dhi)]
        if len(sub) < 20:
            continue
        b = sub['hit'].sum()
        s = len(sub) - b
        m = ' <<<' if s / len(sub) < 0.10 else (' !!!' if s / len(sub) > 0.25 else '')
        print(f"{al:>10} {dl:>14} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")

# ═══════════════════════════════════════════════════════════
# FULL SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("FULL SUMMARY: BEST FILTERS TO AVOID STUCK EVENTS")
print(f"{'='*65}")
filters = {
    'ALL': fp,
    'ATR < 1.0': fp[fp['target_distance_over_atr20'] < 1.0],
    'ATR < 0.8': fp[fp['target_distance_over_atr20'] < 0.8],
    'Stock-spec (excess < -3%)': fp[fp['excess_vs_sector'] < -0.03],
    'Stock-spec (excess < -5%)': fp[fp['excess_vs_sector'] < -0.05],
    'Below 20DMA (< -5%)': fp[fp['d0_dist_20dma'] < -0.05],
    'Below 20DMA (< -10%)': fp[fp['d0_dist_20dma'] < -0.10],
    'ATR<1.0 + excess<-3%': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_sector'] < -0.03)],
    'ATR<1.0 + excess<-5%': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_sector'] < -0.05)],
    'ATR<0.8 + excess<-3%': fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['excess_vs_sector'] < -0.03)],
    'ATR<1.0 + below20DMA<-5%': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['d0_dist_20dma'] < -0.05)],
    'ATR<1.0 + excess<-3% + below20DMA': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_sector'] < -0.03) & (fp['d0_dist_20dma'] < -0.05)],
}
print(f"\n{'Filter':>40} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 68)
for name, sub in filters.items():
    sub = sub.dropna(subset=['hit'])
    if len(sub) < 10:
        continue
    b = sub['hit'].sum()
    s = len(sub) - b
    m = ' <-- best' if s / len(sub) < 0.10 and len(sub) > 50 else (' <--' if s / len(sub) < 0.12 and len(sub) > 100 else '')
    print(f"{name:>40} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")
