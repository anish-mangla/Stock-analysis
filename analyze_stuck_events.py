"""
Deep analysis of stuck events (never bounce +1% within 5 days after entry).
Entry = D+1 close. Bounce check = D+2 through D+6 intraday highs.

Features to analyze:
- ATR (target_distance_over_atr20) — fine-grained bins
- Drop vs 20DMA (how far below moving average)
- Stock-specific vs sector vs market move
- D+1 behavior (our entry day)
- Sector breakdown
- Drop size
- Combos
"""
import pandas as pd
import numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])

# Merge
fp = fp.drop(columns=['stock_event_type', 'stock_event_severity', 'stock_event_description'], errors='ignore')
fp = fp.merge(master[['ticker', 'event_date', 'stock_event_type', 'stock_event_severity',
                       'company_specific_factor', 'sector_etf']].drop_duplicates(),
              on=['ticker', 'event_date'], how='left')
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct',
                   'num_droppers_5pct', 'd0_rebound_low_to_close', 'd0_minute_of_low',
                   'd1_pct_bars_above_vwap', 'd1_return_vs_sector', 'd1_last90_logprice_slope']],
              on=['ticker', 'event_date'], how='left')

# Build 20DMA features from close matrix
print("Building 20DMA features...")
close_mat = pd.read_parquet('backtest_iterations/close_matrix.parquet')
ma20 = close_mat.rolling(20).mean()
# Distance from 20DMA on D+0 (event day)
dist_from_ma20 = (close_mat - ma20) / ma20  # positive = above MA, negative = below

# Map to events
fp['event_date_dt'] = fp['event_date']
dma_rows = []
for _, row in fp.iterrows():
    ticker = row['ticker']
    date = row['event_date']
    if ticker in dist_from_ma20.columns and date in dist_from_ma20.index:
        val = dist_from_ma20.loc[date, ticker]
        dma_rows.append(val)
    else:
        dma_rows.append(np.nan)
fp['d0_dist_from_20dma'] = dma_rows

# Also get SPY return on D+0 and sector ETF return on D+0
daily_rets = close_mat.pct_change()
spy_d0 = []
sector_d0 = []
stock_d0_vs_spy = []
stock_d0_vs_sector = []
for _, row in fp.iterrows():
    date = row['event_date']
    ticker = row['ticker']
    sector = row.get('sector_etf', 'SPY')
    
    spy_ret = daily_rets.loc[date, 'SPY'] if date in daily_rets.index and 'SPY' in daily_rets.columns else np.nan
    sec_ret = daily_rets.loc[date, sector] if date in daily_rets.index and sector in daily_rets.columns else np.nan
    stock_ret = daily_rets.loc[date, ticker] if date in daily_rets.index and ticker in daily_rets.columns else np.nan
    
    spy_d0.append(spy_ret)
    sector_d0.append(sec_ret)
    stock_d0_vs_spy.append(stock_ret - spy_ret if not pd.isna(stock_ret) and not pd.isna(spy_ret) else np.nan)
    stock_d0_vs_sector.append(stock_ret - sec_ret if not pd.isna(stock_ret) and not pd.isna(sec_ret) else np.nan)

fp['spy_d0_return'] = spy_d0
fp['sector_d0_return'] = sector_d0
fp['stock_d0_excess_vs_spy'] = stock_d0_vs_spy
fp['stock_d0_excess_vs_sector'] = stock_d0_vs_sector

# Label stuck
def hit_1pct_5d(row):
    for day in range(2, 7):
        h = row.get(f'd{day}_high_ret', np.nan)
        if not pd.isna(h) and float(h) >= 0.01:
            return True
    return False

fp['hit'] = fp.apply(hit_1pct_5d, axis=1)
bounced = fp[fp['hit']]
stuck = fp[~fp['hit']]
print(f"\nTotal: {len(fp)}, Bounced: {len(bounced)} ({len(bounced)/len(fp):.1%}), Stuck: {len(stuck)} ({len(stuck)/len(fp):.1%})")


# ═══════════════════════════════════════════════════════════
# TABLE 1: ATR (fine-grained)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 1: TARGET DISTANCE / ATR20 (fine-grained)")
print("ATR = how many 'normal daily moves' to reach +5% target")
print("Lower = easier target = more likely to bounce")
print(f"{'='*65}")

atr_bins = [0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0, 999]
atr_labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.8', '0.8-1.0', '1.0-1.2',
              '1.2-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-5.0', '5.0+']

print(f"\n{'ATR':>10} {'n':>6} {'Bounce%':>9} {'Stuck%':>9} {'Stuck n':>8} {'Avg 5d ret':>11}")
print('-' * 58)
for i, label in enumerate(atr_labels):
    lo, hi = atr_bins[i], atr_bins[i+1]
    sub = fp[(fp['target_distance_over_atr20'] >= lo) & (fp['target_distance_over_atr20'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    d6 = sub['d6_close_ret'].dropna()
    avg5 = d6.mean() if len(d6) > 0 else np.nan
    bar = '█' * int(s/len(sub) * 40)
    print(f"{label:>10} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%} {s:>8} {avg5:>+11.2%}  {bar}")

# ═══════════════════════════════════════════════════════════
# TABLE 2: Distance from 20DMA
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 2: DISTANCE FROM 20-DAY MOVING AVERAGE on drop day (D+0)")
print("Negative = stock is below its 20DMA, Positive = above")
print(f"{'='*65}")

dma_bins = [(-999, -0.20), (-0.20, -0.15), (-0.15, -0.10), (-0.10, -0.05),
            (-0.05, -0.02), (-0.02, 0), (0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 999)]
dma_labels = ['<-20%', '-20 to -15%', '-15 to -10%', '-10 to -5%',
              '-5 to -2%', '-2 to 0%', '0 to +2%', '+2 to +5%', '+5 to +10%', '+10%+']

print(f"\n{'vs 20DMA':>14} {'n':>6} {'Bounce%':>9} {'Stuck%':>9} {'Stuck n':>8}")
print('-' * 50)
for (lo, hi), label in zip(dma_bins, dma_labels):
    sub = fp[(fp['d0_dist_from_20dma'] >= lo) & (fp['d0_dist_from_20dma'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '█' * int(s/len(sub) * 40)
    print(f"{label:>14} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%} {s:>8}  {bar}")

# ═══════════════════════════════════════════════════════════
# TABLE 3: Stock-specific vs Sector vs Market move
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 3: WAS THE DROP STOCK-SPECIFIC OR BROAD?")
print("stock_excess_vs_sector = stock D0 return minus sector ETF return")
print("More negative = stock dropped way more than its sector (stock-specific)")
print(f"{'='*65}")

exc_bins = [(-999, -0.08), (-0.08, -0.05), (-0.05, -0.03), (-0.03, -0.01),
            (-0.01, 0), (0, 0.01), (0.01, 999)]
exc_labels = ['<-8% (very stock-specific)', '-8 to -5%', '-5 to -3%', '-3 to -1%',
              '-1 to 0% (moved with sector)', '0 to +1%', '+1%+ (outperformed sector)']

print(f"\n{'Excess vs Sector':>30} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 58)
for (lo, hi), label in zip(exc_bins, exc_labels):
    sub = fp[(fp['stock_d0_excess_vs_sector'] >= lo) & (fp['stock_d0_excess_vs_sector'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '█' * int(s/len(sub) * 40)
    print(f"{label:>30} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")

# SPY on drop day
print(f"\n{'='*65}")
print("TABLE 4: SPY RETURN ON DROP DAY (D+0)")
print("Was the market also down?")
print(f"{'='*65}")

spy_bins = [(-999, -0.03), (-0.03, -0.02), (-0.02, -0.01), (-0.01, 0),
            (0, 0.005), (0.005, 0.01), (0.01, 999)]
spy_labels = ['SPY <-3%', 'SPY -3 to -2%', 'SPY -2 to -1%', 'SPY -1 to 0%',
              'SPY 0 to +0.5%', 'SPY +0.5 to +1%', 'SPY +1%+']

print(f"\n{'SPY D0':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for (lo, hi), label in zip(spy_bins, spy_labels):
    sub = fp[(fp['spy_d0_return'] >= lo) & (fp['spy_d0_return'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '█' * int(s/len(sub) * 40)
    print(f"{label:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")


# ═══════════════════════════════════════════════════════════
# TABLE 5: D+1 behavior (our entry day)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 5: D+1 RETURN (our entry day)")
print("We buy at D+1 close. This is what happened during D+1.")
print(f"{'='*65}")

d1_bins = [(-999, -0.05), (-0.05, -0.03), (-0.03, -0.01), (-0.01, 0),
           (0, 0.01), (0.01, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 999)]
d1_labels = ['D1 <-5%', 'D1 -5 to -3%', 'D1 -3 to -1%', 'D1 -1 to 0%',
             'D1 0 to +1%', 'D1 +1 to +3%', 'D1 +3 to +5%', 'D1 +5 to +8%', 'D1 +8%+']

print(f"\n{'D+1 return':>16} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 44)
for (lo, hi), label in zip(d1_bins, d1_labels):
    sub = fp[(fp['d1_return'] >= lo) & (fp['d1_return'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '█' * int(s/len(sub) * 40)
    print(f"{label:>16} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")

# ═══════════════════════════════════════════════════════════
# TABLE 6: Sector breakdown
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 6: SECTOR")
print(f"{'='*65}")

print(f"\n{'Sector':>8} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 36)
sector_stats = []
for sec in fp['sector_etf'].value_counts().index:
    sub = fp[fp['sector_etf'] == sec]
    if len(sub) < 20: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    sector_stats.append((sec, len(sub), s/len(sub)))
sector_stats.sort(key=lambda x: x[2])
for sec, n, stuck_pct in sector_stats:
    bar = '█' * int(stuck_pct * 40)
    print(f"{sec:>8} {n:>6} {1-stuck_pct:>9.1%} {stuck_pct:>9.1%}  {bar}")

# ═══════════════════════════════════════════════════════════
# TABLE 7: COMBO — ATR + stock-specific excess
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 7: COMBO — ATR bins × Stock-specific excess")
print("Finding the intersection of easy target + stock-specific drop")
print(f"{'='*65}")

atr_coarse = [(0, 0.8, '<0.8'), (0.8, 1.3, '0.8-1.3'), (1.3, 2.0, '1.3-2.0'), (2.0, 999, '2.0+')]
exc_coarse = [(-999, -0.05, 'very stock-spec'), (-0.05, -0.02, 'mostly stock'),
              (-0.02, 0, 'mixed'), (0, 999, 'sector/market')]

print(f"\n{'ATR':>10} {'Drop type':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 56)
for alo, ahi, alabel in atr_coarse:
    for elo, ehi, elabel in exc_coarse:
        sub = fp[(fp['target_distance_over_atr20'] >= alo) & (fp['target_distance_over_atr20'] < ahi) &
                 (fp['stock_d0_excess_vs_sector'] >= elo) & (fp['stock_d0_excess_vs_sector'] < ehi)]
        if len(sub) < 20: continue
        b = sub['hit'].sum()
        s = len(sub) - b
        marker = ' ◄◄◄' if s/len(sub) < 0.10 else (' !!!' if s/len(sub) > 0.25 else '')
        print(f"{alabel:>10} {elabel:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{marker}")

# ═══════════════════════════════════════════════════════════
# TABLE 8: COMBO — ATR + 20DMA distance
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 8: COMBO — ATR bins × Distance from 20DMA")
print(f"{'='*65}")

dma_coarse = [(-999, -0.10, '<-10%'), (-0.10, -0.05, '-10 to -5%'),
              (-0.05, 0, '-5 to 0%'), (0, 0.05, '0 to +5%'), (0.05, 999, '+5%+')]

print(f"\n{'ATR':>10} {'vs 20DMA':>14} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 52)
for alo, ahi, alabel in atr_coarse:
    for dlo, dhi, dlabel in dma_coarse:
        sub = fp[(fp['target_distance_over_atr20'] >= alo) & (fp['target_distance_over_atr20'] < ahi) &
                 (fp['d0_dist_from_20dma'] >= dlo) & (fp['d0_dist_from_20dma'] < dhi)]
        if len(sub) < 20: continue
        b = sub['hit'].sum()
        s = len(sub) - b
        marker = ' ◄◄◄' if s/len(sub) < 0.10 else (' !!!' if s/len(sub) > 0.25 else '')
        print(f"{alabel:>10} {dlabel:>14} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{marker}")

# ═══════════════════════════════════════════════════════════
# TABLE 9: Number of other stocks dropping same day
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("TABLE 9: HOW MANY OTHER STOCKS DROPPED 3%+ SAME DAY")
print("Low = isolated stock event. High = broad market selloff.")
print(f"{'='*65}")

nd_bins = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 40), (40, 60), (60, 100), (100, 999)]
nd_labels = ['0-2 (isolated)', '3-7', '8-14', '15-24', '25-39', '40-59', '60-99 (broad)', '100+ (crash)']

print(f"\n{'Droppers':>20} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 48)
for (lo, hi), label in zip(nd_bins, nd_labels):
    sub = fp[(fp['num_droppers_3pct'] >= lo) & (fp['num_droppers_3pct'] < hi)]
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    bar = '█' * int(s/len(sub) * 40)
    print(f"{label:>20} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}  {bar}")

# ═══════════════════════════════════════════════════════════
# SUMMARY: Best filters to avoid stuck events
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("SUMMARY: CANDIDATE FILTERS TO AVOID STUCK EVENTS")
print(f"{'='*65}")

filters = {
    'ALL events': fp,
    'ATR < 1.0': fp[fp['target_distance_over_atr20'] < 1.0],
    'ATR < 0.8': fp[fp['target_distance_over_atr20'] < 0.8],
    'Stock-spec (excess < -3%)': fp[fp['stock_d0_excess_vs_sector'] < -0.03],
    'Below 20DMA (< -5%)': fp[fp['d0_dist_from_20dma'] < -0.05],
    'ATR<1.0 + stock-spec': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['stock_d0_excess_vs_sector'] < -0.03)],
    'ATR<1.0 + below 20DMA': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['d0_dist_from_20dma'] < -0.05)],
    'ATR<1.0 + droppers<15': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['num_droppers_3pct'] < 15)],
    'ATR<0.8 + stock-spec': fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['stock_d0_excess_vs_sector'] < -0.03)],
    'ATR<1.0 + stock-spec + below20DMA': fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['stock_d0_excess_vs_sector'] < -0.03) & (fp['d0_dist_from_20dma'] < -0.05)],
}

print(f"\n{'Filter':>38} {'n':>6} {'Bounce%':>9} {'Stuck%':>9} {'5d avg':>8}")
print('-' * 75)
for name, sub in filters.items():
    sub = sub.dropna(subset=['hit'])
    if len(sub) < 10: continue
    b = sub['hit'].sum()
    s = len(sub) - b
    d6 = sub['d6_close_ret'].dropna()
    avg = d6.mean() if len(d6) > 0 else np.nan
    marker = ' ◄' if s/len(sub) < 0.12 and len(sub) > 100 else ''
    print(f"{name:>38} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%} {avg:>+8.2%}{marker}")
