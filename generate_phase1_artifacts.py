"""
Phase 1 Artifacts Generator
============================
Generates all tables and heatmaps requested by the expert for the
high-precision mean-reversion screener analysis.

Output: prints tables to console and saves CSV files to phase1_artifacts/
"""

import pandas as pd
import numpy as np
import os

os.makedirs('phase1_artifacts', exist_ok=True)

# ─── LOAD DATA ───
print("Loading data...")
close = pd.read_parquet('backtest_iterations/close_matrix.parquet')
high = pd.read_parquet('backtest_iterations/high_matrix.parquet')
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])
indicators = pd.read_csv('backtest_iterations/intraday_deep/intraday_indicators_features.csv')
indicators['event_date'] = pd.to_datetime(indicators['event_date'])
recovery = pd.read_csv('backtest_iterations/intraday_deep/recovery_curve_data.csv')
recovery['event_date'] = pd.to_datetime(recovery['event_date'])
market = pd.read_csv('outputs/market_daily_features.csv')

# Fix market date column
if 'Unnamed: 0' in market.columns:
    market = market.rename(columns={'Unnamed: 0': 'date'})
market['date'] = pd.to_datetime(market['date'])

# ─── MERGE INTO ONE MASTER DATAFRAME ───
print("Merging datasets...")
df = events.merge(indicators, on=['ticker', 'event_date'], how='left', suffixes=('', '_ind'))
df = df.merge(recovery[['ticker', 'event_date', 'close_d1', 'high_d1', 'low_d1',
                         'close_d2', 'high_d2', 'low_d2', 'close_d3', 'high_d3', 'low_d3',
                         'close_d4', 'high_d4', 'low_d4', 'close_d5', 'high_d5', 'low_d5']],
              on=['ticker', 'event_date'], how='left')
df = df.merge(market, left_on='event_date', right_on='date', how='left')

# ─── CREATE TARGET LABELS ───
print("Creating target labels...")
df['hit_5_60'] = df['success'].astype(bool)
df['hit_5_20'] = (df['days_to_hit'] <= 20) & df['hit_5_60']
df['never_down_8'] = df['max_drawdown'] > -0.08
df['good_trade'] = df['hit_5_60'] & (df['max_drawdown'] > -0.08) & (df['final_return'] >= 0)

# Bad loser definitions
df['bad_loser'] = ~df['hit_5_60']
df['ugly_loser'] = (df['final_return'] < -0.10) | (df['max_drawdown'] < -0.12)

# Year
df['year'] = df['event_date'].dt.year

# ─── HELPER FUNCTIONS ───
def compute_metrics(sub):
    n = len(sub)
    if n == 0:
        return {}
    return {
        'count': n,
        'hit_5_60': sub['hit_5_60'].mean(),
        'hit_5_20': sub['hit_5_20'].mean(),
        'good_trade': sub['good_trade'].mean(),
        'loser_rate': (~sub['hit_5_60']).mean(),
        'ugly_loser_rate': sub['ugly_loser'].mean(),
        'median_days_to_5': sub.loc[sub['hit_5_60'], 'days_to_hit'].median() if sub['hit_5_60'].any() else np.nan,
        'median_day60_return': sub['final_return'].median(),
        'median_max_dd': sub['max_drawdown'].median(),
    }

def univariate_table(df, col, bins, labels=None):
    if labels:
        df['_bin'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    else:
        df['_bin'] = pd.cut(df[col], bins=bins, include_lowest=True)
    rows = []
    for b in df['_bin'].cat.categories:
        sub = df[df['_bin'] == b]
        m = compute_metrics(sub)
        m['bin'] = str(b)
        rows.append(m)
    df.drop(columns='_bin', inplace=True)
    return pd.DataFrame(rows)

def crosstab_table(df, row_col, col_col, metric='good_trade'):
    """Pivot table with row_col as rows, col_col as columns, metric as values."""
    rows = []
    for rv in sorted(df[row_col].dropna().unique()):
        for cv in sorted(df[col_col].dropna().unique()):
            sub = df[(df[row_col] == rv) & (df[col_col] == cv)]
            if len(sub) < 5:
                continue
            rows.append({
                row_col: rv,
                col_col: cv,
                'count': len(sub),
                metric: sub[metric].mean() if metric in sub.columns else np.nan,
                'ugly_loser_rate': sub['ugly_loser'].mean(),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Target Label Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 1: TARGET LABEL SUMMARY")
print("=" * 80)
summary = {
    'hit_5_60 (hit +5% in 60d)': df['hit_5_60'].mean(),
    'hit_5_20 (hit +5% in 20d)': df['hit_5_20'].mean(),
    'never_down_8 (never -8% drawdown)': df['never_down_8'].mean(),
    'good_trade (hit +5%, dd>-8%, day60>=0)': df['good_trade'].mean(),
    'bad_loser (did not hit +5%)': df['bad_loser'].mean(),
    'ugly_loser (day60<-10% or dd<-12%)': df['ugly_loser'].mean(),
}
t1 = pd.DataFrame([{'label': k, 'rate': f"{v*100:.1f}%", 'count': int(v*len(df))} for k, v in summary.items()])
print(t1.to_string(index=False))
t1.to_csv('phase1_artifacts/01_target_label_summary.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2: By Event Type × Severity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 2: BY EVENT TYPE × SEVERITY")
print("=" * 80)
rows = []
for et in sorted(df['stock_event_type'].dropna().unique()):
    for sev in ['low', 'medium', 'high', 'unlabeled']:
        sub = df[(df['stock_event_type'] == et) & (df['stock_event_severity'] == sev)]
        if len(sub) < 10:
            continue
        m = compute_metrics(sub)
        m['event_type'] = et
        m['severity'] = sev
        rows.append(m)
t2 = pd.DataFrame(rows)[['event_type', 'severity', 'count', 'hit_5_60', 'good_trade', 'ugly_loser_rate', 'median_days_to_5', 'median_day60_return']]
t2['hit_5_60'] = t2['hit_5_60'].map(lambda x: f"{x*100:.1f}%")
t2['good_trade'] = t2['good_trade'].map(lambda x: f"{x*100:.1f}%")
t2['ugly_loser_rate'] = t2['ugly_loser_rate'].map(lambda x: f"{x*100:.1f}%")
t2['median_day60_return'] = t2['median_day60_return'].map(lambda x: f"{x*100:.1f}%")
print(t2.to_string(index=False))
t2.to_csv('phase1_artifacts/02_event_type_severity.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3: By Company-Specific Flag
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 3: BY COMPANY-SPECIFIC FLAG (from enriched labels)")
print("=" * 80)
# Load enriched labels
import glob
v2_files = sorted(glob.glob('outputs/stock_level_labels_v2/batch_*_classified.csv'))
v2_dfs = []
for f in v2_files:
    try:
        v2_dfs.append(pd.read_csv(f, on_bad_lines='skip'))
    except:
        pass
if v2_dfs:
    v2 = pd.concat(v2_dfs, ignore_index=True)
    v2['event_date'] = pd.to_datetime(v2['event_date'])
    # Merge company_specific_factor into main df
    df_cs = df.merge(v2[['ticker', 'event_date', 'company_specific_factor']].drop_duplicates(),
                     on=['ticker', 'event_date'], how='left')
    rows = []
    for flag in ['yes', 'no']:
        sub = df_cs[df_cs['company_specific_factor'] == flag]
        m = compute_metrics(sub)
        m['company_specific'] = flag
        rows.append(m)
    t3 = pd.DataFrame(rows)
    print(t3.to_string(index=False))
    t3.to_csv('phase1_artifacts/03_company_specific.csv', index=False)
else:
    print("  No enriched labels available")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4: By D+1 Return Bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 4: BY D+1 RETURN BUCKET")
print("=" * 80)
bins_d1 = [-999, -0.02, 0, 0.01, 0.02, 0.04, 999]
labels_d1 = ['<-2%', '-2 to 0', '0 to 1%', '1 to 2%', '2 to 4%', '4%+']
t4 = univariate_table(df.dropna(subset=['d1_return']), 'd1_return', bins_d1, labels_d1)
print(t4.to_string(index=False))
t4.to_csv('phase1_artifacts/04_d1_return_bucket.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5: By D+1 Close vs VWAP Bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 5: BY D+1 CLOSE VS VWAP BUCKET")
print("=" * 80)
bins_vwap = [-999, -0.01, 0, 0.01, 999]
labels_vwap = ['<-1%', '-1 to 0', '0 to 1%', '1%+']
sub5 = df.dropna(subset=['d1_close_vs_vwap'])
t5 = univariate_table(sub5, 'd1_close_vs_vwap', bins_vwap, labels_vwap)
print(t5.to_string(index=False))
t5.to_csv('phase1_artifacts/05_d1_close_vs_vwap.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 6: By VIX Bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 6: BY VIX BUCKET")
print("=" * 80)
bins_vix = [0, 15, 20, 25, 30, 40, 999]
labels_vix = ['<15', '15-20', '20-25', '25-30', '30-40', '40+']
sub6 = df.dropna(subset=['vix_close'])
t6 = univariate_table(sub6, 'vix_close', bins_vix, labels_vix)
print(t6.to_string(index=False))
t6.to_csv('phase1_artifacts/06_vix_bucket.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 7: By D+0 Close-in-Range Bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 7: BY D+0 CLOSE-IN-RANGE BUCKET")
print("=" * 80)
bins_cir = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels_cir = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
sub7 = df.dropna(subset=['d0_close_in_range'])
t7 = univariate_table(sub7, 'd0_close_in_range', bins_cir, labels_cir)
print(t7.to_string(index=False))
t7.to_csv('phase1_artifacts/07_d0_close_in_range.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 8: By Drop Size
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 8: BY DROP SIZE")
print("=" * 80)
bins_drop = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 1.0]
labels_drop = ['3-4%', '4-5%', '5-6%', '6-8%', '8-10%', '10-15%', '15%+']
t8 = univariate_table(df, 'drop_pct', bins_drop, labels_drop)
print(t8.to_string(index=False))
t8.to_csv('phase1_artifacts/08_drop_size.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 9: Bad-Loser Lift Table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 9: BAD-LOSER LIFT TABLE (what's overrepresented among losers)")
print("=" * 80)
losers = df[df['bad_loser']]
rows = []

# Event type lift
for et in df['stock_event_type'].dropna().unique():
    share_losers = (losers['stock_event_type'] == et).mean()
    share_all = (df['stock_event_type'] == et).mean()
    if share_all > 0:
        rows.append({'feature': f'event_type={et}', 'loser_share': share_losers, 'overall_share': share_all, 'lift': share_losers / share_all})

# Severity lift
for sev in df['stock_event_severity'].dropna().unique():
    share_losers = (losers['stock_event_severity'] == sev).mean()
    share_all = (df['stock_event_severity'] == sev).mean()
    if share_all > 0:
        rows.append({'feature': f'severity={sev}', 'loser_share': share_losers, 'overall_share': share_all, 'lift': share_losers / share_all})

# VIX bucket lift
for lo, hi, label in [(0,15,'vix<15'), (15,20,'vix15-20'), (20,25,'vix20-25'), (25,30,'vix25-30'), (30,40,'vix30-40'), (40,999,'vix40+')]:
    mask_all = (df['vix_close'] >= lo) & (df['vix_close'] < hi)
    mask_losers = (losers['vix_close'] >= lo) & (losers['vix_close'] < hi)
    share_all = mask_all.mean()
    share_losers = mask_losers.mean()
    if share_all > 0:
        rows.append({'feature': label, 'loser_share': share_losers, 'overall_share': share_all, 'lift': share_losers / share_all})

# D+1 return sign
for cond, label in [(df['d1_return'] < 0, 'd1_return<0'), (df['d1_return'] >= 0, 'd1_return>=0')]:
    share_all = cond.mean()
    share_losers = (losers['d1_return'] < 0).mean() if 'd1_return<0' == label else (losers['d1_return'] >= 0).mean()
    if share_all > 0:
        rows.append({'feature': label, 'loser_share': share_losers, 'overall_share': share_all, 'lift': share_losers / share_all})

t9 = pd.DataFrame(rows).sort_values('lift', ascending=False)
t9['lift'] = t9['lift'].round(2)
print(t9.to_string(index=False))
t9.to_csv('phase1_artifacts/09_bad_loser_lift.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP 1: Event Type × D+1 Return Bucket → good_trade rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("HEATMAP 1: EVENT TYPE × D+1 RETURN → GOOD_TRADE RATE")
print("=" * 80)
df['d1_return_bin'] = pd.cut(df['d1_return'], bins=bins_d1, labels=labels_d1)
ct1 = crosstab_table(df.dropna(subset=['d1_return_bin', 'stock_event_type']), 'stock_event_type', 'd1_return_bin', 'good_trade')
if len(ct1) > 0:
    pivot1 = ct1.pivot_table(index='stock_event_type', columns='d1_return_bin', values='good_trade')
    print(pivot1.round(3).to_string())
    pivot1.to_csv('phase1_artifacts/heatmap1_eventtype_x_d1return_goodtrade.csv')
    # Also save counts
    pivot1c = ct1.pivot_table(index='stock_event_type', columns='d1_return_bin', values='count')
    pivot1c.to_csv('phase1_artifacts/heatmap1_eventtype_x_d1return_counts.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP 2: Event Type × D+1 Close vs VWAP → ugly_loser_rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("HEATMAP 2: EVENT TYPE × D+1 CLOSE VS VWAP → UGLY LOSER RATE")
print("=" * 80)
df['d1_vwap_bin'] = pd.cut(df['d1_close_vs_vwap'], bins=bins_vwap, labels=labels_vwap)
ct2 = crosstab_table(df.dropna(subset=['d1_vwap_bin', 'stock_event_type']), 'stock_event_type', 'd1_vwap_bin', 'good_trade')
if len(ct2) > 0:
    pivot2 = ct2.pivot_table(index='stock_event_type', columns='d1_vwap_bin', values='ugly_loser_rate')
    print(pivot2.round(3).to_string())
    pivot2.to_csv('phase1_artifacts/heatmap2_eventtype_x_d1vwap_uglyloser.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP 3: VIX Bucket × D+1 Return → good_trade rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("HEATMAP 3: VIX BUCKET × D+1 RETURN → GOOD_TRADE RATE")
print("=" * 80)
df['vix_bin'] = pd.cut(df['vix_close'], bins=bins_vix, labels=labels_vix)
ct3 = crosstab_table(df.dropna(subset=['vix_bin', 'd1_return_bin']), 'vix_bin', 'd1_return_bin', 'good_trade')
if len(ct3) > 0:
    pivot3 = ct3.pivot_table(index='vix_bin', columns='d1_return_bin', values='good_trade')
    print(pivot3.round(3).to_string())
    pivot3.to_csv('phase1_artifacts/heatmap3_vix_x_d1return_goodtrade.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 10: By Year (stability check)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 10: BY YEAR")
print("=" * 80)
rows = []
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    m = compute_metrics(sub)
    m['year'] = yr
    rows.append(m)
t10 = pd.DataFrame(rows)[['year', 'count', 'hit_5_60', 'good_trade', 'ugly_loser_rate', 'median_days_to_5']]
print(t10.to_string(index=False))
t10.to_csv('phase1_artifacts/10_by_year.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 11: SPY 5d Return Bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 11: BY SPY 5D RETURN BUCKET")
print("=" * 80)
bins_spy = [-999, -0.08, -0.04, -0.02, 0, 0.02, 999]
labels_spy = ['<=-8%', '-8 to -4', '-4 to -2', '-2 to 0', '0 to 2', '2%+']
sub11 = df.dropna(subset=['spy_5d_return'])
t11 = univariate_table(sub11, 'spy_5d_return', bins_spy, labels_spy)
print(t11.to_string(index=False))
t11.to_csv('phase1_artifacts/11_spy_5d_return.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 12: HYG 5d Return Bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 12: BY HYG 5D RETURN BUCKET")
print("=" * 80)
bins_hyg = [-999, -0.02, -0.01, 0, 0.005, 999]
labels_hyg = ['<-2%', '-2 to -1', '-1 to 0', '0 to 0.5%', '0.5%+']
sub12 = df.dropna(subset=['hyg_5d_return'])
t12 = univariate_table(sub12, 'hyg_5d_return', bins_hyg, labels_hyg)
print(t12.to_string(index=False))
t12.to_csv('phase1_artifacts/12_hyg_5d_return.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 13: D+0 Relative Volume
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 13: BY D+0 RELATIVE VOLUME")
print("=" * 80)
bins_rvol = [0, 1.0, 1.5, 2.0, 3.0, 5.0, 999]
labels_rvol = ['<1x', '1-1.5x', '1.5-2x', '2-3x', '3-5x', '5x+']
sub13 = df.dropna(subset=['d0_rvol'])
t13 = univariate_table(sub13, 'd0_rvol', bins_rvol, labels_rvol)
print(t13.to_string(index=False))
t13.to_csv('phase1_artifacts/13_d0_relative_volume.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 14: D+0 Lower Wick
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 14: BY D+0 LOWER WICK")
print("=" * 80)
bins_lw = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
labels_lw = ['0-10%', '10-20%', '20-30%', '30-50%', '50%+']
sub14 = df.dropna(subset=['d0_lower_wick'])
t14 = univariate_table(sub14, 'd0_lower_wick', bins_lw, labels_lw)
print(t14.to_string(index=False))
t14.to_csv('phase1_artifacts/14_d0_lower_wick.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 15: D+2 Return
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 15: BY D+2 FIRST HOUR RETURN")
print("=" * 80)
bins_d2 = [-999, -0.02, -0.01, 0, 0.01, 0.02, 999]
labels_d2 = ['<-2%', '-2 to -1%', '-1 to 0', '0 to 1%', '1 to 2%', '2%+']
sub15 = df.dropna(subset=['d2_first_hour_ret'])
t15 = univariate_table(sub15, 'd2_first_hour_ret', bins_d2, labels_d2)
print(t15.to_string(index=False))
t15.to_csv('phase1_artifacts/15_d2_first_hour_return.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"ALL ARTIFACTS SAVED TO phase1_artifacts/")
print(f"Files: {sorted(os.listdir('phase1_artifacts'))}")
print("=" * 80)
