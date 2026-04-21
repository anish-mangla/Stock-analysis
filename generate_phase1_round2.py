"""
Phase 1 Round 2: Intersection Tables
=====================================
Generates the exact cross-tabs the expert requested:
A) D+1 return × D+1 close-vs-VWAP
B) Same split by company-specific
C) D+1 return × HYG 5d credit regime
D) D+1 × D+1 VWAP × D+2 first-hour
E) Precision frontier table of candidate rules
"""

import pandas as pd
import numpy as np
import os
import glob

os.makedirs('phase1_artifacts', exist_ok=True)

# ─── LOAD & MERGE ───
print("Loading data...")
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])
indicators = pd.read_csv('backtest_iterations/intraday_deep/intraday_indicators_features.csv')
indicators['event_date'] = pd.to_datetime(indicators['event_date'])
market = pd.read_csv('outputs/market_daily_features.csv')
if 'Unnamed: 0' in market.columns:
    market = market.rename(columns={'Unnamed: 0': 'date'})
market['date'] = pd.to_datetime(market['date'])

df = events.merge(indicators, on=['ticker', 'event_date'], how='left', suffixes=('', '_ind'))
df = df.merge(market, left_on='event_date', right_on='date', how='left')

# Load enriched labels for company_specific
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
    df = df.merge(v2[['ticker', 'event_date', 'company_specific_factor']].drop_duplicates(),
                  on=['ticker', 'event_date'], how='left')

# ─── TARGET LABELS ───
df['hit_5_60'] = df['success'].astype(bool)
df['good_trade'] = df['hit_5_60'] & (df['max_drawdown'] > -0.08) & (df['final_return'] >= 0)
df['ugly_loser'] = (df['final_return'] < -0.10) | (df['max_drawdown'] < -0.12)
df['year'] = df['event_date'].dt.year

# ─── BINS ───
bins_d1 = [-999, -0.02, 0, 0.02, 0.04, 999]
labels_d1 = ['<-2%', '-2 to 0', '0 to 2%', '2 to 4%', '4%+']
df['d1_ret_bin'] = pd.cut(df['d1_return'], bins=bins_d1, labels=labels_d1)

bins_vwap = [-999, -0.01, 0, 0.01, 999]
labels_vwap = ['<-1%', '-1 to 0', '0 to 1%', '1%+']
df['d1_vwap_bin'] = pd.cut(df['d1_close_vs_vwap'], bins=bins_vwap, labels=labels_vwap)

bins_hyg = [-999, -0.01, 0, 0.005, 999]
labels_hyg = ['<-1%', '-1 to 0', '0 to 0.5%', '0.5%+']
df['hyg_5d_bin'] = pd.cut(df['hyg_5d_return'], bins=bins_hyg, labels=labels_hyg)

bins_d2fh = [-999, 0, 0.02, 999]
labels_d2fh = ['negative', '0 to 2%', '2%+']
df['d2_fh_bin'] = pd.cut(df['d2_first_hour_ret'], bins=bins_d2fh, labels=labels_d2fh)


def metrics(sub):
    n = len(sub)
    if n == 0:
        return None
    return {
        'count': n,
        'hit_5_60': round(sub['hit_5_60'].mean() * 100, 1),
        'good_trade': round(sub['good_trade'].mean() * 100, 1),
        'loser_rate': round((~sub['hit_5_60']).mean() * 100, 1),
        'ugly_loser': round(sub['ugly_loser'].mean() * 100, 1),
        'med_max_dd': round(sub['max_drawdown'].median() * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# A) D+1 Return × D+1 Close vs VWAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("A) D+1 RETURN × D+1 CLOSE VS VWAP → good_trade% / ugly_loser% / count")
print("=" * 90)

rows_a = []
for r in labels_d1:
    for v in labels_vwap:
        sub = df[(df['d1_ret_bin'] == r) & (df['d1_vwap_bin'] == v)]
        m = metrics(sub)
        if m:
            m['d1_return'] = r
            m['d1_vwap'] = v
            rows_a.append(m)

ta = pd.DataFrame(rows_a)[['d1_return', 'd1_vwap', 'count', 'hit_5_60', 'good_trade', 'loser_rate', 'ugly_loser', 'med_max_dd']]
print(ta.to_string(index=False))
ta.to_csv('phase1_artifacts/A_d1return_x_d1vwap.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# B) Same split by company-specific
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("B) D+1 RETURN × D+1 VWAP — SPLIT BY COMPANY-SPECIFIC")
print("=" * 90)

for cs_flag in ['yes', 'no']:
    print(f"\n  company_specific = {cs_flag}")
    print(f"  {'-'*80}")
    rows_b = []
    sub_cs = df[df['company_specific_factor'] == cs_flag]
    for r in labels_d1:
        for v in labels_vwap:
            sub = sub_cs[(sub_cs['d1_ret_bin'] == r) & (sub_cs['d1_vwap_bin'] == v)]
            m = metrics(sub)
            if m and m['count'] >= 10:
                m['d1_return'] = r
                m['d1_vwap'] = v
                rows_b.append(m)
    tb = pd.DataFrame(rows_b)
    if len(tb) > 0:
        print(tb[['d1_return', 'd1_vwap', 'count', 'hit_5_60', 'good_trade', 'ugly_loser']].to_string(index=False))
        tb.to_csv(f'phase1_artifacts/B_d1_x_vwap_company_{cs_flag}.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# C) D+1 Return × HYG 5d Credit Regime
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("C) D+1 RETURN × HYG 5D CREDIT REGIME → good_trade% / ugly_loser%")
print("=" * 90)

rows_c = []
for r in labels_d1:
    for h in labels_hyg:
        sub = df[(df['d1_ret_bin'] == r) & (df['hyg_5d_bin'] == h)]
        m = metrics(sub)
        if m and m['count'] >= 20:
            m['d1_return'] = r
            m['hyg_5d'] = h
            rows_c.append(m)

tc = pd.DataFrame(rows_c)[['d1_return', 'hyg_5d', 'count', 'hit_5_60', 'good_trade', 'loser_rate', 'ugly_loser']]
print(tc.to_string(index=False))
tc.to_csv('phase1_artifacts/C_d1return_x_hyg5d.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# D) D+1 Return × D+1 VWAP × D+2 First Hour
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("D) D+1 RETURN × D+1 VWAP × D+2 FIRST HOUR → good_trade%")
print("=" * 90)

rows_d = []
for r in labels_d1:
    for v in labels_vwap:
        for d2 in labels_d2fh:
            sub = df[(df['d1_ret_bin'] == r) & (df['d1_vwap_bin'] == v) & (df['d2_fh_bin'] == d2)]
            m = metrics(sub)
            if m and m['count'] >= 15:
                m['d1_return'] = r
                m['d1_vwap'] = v
                m['d2_first_hr'] = d2
                rows_d.append(m)

td = pd.DataFrame(rows_d)[['d1_return', 'd1_vwap', 'd2_first_hr', 'count', 'hit_5_60', 'good_trade', 'ugly_loser']]
print(td.to_string(index=False))
td.to_csv('phase1_artifacts/D_d1_x_vwap_x_d2fh.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# E) Precision Frontier — Candidate Rules
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("E) PRECISION FRONTIER — CANDIDATE RULES")
print("=" * 90)

total_n = len(df)
rules = []

def eval_rule(mask, name):
    sub = df[mask]
    m = metrics(sub)
    if m and m['count'] >= 30:
        m['rule'] = name
        m['coverage'] = round(m['count'] / total_n * 100, 1)
        # By-year stability
        yearly = []
        for yr in sorted(df['year'].unique()):
            yr_sub = sub[sub['year'] == yr]
            if len(yr_sub) >= 5:
                yearly.append(round(yr_sub['good_trade'].mean() * 100, 1))
        m['yearly_good_trade'] = str(yearly)
        rules.append(m)

# Single features
eval_rule(df['d1_return'] >= 0.04, 'D+1 ret >= 4%')
eval_rule(df['d1_return'] >= 0.02, 'D+1 ret >= 2%')
eval_rule(df['d1_close_vs_vwap'] >= 0.01, 'D+1 close > VWAP by 1%+')
eval_rule((df['d1_return'] >= 0) & (df['d1_close_vs_vwap'] >= 0), 'D+1 green + above VWAP')

# Pairs
eval_rule((df['d1_return'] >= 0.04) & (df['d1_close_vs_vwap'] >= 0.01), 'D+1 ret>=4% + VWAP>=1%')
eval_rule((df['d1_return'] >= 0.02) & (df['d1_close_vs_vwap'] >= 0.01), 'D+1 ret>=2% + VWAP>=1%')
eval_rule((df['d1_return'] >= 0.04) & (df['hyg_5d_return'] > 0), 'D+1 ret>=4% + HYG 5d>0')
eval_rule((df['d1_return'] >= 0.02) & (df['d1_close_vs_vwap'] >= 0.01) & (df['hyg_5d_return'] > 0), 'D+1>=2% + VWAP>=1% + HYG>0')

# With company-specific
eval_rule((df['d1_return'] >= 0.04) & (df['company_specific_factor'] == 'no'), 'D+1>=4% + non-company-specific')
eval_rule((df['d1_return'] >= 0.02) & (df['d1_close_vs_vwap'] >= 0.01) & (df['company_specific_factor'] == 'no'), 'D+1>=2% + VWAP>=1% + non-co-specific')

# With credit
eval_rule((df['d1_return'] >= 0.04) & (df['d1_close_vs_vwap'] >= 0.01) & (df['hyg_5d_return'] > 0), 'D+1>=4% + VWAP>=1% + HYG>0')
eval_rule((df['d1_return'] >= 0.04) & (df['d1_close_vs_vwap'] >= 0.01) & (df['hyg_5d_return'] > 0) & (df['company_specific_factor'] == 'no'), 'D+1>=4% + VWAP>=1% + HYG>0 + non-co-spec')

# Negative D+1 as veto check
eval_rule(df['d1_return'] < -0.02, 'D+1 ret < -2% (VETO check)')
eval_rule(df['d1_return'] < 0, 'D+1 ret < 0 (VETO check)')

# D+2 confirmation
eval_rule((df['d1_return'] >= 0.02) & (df['d1_close_vs_vwap'] >= 0.01) & (df['d2_first_hour_ret'] >= 0.02), 'D+1>=2% + VWAP>=1% + D+2 FH>=2%')
eval_rule((df['d1_return'] >= 0.04) & (df['d2_first_hour_ret'] >= 0), 'D+1>=4% + D+2 FH positive')

# VIX filter
eval_rule((df['d1_return'] >= 0.04) & (df['vix_close'] < 30), 'D+1>=4% + VIX<30')
eval_rule((df['d1_return'] >= 0.04) & (df['vix_close'] >= 30), 'D+1>=4% + VIX>=30')

# Drop size interaction
eval_rule((df['d1_return'] >= 0.04) & (df['drop_pct'] >= 0.05), 'D+1>=4% + drop>=5%')
eval_rule((df['d1_return'] >= 0.04) & (df['drop_pct'] < 0.05), 'D+1>=4% + drop<5%')

te = pd.DataFrame(rules)[['rule', 'count', 'coverage', 'hit_5_60', 'good_trade', 'loser_rate', 'ugly_loser', 'med_max_dd', 'yearly_good_trade']]
te = te.sort_values('loser_rate')
print(te.to_string(index=False))
te.to_csv('phase1_artifacts/E_precision_frontier.csv', index=False)


print("\n" + "=" * 90)
print(f"Round 2 artifacts saved to phase1_artifacts/")
print("=" * 90)
