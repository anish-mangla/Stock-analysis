"""
Speed Lane S1 Sanity Checks
============================
Expert requested 3 checks before calling S1 real:
A. Year-by-year stability
B. Date concentration
C. Sector concentration
"""
import pandas as pd
import numpy as np

# Load speed features
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])

# Load forward path for outcome data
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

# Load speed labels
sl = pd.read_csv('phase1_artifacts/event_speed_labels_full.csv')
sl['event_date'] = pd.to_datetime(sl['event_date'])

# Merge speed features with speed labels
df = sf.merge(sl[['ticker', 'event_date', 'speed_label', 'days_to_hit', 'ret_per_day_5d', 'score', 'tier']],
              on=['ticker', 'event_date'], how='inner')

print(f"Total events with speed features + labels: {len(df)}")
print()

# Define S1: target_distance_over_atr20 < 0.8 AND num_droppers_3pct in [61, 93]
s1 = df[(df['target_distance_over_atr20'] < 0.8) & 
        (df['num_droppers_3pct'] >= 61) & (df['num_droppers_3pct'] <= 93)]

print(f"=" * 70)
print(f"S1 CELL: ATR < 0.8 + droppers [61,93]")
print(f"n = {len(s1)}")
print(f"FAST_WIN: {(s1['speed_label'] == 'FAST_WIN').mean():.1%}")
print(f"SLOW_WIN: {(s1['speed_label'] == 'SLOW_WIN').mean():.1%}")
print(f"LOSE: {(s1['speed_label'] == 'LOSE').mean():.1%}")
print()

# ─── CHECK A: Year-by-year stability ───
print(f"=" * 70)
print("CHECK A: Year-by-year stability")
print(f"{'Year':<8} {'n':>5} {'FAST%':>8} {'SLOW%':>8} {'LOSE%':>8} {'ret/day':>10}")
print("-" * 50)
s1['year'] = s1['event_date'].dt.year
for year, g in s1.groupby('year'):
    n = len(g)
    fast = (g['speed_label'] == 'FAST_WIN').mean()
    slow = (g['speed_label'] == 'SLOW_WIN').mean()
    lose = (g['speed_label'] == 'LOSE').mean()
    rpd = g['ret_per_day_5d'].mean()
    print(f"{year:<8} {n:>5} {fast:>8.1%} {slow:>8.1%} {lose:>8.1%} {rpd:>10.3%}")

print()

# ─── CHECK B: Date concentration ───
print(f"=" * 70)
print("CHECK B: Date concentration")
unique_dates = s1['event_date'].nunique()
total_events = len(s1)
print(f"Unique event dates: {unique_dates}")
print(f"Total events: {total_events}")
print(f"Events per unique date: {total_events / unique_dates:.1f}")
print()

date_counts = s1.groupby('event_date').size().sort_values(ascending=False)
top10_dates = date_counts.head(10)
top20_dates = date_counts.head(20)
print(f"Share from top 10 dates: {top10_dates.sum()}/{total_events} = {top10_dates.sum()/total_events:.1%}")
print(f"Share from top 20 dates: {top20_dates.sum()}/{total_events} = {top20_dates.sum()/total_events:.1%}")
print()
print("Top 10 dates:")
for dt, cnt in top10_dates.items():
    print(f"  {dt.strftime('%Y-%m-%d')}: {cnt} events")

print()

# ─── CHECK C: Sector concentration ───
print(f"=" * 70)
print("CHECK C: Sector concentration")

# Load sector info
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])
sector_map = events[['ticker', 'event_date', 'sector_etf']].drop_duplicates()
s1_sec = s1.merge(sector_map, on=['ticker', 'event_date'], how='left')

print(f"\nSector distribution in S1:")
sec_counts = s1_sec['sector_etf'].value_counts()
for sec, cnt in sec_counts.items():
    pct = cnt / len(s1_sec)
    print(f"  {sec}: {cnt} ({pct:.1%})")

print()

# ─── BONUS: Intersection with tradable set ───
print(f"=" * 70)
print("BONUS: S1 intersection with current screener tiers")
tier_counts = s1['tier'].value_counts()
for t, cnt in tier_counts.items():
    print(f"  {t}: {cnt} ({cnt/len(s1):.1%})")

print()

# ─── S2 candidates ───
print(f"=" * 70)
print("S2 CANDIDATES")

# S2a: ATR [0.8, 1.0) + droppers [61, 93]
s2a = df[(df['target_distance_over_atr20'] >= 0.8) & (df['target_distance_over_atr20'] < 1.0) &
         (df['num_droppers_3pct'] >= 61) & (df['num_droppers_3pct'] <= 93)]
print(f"\nS2a: ATR [0.8, 1.0) + droppers [61,93]")
print(f"  n = {len(s2a)}")
if len(s2a) > 0:
    print(f"  FAST: {(s2a['speed_label'] == 'FAST_WIN').mean():.1%}, LOSE: {(s2a['speed_label'] == 'LOSE').mean():.1%}")

# S2b: ATR < 0.8 + droppers [24, 60]
s2b = df[(df['target_distance_over_atr20'] < 0.8) & 
         (df['num_droppers_3pct'] >= 24) & (df['num_droppers_3pct'] <= 60)]
print(f"\nS2b: ATR < 0.8 + droppers [24,60]")
print(f"  n = {len(s2b)}")
if len(s2b) > 0:
    print(f"  FAST: {(s2b['speed_label'] == 'FAST_WIN').mean():.1%}, LOSE: {(s2b['speed_label'] == 'LOSE').mean():.1%}")
    print(f"\n  S2b year-by-year:")
    s2b_y = s2b.copy()
    s2b_y['year'] = s2b_y['event_date'].dt.year
    for year, g in s2b_y.groupby('year'):
        n = len(g)
        fast = (g['speed_label'] == 'FAST_WIN').mean()
        lose = (g['speed_label'] == 'LOSE').mean()
        print(f"    {year}: n={n}, FAST={fast:.1%}, LOSE={lose:.1%}")
