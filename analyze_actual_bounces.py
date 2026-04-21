"""
Work backwards from actual price paths.
For each drop, what was the max high reached within 3/5/7/10 days?
Then figure out what target would have captured most of them.
"""
import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20']], on=['ticker', 'event_date'], how='left')

# Remove crisis windows
clusters = [
    ('2020-02-13', '2020-03-18'), ('2022-04-20', '2022-05-18'),
    ('2022-09-13', '2022-10-07'), ('2025-02-18', '2025-03-13'),
    ('2022-06-01', '2022-06-17'), ('2022-03-29', '2022-04-13'),
    ('2022-01-05', '2022-01-21'), ('2025-03-21', '2025-04-08'),
]
mask = pd.Series(False, index=fp.index)
for s, e in clusters:
    mask |= (fp['event_date'] >= s) & (fp['event_date'] <= e)
clean = fp[~mask].copy()
clean['atr_pct'] = 0.05 / clean['target_distance_over_atr20']

# Compute max intraday high within N days (from D+2 onward, entry at D+1 close)
for window in [3, 5, 7, 10, 15]:
    cols = [f'd{d}_high_ret' for d in range(2, window + 2)]
    existing = [c for c in cols if c in clean.columns]
    clean[f'max_high_{window}d'] = clean[existing].max(axis=1)

# Also compute max drawdown (worst low) within those windows
for window in [3, 5, 7, 10]:
    cols = [f'd{d}_low_ret' for d in range(2, window + 2)]
    existing = [c for c in cols if c in clean.columns]
    clean[f'max_low_{window}d'] = clean[existing].min(axis=1)

print(f"Clean set: {len(clean)} events")
print()

# ═══════════════════════════════════════════════════════════
# PART 1: What does the max high look like within each window?
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: MAX INTRADAY HIGH REACHED (from entry at D+1 close)")
print("=" * 70)

for window in [3, 5, 7, 10, 15]:
    col = f'max_high_{window}d'
    vals = clean[col].dropna()
    print(f"\n  Within {window} days (n={len(vals)}):")
    print(f"    mean: {vals.mean():+.2%}, median: {vals.median():+.2%}")
    for pct in [10, 25, 50, 75, 90, 95]:
        print(f"    p{pct}: {vals.quantile(pct/100):+.2%}")
    print(f"    % reaching +1%: {(vals >= 0.01).mean():.1%}")
    print(f"    % reaching +2%: {(vals >= 0.02).mean():.1%}")
    print(f"    % reaching +3%: {(vals >= 0.03).mean():.1%}")
    print(f"    % reaching +5%: {(vals >= 0.05).mean():.1%}")


# ═══════════════════════════════════════════════════════════
# PART 2: Same but by ATR bucket — what do different vol stocks actually do?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: MAX HIGH WITHIN 5 DAYS — BY ATR BUCKET")
print("This tells us what target each vol bucket can realistically hit")
print("=" * 70)

atr_buckets = [(0, 0.5, '<0.5'), (0.5, 0.8, '0.5-0.8'), (0.8, 1.3, '0.8-1.3'),
               (1.3, 2.0, '1.3-2.0'), (2.0, 999, '2.0+')]

print(f"\n{'Bucket':>10} {'n':>5} {'median':>8} {'p25':>8} {'p75':>8} {'>=1%':>7} {'>=2%':>7} {'>=3%':>7} {'>=5%':>7} {'>=7%':>7} {'>=10%':>7}")
print('-' * 95)
for lo, hi, label in atr_buckets:
    sub = clean[(clean['target_distance_over_atr20'] >= lo) & (clean['target_distance_over_atr20'] < hi)]
    vals = sub['max_high_5d'].dropna()
    if len(vals) < 20: continue
    print(f"{label:>10} {len(vals):>5} {vals.median():>+8.2%} {vals.quantile(0.25):>+8.2%} {vals.quantile(0.75):>+8.2%} "
          f"{(vals>=0.01).mean():>7.1%} {(vals>=0.02).mean():>7.1%} {(vals>=0.03).mean():>7.1%} "
          f"{(vals>=0.05).mean():>7.1%} {(vals>=0.07).mean():>7.1%} {(vals>=0.10).mean():>7.1%}")

# ═══════════════════════════════════════════════════════════
# PART 3: What target captures the MEDIAN bounce for each bucket?
# And what % of events would that target catch?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: OPTIMAL TARGET = PERCENTILE OF ACTUAL MAX HIGH")
print("If we set target at p25 of max_high, we catch 75% of events")
print("If we set target at p50 (median), we catch 50%")
print("=" * 70)

for lo, hi, label in atr_buckets:
    sub = clean[(clean['target_distance_over_atr20'] >= lo) & (clean['target_distance_over_atr20'] < hi)]
    vals = sub['max_high_5d'].dropna()
    if len(vals) < 20: continue
    
    print(f"\n  ATR {label} (n={len(vals)}):")
    print(f"  {'Percentile':>12} {'Target':>8} {'Would catch':>12}")
    print(f"  {'-'*35}")
    for pct in [20, 25, 30, 40, 50, 60, 75]:
        target_val = vals.quantile(pct / 100)
        catch_pct = (vals >= target_val).mean()
        print(f"  p{pct:>10} {target_val:>+8.2%} {catch_pct:>12.1%}")

# ═══════════════════════════════════════════════════════════
# PART 4: Same by drop size
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: MAX HIGH WITHIN 5 DAYS — BY DROP SIZE")
print("=" * 70)

drop_buckets = [(0.03, 0.04, '3-4%'), (0.04, 0.05, '4-5%'), (0.05, 0.07, '5-7%'),
                (0.07, 0.10, '7-10%'), (0.10, 999, '10%+')]

print(f"\n{'Drop':>8} {'n':>5} {'median':>8} {'p25':>8} {'p75':>8} {'>=1%':>7} {'>=2%':>7} {'>=3%':>7} {'>=5%':>7}")
print('-' * 70)
for dlo, dhi, dlabel in drop_buckets:
    sub = clean[(clean['drop_pct'] >= dlo) & (clean['drop_pct'] < dhi)]
    vals = sub['max_high_5d'].dropna()
    if len(vals) < 20: continue
    print(f"{dlabel:>8} {len(vals):>5} {vals.median():>+8.2%} {vals.quantile(0.25):>+8.2%} {vals.quantile(0.75):>+8.2%} "
          f"{(vals>=0.01).mean():>7.1%} {(vals>=0.02).mean():>7.1%} {(vals>=0.03).mean():>7.1%} {(vals>=0.05).mean():>7.1%}")

# ═══════════════════════════════════════════════════════════
# PART 5: The money question — if we set target at p25 of each bucket,
# what's the overall hit rate and return?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: ADAPTIVE TARGET = p25 OF EACH ATR BUCKET'S MAX HIGH")
print("This should catch ~75% of events in each bucket")
print("=" * 70)

# Compute p25 target for each ATR bucket
bucket_targets = {}
for lo, hi, label in atr_buckets:
    sub = clean[(clean['target_distance_over_atr20'] >= lo) & (clean['target_distance_over_atr20'] < hi)]
    vals = sub['max_high_5d'].dropna()
    if len(vals) >= 20:
        bucket_targets[label] = vals.quantile(0.25)

print("\nBucket targets (p25 of max_high_5d):")
for label, target in bucket_targets.items():
    print(f"  ATR {label}: target = {target:+.2%}")

# Now simulate with these targets
def get_atr_label(atr):
    if pd.isna(atr): return None
    if atr < 0.5: return '<0.5'
    if atr < 0.8: return '0.5-0.8'
    if atr < 1.3: return '0.8-1.3'
    if atr < 2.0: return '1.3-2.0'
    return '2.0+'

def simulate(row, target, stop, max_hold):
    for day in range(2, min(max_hold + 2, 61)):
        h = row.get(f'd{day}_high_ret', np.nan)
        l = row.get(f'd{day}_low_ret', np.nan)
        c = row.get(f'd{day}_close_ret', np.nan)
        if pd.isna(h) and pd.isna(l) and pd.isna(c): continue
        if not pd.isna(l) and float(l) <= stop: return day - 1, stop, 'stop'
        if not pd.isna(h) and float(h) >= target: return day - 1, target, 'target'
        if day >= max_hold + 1: return day - 1, float(c) if not pd.isna(c) else 0.0, 'time'
    return max_hold, 0.0, 'time'

# Test different percentile targets
for pct_name, pct_val in [('p20', 0.20), ('p25', 0.25), ('p30', 0.30), ('p33', 0.33), ('p40', 0.40)]:
    # Compute bucket targets at this percentile
    bt = {}
    for lo, hi, label in atr_buckets:
        sub = clean[(clean['target_distance_over_atr20'] >= lo) & (clean['target_distance_over_atr20'] < hi)]
        vals = sub['max_high_5d'].dropna()
        if len(vals) >= 20:
            bt[label] = max(0.01, vals.quantile(pct_val))
    
    rets = []; days = []; tgts = 0; stops = 0
    for _, row in clean.iterrows():
        label = get_atr_label(row['target_distance_over_atr20'])
        target = bt.get(label, 0.03)
        target = max(0.01, target)
        d, r, reason = simulate(row, target, -0.08, 5)
        rets.append(r); days.append(d)
        if reason == 'target': tgts += 1
        if reason == 'stop': stops += 1
    
    rets = np.array(rets); days_arr = np.array(days)
    avg_target = np.mean([bt.get(get_atr_label(a), 0.03) for a in clean['target_distance_over_atr20']])
    print(f"\n  {pct_name} targets (avg target={avg_target:.2%}):")
    print(f"    Hit target: {tgts/len(rets):.1%}, Hit stop: {stops/len(rets):.1%}, Time out: {1-tgts/len(rets)-stops/len(rets):.1%}")
    print(f"    Win rate: {(rets>0).mean():.1%}, Avg ret: {rets.mean():+.2%}, Avg days: {days_arr.mean():.1f}, Ret/day: {rets.mean()/days_arr.mean():+.4%}")
    for label in ['<0.5', '0.5-0.8', '0.8-1.3', '1.3-2.0', '2.0+']:
        if label in bt:
            print(f"      ATR {label}: target={bt[label]:+.2%}")
