"""
Optimal profit target analysis on clean set.
Goal: find the target that maximizes return per day of capital deployed.

Key insight: a stock that normally moves 3%/day should have a different
target than one that moves 0.5%/day. ATR is the natural scaling factor.
"""
import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20']],
              on=['ticker', 'event_date'], how='left')

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
print(f"Clean set: {len(clean)} events")

def simulate(row, target, stop, max_hold):
    """Returns (hold_days, exit_return, reason)"""
    for day in range(2, min(max_hold + 2, 61)):
        h = row.get(f'd{day}_high_ret', np.nan)
        l = row.get(f'd{day}_low_ret', np.nan)
        c = row.get(f'd{day}_close_ret', np.nan)
        if pd.isna(h) and pd.isna(l) and pd.isna(c):
            continue
        if not pd.isna(l) and float(l) <= stop:
            return day - 1, stop, 'stop'
        if not pd.isna(h) and float(h) >= target:
            return day - 1, target, 'target'
        if day >= max_hold + 1:
            return day - 1, float(c) if not pd.isna(c) else 0.0, 'time'
    return max_hold, 0.0, 'time'


# ═══════════════════════════════════════════════════════════
# PART 1: Fixed targets — what's the best flat target?
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("PART 1: FIXED TARGET COMPARISON (stop=-8%, max_hold=20d)")
print(f"{'='*75}")

print(f"\n{'Target':>8} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10} {'Exits: tgt/stop/time':>22}")
print('-' * 70)
for target in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    rets = []; days = []; exits = {'target': 0, 'stop': 0, 'time': 0}
    for _, row in clean.iterrows():
        d, r, reason = simulate(row, target, -0.08, 20)
        rets.append(r); days.append(d); exits[reason] += 1
    rets = np.array(rets); days_arr = np.array(days)
    wr = (rets > 0).mean()
    avg_ret = rets.mean()
    avg_days = days_arr.mean()
    rpd = avg_ret / avg_days if avg_days > 0 else 0
    print(f"{target:>+7.0%} {wr:>8.1%} {avg_ret:>+8.2%} {avg_days:>8.1f} {rpd:>+10.4%}  {exits['target']:>5}/{exits['stop']:>5}/{exits['time']:>5}")

# ═══════════════════════════════════════════════════════════
# PART 2: ATR-scaled targets
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("PART 2: ATR-SCALED TARGET (target = N × ATR20 daily range)")
print("Instead of fixed +5%, set target relative to stock's normal volatility")
print(f"{'='*75}")

# target_distance_over_atr20 = (0.05 * d1_close) / atr20
# So atr20_pct = 0.05 / target_distance_over_atr20 (approx daily range as % of price)
clean['atr_pct'] = 0.05 / clean['target_distance_over_atr20']  # approximate daily ATR as % of price

print(f"\n{'ATR mult':>10} {'Target':>8} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10}")
print('-' * 58)
for atr_mult in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    rets = []; days = []; n_valid = 0
    for _, row in clean.iterrows():
        atr_pct = row.get('atr_pct', np.nan)
        if pd.isna(atr_pct) or atr_pct <= 0:
            continue
        target = atr_pct * atr_mult
        target = max(0.01, min(target, 0.20))  # clamp between 1% and 20%
        d, r, reason = simulate(row, target, -0.08, 20)
        rets.append(r); days.append(d); n_valid += 1
    rets = np.array(rets); days_arr = np.array(days)
    wr = (rets > 0).mean()
    avg_ret = rets.mean()
    avg_days = days_arr.mean()
    rpd = avg_ret / avg_days if avg_days > 0 else 0
    avg_target = clean['atr_pct'].dropna().mean() * atr_mult
    print(f"{atr_mult:>10.1f}x {avg_target:>7.1%} {wr:>8.1%} {avg_ret:>+8.2%} {avg_days:>8.1f} {rpd:>+10.4%}  (n={n_valid})")

# ═══════════════════════════════════════════════════════════
# PART 3: ATR-scaled by ATR bucket (see if different stocks need different multipliers)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("PART 3: OPTIMAL ATR MULTIPLIER BY ATR BUCKET")
print("For each volatility bucket, which multiplier maximizes ret/day?")
print(f"{'='*75}")

atr_buckets = [(0, 0.5, '<0.5 (very volatile)'), (0.5, 0.8, '0.5-0.8'), 
               (0.8, 1.3, '0.8-1.3'), (1.3, 2.0, '1.3-2.0'), (2.0, 999, '2.0+ (low vol)')]

for alo, ahi, alabel in atr_buckets:
    bucket = clean[(clean['target_distance_over_atr20'] >= alo) & (clean['target_distance_over_atr20'] < ahi)]
    if len(bucket) < 50:
        continue
    print(f"\n  ATR bucket: {alabel} (n={len(bucket)})")
    print(f"  {'Target':>10} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10}")
    print(f"  {'-'*50}")
    
    best_rpd = -999
    best_target = None
    
    # Test fixed targets
    for target in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        rets = []; days = []
        for _, row in bucket.iterrows():
            d, r, reason = simulate(row, target, -0.08, 20)
            rets.append(r); days.append(d)
        rets = np.array(rets); days_arr = np.array(days)
        wr = (rets > 0).mean()
        avg_ret = rets.mean()
        avg_days = days_arr.mean()
        rpd = avg_ret / avg_days if avg_days > 0 else 0
        marker = ' <<<' if rpd > best_rpd else ''
        if rpd > best_rpd:
            best_rpd = rpd
            best_target = target
        print(f"  {target:>+9.0%} {wr:>8.1%} {avg_ret:>+8.2%} {avg_days:>8.1f} {rpd:>+10.4%}{marker}")
    
    print(f"  BEST: {best_target:+.0%} target → {best_rpd:+.4%}/day")


# ═══════════════════════════════════════════════════════════
# PART 4: Drop-size-scaled target
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("PART 4: DROP-SIZE-SCALED TARGET")
print("Bigger drops should have bigger targets (more room to bounce)")
print(f"{'='*75}")

drop_buckets = [(0.03, 0.04, '3-4%'), (0.04, 0.05, '4-5%'), (0.05, 0.07, '5-7%'),
                (0.07, 0.10, '7-10%'), (0.10, 999, '10%+')]

for dlo, dhi, dlabel in drop_buckets:
    bucket = clean[(clean['drop_pct'] >= dlo) & (clean['drop_pct'] < dhi)]
    if len(bucket) < 50:
        continue
    print(f"\n  Drop bucket: {dlabel} (n={len(bucket)})")
    print(f"  {'Target':>10} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10}")
    print(f"  {'-'*50}")
    
    best_rpd = -999
    best_target = None
    for target in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
        rets = []; days = []
        for _, row in bucket.iterrows():
            d, r, reason = simulate(row, target, -0.08, 20)
            rets.append(r); days.append(d)
        rets = np.array(rets); days_arr = np.array(days)
        wr = (rets > 0).mean()
        avg_ret = rets.mean()
        avg_days = days_arr.mean()
        rpd = avg_ret / avg_days if avg_days > 0 else 0
        marker = ' <<<' if rpd > best_rpd else ''
        if rpd > best_rpd:
            best_rpd = rpd; best_target = target
        print(f"  {target:>+9.0%} {wr:>8.1%} {avg_ret:>+8.2%} {avg_days:>8.1f} {rpd:>+10.4%}{marker}")
    print(f"  BEST: {best_target:+.0%} target → {best_rpd:+.4%}/day")

# ═══════════════════════════════════════════════════════════
# PART 5: Adaptive target — combine ATR + drop size
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("PART 5: ADAPTIVE TARGET STRATEGIES")
print("Compare: fixed 5% vs ATR-scaled vs drop-scaled vs hybrid")
print(f"{'='*75}")

strategies = {}

# Strategy A: Fixed 5%
rets_a = []; days_a = []
for _, row in clean.iterrows():
    d, r, reason = simulate(row, 0.05, -0.08, 20)
    rets_a.append(r); days_a.append(d)
strategies['A: Fixed +5%'] = (np.array(rets_a), np.array(days_a))

# Strategy B: ATR-scaled (1.5x ATR)
rets_b = []; days_b = []
for _, row in clean.iterrows():
    atr_pct = row.get('atr_pct', np.nan)
    if pd.isna(atr_pct) or atr_pct <= 0:
        target = 0.05
    else:
        target = max(0.02, min(atr_pct * 1.5, 0.15))
    d, r, reason = simulate(row, target, -0.08, 20)
    rets_b.append(r); days_b.append(d)
strategies['B: 1.5x ATR'] = (np.array(rets_b), np.array(days_b))

# Strategy C: ATR-scaled (2x ATR)
rets_c = []; days_c = []
for _, row in clean.iterrows():
    atr_pct = row.get('atr_pct', np.nan)
    if pd.isna(atr_pct) or atr_pct <= 0:
        target = 0.05
    else:
        target = max(0.02, min(atr_pct * 2.0, 0.15))
    d, r, reason = simulate(row, target, -0.08, 20)
    rets_c.append(r); days_c.append(d)
strategies['C: 2.0x ATR'] = (np.array(rets_c), np.array(days_c))

# Strategy D: Drop-scaled (target = 50% of drop)
rets_d = []; days_d = []
for _, row in clean.iterrows():
    target = max(0.02, min(row['drop_pct'] * 0.5, 0.15))
    d, r, reason = simulate(row, target, -0.08, 20)
    rets_d.append(r); days_d.append(d)
strategies['D: 50% of drop'] = (np.array(rets_d), np.array(days_d))

# Strategy E: Drop-scaled (target = 70% of drop)
rets_e = []; days_e = []
for _, row in clean.iterrows():
    target = max(0.02, min(row['drop_pct'] * 0.7, 0.15))
    d, r, reason = simulate(row, target, -0.08, 20)
    rets_e.append(r); days_e.append(d)
strategies['E: 70% of drop'] = (np.array(rets_e), np.array(days_e))

# Strategy F: Hybrid — min(2x ATR, 60% of drop), clamped
rets_f = []; days_f = []
for _, row in clean.iterrows():
    atr_pct = row.get('atr_pct', np.nan)
    atr_target = atr_pct * 2.0 if not pd.isna(atr_pct) and atr_pct > 0 else 0.05
    drop_target = row['drop_pct'] * 0.6
    target = max(0.02, min(atr_target, drop_target, 0.15))
    d, r, reason = simulate(row, target, -0.08, 20)
    rets_f.append(r); days_f.append(d)
strategies['F: min(2xATR, 60%drop)'] = (np.array(rets_f), np.array(days_f))

# Strategy G: Hybrid — use ATR but floor at 3%, cap at drop_pct
rets_g = []; days_g = []
for _, row in clean.iterrows():
    atr_pct = row.get('atr_pct', np.nan)
    if pd.isna(atr_pct) or atr_pct <= 0:
        target = 0.05
    else:
        target = atr_pct * 1.5
        target = max(0.03, min(target, row['drop_pct'] * 0.8, 0.12))
    d, r, reason = simulate(row, target, -0.08, 20)
    rets_g.append(r); days_g.append(d)
strategies['G: 1.5xATR, floor3%, cap80%drop'] = (np.array(rets_g), np.array(days_g))

print(f"\n{'Strategy':>35} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10} {'TotalRet':>10}")
print('-' * 85)
for name, (rets, days_arr) in strategies.items():
    wr = (rets > 0).mean()
    avg_ret = rets.mean()
    avg_days = days_arr.mean()
    rpd = avg_ret / avg_days if avg_days > 0 else 0
    total = rets.sum()
    print(f"{name:>35} {wr:>8.1%} {avg_ret:>+8.2%} {avg_days:>8.1f} {rpd:>+10.4%} {total:>+10.1%}")
