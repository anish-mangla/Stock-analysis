"""
Analyze the day-by-day path of trades that end up losing.
Goal: find early exit signals to cut losses before -8%.
"""
import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20']], on=['ticker', 'event_date'], how='left')

clusters = [('2020-02-13', '2020-03-18'), ('2022-04-20', '2022-05-18'),
    ('2022-09-13', '2022-10-07'), ('2025-02-18', '2025-03-13'),
    ('2022-06-01', '2022-06-17'), ('2022-03-29', '2022-04-13'),
    ('2022-01-05', '2022-01-21'), ('2025-03-21', '2025-04-08')]
mask = pd.Series(False, index=fp.index)
for s, e in clusters: mask |= (fp['event_date'] >= s) & (fp['event_date'] <= e)
clean = fp[~mask].copy()

def get_target(atr):
    if pd.isna(atr): return 0.03
    if atr < 0.5: return 0.1216
    if atr < 0.8: return 0.0542
    if atr < 1.3: return 0.0358
    if atr < 2.0: return 0.0231
    return 0.0158

# For each event, track the full day-by-day path and whether it eventually wins
paths = []
for _, row in clean.iterrows():
    target = get_target(row['target_distance_over_atr20'])
    
    # Track close and high for each day
    day_data = []
    hit_target = False
    hit_target_day = None
    for day in range(2, 7):  # 5-day window
        c = row.get(f'd{day}_close_ret', np.nan)
        h = row.get(f'd{day}_high_ret', np.nan)
        l = row.get(f'd{day}_low_ret', np.nan)
        day_data.append({'day': day, 'close': float(c) if not pd.isna(c) else np.nan,
                         'high': float(h) if not pd.isna(h) else np.nan,
                         'low': float(l) if not pd.isna(l) else np.nan})
        if not pd.isna(h) and float(h) >= target and not hit_target:
            hit_target = True
            hit_target_day = day
    
    paths.append({
        'ticker': row['ticker'], 'date': row['event_date'],
        'target': target, 'hit_target': hit_target, 'hit_day': hit_target_day,
        'd2_close': day_data[0]['close'], 'd3_close': day_data[1]['close'],
        'd4_close': day_data[2]['close'], 'd5_close': day_data[3]['close'],
        'd6_close': day_data[4]['close'],
        'd2_high': day_data[0]['high'], 'd3_high': day_data[1]['high'],
        'd2_low': day_data[0]['low'], 'd3_low': day_data[1]['low'],
    })

pdf = pd.DataFrame(paths)
winners = pdf[pdf['hit_target']]
losers = pdf[~pdf['hit_target']]

print(f"Total: {len(pdf)}, Winners: {len(winners)} ({len(winners)/len(pdf):.1%}), Losers: {len(losers)} ({len(losers)/len(pdf):.1%})")
print()

# What do losers look like at end of each day?
print("=" * 70)
print("LOSER DAY-BY-DAY CLOSE RETURNS (these never hit target in 5 days)")
print("=" * 70)
for day_col in ['d2_close', 'd3_close', 'd4_close', 'd5_close', 'd6_close']:
    vals = losers[day_col].dropna()
    print(f"\n  {day_col} (n={len(vals)}):")
    print(f"    mean={vals.mean():+.2%}, median={vals.median():+.2%}")
    print(f"    p10={vals.quantile(0.1):+.2%}, p25={vals.quantile(0.25):+.2%}, p75={vals.quantile(0.75):+.2%}, p90={vals.quantile(0.9):+.2%}")
    print(f"    % positive: {(vals > 0).mean():.1%}")
    print(f"    % > +1%: {(vals > 0.01).mean():.1%}")
    print(f"    % < -2%: {(vals < -0.02).mean():.1%}")
    print(f"    % < -5%: {(vals < -0.05).mean():.1%}")

# Same for winners
print()
print("=" * 70)
print("WINNER DAY-BY-DAY CLOSE RETURNS (for comparison)")
print("=" * 70)
for day_col in ['d2_close', 'd3_close']:
    vals = winners[day_col].dropna()
    print(f"\n  {day_col} (n={len(vals)}):")
    print(f"    mean={vals.mean():+.2%}, median={vals.median():+.2%}")
    print(f"    % positive: {(vals > 0).mean():.1%}")
    print(f"    % < -2%: {(vals < -0.02).mean():.1%}")


# ═══════════════════════════════════════════════════════════
# KEY QUESTION: Can we tell by D+2 close if it's going to be a loser?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D+2 CLOSE AS EARLY EXIT SIGNAL")
print("If D+2 close is negative, should we exit?")
print("=" * 70)

d2_bins = [(-999, -0.05), (-0.05, -0.03), (-0.03, -0.02), (-0.02, -0.01),
           (-0.01, 0), (0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 999)]
d2_labels = ['<-5%', '-5 to -3%', '-3 to -2%', '-2 to -1%', '-1 to 0%',
             '0 to +1%', '+1 to +2%', '+2 to +3%', '+3%+']

print(f"\n{'D+2 close':>12} {'n':>5} {'Win%':>7} {'Lose%':>7} {'If exit here':>14} {'vs hold to d6':>14}")
print('-' * 65)
for (lo, hi), label in zip(d2_bins, d2_labels):
    sub = pdf[(pdf['d2_close'] >= lo) & (pdf['d2_close'] < hi)]
    if len(sub) < 20: continue
    win_pct = sub['hit_target'].mean()
    lose_pct = 1 - win_pct
    # If we exit at d2 close, our return is d2_close
    exit_here_ret = sub['d2_close'].mean()
    # If we hold to d6, what's the avg return (for losers, it's d6_close; for winners, it's target)
    hold_rets = []
    for _, row in sub.iterrows():
        if row['hit_target']:
            hold_rets.append(row['target'])
        else:
            hold_rets.append(row['d6_close'] if not pd.isna(row['d6_close']) else 0)
    hold_ret = np.mean(hold_rets)
    better = 'EXIT' if exit_here_ret > hold_ret else 'HOLD'
    print(f"{label:>12} {len(sub):>5} {win_pct:>7.1%} {lose_pct:>7.1%} {exit_here_ret:>+14.2%} {hold_ret:>+14.2%}  {better}")

# ═══════════════════════════════════════════════════════════
# Test various early exit rules
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EARLY EXIT RULES — SIMULATED ON ALL EVENTS")
print("=" * 70)

def simulate_with_exit(row, target, stop, max_hold, exit_rules):
    """Simulate with additional early exit rules."""
    for day in range(2, min(max_hold + 2, 61)):
        h = row.get(f'd{day}_high_ret', np.nan)
        l = row.get(f'd{day}_low_ret', np.nan)
        c = row.get(f'd{day}_close_ret', np.nan)
        if pd.isna(h) and pd.isna(l) and pd.isna(c): continue
        
        # Hard stop
        if not pd.isna(l) and float(l) <= stop:
            return day-1, stop, 'stop'
        # Target hit
        if not pd.isna(h) and float(h) >= target:
            return day-1, target, 'target'
        
        # Early exit rules (checked at close)
        if not pd.isna(c):
            c = float(c)
            for rule_name, rule_fn in exit_rules.items():
                if rule_fn(day, c, h, l, target):
                    return day-1, c, f'early_{rule_name}'
        
        # Time stop
        if day >= max_hold + 1:
            return day-1, float(c) if not pd.isna(c) else 0.0, 'time'
    return max_hold, 0.0, 'time'

# Define exit rules
rule_sets = {
    'Baseline (no early exit)': {},
    'Exit if D+2 close < -2%': {
        'd2_neg2': lambda day, c, h, l, t: day == 2 and c < -0.02
    },
    'Exit if D+2 close < -1%': {
        'd2_neg1': lambda day, c, h, l, t: day == 2 and c < -0.01
    },
    'Exit if D+2 close < 0': {
        'd2_neg': lambda day, c, h, l, t: day == 2 and c < 0
    },
    'Exit if any day close < -2%': {
        'any_neg2': lambda day, c, h, l, t: c < -0.02
    },
    'Exit if any day close < -3%': {
        'any_neg3': lambda day, c, h, l, t: c < -0.03
    },
    'Trailing: exit if close drops 2% from peak': {
        'trail2': lambda day, c, h, l, t: False  # need state, handle separately
    },
    'Exit if D+3 close < 0 (gave it 2 days)': {
        'd3_neg': lambda day, c, h, l, t: day == 3 and c < 0
    },
    'Exit if D+3 close < -1%': {
        'd3_neg1': lambda day, c, h, l, t: day == 3 and c < -0.01
    },
    'Tight stop -3% instead of -8%': {},  # handle separately
    'Tight stop -4%': {},
    'Tight stop -5%': {},
}

print(f"\n{'Rule':>40} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10} {'Trades':>7}")
print('-' * 85)

# Baseline
for rule_name, rules in rule_sets.items():
    if 'Tight stop' in rule_name:
        stop_val = float(rule_name.split('-')[1].replace('%', '')) / -100
        rets = []; days = []
        for _, row in clean.iterrows():
            target = get_target(row['target_distance_over_atr20'])
            d, r, reason = simulate_with_exit(row, target, stop_val, 5, {})
            rets.append(r); days.append(d)
    elif 'Trailing' in rule_name:
        # Custom trailing stop
        rets = []; days = []
        for _, row in clean.iterrows():
            target = get_target(row['target_distance_over_atr20'])
            peak_high = 0
            exited = False
            for day in range(2, 7):
                h = row.get(f'd{day}_high_ret', np.nan)
                l = row.get(f'd{day}_low_ret', np.nan)
                c = row.get(f'd{day}_close_ret', np.nan)
                if pd.isna(h) and pd.isna(l) and pd.isna(c): continue
                if not pd.isna(h): peak_high = max(peak_high, float(h))
                if not pd.isna(l) and float(l) <= -0.08:
                    rets.append(-0.08); days.append(day-1); exited = True; break
                if not pd.isna(h) and float(h) >= target:
                    rets.append(target); days.append(day-1); exited = True; break
                # Trailing: if we were up and now close is 2% below peak
                if not pd.isna(c) and peak_high > 0.01 and float(c) < peak_high - 0.02:
                    rets.append(float(c)); days.append(day-1); exited = True; break
                if day == 6:
                    rets.append(float(c) if not pd.isna(c) else 0); days.append(day-1); exited = True; break
            if not exited:
                rets.append(0); days.append(5)
    else:
        rets = []; days = []
        for _, row in clean.iterrows():
            target = get_target(row['target_distance_over_atr20'])
            d, r, reason = simulate_with_exit(row, target, -0.08, 5, rules)
            rets.append(r); days.append(d)
    
    rets = np.array(rets); days_arr = np.array(days)
    wr = (rets > 0).mean()
    avg_ret = rets.mean()
    avg_days = days_arr.mean()
    rpd = avg_ret / avg_days if avg_days > 0 else 0
    print(f"{rule_name:>40} {wr:>8.1%} {avg_ret:>+8.2%} {avg_days:>8.1f} {rpd:>+10.4%} {len(rets):>7}")
