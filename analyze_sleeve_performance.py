"""
Detailed sleeve performance analysis.
Focus on: capital recycling, hold times, and whether speed sleeve
should use different exits in normal vs crisis regime.
"""
import pandas as pd
import numpy as np

# Load the two-sleeve results
# Re-run the best config and capture detailed trade data
from portfolio_two_sleeve import df, simulate_trade, SPEED_EXIT, CORE_EXIT, classify_speed_tier

# Get trade-level data for the speed+crisis config
# We'll analyze the trade log from the saved results
# But easier to just re-analyze the raw events

# Focus on tradable events
tradable = df[df['speed_tier'].isin(['S1_CRISIS', 'S2_SPEED', 'CORE'])].copy()

print(f"Tradable events by sleeve:")
print(tradable['speed_tier'].value_counts())
print()

# Simulate each trade with BOTH exit configs and compare
results = []
for _, row in tradable.iterrows():
    # Speed exit
    s_days, s_ret, s_reason = simulate_trade(row, SPEED_EXIT['target'], SPEED_EXIT['stop'], SPEED_EXIT['max_hold'])
    # Core exit
    c_days, c_ret, c_reason = simulate_trade(row, CORE_EXIT['target'], CORE_EXIT['stop'], CORE_EXIT['max_hold'])
    
    results.append({
        'ticker': row['ticker'],
        'event_date': row['event_date'],
        'speed_tier': row['speed_tier'],
        'score': row['score'],
        'speed_days': s_days,
        'speed_ret': s_ret,
        'speed_reason': s_reason,
        'core_days': c_days,
        'core_ret': c_ret,
        'core_reason': c_reason,
        'speed_ret_per_day': s_ret / s_days if s_days > 0 else 0,
        'core_ret_per_day': c_ret / c_days if c_days > 0 else 0,
    })

rdf = pd.DataFrame(results)

print(f"{'='*70}")
print("SLEEVE COMPARISON: Speed Exit vs Core Exit on same events")
print(f"{'='*70}")

for tier in ['S1_CRISIS', 'S2_SPEED', 'CORE']:
    g = rdf[rdf['speed_tier'] == tier]
    if len(g) == 0:
        continue
    
    print(f"\n{tier} (n={len(g)}):")
    print(f"  Speed exit (+5%/-6%/10d):")
    print(f"    Win rate: {(g['speed_ret'] > 0).mean():.1%}")
    print(f"    Avg return: {g['speed_ret'].mean():+.2%}")
    print(f"    Avg hold: {g['speed_days'].mean():.1f}d")
    print(f"    Ret/day: {g['speed_ret_per_day'].mean():+.4%}")
    
    print(f"  Core exit (+5%/-8%/60d):")
    print(f"    Win rate: {(g['core_ret'] > 0).mean():.1%}")
    print(f"    Avg return: {g['core_ret'].mean():+.2%}")
    print(f"    Avg hold: {g['core_days'].mean():.1f}d")
    print(f"    Ret/day: {g['core_ret_per_day'].mean():+.4%}")

# ─── Key question: does speed exit improve ret/day? ───
print(f"\n{'='*70}")
print("KEY METRIC: Return per day of capital deployed")
print(f"{'='*70}")

for tier in ['S1_CRISIS', 'S2_SPEED', 'CORE']:
    g = rdf[rdf['speed_tier'] == tier]
    if len(g) == 0:
        continue
    
    speed_rpd = g['speed_ret_per_day'].mean()
    core_rpd = g['core_ret_per_day'].mean()
    
    # Capital efficiency: total return / total days
    speed_total_ret = g['speed_ret'].sum()
    speed_total_days = g['speed_days'].sum()
    core_total_ret = g['core_ret'].sum()
    core_total_days = g['core_days'].sum()
    
    print(f"\n{tier}:")
    print(f"  Speed: total_ret={speed_total_ret:+.1%} over {speed_total_days} days = {speed_total_ret/speed_total_days*100:.4f}%/day")
    print(f"  Core:  total_ret={core_total_ret:+.1%} over {core_total_days} days = {core_total_ret/core_total_days*100:.4f}%/day")
    print(f"  Speed exit {'WINS' if speed_total_ret/speed_total_days > core_total_ret/core_total_days else 'LOSES'} on capital efficiency")

# ─── Test intermediate exits for S2_SPEED ───
print(f"\n{'='*70}")
print("EXIT OPTIMIZATION FOR S2_SPEED (normal regime)")
print(f"{'='*70}")

s2 = tradable[tradable['speed_tier'] == 'S2_SPEED']
exit_tests = [
    ('+3%/-4%/5d', 0.03, -0.04, 5),
    ('+4%/-5%/7d', 0.04, -0.05, 7),
    ('+5%/-6%/10d', 0.05, -0.06, 10),
    ('+5%/-6%/15d', 0.05, -0.06, 15),
    ('+5%/-8%/20d', 0.05, -0.08, 20),
    ('+5%/-8%/30d', 0.05, -0.08, 30),
    ('+5%/-8%/60d', 0.05, -0.08, 60),
]

print(f"\n{'Config':<20} {'WinRate':>8} {'AvgRet':>8} {'AvgDays':>8} {'Ret/Day':>10}")
print("-" * 60)
for name, target, stop, max_hold in exit_tests:
    rets = []
    days_list = []
    for _, row in s2.iterrows():
        d, r, reason = simulate_trade(row, target, stop, max_hold)
        rets.append(r)
        days_list.append(d)
    rets = np.array(rets)
    days_arr = np.array(days_list)
    wr = (rets > 0).mean()
    avg_ret = rets.mean()
    avg_days = days_arr.mean()
    rpd = avg_ret / avg_days if avg_days > 0 else 0
    print(f"{name:<20} {wr:>8.1%} {avg_ret:>8.2%} {avg_days:>8.1f} {rpd:>10.4%}")
