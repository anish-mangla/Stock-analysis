"""
Run Screener Backtest
======================
Loads the forward-path data, runs the expert's screener, and outputs results.
"""

import pandas as pd
import numpy as np
from screener_v1 import (
    score_event, size_multiplier_from_tier, should_exit_position,
    BacktestConfig, compute_good_trade, compute_ugly_loser,
    boolish, to_decimal, safe_get, get_day_path,
    COL_SUCCESS, COL_FINAL_RETURN, COL_MAX_DRAWDOWN, COL_D2_FH, COL_D1_VWAP,
    DATE_COL, ensure_datetime
)

# ─── Load data ───
print("Loading events_with_forward_path.parquet...")
df = pd.read_parquet('outputs/events_with_forward_path.parquet')
df = ensure_datetime(df, DATE_COL)
print(f"  {len(df)} events × {len(df.columns)} columns")

# ─── Annotate labels ───
df['success_label'] = df[COL_SUCCESS].apply(boolish)
df['good_trade'] = df.apply(
    lambda r: compute_good_trade(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['ugly_loser'] = df.apply(
    lambda r: compute_ugly_loser(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['year'] = df[DATE_COL].dt.year

# ─── Run screener on all events ───
print("\nRunning screener on all events...")
scores = []
for idx, row in df.iterrows():
    sr = score_event(row)
    scores.append({'score': sr.score, 'tier': sr.tier, 'veto_reason': sr.veto_reason})

scores_df = pd.DataFrame(scores, index=df.index)
df = df.join(scores_df)

print("\nTier distribution:")
print(df['tier'].value_counts())

# ─── Run backtest for entered trades ───
print("\nRunning backtest for entered trades...")
config = BacktestConfig(
    profit_target=0.05,
    partial_profit_target=0.03,
    stop_loss=-0.08,
    ugly_loss_stop=-0.12,
    max_hold_days=60,
    use_partial_profit=True,
    partial_fraction=0.50,
    exit_on_d2_failure=True,
    exit_on_lower_low_after_entry=True,
)

entered = df[df['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])].copy()
print(f"  {len(entered)} trades entered out of {len(df)} events ({len(entered)/len(df)*100:.1f}% coverage)")

results = []
for idx, row in entered.iterrows():
    size_mult = size_multiplier_from_tier(row['tier'], row['score'])
    running_state = {"partial_taken": False, "partial_day": None, "partial_ret": 0.0}
    
    exit_day = None
    exit_reason = None
    exit_ret = None
    
    for day in range(2, config.max_hold_days + 1):
        decision = should_exit_position(row, day, row['tier'], config, running_state)
        if decision.exit_now:
            exit_day = day
            exit_reason = decision.reason
            exit_ret = decision.exit_ret
            break
    
    if exit_day is None:
        exit_day = config.max_hold_days
        exit_reason = "max_hold"
        path = get_day_path(row, config.max_hold_days)
        exit_ret = path["close_ret"] if not pd.isna(path["close_ret"]) else 0.0
    
    partial_taken = running_state["partial_taken"]
    partial_ret = running_state["partial_ret"] if partial_taken else 0.0
    
    if partial_taken:
        remainder_weight = 1.0 - config.partial_fraction
        gross_ret = config.partial_fraction * partial_ret + remainder_weight * (exit_ret if exit_ret is not None else 0.0)
    else:
        gross_ret = exit_ret if exit_ret is not None else 0.0
    
    net_return = gross_ret * size_mult - 0.0005 * size_mult  # fee
    
    results.append({
        'idx': idx,
        'tier': row['tier'],
        'score': row['score'],
        'size_mult': size_mult,
        'exit_day': exit_day,
        'exit_reason': exit_reason,
        'gross_return': gross_ret,
        'net_return': net_return,
        'partial_taken': partial_taken,
    })

bt = pd.DataFrame(results).set_index('idx')
entered = entered.join(bt[['exit_day', 'exit_reason', 'gross_return', 'net_return', 'partial_taken', 'size_mult']])

# ─── Results ───
print("\n" + "=" * 80)
print("SCREENER BACKTEST RESULTS")
print("=" * 80)

# Overall
print(f"\nTotal events: {len(df)}")
print(f"Entered trades: {len(entered)} ({len(entered)/len(df)*100:.1f}% coverage)")
print(f"Vetoed: {(df['tier']=='VETO').sum()} ({(df['tier']=='VETO').mean()*100:.1f}%)")
print(f"NO (not enough confirmation): {(df['tier']=='NO').sum()} ({(df['tier']=='NO').mean()*100:.1f}%)")

# By tier
print("\n" + "-" * 80)
print("BY TIER:")
print("-" * 80)
for tier in ['CONFIDENT_YES', 'SMALLER_YES']:
    sub = entered[entered['tier'] == tier]
    if len(sub) == 0:
        continue
    hit = sub['success_label'].astype(float).mean()
    good = sub['good_trade'].astype(float).mean()
    ugly = sub['ugly_loser'].astype(float).mean()
    avg_ret = sub['net_return'].mean()
    med_ret = sub['net_return'].median()
    print(f"\n  {tier} (n={len(sub)}, coverage={len(sub)/len(df)*100:.1f}%)")
    print(f"    Hit rate (eventually +5%): {hit*100:.1f}%")
    print(f"    Good trade rate: {good*100:.1f}%")
    print(f"    Ugly loser rate: {ugly*100:.1f}%")
    print(f"    Avg net return: {avg_ret*100:.2f}%")
    print(f"    Median net return: {med_ret*100:.2f}%")
    print(f"    Exit reasons: {sub['exit_reason'].value_counts().to_dict()}")

# By year
print("\n" + "-" * 80)
print("BY YEAR (entered trades only):")
print("-" * 80)
print(f"{'Year':>6} {'N':>6} {'Hit%':>7} {'Good%':>7} {'Ugly%':>7} {'AvgRet':>8} {'MedRet':>8}")
for yr in sorted(entered['year'].unique()):
    sub = entered[entered['year'] == yr]
    hit = sub['success_label'].astype(float).mean()
    good = sub['good_trade'].astype(float).mean()
    ugly = sub['ugly_loser'].astype(float).mean()
    avg_ret = sub['net_return'].mean()
    med_ret = sub['net_return'].median()
    print(f"{yr:>6} {len(sub):>6} {hit*100:>6.1f}% {good*100:>6.1f}% {ugly*100:>6.1f}% {avg_ret*100:>7.2f}% {med_ret*100:>7.2f}%")

# Veto analysis
print("\n" + "-" * 80)
print("VETO REASONS:")
print("-" * 80)
vetoed = df[df['tier'] == 'VETO']
print(vetoed['veto_reason'].value_counts().to_string())

# What would have happened if we took the vetoed trades?
print("\n" + "-" * 80)
print("VETOED TRADES — WHAT WOULD HAVE HAPPENED:")
print("-" * 80)
if len(vetoed) > 0:
    v_hit = vetoed['success_label'].astype(float).mean()
    v_good = vetoed['good_trade'].astype(float).mean()
    v_ugly = vetoed['ugly_loser'].astype(float).mean()
    print(f"  N={len(vetoed)}, Hit rate: {v_hit*100:.1f}%, Good trade: {v_good*100:.1f}%, Ugly loser: {v_ugly*100:.1f}%")
    print(f"  (Confirms vetoes are working — these are much worse than entered trades)")

# Walk-forward style: test set only
print("\n" + "-" * 80)
print("WALK-FORWARD: 2024-2026 (out-of-sample if rules were derived from 2020-2023)")
print("-" * 80)
oos = entered[entered['year'] >= 2024]
if len(oos) > 0:
    for tier in ['CONFIDENT_YES', 'SMALLER_YES']:
        sub = oos[oos['tier'] == tier]
        if len(sub) == 0:
            continue
        hit = sub['success_label'].astype(float).mean()
        good = sub['good_trade'].astype(float).mean()
        ugly = sub['ugly_loser'].astype(float).mean()
        avg_ret = sub['net_return'].mean()
        print(f"  {tier} (n={len(sub)}): Hit={hit*100:.1f}%, Good={good*100:.1f}%, Ugly={ugly*100:.1f}%, AvgRet={avg_ret*100:.2f}%")

# Save results
entered.to_csv('phase1_artifacts/backtest_results.csv', index=False)
print(f"\nDetailed results saved to phase1_artifacts/backtest_results.csv")
