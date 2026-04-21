"""
Exit Policy Variants Comparison
================================
Runs 4 backtest variants as the expert requested to isolate screener quality from exit policy.
"""

import pandas as pd
import numpy as np
from screener_v1 import (
    score_event, size_multiplier_from_tier, get_day_path,
    compute_good_trade, compute_ugly_loser, boolish, to_decimal,
    COL_SUCCESS, COL_FINAL_RETURN, COL_MAX_DRAWDOWN, COL_D2_FH,
    DATE_COL, ensure_datetime, safe_get
)

# Load
print("Loading data...")
df = pd.read_parquet('outputs/events_with_forward_path.parquet')
df = ensure_datetime(df, DATE_COL)
df['success_label'] = df[COL_SUCCESS].apply(boolish)
df['good_trade'] = df.apply(lambda r: compute_good_trade(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['ugly_loser'] = df.apply(lambda r: compute_ugly_loser(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['year'] = df[DATE_COL].dt.year

# Score all events
print("Scoring events...")
scores = []
for idx, row in df.iterrows():
    sr = score_event(row)
    scores.append({'score': sr.score, 'tier': sr.tier})
scores_df = pd.DataFrame(scores, index=df.index)
df = df.join(scores_df)

entered = df[df['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])].copy()
print(f"Entered: {len(entered)} trades")


def run_variant(entered_df, variant_name, profit_target=0.05, stop_loss=-0.08,
                ugly_stop=-0.12, max_hold=60, use_partial=False, partial_target=0.03,
                partial_frac=0.50, d2_failure_tiers=None, lower_low_tiers=None,
                d2_failure_threshold=-0.03):
    """Run a backtest variant. d2_failure_tiers/lower_low_tiers = set of tiers where those exits apply."""
    if d2_failure_tiers is None:
        d2_failure_tiers = set()
    if lower_low_tiers is None:
        lower_low_tiers = set()
    
    results = []
    for idx, row in entered_df.iterrows():
        tier = row['tier']
        score = row['score']
        size_mult = size_multiplier_from_tier(tier, score)
        
        partial_taken = False
        partial_ret = 0.0
        exit_day = None
        exit_reason = None
        exit_ret = None
        
        for day in range(2, max_hold + 1):
            path = get_day_path(row, day)
            close_ret = path["close_ret"]
            low_ret = path["low_ret"]
            high_ret = path["high_ret"]
            
            if pd.isna(close_ret) and pd.isna(low_ret) and pd.isna(high_ret):
                continue
            
            # Ugly stop
            if not pd.isna(low_ret) and low_ret <= ugly_stop:
                exit_day, exit_reason, exit_ret = day, "ugly_loss_stop", ugly_stop
                break
            
            # Stop loss
            if not pd.isna(low_ret) and low_ret <= stop_loss:
                exit_day, exit_reason, exit_ret = day, "stop_loss", stop_loss
                break
            
            # Partial profit
            if use_partial and not partial_taken and not pd.isna(high_ret):
                if high_ret >= partial_target:
                    partial_taken = True
                    partial_ret = partial_target
            
            # Profit target
            if not pd.isna(high_ret) and high_ret >= profit_target:
                exit_day, exit_reason, exit_ret = day, "profit_target", profit_target
                break
            
            # D+2 failure (only for specified tiers)
            if tier in d2_failure_tiers and day == 2:
                d2fh = to_decimal(safe_get(row, COL_D2_FH))
                if (not pd.isna(close_ret) and close_ret < d2_failure_threshold):
                    exit_day, exit_reason, exit_ret = day, "d2_failure", close_ret
                    break
            
            # Lower-low (only for specified tiers)
            if tier in lower_low_tiers and day <= 5:
                if not pd.isna(low_ret) and low_ret < -0.04:
                    exit_day, exit_reason, exit_ret = day, "lower_low", low_ret
                    break
            
            # Time stop
            if day >= max_hold:
                exit_day, exit_reason, exit_ret = day, "max_hold", close_ret if not pd.isna(close_ret) else 0.0
                break
        
        if exit_day is None:
            exit_day = max_hold
            exit_reason = "max_hold"
            p = get_day_path(row, max_hold)
            exit_ret = p["close_ret"] if not pd.isna(p["close_ret"]) else 0.0
        
        # Compute return
        if partial_taken:
            remainder_weight = 1.0 - partial_frac
            gross = partial_frac * partial_ret + remainder_weight * (exit_ret if exit_ret else 0.0)
        else:
            gross = exit_ret if exit_ret else 0.0
        
        net = gross * size_mult - 0.0005 * size_mult
        
        results.append({
            'tier': tier, 'exit_day': exit_day, 'exit_reason': exit_reason,
            'gross_return': gross, 'net_return': net, 'size_mult': size_mult,
            'success_label': row['success_label'], 'good_trade': row['good_trade'],
            'ugly_loser': row['ugly_loser'], 'year': row['year'],
        })
    
    res_df = pd.DataFrame(results)
    return res_df


def summarize(res_df, variant_name):
    """Print summary for a variant."""
    rows = []
    for tier in ['CONFIDENT_YES', 'SMALLER_YES', 'ALL']:
        sub = res_df if tier == 'ALL' else res_df[res_df['tier'] == tier]
        if len(sub) == 0:
            continue
        rows.append({
            'variant': variant_name,
            'tier': tier,
            'n': len(sub),
            'hit_rate': sub['success_label'].astype(float).mean(),
            'good_trade': sub['good_trade'].astype(float).mean(),
            'ugly_loser': sub['ugly_loser'].astype(float).mean(),
            'avg_net_ret': sub['net_return'].mean(),
            'median_net_ret': sub['net_return'].median(),
            'pct_profit_target': (sub['exit_reason'] == 'profit_target').mean(),
            'pct_stop_loss': (sub['exit_reason'].isin(['stop_loss', 'ugly_loss_stop'])).mean(),
            'pct_max_hold': (sub['exit_reason'] == 'max_hold').mean(),
            'avg_hold_days': sub['exit_day'].mean(),
        })
    return pd.DataFrame(rows)


# ─── RUN VARIANTS ───
print("\nRunning variant 1: Pure screener baseline (target + stop + time only)...")
v1 = run_variant(entered, "v1_pure_baseline")

print("Running variant 2: Add partial profit...")
v2 = run_variant(entered, "v2_partial_profit", use_partial=True)

print("Running variant 3: Tier-specific (same as v1 for now)...")
v3 = run_variant(entered, "v3_tier_specific")

print("Running variant 4: Mild early exit for SMALLER_YES only...")
v4 = run_variant(entered, "v4_mild_early_smaller",
                 d2_failure_tiers={'SMALLER_YES'},
                 d2_failure_threshold=-0.03,
                 lower_low_tiers={'SMALLER_YES'})

# ─── SUMMARIZE ───
all_summaries = pd.concat([
    summarize(v1, "v1_pure_baseline"),
    summarize(v2, "v2_partial_profit"),
    summarize(v3, "v3_tier_specific"),
    summarize(v4, "v4_mild_early_smaller"),
], ignore_index=True)

print("\n" + "=" * 110)
print("EXIT POLICY VARIANTS COMPARISON")
print("=" * 110)

# Format for display
display = all_summaries.copy()
for col in ['hit_rate', 'good_trade', 'ugly_loser', 'avg_net_ret', 'median_net_ret', 'pct_profit_target', 'pct_stop_loss', 'pct_max_hold']:
    display[col] = display[col].map(lambda x: f"{x*100:.1f}%")
display['avg_hold_days'] = display['avg_hold_days'].map(lambda x: f"{x:.1f}")

print(display.to_string(index=False))

# By year for v1
print("\n" + "-" * 80)
print("V1 PURE BASELINE — BY YEAR:")
print("-" * 80)
print(f"{'Year':>6} {'Tier':>15} {'N':>5} {'Hit%':>7} {'Good%':>7} {'Ugly%':>7} {'AvgRet':>8} {'MedRet':>8} {'AvgHold':>8}")
for yr in sorted(v1['year'].unique()):
    for tier in ['CONFIDENT_YES', 'SMALLER_YES']:
        sub = v1[(v1['year'] == yr) & (v1['tier'] == tier)]
        if len(sub) == 0:
            continue
        print(f"{yr:>6} {tier:>15} {len(sub):>5} {sub['success_label'].astype(float).mean()*100:>6.1f}% "
              f"{sub['good_trade'].astype(float).mean()*100:>6.1f}% {sub['ugly_loser'].astype(float).mean()*100:>6.1f}% "
              f"{sub['net_return'].mean()*100:>7.2f}% {sub['net_return'].median()*100:>7.2f}% {sub['exit_day'].mean():>7.1f}")

# Save
all_summaries.to_csv('phase1_artifacts/exit_variants_comparison.csv', index=False)
v1.to_csv('phase1_artifacts/v1_pure_baseline_trades.csv', index=False)
print(f"\nSaved to phase1_artifacts/exit_variants_comparison.csv")
print(f"Saved to phase1_artifacts/v1_pure_baseline_trades.csv")
