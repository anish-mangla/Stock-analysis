"""
Test impact of merged labels on portfolio performance.
Compare: old screener scores (336 labels) vs new (2513 labels).
"""
import pandas as pd
import numpy as np
from screener_v1 import score_event

# Load updated master
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])

# Merge labels into forward path data
# The forward path has the old labels — we need to update them
label_cols = ['ticker', 'event_date', 'stock_event_type', 'stock_event_severity', 'company_specific_factor']
labels = master[label_cols].drop_duplicates()

# Drop old label columns from fp if they exist, then merge new ones
for col in ['stock_event_type', 'stock_event_severity', 'company_specific_factor']:
    if col in fp.columns:
        fp = fp.drop(columns=[col])

df = fp.merge(labels, on=['ticker', 'event_date'], how='left')
print(f"Events: {len(df)}")
print(f"Labeled: {(df['stock_event_type'] != 'unlabeled').sum()}")
print(f"company_specific_factor coverage: {df['company_specific_factor'].notna().sum()}")
print()

# Score with new labels
print("Scoring with updated labels...")
new_scores = []
for _, row in df.iterrows():
    r = score_event(row)
    new_scores.append({'new_score': r.score, 'new_tier': r.tier, 'new_veto': r.veto_reason})
new_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(new_scores)], axis=1)

# Score with OLD labels (simulate by blanking the new ones)
print("Scoring with old labels (for comparison)...")
df_old = df.copy()
# Revert to unlabeled for events that were newly labeled
newly_labeled_mask = (df_old['stock_event_type'] != 'unlabeled') & (df_old['event_date'] > '2020-01-31')
# Actually easier: just blank company_specific_factor and set event_type to unlabeled for non-original-336
backup = pd.read_csv('outputs/events_fully_labeled_BACKUP.csv')
backup['event_date'] = pd.to_datetime(backup['event_date']).dt.strftime('%Y-%m-%d')
df_old['event_date_str'] = df_old['event_date'].dt.strftime('%Y-%m-%d')

# Get original labeled keys
orig_labeled = backup[backup['stock_event_type'] != 'unlabeled'][['ticker', 'event_date']].copy()
orig_labeled['event_date'] = pd.to_datetime(orig_labeled['event_date']).dt.strftime('%Y-%m-%d')
orig_keys = set(zip(orig_labeled['ticker'], orig_labeled['event_date']))

# Blank new labels
for idx, row in df_old.iterrows():
    key = (row['ticker'], row['event_date_str'])
    if key not in orig_keys:
        df_old.at[idx, 'stock_event_type'] = 'unlabeled'
        df_old.at[idx, 'stock_event_severity'] = 'unlabeled'
        df_old.at[idx, 'company_specific_factor'] = np.nan

old_scores = []
for _, row in df_old.iterrows():
    r = score_event(row)
    old_scores.append({'old_score': r.score, 'old_tier': r.tier, 'old_veto': r.veto_reason})
old_score_df = pd.DataFrame(old_scores)

# Compare
new_df['old_score'] = old_score_df['old_score'].values
new_df['old_tier'] = old_score_df['old_tier'].values
new_df['old_veto'] = old_score_df['old_veto'].values

# ─── Tier changes ───
print(f"\n{'='*60}")
print("TIER CHANGES: Old Labels vs New Labels")
print(f"{'='*60}")

print(f"\nOld tier distribution:")
print(new_df['old_tier'].value_counts().to_dict())
print(f"\nNew tier distribution:")
print(new_df['new_tier'].value_counts().to_dict())

# Events that changed tier
changed = new_df[new_df['old_tier'] != new_df['new_tier']]
print(f"\nEvents that changed tier: {len(changed)}")
if len(changed) > 0:
    print("\nTransition matrix:")
    trans = pd.crosstab(changed['old_tier'], changed['new_tier'])
    print(trans.to_string())

# ─── Impact on tradable set ───
old_tradable = new_df[new_df['old_tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])]
new_tradable = new_df[new_df['new_tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])]
print(f"\nTradable events: {len(old_tradable)} → {len(new_tradable)} (delta: {len(new_tradable) - len(old_tradable)})")

# Events gained (weren't tradable, now are)
gained = new_df[(~new_df['old_tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])) & 
                (new_df['new_tier'].isin(['CONFIDENT_YES', 'SMALLER_YES']))]
# Events lost (were tradable, now aren't)
lost = new_df[(new_df['old_tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])) & 
              (~new_df['new_tier'].isin(['CONFIDENT_YES', 'SMALLER_YES']))]

print(f"  Gained: {len(gained)} events now tradable")
print(f"  Lost: {len(lost)} events no longer tradable")

if len(gained) > 0:
    print(f"\n  Gained events success rate: {gained['success'].mean():.1%}")
    print(f"  Gained events avg final return: {gained['final_return'].mean():+.2%}")
if len(lost) > 0:
    print(f"\n  Lost events success rate: {lost['success'].mean():.1%}")
    print(f"  Lost events avg final return: {lost['final_return'].mean():+.2%}")

# ─── Score changes ───
print(f"\n{'='*60}")
print("SCORE CHANGES")
print(f"{'='*60}")

score_diff = new_df['new_score'] - new_df['old_score']
changed_scores = new_df[score_diff != 0]
print(f"Events with score change: {len(changed_scores)}")
if len(changed_scores) > 0:
    print(f"  Mean score change: {score_diff[score_diff != 0].mean():+.2f}")
    print(f"  Score went UP: {(score_diff > 0).sum()}")
    print(f"  Score went DOWN: {(score_diff < 0).sum()}")

# ─── What drove the changes? ───
print(f"\n{'='*60}")
print("WHAT DROVE THE CHANGES?")
print(f"{'='*60}")

# company_specific_factor = 'no' gives +2 score
cs_no = new_df[new_df['company_specific_factor'] == 'no']
cs_yes = new_df[new_df['company_specific_factor'] == 'yes']
print(f"\ncompany_specific_factor='no' (macro-driven): {len(cs_no)} events")
print(f"  These get +2 score bonus in screener")
print(f"  Success rate: {cs_no['success'].mean():.1%}")
print(f"  Avg final return: {cs_no['final_return'].mean():+.2%}")

print(f"\ncompany_specific_factor='yes' (stock-specific): {len(cs_yes)} events")
print(f"  No bonus (neutral)")
print(f"  Success rate: {cs_yes['success'].mean():.1%}")
print(f"  Avg final return: {cs_yes['final_return'].mean():+.2%}")

# Severe damage events that now get vetoed
severe_veto_types = {'guidance_cut', 'demand_weakness', 'product_service_failure', 
                     'management_governance', 'regulatory_legal'}
newly_vetoed = new_df[(new_df['new_veto'] == 'severe_damage_event_type') & 
                      (new_df['old_veto'] != 'severe_damage_event_type')]
print(f"\nNewly VETO'd by severe_damage_event_type: {len(newly_vetoed)}")
if len(newly_vetoed) > 0:
    print(f"  Their success rate: {newly_vetoed['success'].mean():.1%}")
    print(f"  Their avg final return: {newly_vetoed['final_return'].mean():+.2%}")
    print(f"  (If these are bad trades, the veto is helping us)")

# ─── Quick portfolio comparison ───
print(f"\n{'='*60}")
print("QUICK PORTFOLIO IMPACT (trade-level, no capital constraints)")
print(f"{'='*60}")

def trade_stats(subset, label):
    n = len(subset)
    if n == 0:
        return
    wr = subset['success'].mean()
    avg_ret = subset['final_return'].mean()
    print(f"  {label}: n={n}, win_rate={wr:.1%}, avg_ret={avg_ret:+.2%}")

print("\nOld tradable set:")
trade_stats(old_tradable, "All")
trade_stats(old_tradable[old_tradable['old_tier'] == 'CONFIDENT_YES'], "CONFIDENT_YES")
trade_stats(old_tradable[old_tradable['old_tier'] == 'SMALLER_YES'], "SMALLER_YES")

print("\nNew tradable set:")
trade_stats(new_tradable, "All")
trade_stats(new_tradable[new_tradable['new_tier'] == 'CONFIDENT_YES'], "CONFIDENT_YES")
trade_stats(new_tradable[new_tradable['new_tier'] == 'SMALLER_YES'], "SMALLER_YES")
