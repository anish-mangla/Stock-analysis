"""
Merge ChatGPT Research Labels into Master Dataset
===================================================
Safely merges the 2200 labeled events from batch CSVs into events_fully_labeled.csv.

Steps:
1. Load all batch CSVs from outputs/stock_level_labels_v2/
2. Deduplicate (same ticker+date may appear in multiple batches)
3. Update outputs/stock_level_labels.csv with new labels
4. Update events_fully_labeled.csv with new event types + company_specific_factor
5. Validate nothing was corrupted

Safety:
- Backs up original files before overwriting
- Validates row counts match
- Checks no existing labeled data was lost
"""
import pandas as pd
import numpy as np
import glob
import os
import shutil

OUTPUT_DIR = 'outputs'

# ─── Step 1: Load all batch CSVs ───
print("Step 1: Loading all batch CSVs...")
v2_files = sorted(glob.glob('outputs/stock_level_labels_v2/batch_*_classified.csv'))
print(f"  Found {len(v2_files)} batch files")

all_batches = []
for f in v2_files:
    try:
        d = pd.read_csv(f, on_bad_lines='skip')
        d['source_file'] = os.path.basename(f)
        all_batches.append(d)
    except Exception as e:
        print(f"  ERROR reading {f}: {e}")

v2 = pd.concat(all_batches, ignore_index=True)
v2['event_date'] = pd.to_datetime(v2['event_date']).dt.strftime('%Y-%m-%d')
print(f"  Total rows from batches: {len(v2)}")
print(f"  Unique ticker+date pairs: {v2.groupby(['ticker', 'event_date']).ngroups}")

# ─── Step 2: Deduplicate ───
print("\nStep 2: Deduplicating...")
# Keep the latest batch's label if duplicated (later batches may have corrections)
v2['batch_num'] = v2['source_file'].str.extract(r'batch_(\d+)').astype(int)
v2 = v2.sort_values('batch_num', ascending=False).drop_duplicates(
    subset=['ticker', 'event_date'], keep='first'
).sort_values(['event_date', 'ticker']).reset_index(drop=True)
print(f"  After dedup: {len(v2)} unique events")

# ─── Step 3: Validate label quality ───
print("\nStep 3: Validating labels...")
print(f"  stock_event_type distribution:")
print(f"    {v2['stock_event_type'].value_counts().to_dict()}")
print(f"  stock_event_severity distribution:")
print(f"    {v2['stock_event_severity'].value_counts().to_dict()}")
print(f"  company_specific_factor distribution:")
print(f"    {v2['company_specific_factor'].value_counts().to_dict()}")

# Check for invalid values
valid_types = {
    'earnings_miss', 'guidance_cut', 'demand_weakness', 'product_service_failure',
    'competitive_threat', 'regulatory_legal', 'management_governance',
    'capital_structure', 'analyst_downgrade', 'macro_sensitivity', 'no_clear_catalyst'
}
invalid_types = set(v2['stock_event_type'].dropna().unique()) - valid_types
if invalid_types:
    print(f"  WARNING: Invalid event types found: {invalid_types}")
    # Fix common issues
    v2['stock_event_type'] = v2['stock_event_type'].replace({
        'low': 'no_clear_catalyst',  # likely a parsing error
    })
    invalid_types2 = set(v2['stock_event_type'].dropna().unique()) - valid_types
    if invalid_types2:
        print(f"  Still invalid after fix: {invalid_types2}")
        # Drop rows with invalid types
        v2 = v2[v2['stock_event_type'].isin(valid_types)]
        print(f"  Dropped invalid rows, remaining: {len(v2)}")

valid_severities = {'low', 'medium', 'high'}
invalid_sev = set(v2['stock_event_severity'].dropna().unique()) - valid_severities
if invalid_sev:
    print(f"  WARNING: Invalid severities: {invalid_sev}")

# ─── Step 4: Load and backup master file ───
print("\nStep 4: Backing up master file...")
master_path = 'outputs/events_fully_labeled.csv'
backup_path = 'outputs/events_fully_labeled_BACKUP.csv'
shutil.copy2(master_path, backup_path)
print(f"  Backed up to {backup_path}")

master = pd.read_csv(master_path)
master['event_date'] = pd.to_datetime(master['event_date']).dt.strftime('%Y-%m-%d')
original_shape = master.shape
original_labeled_count = (master['stock_event_type'] != 'unlabeled').sum()
print(f"  Master shape: {original_shape}")
print(f"  Currently labeled: {original_labeled_count}")

# ─── Step 5: Merge new labels into master ───
print("\nStep 5: Merging new labels...")

# Create lookup from v2
v2_lookup = v2.set_index(['ticker', 'event_date'])

# Build description from primary_cause if available
if 'primary_cause' in v2.columns:
    desc_map = dict(zip(
        zip(v2['ticker'], v2['event_date']),
        v2['primary_cause'].fillna('')
    ))
else:
    desc_map = {}

type_map = dict(zip(zip(v2['ticker'], v2['event_date']), v2['stock_event_type']))
sev_map = dict(zip(zip(v2['ticker'], v2['event_date']), v2['stock_event_severity']))
cs_map = dict(zip(zip(v2['ticker'], v2['event_date']), v2['company_specific_factor']))

# Update master
updated_type = 0
updated_sev = 0
updated_cs = 0

# Add company_specific_factor column if not present
if 'company_specific_factor' not in master.columns:
    master['company_specific_factor'] = np.nan

for idx, row in master.iterrows():
    key = (row['ticker'], row['event_date'])
    
    if key in type_map:
        new_type = type_map[key]
        new_sev = sev_map.get(key, 'medium')
        new_desc = desc_map.get(key, '')
        new_cs = cs_map.get(key, np.nan)
        
        # Only update if currently unlabeled OR if new data is from v2 (more recent)
        if row['stock_event_type'] == 'unlabeled' or pd.isna(row['stock_event_type']):
            master.at[idx, 'stock_event_type'] = new_type
            master.at[idx, 'stock_event_severity'] = new_sev
            if new_desc and (pd.isna(row.get('stock_event_description', np.nan)) or row.get('stock_event_description', '') == ''):
                master.at[idx, 'stock_event_description'] = new_desc
            updated_type += 1
        
        # Always update company_specific_factor (wasn't in master before)
        if pd.notna(new_cs):
            master.at[idx, 'company_specific_factor'] = new_cs
            updated_cs += 1

print(f"  Updated event types: {updated_type}")
print(f"  Updated company_specific_factor: {updated_cs}")

# ─── Step 6: Validate ───
print("\nStep 6: Validating...")
new_labeled_count = (master['stock_event_type'] != 'unlabeled').sum()
print(f"  Previously labeled: {original_labeled_count}")
print(f"  Now labeled: {new_labeled_count}")
print(f"  New labels added: {new_labeled_count - original_labeled_count}")
print(f"  Shape unchanged: {master.shape[0] == original_shape[0]} ({master.shape})")

# Verify no rows were lost
assert master.shape[0] == original_shape[0], "ROW COUNT CHANGED! Aborting."

# Verify original labels weren't overwritten
# (we only update 'unlabeled' rows)
master_check = pd.read_csv(backup_path)
master_check['event_date'] = pd.to_datetime(master_check['event_date']).dt.strftime('%Y-%m-%d')
originally_labeled = master_check[master_check['stock_event_type'] != 'unlabeled']
for _, row in originally_labeled.iterrows():
    new_row = master[(master['ticker'] == row['ticker']) & (master['event_date'] == row['event_date'])]
    if len(new_row) > 0:
        assert new_row.iloc[0]['stock_event_type'] == row['stock_event_type'], \
            f"Original label overwritten for {row['ticker']} {row['event_date']}!"

print("  All original labels preserved ✓")

# ─── Step 7: Save ───
print("\nStep 7: Saving...")
master.to_csv(master_path, index=False)
print(f"  Saved {master_path}")

# Also update stock_level_labels.csv for compatibility with build_lookup_table.py
print("\n  Updating stock_level_labels.csv...")
sl_backup = 'outputs/stock_level_labels_BACKUP.csv'
shutil.copy2('outputs/stock_level_labels.csv', sl_backup)

# Build new stock_level_labels from all labeled events
labeled = master[master['stock_event_type'] != 'unlabeled'][
    ['event_date', 'ticker', 'stock_event_type', 'stock_event_severity', 'stock_event_description']
].copy()
labeled = labeled.rename(columns={
    'event_date': 'date',
    'stock_event_type': 'event_type_name',
    'stock_event_severity': 'severity',
    'stock_event_description': 'description',
})
# Add event_type_id (not critical but keep format consistent)
type_id_map = {
    'earnings_miss': 1, 'guidance_cut': 2, 'demand_weakness': 3,
    'product_service_failure': 4, 'competitive_threat': 5, 'regulatory_legal': 6,
    'management_governance': 7, 'capital_structure': 8, 'analyst_downgrade': 9,
    'macro_sensitivity': 10, 'no_clear_catalyst': 11,
}
labeled['event_type_id'] = labeled['event_type_name'].map(type_id_map).fillna(11).astype(int)
labeled = labeled[['date', 'ticker', 'event_type_id', 'event_type_name', 'severity', 'description']]
labeled.to_csv('outputs/stock_level_labels.csv', index=False)
print(f"  Saved stock_level_labels.csv ({len(labeled)} rows)")

# ─── Summary ───
print(f"\n{'='*60}")
print("MERGE COMPLETE")
print(f"{'='*60}")
print(f"  Events labeled: {original_labeled_count} → {new_labeled_count}")
print(f"  company_specific_factor added: {updated_cs} events")
print(f"  Master file rows: {master.shape[0]} (unchanged)")
print(f"  Backups at: {backup_path}, {sl_backup}")

# Final distribution
print(f"\n  Event type distribution (all labeled):")
labeled_final = master[master['stock_event_type'] != 'unlabeled']
print(f"  {labeled_final['stock_event_type'].value_counts().to_dict()}")
print(f"\n  company_specific_factor:")
print(f"  {master['company_specific_factor'].value_counts().to_dict()}")
