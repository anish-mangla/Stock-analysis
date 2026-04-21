import pandas as pd, numpy as np, glob

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

# Get labels from batch CSVs directly (they have the richest descriptions)
v2_files = sorted(glob.glob('outputs/stock_level_labels_v2/batch_*_classified.csv'))
all_batches = []
for f in v2_files:
    try:
        d = pd.read_csv(f, on_bad_lines='skip')
        all_batches.append(d)
    except:
        pass
v2 = pd.concat(all_batches, ignore_index=True)
v2['event_date'] = pd.to_datetime(v2['event_date'])

# Also get from master for description
master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])

# Merge v2 labels into fp
label_cols_v2 = ['ticker', 'event_date', 'stock_event_type', 'stock_event_severity',
                 'company_specific_factor', 'primary_cause', 'secondary_factors']
# Dedupe v2
v2 = v2.sort_values('event_date').drop_duplicates(subset=['ticker', 'event_date'], keep='last')
v2_sub = v2[label_cols_v2].copy()

fp = fp.drop(columns=['stock_event_type', 'stock_event_severity', 'stock_event_description'], errors='ignore')
fp = fp.merge(v2_sub, on=['ticker', 'event_date'], how='left')

# Also get description from master
desc_map = dict(zip(zip(master['ticker'], master['event_date']), master['stock_event_description']))
fp['description'] = [desc_map.get((r['ticker'], r['event_date']), '') for _, r in fp.iterrows()]

def hit_1pct_5d(row):
    for day in range(2, 7):
        h = row.get(f'd{day}_high_ret', np.nan)
        if not pd.isna(h) and float(h) >= 0.01:
            return True
    return False

fp['hit'] = fp.apply(hit_1pct_5d, axis=1)

labeled = fp[fp['stock_event_type'].notna() & (fp['stock_event_type'] != 'unlabeled')]
stuck = labeled[~labeled['hit']]
bounced = labeled[labeled['hit']]

print(f"Labeled: {len(labeled)}, Stuck: {len(stuck)}, Bounced: {len(bounced)}")
print(f"Stuck rate: {len(stuck)/len(labeled):.1%}")

# company_specific — skip if not available
if 'company_specific_factor' in labeled.columns:
    print("\nCOMPANY SPECIFIC:")
    for val in ['yes', 'no']:
        b = len(bounced[bounced['company_specific_factor'] == val])
        s = len(stuck[stuck['company_specific_factor'] == val])
        if b + s > 0:
            print(f"  {val}: bounced={b}, stuck={s}, stuck%={s/(b+s):.1%}")
else:
    # Get from master
    cs_map = dict(zip(zip(master['ticker'], master['event_date']), master.get('company_specific_factor', pd.Series())))
    labeled['cs'] = [cs_map.get((r['ticker'], r['event_date']), '?') for _, r in labeled.iterrows()]
    stuck['cs'] = [cs_map.get((r['ticker'], r['event_date']), '?') for _, r in stuck.iterrows()]
    bounced['cs'] = [cs_map.get((r['ticker'], r['event_date']), '?') for _, r in bounced.iterrows()]
    print("\nCOMPANY SPECIFIC:")
    for val in ['yes', 'no']:
        b = len(bounced[bounced['cs'] == val])
        s = len(stuck[stuck['cs'] == val])
        if b + s > 0:
            print(f"  {val}: bounced={b}, stuck={s}, stuck%={s/(b+s):.1%}")

# Print stuck descriptions grouped by event type
print("\n" + "=" * 80)
print("STUCK EVENT DESCRIPTIONS (grouped by type)")
print("=" * 80)

for et in stuck['stock_event_type'].value_counts().index:
    sub = stuck[stuck['stock_event_type'] == et].sort_values('event_date')
    print(f"\n--- {et.upper()} ({len(sub)} stuck events) ---")
    for _, row in sub.iterrows():
        # Get description from v2 batch data
        v2_match = v2[(v2['ticker'] == row['ticker']) & (v2['event_date'] == row['event_date'])]
        if len(v2_match) > 0:
            v2r = v2_match.iloc[0]
            primary = str(v2r.get('primary_cause', ''))
            secondary = str(v2r.get('secondary_factors', ''))
            cs = str(v2r.get('company_specific_factor', '?'))
        else:
            primary = str(row.get('description', ''))
            secondary = ''
            cs = '?'
        
        sev = row.get('stock_event_severity', '?')
        text = primary[:250] if primary and primary != 'nan' else ''
        sec_text = secondary[:150] if secondary and secondary != 'nan' else ''
        
        print(f"  {row['ticker']:>6} {str(row['event_date'])[:10]} drop={row['drop_pct']:.1%} sev={sev} cs={cs}")
        if text:
            print(f"         PRIMARY: {text}")
        if sec_text:
            print(f"         SECONDARY: {sec_text}")
        print()
