"""
Build synthetic company_specific label from market data.
Logic: if the stock dropped much more than SPY, it's stock-specific.
If it dropped roughly in line with SPY, it's market-driven.

Then test: does filtering out market-driven + severe drops reduce stuck rate?
"""
import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])

# SPY
spy_df = pd.read_parquet('backtest_iterations/spy_benchmark.parquet')
spy_df.index = pd.to_datetime(spy_df.index)
spy_daily = spy_df.iloc[:, 0].pct_change()
fp['spy_d0'] = fp['event_date'].map(spy_daily.to_dict())

# Stock excess vs SPY
fp['excess_vs_spy'] = fp['daily_return'] - fp['spy_d0']

# Speed features
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct']],
              on=['ticker', 'event_date'], how='left')

# Synthetic company_specific label
# Stock-specific = stock dropped much more than market
# Market-driven = stock dropped roughly with market
# Threshold: if excess < -2%, it's at least partially stock-specific
def label_cs(row):
    excess = row['excess_vs_spy']
    spy = row['spy_d0']
    drop = row['drop_pct']
    
    if pd.isna(excess) or pd.isna(spy):
        return 'unknown'
    
    # Stock dropped way more than market = stock-specific
    if excess < -0.03:
        return 'stock_specific'
    # Stock dropped roughly with market on a bad day = market_driven
    elif spy < -0.01 and excess >= -0.02:
        return 'market_driven'
    # In between
    else:
        return 'mixed'

fp['synth_cs'] = fp.apply(label_cs, axis=1)

# Severity proxy: drop size + market context
def label_severity(row):
    drop = row['drop_pct']
    spy = row['spy_d0']
    if pd.isna(drop):
        return 'unknown'
    if drop >= 0.08:
        return 'high'
    elif drop >= 0.05:
        return 'medium'
    else:
        return 'low'

fp['synth_sev'] = fp.apply(label_severity, axis=1)

def hit_1pct_5d(row):
    for day in range(2, 7):
        h = row.get(f'd{day}_high_ret', np.nan)
        if not pd.isna(h) and float(h) >= 0.01:
            return True
    return False

fp['hit'] = fp.apply(hit_1pct_5d, axis=1)

print(f"Total: {len(fp)}")
print(f"\nSynthetic CS distribution:")
print(fp['synth_cs'].value_counts())
print(f"\nSynthetic severity distribution:")
print(fp['synth_sev'].value_counts())


# ═══════════════════════════════════════════════════════════
# Validate synthetic labels against real ChatGPT labels
# ═══════════════════════════════════════════════════════════
master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])
cs_map = dict(zip(zip(master['ticker'], master['event_date']), master['company_specific_factor']))
fp['real_cs'] = [cs_map.get((r['ticker'], r['event_date']), np.nan) for _, r in fp.iterrows()]

has_both = fp[fp['real_cs'].notna() & (fp['synth_cs'] != 'unknown')]
print(f"\n{'='*60}")
print("VALIDATION: Synthetic vs Real ChatGPT labels")
print(f"{'='*60}")
print(f"Events with both labels: {len(has_both)}")
print(f"\nCross-tab:")
ct = pd.crosstab(has_both['synth_cs'], has_both['real_cs'])
print(ct)
print(f"\nAccuracy:")
for synth_val in ['stock_specific', 'mixed', 'market_driven']:
    sub = has_both[has_both['synth_cs'] == synth_val]
    if len(sub) == 0:
        continue
    real_yes = (sub['real_cs'] == 'yes').sum()
    real_no = (sub['real_cs'] == 'no').sum()
    print(f"  synth={synth_val}: real_yes={real_yes} ({real_yes/len(sub):.1%}), real_no={real_no} ({real_no/len(sub):.1%})")

# ═══════════════════════════════════════════════════════════
# Now test: stuck rate by synthetic labels
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STUCK RATE BY SYNTHETIC LABELS (all 7833 events)")
print(f"{'='*60}")

print(f"\n{'Synth CS':>18} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 46)
for val in ['stock_specific', 'mixed', 'market_driven']:
    sub = fp[fp['synth_cs'] == val]
    b = sub['hit'].sum()
    s = len(sub) - b
    print(f"{val:>18} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}")

print(f"\n{'Synth CS':>18} {'Sev':>8} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 55)
for cs in ['stock_specific', 'mixed', 'market_driven']:
    for sev in ['low', 'medium', 'high']:
        sub = fp[(fp['synth_cs'] == cs) & (fp['synth_sev'] == sev)]
        if len(sub) < 20:
            continue
        b = sub['hit'].sum()
        s = len(sub) - b
        marker = ' !!!' if s / len(sub) > 0.25 else (' <<<' if s / len(sub) < 0.12 else '')
        print(f"{cs:>18} {sev:>8} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{marker}")

# ═══════════════════════════════════════════════════════════
# Test different excess thresholds for the synthetic label
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FINE-TUNING: Excess vs SPY threshold for 'market_driven'")
print(f"{'='*60}")

print(f"\n{'Filter':>45} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 73)
print(f"{'ALL':>45} {len(fp):>6} {fp['hit'].mean():>9.1%} {(~fp['hit']).mean():>9.1%}")

# Different ways to define "market-driven drop"
for spy_thresh in [-0.005, -0.01, -0.015, -0.02, -0.03]:
    for excess_thresh in [-0.01, -0.015, -0.02, -0.025, -0.03]:
        # Market-driven = SPY down AND stock didn't drop much more
        market_driven = (fp['spy_d0'] < spy_thresh) & (fp['excess_vs_spy'] >= excess_thresh)
        # Skip these
        kept = fp[~market_driven]
        if len(kept) < 1000:
            continue
        b = kept['hit'].sum()
        s = len(kept) - b
        removed = market_driven.sum()
        removed_stuck = (~fp[market_driven]['hit']).sum() if removed > 0 else 0
        removed_bounce = fp[market_driven]['hit'].sum() if removed > 0 else 0
        # Only show if it improves stuck rate meaningfully
        stuck_pct = s / len(kept)
        if stuck_pct < 0.165:
            print(f"  SPY<{spy_thresh:+.1%} AND excess>={excess_thresh:+.1%} → skip {removed:>5} ({removed_stuck} stuck, {removed_bounce} bounce) → {len(kept):>6} {b/len(kept):>9.1%} {stuck_pct:>9.1%}")

# ═══════════════════════════════════════════════════════════
# Best filter: skip market-driven drops
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("BEST MARKET-DRIVEN FILTERS")
print(f"{'='*60}")

filters = {
    'ALL': fp,
    'Skip: SPY<-1% AND excess>-2%': fp[~((fp['spy_d0'] < -0.01) & (fp['excess_vs_spy'] >= -0.02))],
    'Skip: SPY<-1% AND excess>-1.5%': fp[~((fp['spy_d0'] < -0.01) & (fp['excess_vs_spy'] >= -0.015))],
    'Skip: SPY<-2% AND excess>-2%': fp[~((fp['spy_d0'] < -0.02) & (fp['excess_vs_spy'] >= -0.02))],
    'Skip: SPY<-2% AND excess>-3%': fp[~((fp['spy_d0'] < -0.02) & (fp['excess_vs_spy'] >= -0.03))],
    'Skip: SPY<-1.5% AND excess>-2%': fp[~((fp['spy_d0'] < -0.015) & (fp['excess_vs_spy'] >= -0.02))],
    'Skip: droppers>30 AND excess>-2%': fp[~((fp['num_droppers_3pct'] > 30) & (fp['excess_vs_spy'] >= -0.02))],
    'Skip: droppers>40 AND excess>-2%': fp[~((fp['num_droppers_3pct'] > 40) & (fp['excess_vs_spy'] >= -0.02))],
    'Only: excess < -2% (stock-specific)': fp[fp['excess_vs_spy'] < -0.02],
    'Only: excess < -1.5%': fp[fp['excess_vs_spy'] < -0.015],
}

print(f"\n{'Filter':>45} {'n':>6} {'Bounce%':>9} {'Stuck%':>9} {'Removed':>8}")
print('-' * 80)
for name, sub in filters.items():
    b = sub['hit'].sum()
    s = len(sub) - b
    removed = len(fp) - len(sub)
    print(f"{name:>45} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%} {removed:>8}")

# ═══════════════════════════════════════════════════════════
# Combine with ATR
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COMBINED: ATR + Market-driven filter")
print(f"{'='*60}")

combos = {
    'ALL': fp,
    'ATR<1.0': fp[fp['target_distance_over_atr20'] < 1.0],
    'ATR<1.0 + skip market-driven(SPY<-1%,exc>-2%)':
        fp[(fp['target_distance_over_atr20'] < 1.0) & ~((fp['spy_d0'] < -0.01) & (fp['excess_vs_spy'] >= -0.02))],
    'ATR<1.0 + skip market-driven(SPY<-2%,exc>-2%)':
        fp[(fp['target_distance_over_atr20'] < 1.0) & ~((fp['spy_d0'] < -0.02) & (fp['excess_vs_spy'] >= -0.02))],
    'ATR<0.8': fp[fp['target_distance_over_atr20'] < 0.8],
    'ATR<0.8 + skip market-driven(SPY<-1%,exc>-2%)':
        fp[(fp['target_distance_over_atr20'] < 0.8) & ~((fp['spy_d0'] < -0.01) & (fp['excess_vs_spy'] >= -0.02))],
    'ATR<1.0 + only stock-spec(exc<-2%)':
        fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_spy'] < -0.02)],
    'ATR<0.8 + only stock-spec(exc<-2%)':
        fp[(fp['target_distance_over_atr20'] < 0.8) & (fp['excess_vs_spy'] < -0.02)],
    'ATR<1.0 + only stock-spec(exc<-1.5%)':
        fp[(fp['target_distance_over_atr20'] < 1.0) & (fp['excess_vs_spy'] < -0.015)],
}

print(f"\n{'Filter':>55} {'n':>6} {'Bounce%':>9} {'Stuck%':>9}")
print('-' * 83)
for name, sub in combos.items():
    if len(sub) < 10:
        continue
    b = sub['hit'].sum()
    s = len(sub) - b
    m = ' <<<' if s / len(sub) < 0.10 and len(sub) > 50 else (' <<' if s / len(sub) < 0.12 and len(sub) > 100 else '')
    print(f"{name:>55} {len(sub):>6} {b/len(sub):>9.1%} {s/len(sub):>9.1%}{m}")
