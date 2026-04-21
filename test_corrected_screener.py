"""
Test corrected screener (data-driven label rules) vs old.
"""
import pandas as pd
import numpy as np
import importlib
import screener_v1
importlib.reload(screener_v1)
from screener_v1 import score_event

# Load data with new labels
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
master = pd.read_csv('outputs/events_fully_labeled.csv')
master['event_date'] = pd.to_datetime(master['event_date'])

label_cols = ['ticker', 'event_date', 'stock_event_type', 'stock_event_severity', 'company_specific_factor']
labels = master[label_cols].drop_duplicates()
for col in ['stock_event_type', 'stock_event_severity', 'company_specific_factor']:
    if col in fp.columns:
        fp = fp.drop(columns=[col])
df = fp.merge(labels, on=['ticker', 'event_date'], how='left')

# Score with corrected screener
print("Scoring with corrected screener + new labels...")
scores = []
for _, row in df.iterrows():
    r = score_event(row)
    scores.append({'score': r.score, 'tier': r.tier, 'veto_reason': r.veto_reason})
sdf = pd.DataFrame(scores)
df = pd.concat([df.reset_index(drop=True), sdf], axis=1)

print(f"\nTier distribution:")
print(df['tier'].value_counts().to_dict())

tradable = df[df['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])]
print(f"\nTradable: {len(tradable)}")
print(f"  Success rate: {tradable['success'].mean():.1%}")
print(f"  Avg final return: {tradable['final_return'].mean():+.2%}")
print(f"  CONFIDENT_YES: {len(tradable[tradable['tier']=='CONFIDENT_YES'])}, success={tradable[tradable['tier']=='CONFIDENT_YES']['success'].mean():.1%}")
print(f"  SMALLER_YES: {len(tradable[tradable['tier']=='SMALLER_YES'])}, success={tradable[tradable['tier']=='SMALLER_YES']['success'].mean():.1%}")


# ─── Portfolio simulation ───
print(f"\n{'='*60}")
print("PORTFOLIO SIMULATION: Corrected Screener + New Labels")
print(f"{'='*60}")

sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
df = df.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct']],
              on=['ticker', 'event_date'], how='left')

def is_crisis(row):
    d = row.get('num_droppers_3pct', np.nan)
    atr = row.get('target_distance_over_atr20', np.nan)
    if pd.isna(d) or pd.isna(atr):
        return False
    return 61 <= d <= 93 and atr < 0.8

def simulate_trade(row, target, stop, max_hold):
    for day in range(2, min(max_hold + 2, 61)):
        h = float(row.get(f'd{day}_high_ret', np.nan)) if f'd{day}_high_ret' in row.index else np.nan
        l = float(row.get(f'd{day}_low_ret', np.nan)) if f'd{day}_low_ret' in row.index else np.nan
        c = float(row.get(f'd{day}_close_ret', np.nan)) if f'd{day}_close_ret' in row.index else np.nan
        if pd.isna(h) and pd.isna(l) and pd.isna(c):
            continue
        if not pd.isna(l) and l <= stop:
            return day - 1, stop, 'stop_loss'
        if not pd.isna(h) and h >= target:
            return day - 1, target, 'profit_target'
        if day >= max_hold + 1:
            return day - 1, c if not pd.isna(c) else 0.0, 'time_stop'
    return max_hold, 0.0, 'time_stop'

def run_portfolio(events_df, name, max_positions=10, base_size_pct=0.20,
                  starting_capital=100000, use_crisis=True, adaptive_exits=True):
    events_df = events_df.sort_values('event_date').copy()
    dates = sorted(events_df['event_date'].unique())
    capital = starting_capital
    positions = []
    trade_log = []
    daily_equity = []
    
    for date in dates:
        day_events = events_df[events_df['event_date'] == date]
        
        closing = [p for p in positions if p['days_remaining'] <= 0]
        positions = [p for p in positions if p['days_remaining'] > 0]
        for pos in closing:
            pnl = pos['size'] * pos['exit_ret']
            capital += pos['size'] + pnl
            trade_log.append({**pos, 'exit_date': date, 'pnl': pnl})
        
        available = max_positions - len(positions)
        if available <= 0:
            for p in positions: p['days_remaining'] -= 1
            invested = sum(p['size'] for p in positions)
            daily_equity.append({'date': date, 'equity': capital + invested})
            continue
        
        held = {p['ticker'] for p in positions}
        admitted = []
        
        # Crisis override
        if use_crisis:
            crisis = day_events[day_events.apply(is_crisis, axis=1)]
            crisis = crisis[~crisis['ticker'].isin(held)]
            for _, row in crisis.sort_values('target_distance_over_atr20').iterrows():
                if len(admitted) >= available: break
                if row['ticker'] in [a[0] for a in admitted]: continue
                d, r, reason = simulate_trade(row, 0.05, -0.06, 10)
                admitted.append((row['ticker'], d, r, 'CRISIS', reason, row.get('score', 5)))
        
        # Core trades
        remaining = available - len(admitted)
        core = day_events[day_events['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])]
        core = core[~core['ticker'].isin(held)]
        core = core.sort_values('score', ascending=False)
        
        for _, row in core.iterrows():
            if remaining <= 0: break
            if row['ticker'] in held or row['ticker'] in [a[0] for a in admitted]: continue
            atr = row.get('target_distance_over_atr20', np.nan)
            if adaptive_exits and not pd.isna(atr) and atr < 1.0:
                d, r, reason = simulate_trade(row, 0.05, -0.08, 15)
                sleeve = 'CORE_FAST'
            else:
                d, r, reason = simulate_trade(row, 0.05, -0.08, 60)
                sleeve = 'CORE'
            admitted.append((row['ticker'], d, r, sleeve, reason, row.get('score', 5)))
            remaining -= 1
        
        for ticker, hold_days, exit_ret, sleeve, exit_reason, score in admitted:
            mult = 1.25 if score >= 10 else (1.0 if score >= 8 else 0.75)
            size = capital * base_size_pct * mult
            if size < 500 or capital < size: continue
            capital -= size
            positions.append({
                'ticker': ticker, 'entry_date': date, 'size': size,
                'days_remaining': hold_days, 'total_days': hold_days,
                'exit_ret': exit_ret, 'sleeve': sleeve, 'exit_reason': exit_reason,
            })
            held.add(ticker)
        
        for p in positions: p['days_remaining'] -= 1
        invested = sum(p['size'] for p in positions)
        daily_equity.append({'date': date, 'equity': capital + invested})
    
    # Close remaining
    for pos in positions:
        pnl = pos['size'] * pos['exit_ret']
        capital += pos['size'] + pnl
        trade_log.append({**pos, 'exit_date': 'end', 'pnl': pnl})
    
    tdf = pd.DataFrame(trade_log)
    edf = pd.DataFrame(daily_equity)
    final = capital
    
    if len(edf) > 0:
        years = (edf['date'].max() - edf['date'].min()).days / 365.25
        cagr = (final / starting_capital) ** (1/years) - 1 if years > 0 else 0
        peak = edf['equity'].expanding().max()
        max_dd = ((edf['equity'] - peak) / peak).min()
    else:
        years, cagr, max_dd = 0, 0, 0
    
    return {'name': name, 'final': final, 'cagr': cagr, 'max_dd': max_dd,
            'trades': len(tdf), 'trade_df': tdf, 'equity_df': edf, 'years': years}

# Run configs
configs = [
    ("Corrected screener, 20% size", dict(max_positions=10, base_size_pct=0.20)),
    ("Corrected screener, 25% size", dict(max_positions=10, base_size_pct=0.25)),
    ("Corrected screener, 15 positions", dict(max_positions=15, base_size_pct=0.15)),
    ("Corrected, no crisis override", dict(max_positions=10, base_size_pct=0.20, use_crisis=False)),
    ("Corrected, no adaptive exits", dict(max_positions=10, base_size_pct=0.20, adaptive_exits=False)),
]

for name, kwargs in configs:
    r = run_portfolio(df, name, **kwargs)
    tdf = r['trade_df']
    print(f"\n  {name}:")
    print(f"    ${r['final']:,.0f} | CAGR: {r['cagr']:.1%} | DD: {r['max_dd']:.1%} | Trades: {r['trades']}")
    if len(tdf) > 0:
        print(f"    Win rate: {(tdf['pnl']>0).mean():.1%} | Avg ret: {tdf['exit_ret'].mean():+.2%}")
        for s in sorted(tdf['sleeve'].unique()):
            sg = tdf[tdf['sleeve'] == s]
            print(f"    [{s}] n={len(sg)}, wr={(sg['pnl']>0).mean():.1%}, avg={sg['exit_ret'].mean():+.2%}")

# Year breakdown for best
print(f"\n{'='*60}")
print("YEAR-BY-YEAR: Corrected screener, 25% size")
print(f"{'='*60}")
r = run_portfolio(df, "best", max_positions=10, base_size_pct=0.25)
tdf = r['trade_df'].copy()
tdf['entry_date'] = pd.to_datetime(tdf['entry_date'])
tdf['year'] = tdf['entry_date'].dt.year
for year, yg in tdf.groupby('year'):
    print(f"  {year}: n={len(yg)}, wr={(yg['pnl']>0).mean():.1%}, avg_ret={yg['exit_ret'].mean():+.2%}, pnl=${yg['pnl'].sum():,.0f}")
