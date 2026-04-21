"""
Optimized Portfolio: Adaptive Exits + Crisis Override
=====================================================
Key insight from analysis:
- S2_SPEED sleeve is negative EV — ATR<1.0 alone doesn't predict winners
- But shorter exits on CORE trades improve capital efficiency
- Crisis override (S1) is genuinely profitable

New approach:
- Core trades with ATR < 1.0: use shorter exit (10-15d max hold)
- Core trades with ATR >= 1.0: use standard exit (60d max hold)
- Crisis override: separate sleeve, bypass veto, short exits
- This is "adaptive exits" not "two sleeves"
"""
import pandas as pd
import numpy as np
from screener_v1 import score_event

# Load data
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])

df = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct',
                   'd1_pct_bars_above_vwap']],
              on=['ticker', 'event_date'], how='left')

# Score
print("Scoring events...")
scores = []
for _, row in df.iterrows():
    r = score_event(row)
    scores.append({'score': r.score, 'tier': r.tier, 'veto_reason': r.veto_reason})
df = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)

def is_crisis(row):
    d = row.get('num_droppers_3pct', np.nan)
    atr = row.get('target_distance_over_atr20', np.nan)
    if pd.isna(d) or pd.isna(atr):
        return False
    return 61 <= d <= 93 and atr < 0.8

def simulate_trade(row, target, stop, max_hold):
    for day in range(2, min(max_hold + 2, 61)):
        high_col = f'd{day}_high_ret'
        low_col = f'd{day}_low_ret'
        close_col = f'd{day}_close_ret'
        h = float(row.get(high_col, np.nan)) if high_col in row.index else np.nan
        l = float(row.get(low_col, np.nan)) if low_col in row.index else np.nan
        c = float(row.get(close_col, np.nan)) if close_col in row.index else np.nan
        if pd.isna(h) and pd.isna(l) and pd.isna(c):
            continue
        if not pd.isna(l) and l <= stop:
            return day - 1, stop, 'stop_loss'
        if not pd.isna(h) and h >= target:
            return day - 1, target, 'profit_target'
        if day >= max_hold + 1:
            return day - 1, c if not pd.isna(c) else 0.0, 'time_stop'
    return max_hold, 0.0, 'time_stop'


def run_portfolio(events_df, config_name, max_positions=10, base_size_pct=0.20,
                  starting_capital=100000, use_crisis=True, use_adaptive_exits=True,
                  short_max_hold=15, regime_filter_hyg=-0.01, regime_filter_spy=-0.04):
    events_df = events_df.sort_values('event_date').copy()
    dates = sorted(events_df['event_date'].unique())
    
    capital = starting_capital
    positions = []
    trade_log = []
    daily_equity = []
    
    for date in dates:
        day_events = events_df[events_df['event_date'] == date]
        
        # Close expired
        closing = [p for p in positions if p['days_remaining'] <= 0]
        positions = [p for p in positions if p['days_remaining'] > 0]
        for pos in closing:
            pnl = pos['size'] * pos['exit_ret']
            capital += pos['size'] + pnl
            trade_log.append({
                'ticker': pos['ticker'], 'entry_date': pos['entry_date'],
                'exit_date': date, 'size': pos['size'], 'exit_ret': pos['exit_ret'],
                'pnl': pnl, 'sleeve': pos['sleeve'], 'exit_reason': pos['exit_reason'],
                'hold_days': pos['total_days'],
            })
        
        available = max_positions - len(positions)
        if available <= 0:
            for pos in positions:
                pos['days_remaining'] -= 1
            invested = sum(p['size'] for p in positions)
            daily_equity.append({'date': date, 'equity': capital + invested, 'invested': invested, 'n_pos': len(positions)})
            continue
        
        held = {p['ticker'] for p in positions}
        admitted = []
        
        # Regime filter
        skip_regime = False
        if len(day_events) > 0:
            hyg5 = day_events.iloc[0].get('hyg_5d_return', np.nan)
            spy_ret = day_events.iloc[0].get('spy_5d_return', np.nan) if 'spy_5d_return' in day_events.columns else np.nan
            if not pd.isna(hyg5) and hyg5 <= regime_filter_hyg:
                # Check if also broad market stress
                droppers = day_events.iloc[0].get('num_droppers_3pct', 0)
                if not pd.isna(droppers) and droppers > 40:
                    skip_regime = True
        
        # Crisis override (bypass veto, bypass regime filter)
        if use_crisis:
            crisis_events = day_events[day_events.apply(is_crisis, axis=1)]
            crisis_events = crisis_events[~crisis_events['ticker'].isin(held)]
            if len(crisis_events) > 0:
                crisis_sorted = crisis_events.sort_values('target_distance_over_atr20')
                for _, row in crisis_sorted.iterrows():
                    if len(admitted) >= available:
                        break
                    if row['ticker'] in held or row['ticker'] in [a[0] for a in admitted]:
                        continue
                    d, r, reason = simulate_trade(row, 0.05, -0.06, 10)
                    admitted.append((row['ticker'], d, r, 'CRISIS', reason, row))
        
        # Core trades (must pass screener)
        if not skip_regime:
            core = day_events[day_events['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])]
            core = core[~core['ticker'].isin(held)]
            core = core.sort_values('score', ascending=False)
            
            remaining = available - len(admitted)
            for _, row in core.iterrows():
                if remaining <= 0:
                    break
                if row['ticker'] in held or row['ticker'] in [a[0] for a in admitted]:
                    continue
                
                atr = row.get('target_distance_over_atr20', np.nan)
                
                if use_adaptive_exits and not pd.isna(atr) and atr < 1.0:
                    # Easy target: shorter hold
                    d, r, reason = simulate_trade(row, 0.05, -0.08, short_max_hold)
                    sleeve = 'CORE_FAST'
                else:
                    # Standard exit
                    d, r, reason = simulate_trade(row, 0.05, -0.08, 60)
                    sleeve = 'CORE'
                
                admitted.append((row['ticker'], d, r, sleeve, reason, row))
                remaining -= 1
        
        # Open positions
        for ticker, hold_days, exit_ret, sleeve, exit_reason, row in admitted:
            # Score-weighted sizing
            s = row.get('score', 5)
            if s >= 10:
                size_mult = 1.25
            elif s >= 8:
                size_mult = 1.0
            else:
                size_mult = 0.75
            
            size = capital * base_size_pct * size_mult
            if size < 500 or capital < size:
                continue
            capital -= size
            positions.append({
                'ticker': ticker, 'entry_date': date, 'size': size,
                'days_remaining': hold_days, 'total_days': hold_days,
                'exit_ret': exit_ret, 'sleeve': sleeve, 'exit_reason': exit_reason,
            })
            held.add(ticker)
        
        for pos in positions:
            pos['days_remaining'] -= 1
        
        invested = sum(p['size'] for p in positions)
        daily_equity.append({'date': date, 'equity': capital + invested, 'invested': invested, 'n_pos': len(positions)})
    
    # Close remaining
    for pos in positions:
        pnl = pos['size'] * pos['exit_ret']
        capital += pos['size'] + pnl
        trade_log.append({
            'ticker': pos['ticker'], 'entry_date': pos['entry_date'],
            'exit_date': 'end', 'size': pos['size'], 'exit_ret': pos['exit_ret'],
            'pnl': pnl, 'sleeve': pos['sleeve'], 'exit_reason': pos['exit_reason'],
            'hold_days': pos['total_days'],
        })
    
    tdf = pd.DataFrame(trade_log)
    edf = pd.DataFrame(daily_equity)
    final = capital
    total_ret = final / starting_capital - 1
    
    if len(edf) > 0:
        years = (edf['date'].max() - edf['date'].min()).days / 365.25
        cagr = (final / starting_capital) ** (1 / years) - 1 if years > 0 else 0
        peak = edf['equity'].expanding().max()
        max_dd = ((edf['equity'] - peak) / peak).min()
    else:
        years, cagr, max_dd = 0, 0, 0
    
    return {
        'config': config_name, 'final': final, 'total_ret': total_ret,
        'cagr': cagr, 'max_dd': max_dd, 'years': years,
        'trades': len(tdf), 'trade_df': tdf, 'equity_df': edf,
    }


def show(r):
    print(f"\n  {r['config']}:")
    print(f"    ${r['final']:,.0f} | CAGR: {r['cagr']:.1%} | DD: {r['max_dd']:.1%} | Trades: {r['trades']}")
    tdf = r['trade_df']
    if len(tdf) > 0:
        print(f"    Win rate: {(tdf['pnl']>0).mean():.1%} | Avg ret: {tdf['exit_ret'].mean():+.2%}")
        for s in tdf['sleeve'].unique():
            sg = tdf[tdf['sleeve'] == s]
            print(f"    [{s}] n={len(sg)}, wr={(sg['pnl']>0).mean():.1%}, avg={sg['exit_ret'].mean():+.2%}, avg_hold={sg['hold_days'].mean():.1f}d")
        edf = r['equity_df']
        if len(edf) > 0:
            dep = edf['invested'] / edf['equity']
            print(f"    Capital deployed: mean={dep.mean():.1%}, median={dep.median():.1%}")


# ─── Run all configs ───
print(f"\n{'='*70}")
print("OPTIMIZED PORTFOLIO COMPARISON")
print(f"{'='*70}")

configs = [
    ("1. Baseline (core 60d, no crisis)", dict(use_crisis=False, use_adaptive_exits=False, max_positions=10, base_size_pct=0.20)),
    ("2. + Crisis override only", dict(use_crisis=True, use_adaptive_exits=False, max_positions=10, base_size_pct=0.20)),
    ("3. + Adaptive exits (15d for ATR<1)", dict(use_crisis=True, use_adaptive_exits=True, short_max_hold=15, max_positions=10, base_size_pct=0.20)),
    ("4. + Adaptive exits (20d for ATR<1)", dict(use_crisis=True, use_adaptive_exits=True, short_max_hold=20, max_positions=10, base_size_pct=0.20)),
    ("5. + Adaptive exits (30d for ATR<1)", dict(use_crisis=True, use_adaptive_exits=True, short_max_hold=30, max_positions=10, base_size_pct=0.20)),
    ("6. Config 3 + 25% size", dict(use_crisis=True, use_adaptive_exits=True, short_max_hold=15, max_positions=10, base_size_pct=0.25)),
    ("7. Config 3 + 15 positions", dict(use_crisis=True, use_adaptive_exits=True, short_max_hold=15, max_positions=15, base_size_pct=0.15)),
    ("8. Config 4 + 25% size", dict(use_crisis=True, use_adaptive_exits=True, short_max_hold=20, max_positions=10, base_size_pct=0.25)),
]

results = []
for name, kwargs in configs:
    r = run_portfolio(df, name, **kwargs)
    show(r)
    results.append(r)

# ─── Year breakdown for best ───
print(f"\n{'='*70}")
print("YEAR-BY-YEAR: Config 6 (Crisis + Adaptive 15d + 25% size)")
print(f"{'='*70}")

best = results[5]
tdf = best['trade_df'].copy()
tdf['entry_date'] = pd.to_datetime(tdf['entry_date'])
tdf['year'] = tdf['entry_date'].dt.year

print(f"\n{'Year':<6} {'N':>5} {'Win%':>7} {'AvgRet':>8} {'PnL':>12}")
print("-" * 45)
for year, yg in tdf.groupby('year'):
    print(f"{year:<6} {len(yg):>5} {(yg['pnl']>0).mean():>7.1%} {yg['exit_ret'].mean():>8.2%} ${yg['pnl'].sum():>11,.0f}")

# Save
summary = []
for r in results:
    summary.append({
        'config': r['config'], 'final_equity': r['final'],
        'cagr': r['cagr'], 'max_dd': r['max_dd'], 'trades': r['trades'],
    })
pd.DataFrame(summary).to_csv('phase1_artifacts/optimized_portfolio_comparison.csv', index=False)
print("\nSaved to phase1_artifacts/optimized_portfolio_comparison.csv")
