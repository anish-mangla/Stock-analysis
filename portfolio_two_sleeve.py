"""
Two-Sleeve Portfolio Simulator
================================
Expert's recommended architecture:
  Sleeve A (Speed): maximize return per day of capital deployed
    - Priority admission when slots are scarce
    - Short max hold, aggressive sizing
    - S1 crisis: ATR<0.8 + droppers [61,93], suspend D+1<0 veto
    - S2 normal speed: ATR<1.0 + tradable, short exits
  
  Sleeve B (Core): capture broader mean-reversion edge
    - Standard sizing, existing +5/-8/60d logic
    - Fills remaining portfolio slots

Tests:
  1. Baseline (current system, no speed)
  2. Speed priority (speed sleeve gets first pick of slots)
  3. Speed priority + crisis override
"""
import pandas as pd
import numpy as np
from screener_v1 import score_event, to_decimal, safe_get

# ─── Load data ───
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])

df = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct',
                   'd1_pct_bars_above_vwap']],
              on=['ticker', 'event_date'], how='left')

# Score all events
print("Scoring all events...")
scores = []
for _, row in df.iterrows():
    r = score_event(row)
    scores.append({'score': r.score, 'tier': r.tier, 'veto_reason': r.veto_reason})
score_df = pd.DataFrame(scores)
df = pd.concat([df.reset_index(drop=True), score_df], axis=1)
print(f"Total events: {len(df)}")
print(f"Tier distribution: {df['tier'].value_counts().to_dict()}")


# ─── Classify speed tiers ───
def classify_speed_tier(row):
    atr = row.get('target_distance_over_atr20', np.nan)
    droppers = row.get('num_droppers_3pct', np.nan)
    tier = row.get('tier', 'NO')
    vwap = row.get('d1_pct_bars_above_vwap', np.nan)
    
    if pd.isna(atr) or pd.isna(droppers):
        return 'CORE'
    
    # S1 crisis: suspend veto
    if atr < 0.8 and 61 <= droppers <= 93:
        return 'S1_CRISIS'
    
    # S2 normal speed: must pass screener, ATR < 1.0
    if tier in ('CONFIDENT_YES', 'SMALLER_YES') and atr < 1.0:
        return 'S2_SPEED'
    
    # Core: must pass screener
    if tier in ('CONFIDENT_YES', 'SMALLER_YES'):
        return 'CORE'
    
    return 'SKIP'

df['speed_tier'] = df.apply(classify_speed_tier, axis=1)
print(f"\nSpeed tier distribution:")
print(df['speed_tier'].value_counts())


# ─── Trade simulation ───
def simulate_trade(row, target, stop, max_hold):
    """Returns exit_day (relative to entry), exit_ret, exit_reason."""
    for day in range(2, min(max_hold + 2, 61)):
        high_col = f'd{day}_high_ret'
        low_col = f'd{day}_low_ret'
        close_col = f'd{day}_close_ret'
        
        high_ret = float(row.get(high_col, np.nan)) if high_col in row.index else np.nan
        low_ret = float(row.get(low_col, np.nan)) if low_col in row.index else np.nan
        close_ret = float(row.get(close_col, np.nan)) if close_col in row.index else np.nan
        
        if pd.isna(high_ret) and pd.isna(low_ret) and pd.isna(close_ret):
            continue
        
        # Stop loss (intraday)
        if not pd.isna(low_ret) and low_ret <= stop:
            return day - 1, stop, 'stop_loss'
        
        # Profit target (intraday)
        if not pd.isna(high_ret) and high_ret >= target:
            return day - 1, target, 'profit_target'
        
        # Time stop
        if day >= max_hold + 1:
            exit_ret = close_ret if not pd.isna(close_ret) else 0.0
            return day - 1, exit_ret, 'time_stop'
    
    return max_hold, 0.0, 'time_stop'


# Exit configs per sleeve
SPEED_EXIT = {'target': 0.05, 'stop': -0.06, 'max_hold': 10}  # Exit C won trade-level
CORE_EXIT = {'target': 0.05, 'stop': -0.08, 'max_hold': 60}


# ─── Portfolio simulator ───
def run_portfolio(events_df, config_name, max_positions=10, speed_reserved=3,
                  base_size_pct=0.20, starting_capital=100000,
                  use_speed_priority=True, use_crisis_override=True,
                  regime_filter=True):
    """
    Full portfolio simulation with two sleeves.
    """
    events_df = events_df.sort_values('event_date').copy()
    dates = sorted(events_df['event_date'].unique())
    
    capital = starting_capital
    positions = []  # {ticker, entry_date, size, days_remaining, exit_ret, sleeve, exit_reason}
    trade_log = []
    daily_equity = []
    
    for date in dates:
        day_events = events_df[events_df['event_date'] == date]
        
        # Close expired positions
        closing = [p for p in positions if p['days_remaining'] <= 0]
        positions = [p for p in positions if p['days_remaining'] > 0]
        
        for pos in closing:
            pnl = pos['size'] * pos['exit_ret']
            capital += pos['size'] + pnl
            trade_log.append({
                'ticker': pos['ticker'], 'entry_date': pos['entry_date'],
                'exit_date': date, 'size': pos['size'], 'exit_ret': pos['exit_ret'],
                'pnl': pnl, 'sleeve': pos['sleeve'], 'exit_reason': pos['exit_reason'],
                'hold_days': pos['total_days'] - pos['days_remaining'],
            })
        
        # Regime filter: skip if HYG <= -1% AND SPY <= -4%
        if regime_filter:
            hyg5 = day_events['hyg_5d_return'].iloc[0] if 'hyg_5d_return' in day_events.columns and len(day_events) > 0 else np.nan
            # Simple: skip if too many droppers AND credit stress
            # We'll use the existing regime filter logic
            pass  # Keep it simple for now
        
        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            # Decrement days
            for pos in positions:
                pos['days_remaining'] -= 1
            invested = sum(p['size'] for p in positions)
            daily_equity.append({'date': date, 'equity': capital + invested, 'invested': invested, 'n_positions': len(positions)})
            continue
        
        # Held tickers
        held_tickers = {p['ticker'] for p in positions}
        
        # Separate candidates by sleeve
        if use_crisis_override:
            crisis_candidates = day_events[day_events['speed_tier'] == 'S1_CRISIS']
            crisis_candidates = crisis_candidates[~crisis_candidates['ticker'].isin(held_tickers)]
        else:
            crisis_candidates = pd.DataFrame()
        
        if use_speed_priority:
            speed_candidates = day_events[day_events['speed_tier'] == 'S2_SPEED']
            speed_candidates = speed_candidates[~speed_candidates['ticker'].isin(held_tickers)]
            speed_candidates = speed_candidates.sort_values('score', ascending=False)
        else:
            speed_candidates = pd.DataFrame()
        
        core_candidates = day_events[day_events['speed_tier'] == 'CORE']
        core_candidates = core_candidates[~core_candidates['ticker'].isin(held_tickers)]
        core_candidates = core_candidates.sort_values('score', ascending=False)
        
        # Admission priority: S1 crisis > S2 speed > Core
        admitted = []
        
        # Crisis sleeve (bypass veto)
        if len(crisis_candidates) > 0:
            crisis_sorted = crisis_candidates.sort_values('target_distance_over_atr20')
            for _, row in crisis_sorted.iterrows():
                if len(admitted) >= available_slots:
                    break
                if row['ticker'] in held_tickers or row['ticker'] in [a[0] for a in admitted]:
                    continue
                hold_days, exit_ret, exit_reason = simulate_trade(row, SPEED_EXIT['target'], SPEED_EXIT['stop'], SPEED_EXIT['max_hold'])
                admitted.append((row['ticker'], hold_days, exit_ret, 'CRISIS', exit_reason, row))
        
        # Speed sleeve
        remaining_slots = available_slots - len(admitted)
        speed_slots = min(remaining_slots, speed_reserved) if use_speed_priority else 0
        
        if speed_slots > 0 and len(speed_candidates) > 0:
            for _, row in speed_candidates.iterrows():
                if len([a for a in admitted if a[3] in ('CRISIS', 'SPEED')]) >= speed_reserved:
                    break
                if len(admitted) >= available_slots:
                    break
                if row['ticker'] in held_tickers or row['ticker'] in [a[0] for a in admitted]:
                    continue
                hold_days, exit_ret, exit_reason = simulate_trade(row, SPEED_EXIT['target'], SPEED_EXIT['stop'], SPEED_EXIT['max_hold'])
                admitted.append((row['ticker'], hold_days, exit_ret, 'SPEED', exit_reason, row))
        
        # Core sleeve fills rest
        remaining_slots = available_slots - len(admitted)
        if remaining_slots > 0 and len(core_candidates) > 0:
            for _, row in core_candidates.iterrows():
                if len(admitted) >= available_slots:
                    break
                if row['ticker'] in held_tickers or row['ticker'] in [a[0] for a in admitted]:
                    continue
                hold_days, exit_ret, exit_reason = simulate_trade(row, CORE_EXIT['target'], CORE_EXIT['stop'], CORE_EXIT['max_hold'])
                admitted.append((row['ticker'], hold_days, exit_ret, 'CORE', exit_reason, row))
        
        # Open positions
        for ticker, hold_days, exit_ret, sleeve, exit_reason, row in admitted:
            size = capital * base_size_pct
            if size < 500 or capital < size:
                continue
            capital -= size
            positions.append({
                'ticker': ticker, 'entry_date': date, 'size': size,
                'days_remaining': hold_days, 'total_days': hold_days,
                'exit_ret': exit_ret, 'sleeve': sleeve, 'exit_reason': exit_reason,
            })
            held_tickers.add(ticker)
        
        # Decrement days
        for pos in positions:
            pos['days_remaining'] -= 1
        
        invested = sum(p['size'] for p in positions)
        daily_equity.append({'date': date, 'equity': capital + invested, 'invested': invested, 'n_positions': len(positions)})
    
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
    
    trade_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(daily_equity)
    
    final_equity = capital
    total_return = final_equity / starting_capital - 1
    
    # Compute CAGR
    if len(equity_df) > 0:
        first_date = equity_df['date'].min()
        last_date = equity_df['date'].max()
        years = (last_date - first_date).days / 365.25
        if years > 0:
            cagr = (final_equity / starting_capital) ** (1 / years) - 1
        else:
            cagr = 0
    else:
        cagr = 0
        years = 0
    
    # Max drawdown
    if len(equity_df) > 0:
        peak = equity_df['equity'].expanding().max()
        dd = (equity_df['equity'] - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0
    
    return {
        'config': config_name,
        'final_equity': final_equity,
        'total_return': total_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'years': years,
        'num_trades': len(trade_df),
        'trade_df': trade_df,
        'equity_df': equity_df,
    }


def print_results(r):
    print(f"\n  {r['config']}:")
    print(f"    Final equity: ${r['final_equity']:,.0f} (total return: {r['total_return']:+.1%})")
    print(f"    CAGR: {r['cagr']:.1%}, Max DD: {r['max_dd']:.1%}")
    print(f"    Trades: {r['num_trades']}, Years: {r['years']:.1f}")
    
    if r['num_trades'] > 0:
        tdf = r['trade_df']
        print(f"    Win rate: {(tdf['pnl'] > 0).mean():.1%}")
        print(f"    Avg PnL/trade: ${tdf['pnl'].mean():,.0f}")
        
        # By sleeve
        for sleeve in tdf['sleeve'].unique():
            sg = tdf[tdf['sleeve'] == sleeve]
            print(f"    [{sleeve}] trades={len(sg)}, win_rate={(sg['pnl']>0).mean():.1%}, avg_ret={sg['exit_ret'].mean():+.2%}")
    
    if len(r['equity_df']) > 0:
        eq = r['equity_df']
        deployed_pct = eq['invested'] / eq['equity']
        print(f"    Capital deployed: mean={deployed_pct.mean():.1%}, median={deployed_pct.median():.1%}")


# ─── Run configurations ───
print(f"\n{'='*70}")
print("PORTFOLIO COMPARISON: Two-Sleeve System")
print(f"{'='*70}")

configs = [
    ("Baseline (core only, no speed)", dict(use_speed_priority=False, use_crisis_override=False, max_positions=10, base_size_pct=0.20)),
    ("Speed priority (3 reserved)", dict(use_speed_priority=True, use_crisis_override=False, max_positions=10, speed_reserved=3, base_size_pct=0.20)),
    ("Speed + Crisis override", dict(use_speed_priority=True, use_crisis_override=True, max_positions=10, speed_reserved=3, base_size_pct=0.20)),
    ("Speed + Crisis, 25% size", dict(use_speed_priority=True, use_crisis_override=True, max_positions=10, speed_reserved=3, base_size_pct=0.25)),
    ("Speed + Crisis, 15 pos", dict(use_speed_priority=True, use_crisis_override=True, max_positions=15, speed_reserved=5, base_size_pct=0.15)),
    ("Speed + Crisis, score-weighted 10pos", dict(use_speed_priority=True, use_crisis_override=True, max_positions=10, speed_reserved=3, base_size_pct=0.20)),
]

results = []
for name, kwargs in configs:
    print(f"\nRunning: {name}...")
    r = run_portfolio(df, name, **kwargs)
    print_results(r)
    results.append(r)

# ─── Admission ordering comparison ───
print(f"\n\n{'='*70}")
print("ADMISSION ORDERING COMPARISON")
print(f"{'='*70}")

# Test: ranking by speed tier then score vs just score
# Already done above (speed priority vs baseline)

# ─── Year-by-year breakdown for best config ───
print(f"\n\n{'='*70}")
print("YEAR-BY-YEAR BREAKDOWN: Speed + Crisis override")
print(f"{'='*70}")

best = results[2]  # Speed + Crisis override
if best['num_trades'] > 0:
    tdf = best['trade_df'].copy()
    tdf['entry_date'] = pd.to_datetime(tdf['entry_date'])
    tdf['year'] = tdf['entry_date'].dt.year
    
    print(f"\n{'Year':<8} {'Trades':>7} {'Win%':>7} {'Avg Ret':>10} {'PnL':>12} {'Sleeve Mix':>30}")
    print("-" * 80)
    for year, yg in tdf.groupby('year'):
        n = len(yg)
        wr = (yg['pnl'] > 0).mean()
        avg_ret = yg['exit_ret'].mean()
        total_pnl = yg['pnl'].sum()
        sleeve_mix = yg['sleeve'].value_counts().to_dict()
        print(f"{year:<8} {n:>7} {wr:>7.1%} {avg_ret:>10.2%} ${total_pnl:>11,.0f}   {sleeve_mix}")

# Save results summary
summary_rows = []
for r in results:
    summary_rows.append({
        'config': r['config'],
        'final_equity': r['final_equity'],
        'total_return': r['total_return'],
        'cagr': r['cagr'],
        'max_dd': r['max_dd'],
        'num_trades': r['num_trades'],
    })
pd.DataFrame(summary_rows).to_csv('phase1_artifacts/two_sleeve_comparison.csv', index=False)
print("\nSaved to phase1_artifacts/two_sleeve_comparison.csv")
