"""
Crisis Snapback Sleeve Backtest
================================
Expert's recommended architecture:
- Separate sleeve activated only during rare broad-flush regime
- Trigger: num_droppers_3pct in [61, 93] (date-level)
- Entry universe: all events on trigger dates with target_distance_over_atr20 < 0.8
- Suspends D+1<0 veto
- Short-horizon exits: test 3 families (A/B/C)
- Separate PnL tracking

Tests:
  Exit A: +3% / -4% / 5d
  Exit B: +4% / -5% / 7d
  Exit C: +5% / -6% / 10d
"""
import pandas as pd
import numpy as np
from collections import defaultdict

# Load data
fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])

# Merge speed features
df = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20', 'num_droppers_3pct']],
              on=['ticker', 'event_date'], how='left')

print(f"Total events: {len(df)}")

# ─── Crisis regime filter ───
crisis = df[(df['num_droppers_3pct'] >= 61) & (df['num_droppers_3pct'] <= 93)].copy()
print(f"Events on crisis dates (61-93 droppers): {len(crisis)}")

# S1 filter: easy target
s1 = crisis[crisis['target_distance_over_atr20'] < 0.8].copy()
print(f"S1 events (ATR < 0.8): {len(s1)}")

# Also test without ATR filter (equal-weight basket of all crisis-date events)
print(f"All crisis events (no ATR filter): {len(crisis)}")
print()

def to_dec(x):
    if pd.isna(x): return np.nan
    return float(x)

def get_col(row, col):
    return row[col] if col in row.index else np.nan


def simulate_trade(row, target, stop, max_hold):
    """Simulate a single trade with given exit params. Entry at D+1 close."""
    for day in range(2, max_hold + 2):  # d2 through d(max_hold+1)
        high_col = f'd{day}_high_ret'
        low_col = f'd{day}_low_ret'
        close_col = f'd{day}_close_ret'
        
        high_ret = to_dec(get_col(row, high_col))
        low_ret = to_dec(get_col(row, low_col))
        close_ret = to_dec(get_col(row, close_col))
        
        # Check stop first (intraday)
        if not pd.isna(low_ret) and low_ret <= stop:
            return {'exit_day': day, 'exit_ret': stop, 'exit_reason': 'stop_loss'}
        
        # Check target (intraday)
        if not pd.isna(high_ret) and high_ret >= target:
            return {'exit_day': day, 'exit_ret': target, 'exit_reason': 'profit_target'}
        
        # Time stop on last day
        if day == max_hold + 1:
            exit_ret = close_ret if not pd.isna(close_ret) else 0.0
            return {'exit_day': day, 'exit_ret': exit_ret, 'exit_reason': 'time_stop'}
    
    # Fallback
    return {'exit_day': max_hold + 1, 'exit_ret': 0.0, 'exit_reason': 'time_stop'}


EXIT_CONFIGS = {
    'A_3_4_5d':  {'target': 0.03, 'stop': -0.04, 'max_hold': 5},
    'B_4_5_7d':  {'target': 0.04, 'stop': -0.05, 'max_hold': 7},
    'C_5_6_10d': {'target': 0.05, 'stop': -0.06, 'max_hold': 10},
}


def run_backtest(events_df, exit_configs, label):
    """Run backtest for a set of events across all exit configs."""
    print(f"\n{'='*70}")
    print(f"BACKTEST: {label} (n={len(events_df)})")
    print(f"{'='*70}")
    
    results = {}
    for config_name, params in exit_configs.items():
        trades = []
        for _, row in events_df.iterrows():
            result = simulate_trade(row, params['target'], params['stop'], params['max_hold'])
            result['ticker'] = row['ticker']
            result['event_date'] = row['event_date']
            trades.append(result)
        
        trades_df = pd.DataFrame(trades)
        
        n = len(trades_df)
        winners = trades_df[trades_df['exit_ret'] > 0]
        losers = trades_df[trades_df['exit_ret'] < 0]
        
        win_rate = len(winners) / n if n > 0 else 0
        avg_win = winners['exit_ret'].mean() if len(winners) > 0 else 0
        avg_loss = losers['exit_ret'].mean() if len(losers) > 0 else 0
        avg_ret = trades_df['exit_ret'].mean()
        avg_days = trades_df['exit_day'].mean() - 1  # subtract 1 since day 2 = 1 day held
        ret_per_day = avg_ret / avg_days if avg_days > 0 else 0
        
        # Exit reason breakdown
        reasons = trades_df['exit_reason'].value_counts()
        
        print(f"\n  {config_name}: target={params['target']:+.0%} stop={params['stop']:+.0%} max={params['max_hold']}d")
        print(f"    Win rate: {win_rate:.1%} ({len(winners)}/{n})")
        print(f"    Avg win: {avg_win:+.2%}, Avg loss: {avg_loss:+.2%}")
        print(f"    Avg return: {avg_ret:+.3%}")
        print(f"    Avg hold days: {avg_days:.1f}")
        print(f"    Return/day: {ret_per_day:+.4%}")
        print(f"    Exits: {dict(reasons)}")
        
        results[config_name] = {
            'n': n, 'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'avg_ret': avg_ret, 'avg_days': avg_days, 'ret_per_day': ret_per_day,
            'trades_df': trades_df,
        }
    
    return results


# ─── Run backtests ───

# 1. S1 only (ATR < 0.8 + crisis dates)
s1_results = run_backtest(s1, EXIT_CONFIGS, "S1: ATR<0.8 + Crisis Dates")

# 2. All crisis events (equal-weight basket test)
crisis_results = run_backtest(crisis, EXIT_CONFIGS, "All Crisis Events (no ATR filter)")

# 3. S1 that currently pass screener (CONFIDENT_YES or SMALLER_YES)
sl = pd.read_csv('phase1_artifacts/event_speed_labels_full.csv')
sl['event_date'] = pd.to_datetime(sl['event_date'])
s1_tradable = s1.merge(sl[['ticker', 'event_date', 'tier']], on=['ticker', 'event_date'], how='left')
s1_pass = s1_tradable[s1_tradable['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])]
s1_veto = s1_tradable[s1_tradable['tier'] == 'VETO']

print(f"\n\nS1 screener breakdown:")
print(f"  Pass screener (CONF+SMALL): {len(s1_pass)} ({len(s1_pass)/len(s1):.1%})")
print(f"  VETO'd: {len(s1_veto)} ({len(s1_veto)/len(s1):.1%})")

s1_pass_results = run_backtest(s1_pass, EXIT_CONFIGS, "S1 Tradable Only (pass screener)")
s1_veto_results = run_backtest(s1_veto, EXIT_CONFIGS, "S1 VETO'd Only (currently blocked)")


# ─── Portfolio-level simulation for crisis sleeve ───
print(f"\n\n{'='*70}")
print("PORTFOLIO SIMULATION: Crisis Sleeve with Capital Constraints")
print(f"{'='*70}")

# Use best exit config from trade-level results
# Simulate with: 3 reserved slots for crisis, $100K starting capital

def portfolio_sim_crisis(events_df, exit_params, max_crisis_slots=3, 
                          base_size_pct=0.20, starting_capital=100000):
    """
    Simulate crisis sleeve as portfolio overlay.
    On crisis dates, admit up to max_crisis_slots positions.
    """
    events_df = events_df.sort_values('event_date').copy()
    
    capital = starting_capital
    positions = []  # list of {ticker, entry_date, exit_date, size, exit_ret}
    daily_equity = {}
    trade_log = []
    
    # Group by event date
    for date, group in events_df.groupby('event_date'):
        # Close expired positions
        new_positions = []
        for pos in positions:
            if pos['days_remaining'] <= 0:
                pnl = pos['size'] * pos['exit_ret']
                capital += pos['size'] + pnl
                trade_log.append({
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'size': pos['size'],
                    'exit_ret': pos['exit_ret'],
                    'pnl': pnl,
                })
            else:
                new_positions.append(pos)
        positions = new_positions
        
        # Admit new positions (up to max_crisis_slots total)
        available_slots = max_crisis_slots - len(positions)
        if available_slots > 0:
            # Sort by target_distance_over_atr20 (lower = easier target)
            candidates = group.sort_values('target_distance_over_atr20')
            for _, row in candidates.head(available_slots).iterrows():
                result = simulate_trade(row, exit_params['target'], exit_params['stop'], exit_params['max_hold'])
                size = capital * base_size_pct
                if size < 1000:
                    continue
                capital -= size
                positions.append({
                    'ticker': row['ticker'],
                    'entry_date': date,
                    'size': size,
                    'exit_ret': result['exit_ret'],
                    'days_remaining': result['exit_day'] - 1,
                    'exit_reason': result['exit_reason'],
                })
        
        # Decrement days
        for pos in positions:
            pos['days_remaining'] -= 1
        
        # Track equity
        invested = sum(p['size'] for p in positions)
        daily_equity[date] = capital + invested
    
    # Close remaining
    for pos in positions:
        pnl = pos['size'] * pos['exit_ret']
        capital += pos['size'] + pnl
        trade_log.append({
            'ticker': pos['ticker'],
            'entry_date': pos['entry_date'],
            'exit_date': 'end',
            'size': pos['size'],
            'exit_ret': pos['exit_ret'],
            'pnl': pnl,
        })
    
    final_capital = capital
    total_return = (final_capital / starting_capital - 1)
    trade_df = pd.DataFrame(trade_log)
    
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'num_trades': len(trade_df),
        'trade_df': trade_df,
        'daily_equity': daily_equity,
    }


# Run portfolio sim for each exit config on S1
for config_name, params in EXIT_CONFIGS.items():
    for slots in [3, 5]:
        result = portfolio_sim_crisis(s1, params, max_crisis_slots=slots, base_size_pct=0.25)
        print(f"\n  {config_name} | {slots} slots | 25% size:")
        print(f"    Trades: {result['num_trades']}")
        print(f"    Final capital: ${result['final_capital']:,.0f}")
        print(f"    Total return: {result['total_return']:+.1%}")
        if result['num_trades'] > 0:
            tdf = result['trade_df']
            print(f"    Avg PnL/trade: ${tdf['pnl'].mean():,.0f}")
            print(f"    Win rate: {(tdf['pnl'] > 0).mean():.1%}")


# ─── Year-by-year trade-level results for best config ───
print(f"\n\n{'='*70}")
print("YEAR-BY-YEAR TRADE RESULTS (Exit B: +4%/-5%/7d on S1)")
print(f"{'='*70}")

best = s1_results['B_4_5_7d']['trades_df'].copy()
best['year'] = best['event_date'].dt.year
for year, g in best.groupby('year'):
    n = len(g)
    wr = (g['exit_ret'] > 0).mean()
    avg = g['exit_ret'].mean()
    print(f"  {year}: n={n}, win_rate={wr:.1%}, avg_ret={avg:+.3%}")
