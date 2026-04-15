#!/usr/bin/env python3
"""
V02 — Buy at specific times vs close.
Optimized: pre-compute entry prices for all times, then run backtests.
"""
import pandas as pd
import numpy as np
import os, json, sys, time as timer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, print_results, save_results, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

# Entry times (ET -> UTC): 9:35=13:35=815, etc.
ENTRY_TIMES = {
    '09:35': 815, '09:45': 825, '10:00': 840, '10:30': 870,
    '11:00': 900, '12:00': 960, '13:00': 1020, '14:00': 1080,
    '15:00': 1140, '15:30': 1170,
}

# Pre-compute: for each (ticker, date), get the price at each entry time
# This is the expensive part — do it once
print("Loading intraday data and building price matrices...")
t0 = timer.time()

all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'close']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute
print(f"Loaded {len(bars):,} bars in {timer.time()-t0:.1f}s")

# For each entry time, build a (date, ticker) -> price matrix
# that matches the format of close_mat
t0 = timer.time()
entry_price_matrices = {}

for label, target_min in ENTRY_TIMES.items():
    # Get bars within +/- 3 minutes of target
    nearby = bars[(bars['utc_min'] >= target_min - 3) & (bars['utc_min'] <= target_min + 3)].copy()
    # For each (symbol, date), pick the bar closest to target_min
    nearby['dist'] = (nearby['utc_min'] - target_min).abs()
    best = nearby.sort_values('dist').groupby(['symbol', 'date']).first().reset_index()
    
    # Pivot to matrix format matching close_mat
    price_df = best.pivot(index='date', columns='symbol', values='close')
    price_df.index = pd.to_datetime(price_df.index)
    
    # Rename BRK.B back to BRK-B to match events
    if 'BRK.B' in price_df.columns:
        price_df = price_df.rename(columns={'BRK.B': 'BRK-B'})
    
    entry_price_matrices[label] = price_df

print(f"Built {len(entry_price_matrices)} price matrices in {timer.time()-t0:.1f}s")
del bars  # free memory

# Entry filter (no look-ahead biases)
def entry_filter(row, ctx):
    d = row['drop_pct']
    out = row.get('sector_rotation_direction', '') == 'outflow'
    stress = row.get('liquidity_credit_stress_severity', 'none')
    fri = row['event_date'].dayofweek == 4
    if d >= 0.07: return True
    if d >= 0.05 and (out or fri or stress in ['medium', 'low']): return True
    if d >= 0.03 and out and fri: return True
    if d >= 0.03 and stress == 'medium': return True
    return False

# Now run backtests — we need a modified engine that uses our price matrix for entry
# but still uses close_mat/high_mat for exits (since we hold for days)

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_timed_backtest(entry_prices, label):
    """Like engine.run_backtest but uses entry_prices matrix for entry."""
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    pos_size = INITIAL_CAPITAL / 10
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades: List[Dict] = []
    pending = []
    
    for today in trading_dates:
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
        
        # Exits
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in close_mat.columns: continue
            tc = close_mat.at[today, ticker]
            th = high_mat.at[today, ticker] if ticker in high_mat.columns else None
            if pd.isna(tc): continue
            days_held = len(trading_dates[(trading_dates > pos.entry_date) & (trading_dates <= today)])
            er, ep = None, None
            if th is not None and not pd.isna(th) and float(th) >= pos.target_price:
                er, ep = 'target', pos.target_price
            elif today >= pos.max_exit_date:
                er, ep = 'max_hold', float(tc)
            if er:
                pnl = (ep - pos.entry_price) * pos.shares
                trades.append({'pnl': pnl, 'entry_date': pos.entry_date, 'exit_date': today,
                    'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        # Process entries
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 10 or cash < pos_size: continue
            # Get entry price from our time-specific matrix
            ep = None
            if today in entry_prices.index and ticker in entry_prices.columns:
                v = entry_prices.at[today, ticker]
                if pd.notna(v):
                    ep = float(v)
            # Fallback to close
            if ep is None and ticker in close_mat.columns and today in close_mat.index:
                v = close_mat.at[today, ticker]
                if pd.notna(v):
                    ep = float(v)
            if ep is None or ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < 60: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*1.05, max_exit_date=future[59], shares=pos_size/ep, allocated=pos_size)
            cash -= pos_size
        
        # New candidates
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row, {}): continue
                ticker = row['ticker']
                if ticker in positions: continue
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1]})
    
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n = len(tdf)
    total_pnl = tdf['pnl'].sum() if n > 0 else 0
    ret = total_pnl / INITIAL_CAPITAL * 100
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    
    # Annual
    annual = {}
    if n > 0:
        tdf['year'] = pd.to_datetime(tdf['entry_date']).dt.year
        for yr, g in tdf.groupby('year'):
            annual[int(yr)] = round(float(g['pnl'].sum() / INITIAL_CAPITAL * 100), 1)
    
    return {'time': label, 'return': round(ret, 1), 'trades': n, 'wr': round(wr, 1),
            'losers': losers, 'hold': round(avg_hold, 1), 'annual': annual}

# Also run with close_mat as baseline
print(f"\n{'='*70}")
print(f"  V02: ENTRY TIME BACKTEST (+5%/60d, fixed 1/10, no biases)")
print(f"{'='*70}\n")

results = []

# Baseline: buy at close (use engine directly)
t0 = timer.time()
r_close = run_backtest(events, close_mat, high_mat, low_mat, entry_filter,
    target_return=0.05, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)
results.append({'time': 'close', 'return': r_close['total_return'], 'trades': r_close['trades'],
    'wr': r_close['win_rate'], 'losers': r_close['losers'], 'hold': r_close['avg_hold_days'],
    'annual': {int(k): round(v['return'], 1) for k, v in r_close['annual'].items()}})
print(f"  {'close':>8}: {r_close['total_return']:>+7.1f}% | {r_close['trades']} trades | WR {r_close['win_rate']:.0f}% | {timer.time()-t0:.0f}s")

# Each entry time
for label, _ in ENTRY_TIMES.items():
    t0 = timer.time()
    r = run_timed_backtest(entry_price_matrices[label], label)
    results.append(r)
    print(f"  {label:>8}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | {timer.time()-t0:.0f}s")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"{'Time':>8} {'Return':>8} {'Trades':>7} {'WR':>5} {'Losers':>7}")
print('-'*40)
for r in sorted(results, key=lambda x: -x['return']):
    print(f"{r['time']:>8} {r['return']:>+7.1f}% {r['trades']:>7} {r['wr']:>4.0f}% {r['losers']:>7}")

best = max(results, key=lambda x: x['return'])
close_r = [r for r in results if r['time'] == 'close'][0]
print(f"\nBest: {best['time']} ({best['return']:+.1f}%)")
print(f"vs close: {best['return'] - close_r['return']:+.1f}% difference")

# Save
with open(os.path.join(OUT_DIR, 'v02_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
with open(os.path.join(OUT_DIR, 'v02_results.txt'), 'w') as f:
    f.write("V02: ENTRY TIME BACKTEST\n+5%/60d, fixed 1/10, no look-ahead biases\n\n")
    f.write(f"{'Time':>8} {'Return':>8} {'Trades':>7} {'WR':>5} {'Losers':>7}\n")
    f.write('-'*40 + '\n')
    for r in sorted(results, key=lambda x: -x['return']):
        f.write(f"{r['time']:>8} {r['return']:>+7.1f}% {r['trades']:>7} {r['wr']:>4.0f}% {r['losers']:>7}\n")
    f.write(f"\nBest: {best['time']} ({best['return']:+.1f}%)\n")
    f.write(f"vs close: {best['return'] - close_r['return']:+.1f}%\n")
    f.write(f"\nAnnual breakdown (best = {best['time']}):\n")
    for yr, ret in sorted(best['annual'].items()):
        f.write(f"  {yr}: {ret:+.1f}%\n")
print(f"\nSaved to {OUT_DIR}/v02_results.txt")
