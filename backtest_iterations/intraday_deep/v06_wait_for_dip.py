#!/usr/bin/env python3
"""
V06 — Wait for intraday dip before buying.
Instead of buying immediately, wait for stock to dip X% below open during day.
If it never dips, buy at close. Test X = 0.5%, 1%, 1.5%, 2%.
Also test: wait for dip, then buy when it recovers 0.5% from the dip low.
"""
import pandas as pd
import numpy as np
import os, json, sys, time as timer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

print("Loading intraday data...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'low', 'close']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date

# For each (symbol, date): compute open, low, close, and "dip then recover" prices
print("Computing dip metrics...")
daily = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'),
    day_low=('low', 'min'),
    day_close=('close', 'last'),
).reset_index()

# For "dip then recover": need to find the price AFTER the low
# This requires per-bar processing. Let's compute for each day:
# - The time of the intraday low
# - The price 30 minutes after the low
print("Computing post-dip recovery prices...")
bars_sorted = bars.sort_values(['symbol', 'date', 'timestamp'])
recovery_data = []

for (sym, date), group in bars_sorted.groupby(['symbol', 'date']):
    if len(group) < 50: continue
    low_idx = group['low'].idxmin()
    low_price = float(group.loc[low_idx, 'low'])
    low_time = group.loc[low_idx, 'timestamp']
    open_price = float(group.iloc[0]['open'])
    close_price = float(group.iloc[-1]['close'])
    
    # Price 30 min after the low
    after_low = group[group['timestamp'] > low_time]
    if len(after_low) >= 30:
        price_30m_after = float(after_low.iloc[29]['close'])
    elif len(after_low) > 0:
        price_30m_after = float(after_low.iloc[-1]['close'])
    else:
        price_30m_after = close_price
    
    recovery_data.append({
        'symbol': sym, 'date': date,
        'open': open_price, 'low': low_price, 'close': close_price,
        'dip_pct': (low_price / open_price - 1) * 100,
        'recovery_price': price_30m_after,
        'recovery_from_low': (price_30m_after / low_price - 1) * 100 if low_price > 0 else 0,
    })

rdf = pd.DataFrame(recovery_data)
del bars, bars_sorted

# Build entry price matrices for each strategy
# Strategy: if dip >= X%, entry = low * 1.005 (buy on recovery). Else entry = close.
for dip_thresh in [0.5, 1.0, 1.5, 2.0]:
    col = f'entry_dip{dip_thresh}'
    rdf[col] = np.where(
        rdf['dip_pct'] <= -dip_thresh,
        rdf['low'] * 1.005,  # buy at low + 0.5% (confirmation of recovery)
        rdf['close']  # fallback
    )

# Pivot all entry strategies
entry_mats = {}
for dip_thresh in [0.5, 1.0, 1.5, 2.0]:
    col = f'entry_dip{dip_thresh}'
    mat = rdf.pivot(index='date', columns='symbol', values=col)
    mat.index = pd.to_datetime(mat.index)
    if 'BRK.B' in mat.columns:
        mat.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)
    entry_mats[dip_thresh] = mat

# Also: "buy at recovery price" (30 min after low, regardless of dip size)
mat_recovery = rdf.pivot(index='date', columns='symbol', values='recovery_price')
mat_recovery.index = pd.to_datetime(mat_recovery.index)
if 'BRK.B' in mat_recovery.columns:
    mat_recovery.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

print("Done")

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

from dataclasses import dataclass
from typing import Dict

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_bt(entry_mat, label):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    pos_size = INITIAL_CAPITAL / 10
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades, pending = [], []
    
    for today in trading_dates:
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
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
                trades.append({'pnl': pnl})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 10 or cash < pos_size: continue
            ep = None
            if today in entry_mat.index and ticker in entry_mat.columns:
                v = entry_mat.at[today, ticker]
                if pd.notna(v): ep = float(v)
            if ep is None or ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < 60: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*1.05, max_exit_date=future[59], shares=pos_size/ep, allocated=pos_size)
            cash -= pos_size
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row, {}): continue
                if row['ticker'] in positions: continue
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': row['ticker'], 'enter_on': future[1]})
    
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n = len(tdf)
    ret = tdf['pnl'].sum() / INITIAL_CAPITAL * 100 if n > 0 else 0
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    return {'label': label, 'return': round(ret, 1), 'trades': n, 'wr': round(wr, 1)}

print(f"\n{'='*70}")
print(f"  V06: WAIT FOR INTRADAY DIP")
print(f"{'='*70}\n")

results = []
# Baseline
from engine import run_backtest as rb
r_close = rb(events, close_mat, high_mat, low_mat, entry_filter,
    target_return=0.05, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)
results.append({'label': 'Close (baseline)', 'return': r_close['total_return'], 'trades': r_close['trades'], 'wr': r_close['win_rate']})
print(f"  {'Close (baseline)':>35}: {r_close['total_return']:>+7.1f}% | {r_close['trades']} trades | WR {r_close['win_rate']:.0f}%")

for dip in [0.5, 1.0, 1.5, 2.0]:
    r = run_bt(entry_mats[dip], f"Dip {dip}% then buy (or close)")
    results.append(r)
    print(f"  {r['label']:>35}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}%")

r = run_bt(mat_recovery, "Buy 30min after intraday low")
results.append(r)
print(f"  {r['label']:>35}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}%")

# Dip analysis
print(f"\n  Dip frequency on entry days:")
for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
    pct = (rdf['dip_pct'] <= -thresh).mean() * 100
    print(f"    Dips >= {thresh}%: {pct:.1f}% of days")

with open(os.path.join(OUT_DIR, 'v06_results.txt'), 'w') as f:
    f.write("V06: WAIT FOR INTRADAY DIP\n\n")
    for r in results:
        f.write(f"{r['label']:>35}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}%\n")
with open(os.path.join(OUT_DIR, 'v06_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/v06_results.txt")
