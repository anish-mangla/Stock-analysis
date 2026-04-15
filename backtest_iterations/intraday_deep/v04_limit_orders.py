#!/usr/bin/env python3
"""
V04 — Limit order at open minus X%.
Place limit at open-0.5%, -1%, -1.5%, -2%. If price dips to limit during day,
we get filled. If not, buy at close as fallback. Test fill rates and improvement.
"""
import pandas as pd
import numpy as np
import os, json, sys, time as timer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, INITIAL_CAPITAL

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

# For each (symbol, date): get open price and intraday low
print("Computing daily open and low...")
daily = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'),
    day_low=('low', 'min'),
    day_close=('close', 'last'),
).reset_index()

# Pivot
open_mat = daily.pivot(index='date', columns='symbol', values='day_open')
open_mat.index = pd.to_datetime(open_mat.index)
low_intraday = daily.pivot(index='date', columns='symbol', values='day_low')
low_intraday.index = pd.to_datetime(low_intraday.index)
close_intraday = daily.pivot(index='date', columns='symbol', values='day_close')
close_intraday.index = pd.to_datetime(close_intraday.index)

for m in [open_mat, low_intraday, close_intraday]:
    if 'BRK.B' in m.columns:
        m.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

del bars, daily
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

def run_limit_bt(pct_below_open, label):
    """Place limit at open*(1-pct). If low <= limit, filled at limit. Else buy at close."""
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    pos_size = INITIAL_CAPITAL / 10
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades, pending = [], []
    filled_at_limit = 0
    filled_at_close = 0
    
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
                trades.append({'pnl': pnl, 'entry_date': pos.entry_date, 'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 10 or cash < pos_size: continue
            
            # Get open and low for today
            o = open_mat.at[today, ticker] if today in open_mat.index and ticker in open_mat.columns else None
            l = low_intraday.at[today, ticker] if today in low_intraday.index and ticker in low_intraday.columns else None
            c = close_intraday.at[today, ticker] if today in close_intraday.index and ticker in close_intraday.columns else None
            
            if o is None or pd.isna(o):
                continue
            o = float(o)
            
            limit_price = o * (1 - pct_below_open)
            
            # Did the low reach our limit?
            if l is not None and not pd.isna(l) and float(l) <= limit_price:
                ep = limit_price
                filled_at_limit += 1
            elif c is not None and not pd.isna(c):
                ep = float(c)  # fallback to close
                filled_at_close += 1
            else:
                continue
            
            if ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < 60: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*1.05, max_exit_date=future[59], shares=pos_size/ep, allocated=pos_size)
            cash -= pos_size
        
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
    ret = tdf['pnl'].sum() / INITIAL_CAPITAL * 100 if n > 0 else 0
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    fill_rate = filled_at_limit / max(filled_at_limit + filled_at_close, 1) * 100
    
    return {'label': label, 'return': round(ret, 1), 'trades': n, 'wr': round(wr, 1),
            'losers': losers, 'limit_fills': filled_at_limit, 'close_fills': filled_at_close,
            'fill_rate': round(fill_rate, 1)}

print(f"\n{'='*70}")
print(f"  V04: LIMIT ORDER STRATEGIES")
print(f"{'='*70}\n")

results = []

# Baseline
r_base = run_limit_bt(0.0, "Open price (no limit)")
results.append(r_base)
print(f"  {'Open (no limit)':>25}: {r_base['return']:>+7.1f}% | {r_base['trades']} trades | WR {r_base['wr']:.0f}%")

# Close baseline
from engine import run_backtest as rb
r_close = rb(events, close_mat, high_mat, low_mat, entry_filter,
    target_return=0.05, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)
results.append({'label': 'Close (baseline)', 'return': r_close['total_return'], 'trades': r_close['trades'],
    'wr': r_close['win_rate'], 'losers': r_close['losers'], 'limit_fills': 0, 'close_fills': r_close['trades'], 'fill_rate': 0})
print(f"  {'Close (baseline)':>25}: {r_close['total_return']:>+7.1f}% | {r_close['trades']} trades | WR {r_close['win_rate']:.0f}%")

for pct in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    label = f"Limit open-{pct*100:.1f}%"
    r = run_limit_bt(pct, label)
    results.append(r)
    print(f"  {label:>25}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | fill rate {r['fill_rate']:.0f}%")

with open(os.path.join(OUT_DIR, 'v04_results.txt'), 'w') as f:
    f.write("V04: LIMIT ORDER STRATEGIES\n+5%/60d, fixed 1/10, fallback to close if limit not hit\n\n")
    f.write(f"{'Strategy':>25} {'Return':>8} {'Trades':>7} {'WR':>5} {'Limit%':>7}\n")
    f.write('-'*55 + '\n')
    for r in results:
        f.write(f"{r['label']:>25} {r['return']:>+7.1f}% {r['trades']:>7} {r['wr']:>4.0f}% {r.get('fill_rate',0):>6.0f}%\n")
with open(os.path.join(OUT_DIR, 'v04_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/v04_results.txt")
