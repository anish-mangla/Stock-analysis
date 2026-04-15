#!/usr/bin/env python3
"""
V03 — VWAP entry vs close entry.
Buy at the day's VWAP instead of close. Also test: buy only if price is
below VWAP at 2pm (our best time from V02).
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
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'close', 'vwap', 'volume']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date

# Compute daily VWAP for each (symbol, date)
print("Computing daily VWAP...")
t0 = timer.time()
# VWAP = sum(price * volume) / sum(volume) across all bars
bars['pv'] = bars['vwap'] * bars['volume']
daily_vwap = bars.groupby(['symbol', 'date']).agg(
    total_pv=('pv', 'sum'),
    total_vol=('volume', 'sum'),
    close=('close', 'last'),
).reset_index()
daily_vwap['vwap'] = daily_vwap['total_pv'] / daily_vwap['total_vol'].replace(0, np.nan)

# Pivot to matrix
vwap_matrix = daily_vwap.pivot(index='date', columns='symbol', values='vwap')
vwap_matrix.index = pd.to_datetime(vwap_matrix.index)
if 'BRK.B' in vwap_matrix.columns:
    vwap_matrix = vwap_matrix.rename(columns={'BRK.B': 'BRK-B'})

# Also get price at 2pm (14:00 ET = 18:00 UTC = minute 1080)
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute
pm2 = bars[(bars['utc_min'] >= 1077) & (bars['utc_min'] <= 1083)]
pm2_best = pm2.sort_values('utc_min').groupby(['symbol', 'date']).first().reset_index()
pm2_matrix = pm2_best.pivot(index='date', columns='symbol', values='close')
pm2_matrix.index = pd.to_datetime(pm2_matrix.index)
if 'BRK.B' in pm2_matrix.columns:
    pm2_matrix = pm2_matrix.rename(columns={'BRK.B': 'BRK-B'})

# Running VWAP at 2pm (for "buy at 2pm only if below VWAP" strategy)
# Approximate: VWAP of bars up to 2pm
bars_before_2pm = bars[bars['utc_min'] <= 1080]
vwap_at_2pm = bars_before_2pm.groupby(['symbol', 'date']).agg(
    total_pv=('pv', 'sum'), total_vol=('volume', 'sum')
).reset_index()
vwap_at_2pm['vwap_2pm'] = vwap_at_2pm['total_pv'] / vwap_at_2pm['total_vol'].replace(0, np.nan)
vwap_2pm_matrix = vwap_at_2pm.pivot(index='date', columns='symbol', values='vwap_2pm')
vwap_2pm_matrix.index = pd.to_datetime(vwap_2pm_matrix.index)
if 'BRK.B' in vwap_2pm_matrix.columns:
    vwap_2pm_matrix = vwap_2pm_matrix.rename(columns={'BRK.B': 'BRK-B'})

print(f"Done in {timer.time()-t0:.1f}s")
del bars

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
from typing import Dict, List

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_bt(entry_price_mat, label, filter_fn=None):
    """filter_fn(ticker, today) -> bool, extra filter on entry day"""
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
                trades.append({'pnl': pnl, 'entry_date': pos.entry_date, 'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 10 or cash < pos_size: continue
            if filter_fn and not filter_fn(ticker, today): continue
            ep = None
            if today in entry_price_mat.index and ticker in entry_price_mat.columns:
                v = entry_price_mat.at[today, ticker]
                if pd.notna(v): ep = float(v)
            if ep is None and ticker in close_mat.columns:
                v = close_mat.at[today, ticker]
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
    return {'label': label, 'return': round(ret, 1), 'trades': n, 'wr': round(wr, 1), 'losers': losers}

# Filter: only buy at 2pm if price < running VWAP
def below_vwap_filter(ticker, today):
    if today in pm2_matrix.index and ticker in pm2_matrix.columns and \
       today in vwap_2pm_matrix.index and ticker in vwap_2pm_matrix.columns:
        price = pm2_matrix.at[today, ticker]
        vwap = vwap_2pm_matrix.at[today, ticker]
        if pd.notna(price) and pd.notna(vwap):
            return float(price) < float(vwap)
    return True  # fallback: allow

print(f"\n{'='*70}")
print(f"  V03: VWAP ENTRY STRATEGIES")
print(f"{'='*70}\n")

results = []
for mat, label, filt in [
    (close_mat, "Close (baseline)", None),
    (vwap_matrix, "Daily VWAP", None),
    (pm2_matrix, "2:00 PM price", None),
    (pm2_matrix, "2:00 PM, only if < VWAP", below_vwap_filter),
]:
    t0 = timer.time()
    r = run_bt(mat, label, filt)
    results.append(r)
    print(f"  {label:>30}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | {timer.time()-t0:.0f}s")

with open(os.path.join(OUT_DIR, 'v03_results.txt'), 'w') as f:
    f.write("V03: VWAP ENTRY STRATEGIES\n+5%/60d, fixed 1/10, no biases\n\n")
    for r in results:
        f.write(f"{r['label']:>30}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | {r['losers']} losers\n")
with open(os.path.join(OUT_DIR, 'v03_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/v03_results.txt")
