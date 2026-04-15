#!/usr/bin/env python3
"""
INTRADAY V04: Realistic slippage testing on best configs.
The open price is theoretical - in practice you'll get filled slightly worse.
Test 0.1%, 0.2%, 0.3%, 0.5% slippage on the top configs.
Also test: what if we use VWAP proxy (weighted avg of OHLC)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict

close, high, low, spy, events = load_cached_data()
open_prices = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'open_matrix.parquet'))

spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()
trading_dates = pd.DatetimeIndex(sorted(close.index))

CRISIS_PERIODS = [(pd.Timestamp('2020-02-20'),pd.Timestamp('2020-04-15')),(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-10-31')),(pd.Timestamp('2025-03-01'),pd.Timestamp('2025-04-30'))]
BAD_TICKERS = {'PFE','TMUS','ACN','CVS','AMT','BLK','IBM','LMT','DHR','MSFT'}
BEST_TICKERS = {'NVDA','AMZN','AMAT','AVGO','GE','INTU','MA','AXP','LOW','MO','COF','BMY','MS','MDT','HON'}
BEST_SECTORS = {'SMH','XLK'}

def entry_filter(row):
    if row['ticker'] in BAD_TICKERS: return False
    dt = row['event_date']
    for s,e in CRISIS_PERIODS:
        if s <= dt <= e: return False
    t,sec,d = row['ticker'],row.get('sector_etf',''),row['drop_pct']
    out = row.get('sector_rotation_direction','')=='outflow'
    stress = row.get('liquidity_credit_stress_severity','none')
    fri = dt.dayofweek==4
    if t in BEST_TICKERS: return True
    if d>=0.07: return True
    if d>=0.05 and (out or sec in BEST_SECTORS or fri or stress in ['medium','low']): return True
    if sec in BEST_SECTORS and out: return True
    if fri and stress=='medium': return True
    return False

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_bt(entry_price_fn, target_ret, max_hold, label):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades, daily, pending = [], [], []
    
    for today in trading_dates:
        equity = cash + sum(
            pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker]))
        
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
        
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in close.columns: continue
            tc = close.at[today, ticker]
            th = high.at[today, ticker] if ticker in high.columns else None
            if pd.isna(tc): continue
            days_held = len(trading_dates[(trading_dates > pos.entry_date) & (trading_dates <= today)])
            er, ep = None, None
            if th is not None and not pd.isna(th) and float(th) >= pos.target_price:
                er, ep = 'target', pos.target_price
            elif today >= pos.max_exit_date:
                er, ep = 'max_hold', float(tc)
            if er:
                pnl = (ep - pos.entry_price) * pos.shares
                trades.append({'ticker': ticker, 'pnl': pnl, 'entry_date': pos.entry_date,
                    'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 12 or cash < entry['size']: continue
            ep = entry_price_fn(ticker, today)
            if ep is None or ep <= 0: continue
            size = min(entry['size'], cash)
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep * (1 + target_ret), max_exit_date=future[max_hold - 1],
                shares=size / ep, allocated=size)
            cash -= size
        
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row): continue
                ticker = row['ticker']
                if ticker in positions: continue
                size = equity / 6 if (today in spy_close.index and today in spy_20ma.index and 
                    pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and 
                    spy_close.at[today] > spy_20ma.at[today]) else equity / 14
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1], 'size': size})
        
        mv = sum(pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker]))
        daily.append({'date': today, 'equity': cash + mv})
    
    edf = pd.DataFrame(daily)
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total_ret = (edf['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n = len(tdf)
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    
    print(f"  {label}: {total_ret:+.1f}% | {n} trades | WR {wr:.0f}% | hold {avg_hold:.0f}d")
    return {'label': label, 'return': round(total_ret, 2), 'trades': n, 'wr': round(wr, 1)}

def make_open_slip(slip):
    def fn(t, d):
        if t not in open_prices.columns or d not in open_prices.index: return None
        v = open_prices.at[d, t]
        return float(v) * (1 + slip) if pd.notna(v) else None
    return fn

# VWAP proxy: (Open + High + Low + Close) / 4 (typical price)
def vwap_proxy(t, d):
    if t not in open_prices.columns or d not in open_prices.index: return None
    o = open_prices.at[d, t]
    h = high.at[d, t] if t in high.columns and d in high.index else None
    l = low.at[d, t] if t in low.columns and d in low.index else None
    c = close.at[d, t] if t in close.columns and d in close.index else None
    if any(x is None or pd.isna(x) for x in [o, h, l, c]): return None
    return (float(o) + float(h) + float(l) + float(c)) / 4

print("="*60)
print("  INTRADAY V04: SLIPPAGE TESTING")
print("="*60)

results = []

# Best config: +7% target, 30d hold
print("\n--- +7% target, 30d hold ---")
for slip in [0, 0.001, 0.002, 0.003, 0.005, 0.01]:
    r = run_bt(make_open_slip(slip), 0.07, 30, f"+7%/30d, open+{slip*100:.1f}%")
    results.append(r)

# +5% target, 60d hold
print("\n--- +5% target, 60d hold ---")
for slip in [0, 0.001, 0.002, 0.003, 0.005, 0.01]:
    r = run_bt(make_open_slip(slip), 0.05, 60, f"+5%/60d, open+{slip*100:.1f}%")
    results.append(r)

# VWAP proxy
print("\n--- VWAP proxy entry ---")
r = run_bt(vwap_proxy, 0.05, 60, "VWAP proxy, +5%/60d")
results.append(r)
r = run_bt(vwap_proxy, 0.07, 30, "VWAP proxy, +7%/30d")
results.append(r)

# Save
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'v04_slippage_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(outdir, 'v04_slippage_results.txt'), 'w') as f:
    f.write("INTRADAY V04: SLIPPAGE TESTING\n\n")
    for r in results:
        f.write(f"{r['label']}: {r['return']:+.1f}% | {r['trades']} trades | WR {r['wr']:.0f}%\n")
print(f"\nSaved to {outdir}/v04_slippage_results.txt")
