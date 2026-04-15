#!/usr/bin/env python3
"""
INTRADAY V07: Re-run with FIXED open_matrix (same adjustment as close/high/low).
Previous results were INVALID due to mismatched price adjustments.
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

# First verify the data is now consistent
print("DATA CONSISTENCY CHECK:")
d = pd.Timestamp('2020-02-04')
for t in ['CVX', 'XOM', 'AAPL', 'NVDA']:
    o = float(open_prices.at[d, t])
    c = float(close.at[d, t])
    h = float(high.at[d, t])
    l = float(low.at[d, t])
    print(f"  {t}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} | O/C={o/c:.4f}")
print()

# Check the real open-vs-close gap
gaps = []
for _, row in events.iterrows():
    t = row['ticker']
    dt = row['event_date']
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    entry_day = future[1]
    if t not in close.columns or t not in open_prices.columns: continue
    if entry_day not in close.index or entry_day not in open_prices.index: continue
    c = close.at[entry_day, t]
    o = open_prices.at[entry_day, t]
    if pd.isna(c) or pd.isna(o): continue
    c, o = float(c), float(o)
    if c > 0:
        gaps.append((o - c) / c * 100)

gs = pd.Series(gaps)
print(f"REAL open vs close gap on entry days (T+2):")
print(f"  Mean: {gs.mean():+.2f}%  Median: {gs.median():+.2f}%")
print(f"  This is the TRUE advantage of buying at open vs close")
print()

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_bt(entry_type, target_ret, max_hold, sizing_mode, label):
    """entry_type: 'close' or 'open'"""
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
            
            if entry_type == 'open':
                if ticker not in open_prices.columns or today not in open_prices.index: continue
                ep = open_prices.at[today, ticker]
            else:
                if ticker not in close.columns or today not in close.index: continue
                ep = close.at[today, ticker]
            
            if pd.isna(ep): continue
            ep = float(ep)
            if ep <= 0: continue
            size = min(entry['size'], cash)
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*(1+target_ret), max_exit_date=future[max_hold-1],
                shares=size/ep, allocated=size)
            cash -= size
        
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row): continue
                ticker = row['ticker']
                if ticker in positions: continue
                if sizing_mode == 'adaptive':
                    is_bull = (today in spy_close.index and today in spy_20ma.index and 
                        pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and 
                        spy_close.at[today] > spy_20ma.at[today])
                    size = equity / 6 if is_bull else equity / 14
                else:
                    size = INITIAL_CAPITAL / 8
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1], 'size': size})
        
        mv = sum(pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker]))
        daily.append({'date': today, 'equity': cash + mv})
    
    edf = pd.DataFrame(daily)
    edf['date'] = pd.to_datetime(edf['date'])
    edf['year'] = edf['date'].dt.year
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total_ret = (edf['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n = len(tdf)
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    
    print(f"  {label}: {total_ret:+.1f}% | {n} trades | WR {wr:.0f}% | hold {avg_hold:.0f}d | {losers} losers")
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        print(f"    {year}: {yr_ret:+.1f}% ({yr_trades} trades, {yr_losers} losers)")
    
    return {'label': label, 'return': round(total_ret, 2), 'trades': n, 'wr': round(wr, 1), 'losers': losers}

print("="*60)
print("  INTRADAY V07: CORRECTED DATA")
print("="*60)

results = []

# Fixed sizing comparisons
print("\n--- FIXED SIZING (1/8, max 8) ---")
results.append(run_bt('close', 0.05, 60, 'fixed', "Close entry, +5%/60d, fixed"))
results.append(run_bt('open', 0.05, 60, 'fixed', "Open entry, +5%/60d, fixed"))
results.append(run_bt('close', 0.07, 45, 'fixed', "Close entry, +7%/45d, fixed"))
results.append(run_bt('open', 0.07, 45, 'fixed', "Open entry, +7%/45d, fixed"))

# Adaptive sizing comparisons
print("\n--- ADAPTIVE SIZING (1/6 bull, 1/14 bear, 20MA, max 12) ---")
results.append(run_bt('close', 0.05, 60, 'adaptive', "Close entry, +5%/60d, adaptive"))
results.append(run_bt('open', 0.05, 60, 'adaptive', "Open entry, +5%/60d, adaptive"))
results.append(run_bt('close', 0.07, 45, 'adaptive', "Close entry, +7%/45d, adaptive"))
results.append(run_bt('open', 0.07, 45, 'adaptive', "Open entry, +7%/45d, adaptive"))

# Save
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'v07_fixed_data_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(outdir, 'v07_fixed_data_results.txt'), 'w') as f:
    f.write("INTRADAY V07: CORRECTED DATA (open_matrix fixed to match close/high/low)\n")
    f.write("Previous V01-V05 results were INVALID due to price adjustment mismatch\n\n")
    for r in results:
        f.write(f"{r['label']}: {r['return']:+.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | {r['losers']} losers\n")
print(f"\nSaved to {outdir}/v07_fixed_data_results.txt")
