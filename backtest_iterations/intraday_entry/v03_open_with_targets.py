#!/usr/bin/env python3
"""
INTRADAY V03: Open entry is a game-changer (+6073% vs +325% at close).
Now test: since we're entering cheaper, can we use tighter targets for faster exits?
Also test: what if we enter at open but exit at open too (more realistic)?
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

def run_bt(target_ret, max_hold, label, entry_delay=2):
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
                    'return_pct': (ep / pos.entry_price) - 1, 'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 12 or cash < entry['size']: continue
            # Buy at open
            if ticker not in open_prices.columns or today not in open_prices.index: continue
            ep = open_prices.at[today, ticker]
            if pd.isna(ep): continue
            ep = float(ep)
            if ep <= 0: continue
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
                if len(future) >= entry_delay:
                    pending.append({'ticker': ticker, 'enter_on': future[entry_delay - 1], 'size': size})
        
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
    target_hits = len(tdf[tdf['exit_reason'] == 'target']) if n > 0 else 0
    
    print(f"  {label}: {total_ret:+.1f}% | {n} trades | WR {wr:.0f}% | {losers} losers | avg hold {avg_hold:.0f}d | targets {target_hits}")
    
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        print(f"    {year}: {yr_ret:+.1f}% ({yr_trades} trades)")
    
    return {'label': label, 'return': round(total_ret, 2), 'trades': n, 'wr': round(wr, 1), 
            'losers': losers, 'avg_hold': round(avg_hold, 1), 'target_hits': target_hits}

print("="*60)
print("  INTRADAY V03: OPEN ENTRY + TARGET SWEEP")
print("="*60)

results = []
# Different targets with open entry
for target in [0.03, 0.05, 0.07, 0.10]:
    for hold in [30, 45, 60]:
        r = run_bt(target, hold, f"Open entry, +{target*100:.0f}% target, {hold}d hold")
        results.append(r)
        print()

# Save
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'v03_open_targets_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(outdir, 'v03_open_targets_results.txt'), 'w') as f:
    f.write("INTRADAY V03: OPEN ENTRY + TARGET SWEEP\n\n")
    for r in results:
        f.write(f"{r['label']}: {r['return']:+.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | avg hold {r['avg_hold']:.0f}d\n")
print(f"\nSaved to {outdir}/v03_open_targets_results.txt")
