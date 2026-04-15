#!/usr/bin/env python3
"""
INTRADAY V02: Deep dive into "buy at open" strategy. V01 showed +6073% which
seems too good. Need to verify:
1. Is the open price actually available at entry time? (Yes - 2 day delay)
2. What's the slippage impact? (Test open + 0.1%, 0.2%, 0.5%)
3. Year-by-year breakdown
4. What if we use 1-day delay instead of 2-day?
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

def run_backtest(entry_price_fn, label, entry_delay=2):
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
                trades.append({'ticker': ticker, 'entry_date': pos.entry_date, 'exit_date': today,
                    'entry_price': pos.entry_price, 'exit_price': ep, 'pnl': pnl,
                    'return_pct': (ep / pos.entry_price) - 1, 'hold_days': days_held, 'exit_reason': er})
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
            if len(future) < 60: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep * 1.05, max_exit_date=future[59], shares=size / ep, allocated=size)
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
    
    print(f"\n  {label}: {total_ret:+.1f}% | {n} trades | WR {wr:.0f}% | {losers} losers")
    
    # Year breakdown
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        print(f"    {year}: {yr_ret:+.1f}% ({yr_trades} trades, {yr_losers} losers)")
    
    return {'label': label, 'return': round(total_ret, 2), 'trades': n, 'win_rate': round(wr, 1), 'losers': losers}

print("="*60)
print("  INTRADAY V02: OPEN ENTRY DEEP DIVE")
print("="*60)

# Buy at close (reference)
results = []
results.append(run_backtest(
    lambda t, d: float(close.at[d, t]) if t in close.columns and d in close.index and pd.notna(close.at[d, t]) else None,
    "Buy at CLOSE (2d delay)", entry_delay=2))

# Buy at open with various slippage
for slip in [0, 0.001, 0.002, 0.005, 0.01]:
    def make_open_fn(s):
        def fn(t, d):
            if t not in open_prices.columns or d not in open_prices.index: return None
            v = open_prices.at[d, t]
            return float(v) * (1 + s) if pd.notna(v) else None
        return fn
    slip_label = f"+{slip*100:.1f}%" if slip > 0 else ""
    results.append(run_backtest(make_open_fn(slip), f"Buy at OPEN{slip_label} (2d delay)", entry_delay=2))

# Buy at open with 1-day delay
results.append(run_backtest(
    lambda t, d: float(open_prices.at[d, t]) if t in open_prices.columns and d in open_prices.index and pd.notna(open_prices.at[d, t]) else None,
    "Buy at OPEN (1d delay)", entry_delay=1))

# Buy at open with 0-day delay (same day as event - probably not realistic)
results.append(run_backtest(
    lambda t, d: float(open_prices.at[d, t]) if t in open_prices.columns and d in open_prices.index and pd.notna(open_prices.at[d, t]) else None,
    "Buy at OPEN (0d delay - unrealistic)", entry_delay=0))

# Save
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'v02_open_entry_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(outdir, 'v02_open_entry_results.txt'), 'w') as f:
    f.write("INTRADAY V02: OPEN ENTRY DEEP DIVE\n\n")
    for r in results:
        f.write(f"{r['label']}: {r['return']:+.1f}% | {r['trades']} trades | WR {r['win_rate']:.0f}% | {r['losers']} losers\n")
print(f"\nSaved to {outdir}/v02_open_entry_results.txt")
