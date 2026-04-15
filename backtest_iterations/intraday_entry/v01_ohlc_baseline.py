#!/usr/bin/env python3
"""
INTRADAY V01: Baseline analysis using daily OHLC data. Compare entry strategies:
1. Buy at close (current approach)
2. Buy at open (next day)
3. Buy at midpoint of day's range
4. Limit order at various levels between open and low
Uses best config: 20MA regime, 1/6 bull, 1/14 bear, equity-based, 2-day delay.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL
import pandas as pd
import numpy as np
import json

close, high, low, spy, events = load_cached_data()

# Also need open prices
import yfinance as yf
# Check if we have open matrix cached
open_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'open_matrix.parquet')
if os.path.exists(open_path):
    open_prices = pd.read_parquet(open_path)
else:
    print("Caching open prices...")
    tickers = list(close.columns)
    data = yf.download(tickers, start='2019-01-01', end='2026-01-01', progress=False)
    open_prices = data['Open']
    open_prices.index = open_prices.index.tz_localize(None)
    open_prices.to_parquet(open_path)
    print(f"Saved open matrix: {open_prices.shape}")

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

# First, let's analyze the OHLC patterns on entry days
# For each trade in our best config, what does the entry day look like?
print("="*60)
print("  INTRADAY ENTRY ANALYSIS - OHLC PATTERNS")
print("="*60)

# Get all candidate events that pass our filter
candidates = []
for _, row in events.iterrows():
    if not entry_filter(row): continue
    ticker = row['ticker']
    dt = row['event_date']
    # Entry is 2 days later
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    entry_day = future[1]  # 2-day delay
    
    if ticker not in close.columns or ticker not in open_prices.columns: continue
    if ticker not in high.columns or ticker not in low.columns: continue
    
    c = close.at[entry_day, ticker] if entry_day in close.index else None
    o = open_prices.at[entry_day, ticker] if entry_day in open_prices.index else None
    h = high.at[entry_day, ticker] if entry_day in high.index else None
    l = low.at[entry_day, ticker] if entry_day in low.index else None
    
    if any(x is None or pd.isna(x) for x in [c, o, h, l]): continue
    c, o, h, l = float(c), float(o), float(h), float(l)
    if o <= 0 or c <= 0: continue
    
    candidates.append({
        'ticker': ticker, 'event_date': dt, 'entry_day': entry_day,
        'open': o, 'high': h, 'low': l, 'close': c,
        'range_pct': (h - l) / o * 100,
        'open_vs_close': (o - c) / c * 100,  # positive = open above close
        'low_vs_close': (l - c) / c * 100,   # negative = low below close
        'midpoint': (h + l) / 2,
    })

cdf = pd.DataFrame(candidates)
print(f"\nAnalyzed {len(cdf)} entry days")
print(f"\nEntry day OHLC stats:")
print(f"  Avg daily range: {cdf['range_pct'].mean():.2f}%")
print(f"  Avg open vs close: {cdf['open_vs_close'].mean():+.2f}%")
print(f"  Avg low vs close: {cdf['low_vs_close'].mean():+.2f}%")
print(f"  Median low vs close: {cdf['low_vs_close'].median():+.2f}%")

# What % of the time is the open below the close?
open_below_close = (cdf['open'] < cdf['close']).mean() * 100
print(f"\n  Open < Close (gap up day): {open_below_close:.1f}%")
print(f"  Open > Close (gap down day): {100-open_below_close:.1f}%")

# How much cheaper could we buy at the low vs close?
savings = ((cdf['close'] - cdf['low']) / cdf['close'] * 100)
print(f"\n  If we bought at LOW instead of CLOSE:")
print(f"    Avg savings: {savings.mean():.2f}%")
print(f"    Median savings: {savings.median():.2f}%")
print(f"    75th percentile: {savings.quantile(0.75):.2f}%")

# How much cheaper at open vs close?
open_savings = ((cdf['close'] - cdf['open']) / cdf['close'] * 100)
print(f"\n  If we bought at OPEN instead of CLOSE:")
print(f"    Avg savings: {open_savings.mean():+.2f}%")
print(f"    Median savings: {open_savings.median():+.2f}%")

# Limit order fill rates at various levels below close
print(f"\n  LIMIT ORDER FILL RATES (below previous close):")
for pct in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    # Would a limit order at close - X% have filled?
    limit_price = cdf['close'] * (1 - pct/100)  # this is wrong, should be entry day
    # Actually: limit at open - X% of open
    limit = cdf['open'] * (1 - pct/100)
    filled = (cdf['low'] <= limit).mean() * 100
    avg_savings_if_filled = ((cdf['close'] - limit) / cdf['close'])[cdf['low'] <= limit].mean() * 100
    print(f"    Limit at open-{pct}%: fills {filled:.1f}% of time, avg savings {avg_savings_if_filled:+.2f}% vs close")

# Now simulate different entry strategies with full backtest
print(f"\n{'='*60}")
print(f"  FULL BACKTEST WITH DIFFERENT ENTRY PRICES")
print(f"{'='*60}")

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_entry_backtest(entry_price_fn, label):
    """entry_price_fn(ticker, entry_day) -> price or None (skip if limit not filled)"""
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades = []
    daily = []
    pending = []
    
    for today in trading_dates:
        equity = cash + sum(
            pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker])
        )
        
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
        
        # Exits
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
        
        # Process entries
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 12 or cash < entry['size']:
                continue
            ep = entry_price_fn(ticker, today)
            if ep is None: continue  # limit order not filled
            if ep <= 0: continue
            size = min(entry['size'], cash)
            future = trading_dates[trading_dates > today]
            if len(future) < 60: continue
            positions[ticker] = Pos(
                ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep * 1.05, max_exit_date=future[59],
                shares=size / ep, allocated=size)
            cash -= size
        
        # New candidates
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
    
    print(f"  {label}: {total_ret:+.1f}% | {n} trades | WR {wr:.0f}% | {losers} losers")
    return {'label': label, 'return': round(total_ret, 2), 'trades': n, 'win_rate': round(wr, 1), 'losers': losers}

# Strategy 1: Buy at close (current)
def buy_at_close(ticker, day):
    if ticker in close.columns and day in close.index:
        v = close.at[day, ticker]
        return float(v) if pd.notna(v) else None
    return None

# Strategy 2: Buy at open
def buy_at_open(ticker, day):
    if ticker in open_prices.columns and day in open_prices.index:
        v = open_prices.at[day, ticker]
        return float(v) if pd.notna(v) else None
    return None

# Strategy 3: Buy at midpoint
def buy_at_mid(ticker, day):
    if ticker in high.columns and ticker in low.columns and day in high.index:
        h = high.at[day, ticker]
        l = low.at[day, ticker]
        if pd.notna(h) and pd.notna(l):
            return (float(h) + float(l)) / 2
    return None

# Strategy 4: Limit order at open - 1%
def make_limit_fn(pct_below_open):
    def fn(ticker, day):
        if ticker not in open_prices.columns or day not in open_prices.index: return None
        if ticker not in low.columns or day not in low.index: return None
        o = open_prices.at[day, ticker]
        l = low.at[day, ticker]
        if pd.isna(o) or pd.isna(l): return None
        limit = float(o) * (1 - pct_below_open)
        if float(l) <= limit:
            return limit  # filled
        return None  # not filled
    return fn

results = []
results.append(run_entry_backtest(buy_at_close, "Buy at CLOSE (current)"))
results.append(run_entry_backtest(buy_at_open, "Buy at OPEN"))
results.append(run_entry_backtest(buy_at_mid, "Buy at MIDPOINT"))
for pct in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    results.append(run_entry_backtest(make_limit_fn(pct), f"Limit at open-{pct*100:.1f}%"))

# Save
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'v01_ohlc_baseline_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(outdir, 'v01_ohlc_baseline_results.txt'), 'w') as f:
    f.write("INTRADAY ENTRY OPTIMIZATION V01 - OHLC BASELINE\n\n")
    for r in results:
        f.write(f"{r['label']}: {r['return']:+.1f}% | {r['trades']} trades | WR {r['win_rate']:.0f}% | {r['losers']} losers\n")
print(f"\nResults saved to {outdir}/v01_ohlc_baseline_results.txt")
