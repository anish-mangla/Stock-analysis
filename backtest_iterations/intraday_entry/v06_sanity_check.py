#!/usr/bin/env python3
"""
SANITY CHECK: Isolate the open-entry improvement from compounding.
Run with FIXED position sizing (1/8 of initial capital) to see the
true impact of open vs close entry without equity-based amplification.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results, INITIAL_CAPITAL
import pandas as pd

close, high, low, spy, events = load_cached_data()
open_prices = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'open_matrix.parquet'))

CRISIS_PERIODS = [(pd.Timestamp('2020-02-20'),pd.Timestamp('2020-04-15')),(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-10-31')),(pd.Timestamp('2025-03-01'),pd.Timestamp('2025-04-30'))]
BAD_TICKERS = {'PFE','TMUS','ACN','CVS','AMT','BLK','IBM','LMT','DHR','MSFT'}
BEST_TICKERS = {'NVDA','AMZN','AMAT','AVGO','GE','INTU','MA','AXP','LOW','MO','COF','BMY','MS','MDT','HON'}
BEST_SECTORS = {'SMH','XLK'}

def entry_filter(row, ctx):
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

# Test 1: Fixed sizing, close entry, +5% target (our original Phase 1 best)
print("="*60)
print("  SANITY CHECK: FIXED SIZING (no compounding)")
print("="*60)

print("\n--- CLOSE ENTRY (original Phase 1 config) ---")
r1 = run_backtest(events, close, high, low, entry_filter,
    target_return=0.05, max_hold_days=60, position_fraction=8,
    max_positions=8, entry_delay=2)
print_results(r1, "FIXED 1/8, close entry, +5%/60d")

print("\n--- CLOSE ENTRY, +7% target ---")
r2 = run_backtest(events, close, high, low, entry_filter,
    target_return=0.07, max_hold_days=45, position_fraction=8,
    max_positions=8, entry_delay=2)
print_results(r2, "FIXED 1/8, close entry, +7%/45d")

# Now we need to hack the engine to use open prices for entry
# The engine uses entry_price from events or close price
# We need a custom backtest that uses open prices

from dataclasses import dataclass
from typing import Dict, List
import json

trading_dates = pd.DatetimeIndex(sorted(close.index))

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_fixed_open(target_ret, max_hold, pos_frac, max_pos, slippage, label):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    fixed_size = INITIAL_CAPITAL / pos_frac
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades, daily, pending = [], [], []
    
    for today in trading_dates:
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
                    'exit_date': today, 'entry_price': pos.entry_price, 'exit_price': ep,
                    'return_pct': (ep/pos.entry_price)-1, 'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= max_pos or cash < fixed_size: continue
            if ticker not in open_prices.columns or today not in open_prices.index: continue
            v = open_prices.at[today, ticker]
            if pd.isna(v): continue
            ep = float(v) * (1 + slippage)
            if ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*(1+target_ret), max_exit_date=future[max_hold-1],
                shares=fixed_size/ep, allocated=fixed_size)
            cash -= fixed_size
        
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row, {}): continue
                ticker = row['ticker']
                if ticker in positions: continue
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1]})
        
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
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  TOTAL RETURN: {total_ret:+.1f}%")
    print(f"  Trades: {n} | WR: {wr:.0f}% | Losers: {losers} | Avg hold: {avg_hold:.0f}d")
    
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        print(f"    {year}: {yr_ret:+.1f}% ({yr_trades} trades, {yr_losers} losers)")
    
    return total_ret

print("\n\n--- OPEN ENTRY, FIXED SIZING (isolating open-entry impact) ---")

# Same as Phase 1 best but with open entry
run_fixed_open(0.05, 60, 8, 8, 0.0, "FIXED 1/8, OPEN entry, +5%/60d, 0% slip")
run_fixed_open(0.05, 60, 8, 8, 0.002, "FIXED 1/8, OPEN entry, +5%/60d, 0.2% slip")
run_fixed_open(0.07, 45, 8, 8, 0.0, "FIXED 1/8, OPEN entry, +7%/45d, 0% slip")
run_fixed_open(0.07, 45, 8, 8, 0.002, "FIXED 1/8, OPEN entry, +7%/45d, 0.2% slip")

# Now layer on adaptive sizing with open entry
print("\n\n--- OPEN ENTRY + ADAPTIVE SIZING (the full combo) ---")
run_fixed_open(0.07, 45, 8, 12, 0.002, "FIXED 1/8, OPEN, +7%/45d, 0.2% slip, max12")
