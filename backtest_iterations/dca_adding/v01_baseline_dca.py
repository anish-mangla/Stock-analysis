#!/usr/bin/env python3
"""
DCA V01 BASELINE: Test adding to positions when stock drops further after entry.
Uses the best sizing config (20MA, 1/6 bull, 1/14 bear, equity-based).
Strategy: If a position drops 3%+ from entry, add another half-position.
Max 1 add per position. This lowers average cost and increases recovery odds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

close, high, low, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()
trading_dates = pd.DatetimeIndex(sorted(close.index))

CRISIS_PERIODS = [(pd.Timestamp('2020-02-20'),pd.Timestamp('2020-04-15')),(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-10-31')),(pd.Timestamp('2025-03-01'),pd.Timestamp('2025-04-30'))]
BAD_TICKERS = {'PFE','TMUS','ACN','CVS','AMT','BLK','IBM','LMT','DHR','MSFT'}
BEST_TICKERS = {'NVDA','AMZN','AMAT','AVGO','GE','INTU','MA','AXP','LOW','MO','COF','BMY','MS','MDT','HON'}
BEST_SECTORS = {'SMH','XLK'}

@dataclass
class DCAPosition:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_return: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float; adds: int = 0
    avg_price: float = 0.0
    
    def __post_init__(self):
        if self.avg_price == 0:
            self.avg_price = self.entry_price

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

def get_position_size(today, equity):
    if today in spy_close.index and today in spy_20ma.index:
        s = spy_close.at[today]
        m = spy_20ma.at[today]
        if pd.notna(s) and pd.notna(m) and s > m:
            return equity / 6
    return equity / 14

def run_dca_backtest(dca_threshold=-0.03, dca_fraction=0.5, max_adds=1):
    """
    dca_threshold: how much the position must drop before adding (e.g., -0.03 = -3%)
    dca_fraction: fraction of original position size to add (0.5 = half position)
    max_adds: max number of times to add to a position
    """
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    positions: Dict[str, DCAPosition] = {}
    trades = []
    daily = []
    pending = []
    max_positions = 12
    target_return = 0.05
    max_hold_days = 60

    for today in trading_dates:
        equity = cash + sum(
            pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker])
        )
        
        # Process delayed entries
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]

        # Check exits
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in close.columns: continue
            tc = close.at[today, ticker]
            th = high.at[today, ticker] if ticker in high.columns else None
            if pd.isna(tc): continue
            tc = float(tc)
            days_held = len(trading_dates[(trading_dates > pos.entry_date) & (trading_dates <= today)])
            
            target_price = pos.avg_price * (1 + target_return)
            er, ep = None, None
            if th is not None and not pd.isna(th) and float(th) >= target_price:
                er, ep = 'target', target_price
            elif today >= pos.max_exit_date:
                er, ep = 'max_hold', tc
            
            if er:
                pnl = (ep - pos.avg_price) * pos.shares
                trades.append({
                    'ticker': ticker, 'entry_date': pos.entry_date, 'exit_date': today,
                    'entry_price': pos.entry_price, 'avg_price': pos.avg_price,
                    'exit_price': ep, 'pnl': pnl,
                    'return_pct': (ep / pos.avg_price) - 1,
                    'hold_days': days_held, 'exit_reason': er, 'adds': pos.adds,
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)
            else:
                # Check if we should add to this position (DCA)
                current_return = (tc / pos.avg_price) - 1
                if pos.adds < max_adds and current_return <= dca_threshold:
                    add_size = get_position_size(today, equity) * dca_fraction
                    if cash >= add_size and tc > 0:
                        new_shares = add_size / tc
                        total_cost = pos.avg_price * pos.shares + tc * new_shares
                        pos.shares += new_shares
                        pos.avg_price = total_cost / pos.shares
                        pos.allocated += add_size
                        pos.adds += 1
                        cash -= add_size
        
        for t in to_close: del positions[t]

        # Process ready delayed entries
        for entry in ready:
            ticker = entry['ticker']
            size = entry['size']
            if ticker in positions or len(positions) >= max_positions or cash < size:
                continue
            if ticker not in close.columns: continue
            ep = close.at[today, ticker]
            if pd.isna(ep): continue
            ep = float(ep)
            if ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold_days: continue
            positions[ticker] = DCAPosition(
                ticker=ticker, entry_date=today, entry_price=ep,
                target_return=target_return, max_exit_date=future[max_hold_days - 1],
                shares=size / ep, allocated=size,
            )
            cash -= size

        # New candidates
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row): continue
                ticker = row['ticker']
                if ticker in positions: continue
                size = get_position_size(today, equity)
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1], 'size': size})

        # Daily equity
        mv = sum(
            pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker])
        )
        daily.append({'date': today, 'equity': cash + mv, 'cash': cash,
                      'open_positions': len(positions),
                      'capital_utilization': 1.0 - (cash / (cash + mv)) if (cash + mv) > 0 else 0})

    return _format_results(trades, daily, dca_threshold, dca_fraction, max_adds)

def _format_results(trades, daily, thresh, frac, max_adds):
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    edf = pd.DataFrame(daily)
    edf['date'] = pd.to_datetime(edf['date'])
    edf['year'] = edf['date'].dt.year
    
    total_return = (edf['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n = len(tdf)
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    avg_util = edf['capital_utilization'].mean() * 100
    
    # DCA stats
    dca_trades = len(tdf[tdf['adds'] > 0]) if n > 0 and 'adds' in tdf.columns else 0
    dca_wr = (tdf[tdf['adds'] > 0]['pnl'] > 0).mean() * 100 if dca_trades > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"  DCA: thresh={thresh:.0%}, add={frac:.0%} of pos, max_adds={max_adds}")
    print(f"{'='*60}")
    print(f"  TOTAL RETURN: {total_return:+.1f}%")
    print(f"  Trades: {n} | WR: {wr:.0f}% | Losers: {losers}")
    print(f"  Avg hold: {avg_hold:.1f}d | Capital util: {avg_util:.0f}%")
    print(f"  DCA trades: {dca_trades} ({dca_wr:.0f}% win rate)")
    
    winners = tdf[tdf['pnl'] > 0] if n > 0 else pd.DataFrame()
    losers_df = tdf[tdf['pnl'] <= 0] if n > 0 else pd.DataFrame()
    print(f"  Avg win: ${winners['pnl'].mean():.0f} | Avg loss: ${losers_df['pnl'].mean():.0f}" if n > 0 else "")
    
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        print(f"    {year}: {yr_ret:+.1f}% ({yr_trades} trades, {yr_losers} losers)")
    
    return {'total_return': round(total_return, 2), 'trades': n, 'win_rate': round(wr, 1),
            'losers': losers, 'dca_trades': dca_trades, 'dca_win_rate': round(dca_wr, 1)}

# Run different DCA configurations
print("="*60)
print("  REFERENCE: No DCA (V10d best) = +401.5%")
print("="*60)

configs = [
    (-0.03, 0.5, 1, "3% drop, add 50%, max 1 add"),
    (-0.05, 0.5, 1, "5% drop, add 50%, max 1 add"),
    (-0.03, 1.0, 1, "3% drop, add 100%, max 1 add"),
    (-0.05, 1.0, 1, "5% drop, add 100%, max 1 add"),
    (-0.03, 0.5, 2, "3% drop, add 50%, max 2 adds"),
    (-0.05, 0.5, 2, "5% drop, add 50%, max 2 adds"),
]

results_all = {}
for thresh, frac, max_adds, label in configs:
    r = run_dca_backtest(thresh, frac, max_adds)
    results_all[label] = r

# Save summary
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'v01_baseline_dca_results.txt'), 'w') as f:
    f.write("DCA EXPLORATION V01\n")
    f.write(f"Reference (no DCA): +401.5%\n\n")
    for label, r in results_all.items():
        f.write(f"{label}: {r['total_return']:+.1f}% | {r['trades']} trades | WR {r['win_rate']:.0f}% | {r['dca_trades']} DCA'd\n")

with open(os.path.join(outdir, 'v01_baseline_dca_results.json'), 'w') as f:
    json.dump(results_all, f, indent=2)

print(f"\nResults saved to {outdir}/v01_baseline_dca_results.txt")
