#!/usr/bin/env python3
"""
TINY TARGET EXPLORATION: What if we aim for 0.25%, 0.5%, 1%, 2%, 3%?
With D+1 positive filter, regime gate (SPY>20MA), equity-based compounding.
The idea: tiny target = very high WR + very fast turnover = more compounding.
"""
import pandas as pd
import numpy as np
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

# Compute D+1 return for each event (close-to-close)
print("Computing D+1 returns...")
event_d1 = {}
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    future = trading_dates[trading_dates > dt]
    if len(future) < 1: continue
    t1 = future[0]
    if ticker in close_mat.columns and t1 in close_mat.index:
        c0 = close_mat.at[dt, ticker]
        c1 = close_mat.at[t1, ticker]
        if pd.notna(c0) and pd.notna(c1) and float(c0) > 0:
            event_d1[(ticker, dt)] = (float(c1) / float(c0) - 1) * 100
print(f"Computed {len(event_d1)} D+1 returns")

def base_filter(row):
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

def run_tiny(target_ret, max_hold, bull_frac, bear_frac, max_pos, d1_threshold, label):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades, pending, daily_eq = [], [], []
    
    for today in trading_dates:
        equity = cash + sum(
            pos.shares * float(close_mat.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close_mat.columns and pd.notna(close_mat.at[today, pos.ticker]))
        
        # Regime gate
        is_bull = (today in spy_close.index and today in spy_20ma.index and
            pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and
            spy_close.at[today] > spy_20ma.at[today])
        
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
        
        # Exits
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
                trades.append({'pnl': pnl, 'entry_date': pos.entry_date, 'exit_date': today,
                    'hold_days': days_held, 'exit_reason': er})
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]
        
        # Entries (only when bull)
        if is_bull:
            for entry in ready:
                ticker = entry['ticker']
                if ticker in positions or len(positions) >= max_pos: continue
                size = equity / bull_frac
                if cash < size: continue
                if ticker not in close_mat.columns: continue
                ep = close_mat.at[today, ticker]
                if pd.isna(ep): continue
                ep = float(ep)
                if ep <= 0: continue
                future = trading_dates[trading_dates > today]
                if len(future) < max_hold: continue
                positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                    target_price=ep*(1+target_ret), max_exit_date=future[max_hold-1],
                    shares=size/ep, allocated=size)
                cash -= size
            
            # New candidates
            cands = events_by_date.get(today)
            if cands is not None:
                for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                    if not base_filter(row): continue
                    ticker = row['ticker']
                    if ticker in positions: continue
                    # D+1 filter
                    d1 = event_d1.get((ticker, row['event_date']))
                    if d1 is not None and d1 < d1_threshold:
                        continue
                    future = trading_dates[trading_dates > today]
                    if len(future) >= 2:
                        pending.append({'ticker': ticker, 'enter_on': future[1]})
        
        mv = sum(pos.shares * float(close_mat.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close_mat.columns and pd.notna(close_mat.at[today, pos.ticker]))
        daily_eq.append({'date': today, 'equity': cash + mv})
    
    edf = pd.DataFrame(daily_eq)
    edf['date'] = pd.to_datetime(edf['date'])
    edf['year'] = edf['date'].dt.year
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n = len(tdf)
    total_ret = (edf['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr = ((1 + total_ret/100) ** (1/6) - 1) * 100
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    targets = len(tdf[tdf['exit_reason'] == 'target']) if n > 0 else 0
    
    annual = {}
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        annual[year] = {'return': round(yr_ret, 1), 'trades': yr_trades, 'losers': yr_losers}
    
    return {'label': label, 'total': round(total_ret, 1), 'cagr': round(cagr, 1),
            'trades': n, 'wr': round(wr, 1), 'losers': losers, 'avg_hold': round(avg_hold, 1),
            'targets': targets, 'annual': annual}

print(f"\n{'='*80}")
print(f"  TINY TARGET EXPLORATION")
print(f"  Regime gate (SPY>20MA) + D+1 positive filter + equity-based compounding")
print(f"{'='*80}\n")

results = []

# Sweep targets with different sizing
configs = [
    # (target, hold, bull_frac, bear_frac, max_pos, d1_thresh, label)
    (0.0025, 60, 3, 8, 15, 0, "+0.25%/60d, 1/3 bull, D+1>0%"),
    (0.005, 60, 3, 8, 15, 0, "+0.5%/60d, 1/3 bull, D+1>0%"),
    (0.01, 60, 3, 8, 15, 0, "+1%/60d, 1/3 bull, D+1>0%"),
    (0.02, 60, 3, 8, 15, 0, "+2%/60d, 1/3 bull, D+1>0%"),
    (0.03, 60, 3, 8, 15, 0, "+3%/60d, 1/3 bull, D+1>0%"),
    (0.05, 60, 3, 8, 12, 0, "+5%/60d, 1/3 bull, D+1>0%"),
    
    # Same but with D+1 > 1% (stricter filter, higher confidence)
    (0.0025, 60, 3, 8, 15, 1, "+0.25%/60d, 1/3 bull, D+1>1%"),
    (0.005, 60, 3, 8, 15, 1, "+0.5%/60d, 1/3 bull, D+1>1%"),
    (0.01, 60, 3, 8, 15, 1, "+1%/60d, 1/3 bull, D+1>1%"),
    (0.02, 60, 3, 8, 15, 1, "+2%/60d, 1/3 bull, D+1>1%"),
    (0.03, 60, 3, 8, 15, 1, "+3%/60d, 1/3 bull, D+1>1%"),
    (0.05, 60, 3, 8, 12, 1, "+5%/60d, 1/3 bull, D+1>1%"),
    
    # Shorter hold periods with tiny targets
    (0.005, 10, 3, 8, 20, 0, "+0.5%/10d, 1/3 bull, D+1>0%"),
    (0.005, 20, 3, 8, 20, 0, "+0.5%/20d, 1/3 bull, D+1>0%"),
    (0.01, 10, 3, 8, 20, 0, "+1%/10d, 1/3 bull, D+1>0%"),
    (0.01, 20, 3, 8, 20, 0, "+1%/20d, 1/3 bull, D+1>0%"),
    (0.02, 10, 3, 8, 20, 0, "+2%/10d, 1/3 bull, D+1>0%"),
    (0.02, 20, 3, 8, 20, 0, "+2%/20d, 1/3 bull, D+1>0%"),
]

for target, hold, bf, brf, mp, d1t, label in configs:
    r = run_tiny(target, hold, bf, brf, mp, d1t, label)
    results.append(r)
    
    yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
    worst = min(yearly) if yearly else 0
    
    print(f"  {label}")
    print(f"    Total: {r['total']:>+8.1f}% | CAGR: {r['cagr']:>+5.1f}% | WR: {r['wr']:.0f}% | {r['trades']} trades | hold: {r['avg_hold']:.0f}d | targets: {r['targets']} | worst yr: {worst:+.1f}%")
    for yr, d in sorted(r['annual'].items()):
        if yr >= 2026: continue
        print(f"      {yr}: {d['return']:>+7.1f}% ({d['trades']} trades, {d['losers']} losers)")
    print()

# Save
with open(os.path.join(OUT_DIR, 'v_tiny_target_results.txt'), 'w') as f:
    f.write("TINY TARGET EXPLORATION\nRegime gate + D+1 filter + compounding\n\n")
    f.write(f"{'Label':>40} {'Total':>8} {'CAGR':>6} {'WR':>5} {'Trades':>7} {'Hold':>5} {'Worst':>7}\n")
    f.write('-'*85 + '\n')
    for r in results:
        yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
        worst = min(yearly) if yearly else 0
        f.write(f"{r['label']:>40} {r['total']:>+7.1f}% {r['cagr']:>+5.1f}% {r['wr']:>4.0f}% {r['trades']:>7} {r['avg_hold']:>4.0f}d {worst:>+6.1f}%\n")

print(f"Saved to {OUT_DIR}/v_tiny_target_results.txt")
