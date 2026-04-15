#!/usr/bin/env python3
"""
REGIME GATING: Don't trade in bad markets at all.

Current system: trade in all markets, just size down in bear.
New idea: STOP trading entirely when market is bad. Capital is freed up.

Test multiple regime detectors (all observable in real-time, no look-ahead):
1. SPY below 20MA → stop trading
2. SPY below 50MA → stop trading
3. VIX above 25/30/35 → stop trading
4. SPY drawdown from 52-week high > 10%/15%/20% → stop trading
5. Combination: SPY below 20MA AND VIX > 25

For each: report CAGR on capital deployed, time in market, and idle capital periods.
"""
import pandas as pd
import numpy as np
import os, json, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()
spy_50ma = spy_close.rolling(50).mean()
spy_200ma = spy_close.rolling(200).mean()
spy_52w_high = spy_close.rolling(252).max()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

# Load VIX
import yfinance as yf
vix = yf.download('^VIX', start='2019-01-01', end='2026-01-01', progress=False)['Close']
if isinstance(vix, pd.DataFrame):
    vix = vix.iloc[:, 0]
vix.index = vix.index.tz_localize(None)

# Load dip entry matrix
print("Loading intraday dip entry data...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'low', 'close']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
daily = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'), day_low=('low', 'min'), day_close=('close', 'last'),
).reset_index()
daily['dip_entry'] = np.where(
    daily['day_low'] <= daily['day_open'] * 0.995,
    daily['day_open'] * 0.995, daily['day_close'])
dip_mat = daily.pivot(index='date', columns='symbol', values='dip_entry')
dip_mat.index = pd.to_datetime(dip_mat.index)
if 'BRK.B' in dip_mat.columns: dip_mat.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

# Overnight gaps
open_lookup = daily.set_index(['symbol', 'date'])['day_open'].to_dict()
event_gaps = {}
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    t1 = future[0].date()
    if ticker not in close_mat.columns or dt not in close_mat.index: continue
    c0 = close_mat.at[dt, ticker]
    if pd.isna(c0): continue
    o1 = open_lookup.get((alpaca_t, t1))
    if o1 is None: continue
    event_gaps[(ticker, dt)] = (o1 / float(c0) - 1) * 100

del bars, daily
print("Done")

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
from typing import Dict, List

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_gated(label, regime_fn, bull_frac=4, bear_frac=10, max_pos=12,
              target_ret=0.05, max_hold=60, use_gap_filter=True):
    """regime_fn(today) -> 'trade' or 'stop'. When 'stop', no new entries."""
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades = []
    pending = []
    daily_eq = []
    trading_days = 0
    stopped_days = 0
    
    for today in trading_dates:
        equity = cash + sum(
            pos.shares * float(close_mat.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close_mat.columns and pd.notna(close_mat.at[today, pos.ticker]))
        
        regime = regime_fn(today)
        if regime == 'trade':
            trading_days += 1
        else:
            stopped_days += 1
        
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
        
        # Exits always happen (even when stopped)
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
        
        # Entries only when regime says trade
        if regime == 'trade':
            for entry in ready:
                ticker = entry['ticker']
                if ticker in positions or len(positions) >= max_pos: continue
                
                is_bull = (today in spy_close.index and today in spy_20ma.index and
                    pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and
                    spy_close.at[today] > spy_20ma.at[today])
                size = equity / bull_frac if is_bull else equity / bear_frac
                if cash < size: continue
                
                # Dip entry
                ep = None
                if today in dip_mat.index and ticker in dip_mat.columns:
                    v = dip_mat.at[today, ticker]
                    if pd.notna(v): ep = float(v)
                if ep is None and ticker in close_mat.columns:
                    v = close_mat.at[today, ticker]
                    if pd.notna(v): ep = float(v)
                if ep is None or ep <= 0: continue
                
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
                    if use_gap_filter:
                        gap = event_gaps.get((ticker, row['event_date']))
                        if gap is not None and gap < 0: continue
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
    years = 6
    cagr = ((1 + total_ret/100) ** (1/years) - 1) * 100
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    time_in_market = trading_days / (trading_days + stopped_days) * 100
    
    annual = {}
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        annual[year] = {'return': round(yr_ret, 1), 'trades': yr_trades, 'losers': yr_losers}
    
    return {'label': label, 'total': round(total_ret, 1), 'cagr': round(cagr, 1),
            'trades': n, 'wr': round(wr, 1), 'losers': losers,
            'time_in_market': round(time_in_market, 1), 'annual': annual}

# ============================================================
# REGIME FUNCTIONS (all real-time observable)
# ============================================================

def always_trade(today):
    return 'trade'

def gate_spy_below_20ma(today):
    if today in spy_close.index and today in spy_20ma.index:
        s, m = spy_close.get(today), spy_20ma.get(today)
        if pd.notna(s) and pd.notna(m) and s < m:
            return 'stop'
    return 'trade'

def gate_spy_below_50ma(today):
    if today in spy_close.index and today in spy_50ma.index:
        s, m = spy_close.get(today), spy_50ma.get(today)
        if pd.notna(s) and pd.notna(m) and s < m:
            return 'stop'
    return 'trade'

def gate_vix_above(threshold):
    def fn(today):
        if today in vix.index:
            v = vix.get(today)
            if pd.notna(v) and float(v) > threshold:
                return 'stop'
        return 'trade'
    return fn

def gate_spy_drawdown(threshold):
    def fn(today):
        if today in spy_close.index and today in spy_52w_high.index:
            s, h = spy_close.get(today), spy_52w_high.get(today)
            if pd.notna(s) and pd.notna(h) and h > 0:
                dd = (s / h - 1) * 100
                if dd < -threshold:
                    return 'stop'
        return 'trade'
    return fn

def gate_combo_20ma_vix25(today):
    below_ma = False
    high_vix = False
    if today in spy_close.index and today in spy_20ma.index:
        s, m = spy_close.get(today), spy_20ma.get(today)
        if pd.notna(s) and pd.notna(m) and s < m:
            below_ma = True
    if today in vix.index:
        v = vix.get(today)
        if pd.notna(v) and float(v) > 25:
            high_vix = True
    if below_ma and high_vix:
        return 'stop'
    return 'trade'

def gate_combo_50ma_vix30(today):
    below_ma = False
    high_vix = False
    if today in spy_close.index and today in spy_50ma.index:
        s, m = spy_close.get(today), spy_50ma.get(today)
        if pd.notna(s) and pd.notna(m) and s < m:
            below_ma = True
    if today in vix.index:
        v = vix.get(today)
        if pd.notna(v) and float(v) > 30:
            high_vix = True
    if below_ma and high_vix:
        return 'stop'
    return 'trade'

# ============================================================
# RUN ALL CONFIGS
# ============================================================
print(f"\n{'='*70}")
print(f"  REGIME GATING — STOP TRADING IN BAD MARKETS")
print(f"{'='*70}\n")

configs = [
    ("No gating (always trade)", always_trade, 4, 10),
    ("Gate: SPY < 20MA", gate_spy_below_20ma, 4, 10),
    ("Gate: SPY < 50MA", gate_spy_below_50ma, 4, 10),
    ("Gate: VIX > 25", gate_vix_above(25), 4, 10),
    ("Gate: VIX > 30", gate_vix_above(30), 4, 10),
    ("Gate: SPY DD > 10%", gate_spy_drawdown(10), 4, 10),
    ("Gate: SPY DD > 15%", gate_spy_drawdown(15), 4, 10),
    ("Gate: SPY<20MA + VIX>25", gate_combo_20ma_vix25, 4, 10),
    ("Gate: SPY<50MA + VIX>30", gate_combo_50ma_vix30, 4, 10),
    # More aggressive sizing when gated (since we avoid bad periods)
    ("Gate: SPY<20MA+VIX>25, 1/3 bull", gate_combo_20ma_vix25, 3, 8),
    ("Gate: SPY<50MA+VIX>30, 1/3 bull", gate_combo_50ma_vix30, 3, 8),
    ("Gate: SPY<20MA, 1/3 bull", gate_spy_below_20ma, 3, 8),
    ("Gate: SPY DD>10%, 1/3 bull", gate_spy_drawdown(10), 3, 8),
]

results = []
for label, regime_fn, bf, brf in configs:
    r = run_gated(label, regime_fn, bull_frac=bf, bear_frac=brf)
    results.append(r)
    
    yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
    worst = min(yearly) if yearly else 0
    best = max(yearly) if yearly else 0
    
    print(f"  {label}")
    print(f"    Total: {r['total']:>+8.1f}% | CAGR: {r['cagr']:>+5.1f}% | WR {r['wr']:.0f}% | {r['trades']} trades | In market: {r['time_in_market']:.0f}%")
    print(f"    Worst: {worst:+.1f}% | Best: {best:+.1f}%")
    for yr, d in sorted(r['annual'].items()):
        if yr >= 2026: continue
        print(f"      {yr}: {d['return']:>+7.1f}% ({d['trades']} trades, {d['losers']} losers)")
    print()

# Save
with open(os.path.join(OUT_DIR, 'v_regime_gating_results.txt'), 'w') as f:
    f.write("REGIME GATING — STOP TRADING IN BAD MARKETS\n")
    f.write("All use dip entry + gap>0% filter. Adaptive sizing.\n\n")
    f.write(f"{'Label':>40} {'Total':>8} {'CAGR':>6} {'WR':>5} {'Trades':>7} {'InMkt':>6} {'Worst':>7}\n")
    f.write('-'*85 + '\n')
    for r in results:
        yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
        worst = min(yearly) if yearly else 0
        f.write(f"{r['label']:>40} {r['total']:>+7.1f}% {r['cagr']:>+5.1f}% {r['wr']:>4.0f}% {r['trades']:>7} {r['time_in_market']:>5.0f}% {worst:>+6.1f}%\n")

print(f"Saved to {OUT_DIR}/v_regime_gating_results.txt")
