#!/usr/bin/env python3
"""
V08 — Intraday profile of exit days (winners that hit target).
When our +5% target is hit, what does the day look like? Does the stock overshoot?
"""
import pandas as pd
import numpy as np
import os, json, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, INITIAL_CAPITAL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRADAY_DIR = os.path.join(os.path.dirname(BASE), 'intraday_data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

close_mat, high_mat, low_mat, spy, events = load_cached_data()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

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

# First run a backtest to get the actual trades with exit dates
r = run_backtest(events, close_mat, high_mat, low_mat, entry_filter,
    target_return=0.05, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)

# Extract trades from the equity curve... actually we need the trades list
# Let me re-run with a custom version that captures trade details
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

events_by_date = {dt: g for dt, g in events.groupby('event_date')}
cash = INITIAL_CAPITAL
positions: Dict[str, Pos] = {}
trades = []
pending = []

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
            trades.append({'ticker': ticker, 'entry_date': pos.entry_date, 'exit_date': today,
                'entry_price': pos.entry_price, 'exit_price': ep, 'exit_reason': er,
                'hold_days': days_held, 'pnl': pnl})
            cash += pos.allocated + pnl
            to_close.append(ticker)
    for t in to_close: del positions[t]
    for entry in ready:
        ticker = entry['ticker']
        if ticker in positions or len(positions) >= 10 or cash < INITIAL_CAPITAL/10: continue
        if ticker not in close_mat.columns: continue
        ep = close_mat.at[today, ticker]
        if pd.isna(ep): continue
        ep = float(ep)
        if ep <= 0: continue
        future = trading_dates[trading_dates > today]
        if len(future) < 60: continue
        pos_size = INITIAL_CAPITAL / 10
        positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
            target_price=ep*1.05, max_exit_date=future[59], shares=pos_size/ep, allocated=pos_size)
        cash -= pos_size
    cands = events_by_date.get(today)
    if cands is not None:
        for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
            if not entry_filter(row, {}): continue
            if row['ticker'] in positions: continue
            future = trading_dates[trading_dates > today]
            if len(future) >= 2:
                pending.append({'ticker': row['ticker'], 'enter_on': future[1]})

tdf = pd.DataFrame(trades)
winners = tdf[tdf['exit_reason'] == 'target']
losers = tdf[tdf['exit_reason'] == 'max_hold']
print(f"Total trades: {len(tdf)} | Winners (target): {len(winners)} | Losers (max_hold): {len(losers)}")

# Load intraday data
print("Loading intraday data...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'high', 'low', 'close']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['minute'] = (bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute) - (13*60+30)
bars_index = {k: v for k, v in bars.groupby(['symbol', 'date'])}
del bars

# Analyze winner exit days
print("Analyzing winner exit days...")
overshoot_data = []
for _, trade in winners.iterrows():
    ticker = trade['ticker']
    exit_date = trade['exit_date'].date()
    entry_price = trade['entry_price']
    target_price = trade['exit_price']  # = entry * 1.05
    
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    key = (alpaca_t, exit_date)
    if key not in bars_index: continue
    
    day_bars = bars_index[key].sort_values('timestamp')
    if len(day_bars) < 50: continue
    
    open_price = float(day_bars.iloc[0]['open'])
    day_high = float(day_bars['high'].max())
    day_close = float(day_bars.iloc[-1]['close'])
    
    # How much did it overshoot the target?
    overshoot = (day_high / target_price - 1) * 100
    close_vs_target = (day_close / target_price - 1) * 100
    
    # At what minute did it first hit the target?
    hit_minute = None
    for _, bar in day_bars.iterrows():
        if float(bar['high']) >= target_price:
            hit_minute = int(bar['minute'])
            break
    
    overshoot_data.append({
        'ticker': ticker, 'exit_date': exit_date,
        'overshoot_pct': overshoot,
        'close_vs_target': close_vs_target,
        'hit_minute': hit_minute,
    })

odf = pd.DataFrame(overshoot_data)
print(f"Analyzed {len(odf)} winner exit days")

print(f"\n{'='*70}")
print(f"  V08: WINNER EXIT DAY ANALYSIS")
print(f"{'='*70}")

print(f"\n  Overshoot beyond +5% target:")
print(f"    Mean: {odf['overshoot_pct'].mean():+.2f}%")
print(f"    Median: {odf['overshoot_pct'].median():+.2f}%")
print(f"    P75: {odf['overshoot_pct'].quantile(0.75):+.2f}%")
print(f"    P90: {odf['overshoot_pct'].quantile(0.90):+.2f}%")

print(f"\n  Close price vs target on exit day:")
print(f"    Mean: {odf['close_vs_target'].mean():+.2f}%")
print(f"    Median: {odf['close_vs_target'].median():+.2f}%")
print(f"    % that close ABOVE target: {(odf['close_vs_target'] > 0).mean()*100:.1f}%")
print(f"    % that close BELOW target: {(odf['close_vs_target'] < 0).mean()*100:.1f}%")

print(f"\n  Time target is first hit:")
if odf['hit_minute'].notna().any():
    hits = odf['hit_minute'].dropna()
    print(f"    Mean: minute {hits.mean():.0f} ({int((hits.mean()+9*60+30)//60)}:{int((hits.mean()+9*60+30)%60):02d})")
    print(f"    Median: minute {hits.median():.0f}")
    print(f"    Hit in first hour: {(hits <= 60).mean()*100:.1f}%")
    print(f"    Hit in last hour: {(hits >= 330).mean()*100:.1f}%")

# Save
results = {
    'winners_analyzed': len(odf),
    'avg_overshoot': round(odf['overshoot_pct'].mean(), 2),
    'median_overshoot': round(odf['overshoot_pct'].median(), 2),
    'avg_close_vs_target': round(odf['close_vs_target'].mean(), 2),
    'pct_close_above_target': round((odf['close_vs_target'] > 0).mean() * 100, 1),
}
with open(os.path.join(OUT_DIR, 'v08_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, 'v08_results.txt'), 'w') as f:
    f.write("V08: WINNER EXIT DAY ANALYSIS\n\n")
    for k, v in results.items():
        f.write(f"{k}: {v}\n")
print(f"\nSaved to {OUT_DIR}/v08_results.txt")
