#!/usr/bin/env python3
"""
V09-V16 BATCH: Run remaining entry/exit/stop-loss investigations.
V09: Higher targets (since V06 showed we can get better entry prices)
V10-V12: Combine best findings
V13-V16: Intraday drawdown analysis for stop loss design
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

# Load intraday for dip-entry strategy
print("Loading intraday data for dip-entry matrices...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'low', 'close', 'high', 'volume']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date

# Build dip-entry matrix (V06 best: 0.5% dip)
daily_agg = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'), day_low=('low', 'min'), day_close=('close', 'last'),
    day_high=('high', 'max'), total_vol=('volume', 'sum'),
).reset_index()
daily_agg['dip_entry'] = np.where(
    daily_agg['day_low'] <= daily_agg['day_open'] * 0.995,
    daily_agg['day_open'] * 0.995,
    daily_agg['day_close']
)
dip_mat = daily_agg.pivot(index='date', columns='symbol', values='dip_entry')
dip_mat.index = pd.to_datetime(dip_mat.index)
if 'BRK.B' in dip_mat.columns:
    dip_mat.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

# Intraday drawdown data for stop loss analysis
print("Computing intraday drawdown stats...")
# For each (symbol, date): max drawdown from open
daily_agg['max_dd_from_open'] = (daily_agg['day_low'] / daily_agg['day_open'] - 1) * 100
daily_agg['range_pct'] = (daily_agg['day_high'] - daily_agg['day_low']) / daily_agg['day_open'] * 100

del bars
print("Done")

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

# ============================================================
# V09: With dip entry, test higher targets (since entry is cheaper)
# ============================================================
print(f"\n{'='*70}")
print(f"  V09: DIP ENTRY + TARGET SWEEP")
print(f"{'='*70}\n")

from dataclasses import dataclass
from typing import Dict

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_dip_bt(target_ret, max_hold, label):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    pos_size = INITIAL_CAPITAL / 10
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades, pending = [], []
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
                trades.append({'pnl': (ep - pos.entry_price) * pos.shares, 'exit_reason': er,
                    'entry_date': pos.entry_date, 'hold_days': days_held})
                cash += pos.allocated + (ep - pos.entry_price) * pos.shares
                to_close.append(ticker)
        for t in to_close: del positions[t]
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions or len(positions) >= 10 or cash < pos_size: continue
            ep = None
            if today in dip_mat.index and ticker in dip_mat.columns:
                v = dip_mat.at[today, ticker]
                if pd.notna(v): ep = float(v)
            if ep is None or ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold: continue
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*(1+target_ret), max_exit_date=future[max_hold-1],
                shares=pos_size/ep, allocated=pos_size)
            cash -= pos_size
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row, {}): continue
                if row['ticker'] in positions: continue
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': row['ticker'], 'enter_on': future[1]})
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n = len(tdf)
    ret = tdf['pnl'].sum() / INITIAL_CAPITAL * 100 if n > 0 else 0
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    targets = len(tdf[tdf['exit_reason'] == 'target']) if n > 0 else 0
    return {'label': label, 'return': round(ret, 1), 'trades': n, 'wr': round(wr, 1),
            'losers': losers, 'avg_hold': round(avg_hold, 1), 'target_hits': targets}

v09_results = []
for target in [0.03, 0.05, 0.07, 0.10]:
    for hold in [30, 45, 60]:
        r = run_dip_bt(target, hold, f"Dip entry, +{target*100:.0f}%/{hold}d")
        v09_results.append(r)
        print(f"  {r['label']:>25}: {r['return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | hold {r['avg_hold']:.0f}d | targets {r['target_hits']}")

# Also close-entry baseline for comparison
for target in [0.05, 0.07]:
    r_base = run_backtest(events, close_mat, high_mat, low_mat, entry_filter,
        target_return=target, max_hold_days=60, position_fraction=10, max_positions=10, entry_delay=2)
    print(f"  {'Close +' + str(int(target*100)) + '%/60d':>25}: {r_base['total_return']:>+7.1f}% | {r_base['trades']} trades | WR {r_base['win_rate']:.0f}%")

# ============================================================
# V16: Intraday drawdown analysis
# ============================================================
print(f"\n{'='*70}")
print(f"  V16: INTRADAY DRAWDOWN ANALYSIS")
print(f"{'='*70}")

# Match events to their entry days and check intraday drawdowns during hold period
# For now, just analyze the entry day drawdowns
events_dd = []
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    entry_day = future[1].date()
    
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    dd_row = daily_agg[(daily_agg['symbol'] == alpaca_t) & (daily_agg['date'] == entry_day)]
    if len(dd_row) == 0: continue
    
    events_dd.append({
        'ticker': ticker, 'success': bool(row.get('success', False)),
        'drop_pct': float(row['drop_pct']),
        'entry_day_dd': float(dd_row.iloc[0]['max_dd_from_open']),
        'entry_day_range': float(dd_row.iloc[0]['range_pct']),
    })

eddf = pd.DataFrame(events_dd)
print(f"\n  Entry day intraday drawdown (from open):")
print(f"    Mean: {eddf['entry_day_dd'].mean():.2f}%")
print(f"    Median: {eddf['entry_day_dd'].median():.2f}%")
print(f"    P10 (worst 10%): {eddf['entry_day_dd'].quantile(0.10):.2f}%")

print(f"\n  Entry day range:")
print(f"    Mean: {eddf['entry_day_range'].mean():.2f}%")
print(f"    Median: {eddf['entry_day_range'].median():.2f}%")

print(f"\n  Winners vs losers intraday drawdown:")
print(f"    Winners: {eddf[eddf['success']]['entry_day_dd'].mean():.2f}% avg dd, {eddf[eddf['success']]['entry_day_range'].mean():.2f}% range")
print(f"    Losers:  {eddf[~eddf['success']]['entry_day_dd'].mean():.2f}% avg dd, {eddf[~eddf['success']]['entry_day_range'].mean():.2f}% range")

# Save all results
all_results = {'v09': v09_results, 'v16_drawdown': {
    'mean_dd': round(eddf['entry_day_dd'].mean(), 2),
    'median_dd': round(eddf['entry_day_dd'].median(), 2),
    'winners_dd': round(eddf[eddf['success']]['entry_day_dd'].mean(), 2),
    'losers_dd': round(eddf[~eddf['success']]['entry_day_dd'].mean(), 2),
}}

with open(os.path.join(OUT_DIR, 'v09_v16_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

with open(os.path.join(OUT_DIR, 'v09_results.txt'), 'w') as f:
    f.write("V09: DIP ENTRY + TARGET SWEEP\n\n")
    f.write(f"{'Config':>25} {'Return':>8} {'Trades':>7} {'WR':>5} {'Hold':>5} {'Targets':>8}\n")
    f.write('-'*60 + '\n')
    for r in v09_results:
        f.write(f"{r['label']:>25} {r['return']:>+7.1f}% {r['trades']:>7} {r['wr']:>4.0f}% {r['avg_hold']:>4.0f}d {r['target_hits']:>8}\n")

with open(os.path.join(OUT_DIR, 'v16_results.txt'), 'w') as f:
    f.write("V16: INTRADAY DRAWDOWN ANALYSIS\n\n")
    f.write(f"Entry day drawdown from open:\n")
    f.write(f"  Mean: {eddf['entry_day_dd'].mean():.2f}%\n")
    f.write(f"  Winners: {eddf[eddf['success']]['entry_day_dd'].mean():.2f}%\n")
    f.write(f"  Losers: {eddf[~eddf['success']]['entry_day_dd'].mean():.2f}%\n")

print(f"\nSaved to {OUT_DIR}/v09_results.txt and v16_results.txt")
