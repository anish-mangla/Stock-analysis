#!/usr/bin/env python3
"""
AGGRESSIVE STRATEGY EXPLORATION: How do we get to 50%+ CAGR?

The bottleneck is clear: we're too conservative. Let's test:
1. More aggressive sizing (1/3 bull, 1/6 bear)
2. More positions (max 15-20)
3. Faster turnover (shorter hold = more trades = more compounding)
4. Use the gap filter to be AGGRESSIVE on high-confidence trades, not just filter
5. Tiered sizing: high-confidence trades get 2x the position size
6. Deploy idle capital in SPY when not trading (capture bull market returns)
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
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

# Load intraday signals
print("Loading intraday signals...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path, columns=['symbol', 'timestamp', 'open', 'low', 'close', 'volume']))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date

daily = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'), day_low=('low', 'min'), day_close=('close', 'last'),
    total_vol=('volume', 'sum'),
).reset_index()
daily['dip_entry'] = np.where(
    daily['day_low'] <= daily['day_open'] * 0.995,
    daily['day_open'] * 0.995, daily['day_close']
)
daily = daily.sort_values(['symbol', 'date'])
daily['avg_vol_20d'] = daily.groupby('symbol')['total_vol'].transform(
    lambda x: x.rolling(20, min_periods=10).mean().shift(1))
daily['rel_volume'] = daily['total_vol'] / daily['avg_vol_20d']

dip_mat = daily.pivot(index='date', columns='symbol', values='dip_entry')
dip_mat.index = pd.to_datetime(dip_mat.index)
relvol_mat = daily.pivot(index='date', columns='symbol', values='rel_volume')
relvol_mat.index = pd.to_datetime(relvol_mat.index)
for m in [dip_mat, relvol_mat]:
    if 'BRK.B' in m.columns: m.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

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
    is_spy: bool = False

def run_aggressive(label, bull_frac, bear_frac, max_pos, target_ret, max_hold,
                   use_gap_filter=True, gap_threshold=0.0,
                   use_vol_filter=False, vol_max=2.0,
                   use_confidence_sizing=False,
                   deploy_idle_in_spy=False):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades = []
    pending = []
    daily_eq = []
    
    for today in trading_dates:
        equity = cash + sum(
            pos.shares * float(close_mat.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close_mat.columns and pd.notna(close_mat.at[today, pos.ticker]))
        
        ready = [p for p in pending if p['enter_on'] <= today]
        pending = [p for p in pending if p['enter_on'] > today]
        
        # Exits
        to_close = []
        for ticker, pos in positions.items():
            if pos.is_spy:
                # SPY position: close if we need capital for trades
                # (handled below when we need to enter)
                continue
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
        
        # Close SPY positions if we have pending entries
        if deploy_idle_in_spy and 'SPY_IDLE' in positions and len(ready) > 0:
            spy_pos = positions['SPY_IDLE']
            if 'SPY' in close_mat.columns and today in close_mat.index:
                spy_price = close_mat.at[today, 'SPY']
                if pd.notna(spy_price):
                    pnl = (float(spy_price) - spy_pos.entry_price) * spy_pos.shares
                    cash += spy_pos.allocated + pnl
                    del positions['SPY_IDLE']
        
        # Entries
        non_spy_positions = sum(1 for p in positions.values() if not p.is_spy)
        
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions: continue
            
            is_bull = (today in spy_close.index and today in spy_20ma.index and
                pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and
                spy_close.at[today] > spy_20ma.at[today])
            base_size = equity / bull_frac if is_bull else equity / bear_frac
            
            # Confidence-based sizing
            if use_confidence_sizing:
                gap = entry.get('gap', 0)
                if gap > 3:
                    base_size *= 1.5  # high confidence: 50% bigger
                elif gap > 1:
                    base_size *= 1.2  # medium confidence: 20% bigger
            
            if non_spy_positions >= max_pos or cash < base_size: continue
            
            # Volume filter
            if use_vol_filter:
                if today in relvol_mat.index and ticker in relvol_mat.columns:
                    rv = relvol_mat.at[today, ticker]
                    if pd.notna(rv) and float(rv) > vol_max: continue
            
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
                shares=base_size/ep, allocated=base_size)
            cash -= base_size
            non_spy_positions += 1
        
        # New candidates
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not base_filter(row): continue
                ticker = row['ticker']
                if ticker in positions: continue
                
                gap = event_gaps.get((ticker, row['event_date']), None)
                if use_gap_filter and gap is not None and gap < gap_threshold:
                    continue
                
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1], 'gap': gap or 0})
        
        # Deploy idle capital in SPY
        if deploy_idle_in_spy and 'SPY_IDLE' not in positions:
            idle = cash
            if idle > equity * 0.2 and 'SPY' in close_mat.columns and today in close_mat.index:
                spy_price = close_mat.at[today, 'SPY']
                if pd.notna(spy_price):
                    spy_price = float(spy_price)
                    deploy = idle * 0.8  # keep 20% cash buffer
                    if deploy > 1000:
                        future = trading_dates[trading_dates > today]
                        if len(future) > 0:
                            positions['SPY_IDLE'] = Pos(
                                ticker='SPY', entry_date=today, entry_price=spy_price,
                                target_price=spy_price*100, max_exit_date=future[-1],
                                shares=deploy/spy_price, allocated=deploy, is_spy=True)
                            cash -= deploy
        
        # Daily equity
        mv = sum(
            pos.shares * float(close_mat.at[today, pos.ticker])
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
    
    annual = {}
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        annual[year] = {'return': round(yr_ret, 1), 'trades': yr_trades, 'losers': yr_losers}
    
    return {'label': label, 'total': round(total_ret, 1), 'cagr': round(cagr, 1),
            'trades': n, 'wr': round(wr, 1), 'losers': losers, 'annual': annual}

# ============================================================
# RUN AGGRESSIVE CONFIGS
# ============================================================
print(f"\n{'='*70}")
print(f"  AGGRESSIVE STRATEGY EXPLORATION — TARGETING 50%+ CAGR")
print(f"{'='*70}\n")

configs = [
    # Baseline for reference
    ("A1. Current best (1/6 bull, 1/14 bear)", dict(bull_frac=6, bear_frac=14, max_pos=12, target_ret=0.05, max_hold=60)),
    
    # More aggressive sizing
    ("A2. Aggressive sizing (1/4 bull, 1/8 bear)", dict(bull_frac=4, bear_frac=8, max_pos=12, target_ret=0.05, max_hold=60)),
    ("A3. Very aggressive (1/3 bull, 1/6 bear)", dict(bull_frac=3, bear_frac=6, max_pos=15, target_ret=0.05, max_hold=60)),
    
    # Faster turnover
    ("A4. Fast turnover (1/4, +3%/30d)", dict(bull_frac=4, bear_frac=8, max_pos=15, target_ret=0.03, max_hold=30)),
    ("A5. Fast turnover (1/3, +3%/20d)", dict(bull_frac=3, bear_frac=6, max_pos=20, target_ret=0.03, max_hold=20)),
    
    # Confidence-based sizing (bigger on high-gap trades)
    ("A6. Confidence sizing (1/4, gap>0%)", dict(bull_frac=4, bear_frac=8, max_pos=12, target_ret=0.05, max_hold=60, use_confidence_sizing=True)),
    
    # Deploy idle capital in SPY
    ("A7. Idle in SPY (1/6 bull, 1/14 bear)", dict(bull_frac=6, bear_frac=14, max_pos=12, target_ret=0.05, max_hold=60, deploy_idle_in_spy=True)),
    ("A8. Aggressive + idle in SPY", dict(bull_frac=4, bear_frac=8, max_pos=15, target_ret=0.05, max_hold=60, deploy_idle_in_spy=True)),
    ("A9. Very aggressive + idle in SPY", dict(bull_frac=3, bear_frac=6, max_pos=15, target_ret=0.05, max_hold=60, deploy_idle_in_spy=True)),
    
    # Aggressive + gap filter + vol filter (quality trades only, but big)
    ("A10. Quality+size (1/3, gap>1%, vol<2x)", dict(bull_frac=3, bear_frac=6, max_pos=15, target_ret=0.05, max_hold=60, gap_threshold=1.0, use_vol_filter=True)),
    
    # All-in on high confidence
    ("A11. All-in high conf (1/3, gap>1%, +SPY)", dict(bull_frac=3, bear_frac=6, max_pos=15, target_ret=0.05, max_hold=60, gap_threshold=1.0, deploy_idle_in_spy=True)),
    
    # Fastest possible turnover with aggressive sizing
    ("A12. Turbo (1/3, +3%/15d, max 20)", dict(bull_frac=3, bear_frac=6, max_pos=20, target_ret=0.03, max_hold=15)),
    ("A13. Turbo + SPY idle", dict(bull_frac=3, bear_frac=6, max_pos=20, target_ret=0.03, max_hold=15, deploy_idle_in_spy=True)),
]

results = []
for label, kwargs in configs:
    r = run_aggressive(label, **kwargs)
    results.append(r)
    
    yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
    worst = min(yearly) if yearly else 0
    best = max(yearly) if yearly else 0
    
    print(f"  {label}")
    print(f"    Total: {r['total']:>+8.1f}% | CAGR: {r['cagr']:>+5.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | Worst yr: {worst:+.1f}%")
    for yr, d in sorted(r['annual'].items()):
        if yr >= 2026: continue
        print(f"      {yr}: {d['return']:>+7.1f}% ({d['trades']} trades, {d['losers']} losers)")
    print()

# Save
with open(os.path.join(OUT_DIR, 'v_aggressive_results.txt'), 'w') as f:
    f.write("AGGRESSIVE STRATEGY EXPLORATION\nTarget: 50%+ CAGR\n\n")
    f.write(f"{'#':>3} {'Label':>45} {'Total':>8} {'CAGR':>6} {'Trades':>7} {'WR':>5} {'Worst':>7}\n")
    f.write('-'*85 + '\n')
    for r in results:
        yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
        worst = min(yearly) if yearly else 0
        f.write(f"    {r['label']:>45} {r['total']:>+7.1f}% {r['cagr']:>+5.1f}% {r['trades']:>7} {r['wr']:>4.0f}% {worst:>+6.1f}%\n")

print(f"Saved to {OUT_DIR}/v_aggressive_results.txt")
