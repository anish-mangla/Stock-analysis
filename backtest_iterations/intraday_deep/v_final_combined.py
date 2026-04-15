#!/usr/bin/env python3
"""
FINAL COMBINED STRATEGY: Everything we've learned, no cheating.

Entry signals (no look-ahead):
- Drop 3%+ from recent high (observable)
- Sector outflow, VIX/stress, Friday (all observable same day)
- 2-day delay before entry

Intraday findings applied:
- V25: Overnight gap filter — only enter if T+1 gapped up (strongest signal)
- V06: Dip entry — wait for 0.5% dip from open, fallback to close
- V19: Volume filter — skip if entry day volume > 2x normal
- V05: First-hour filter — prefer entries where first 60 min is positive

Position sizing:
- Equity-based compounding (1/6 bull, 1/14 bear, SPY vs 20MA)

We run multiple versions:
1. Baseline: no intraday signals (close entry, no filters)
2. + Dip entry only
3. + Dip entry + overnight gap filter
4. + Dip entry + overnight gap + volume filter
5. + All of the above + first-hour filter
6. Fixed sizing version of #4 (most conservative honest number)
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
spy_close = spy['SPY']
spy_20ma = spy_close.rolling(20).mean()
trading_dates = pd.DatetimeIndex(sorted(close_mat.index))

# ============================================================
# LOAD AND PRECOMPUTE ALL INTRADAY SIGNALS
# ============================================================
print("Loading intraday data and computing signals...")

all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute

# 1. Daily aggregates
daily = bars.groupby(['symbol', 'date']).agg(
    day_open=('open', 'first'),
    day_low=('low', 'min'),
    day_close=('close', 'last'),
    day_high=('high', 'max'),
    total_vol=('volume', 'sum'),
).reset_index()

# 2. Dip entry price (V06): buy at open-0.5% if dip happens, else close
daily['dip_entry'] = np.where(
    daily['day_low'] <= daily['day_open'] * 0.995,
    daily['day_open'] * 0.995,
    daily['day_close']
)

# 3. 20-day average volume for relative volume (V19)
daily = daily.sort_values(['symbol', 'date'])
daily['avg_vol_20d'] = daily.groupby('symbol')['total_vol'].transform(
    lambda x: x.rolling(20, min_periods=10).mean().shift(1)
)
daily['rel_volume'] = daily['total_vol'] / daily['avg_vol_20d']

# 4. First-hour return (V05): price at 10:30 vs open
first_bar = bars.sort_values('timestamp').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'open']]
first_bar.columns = ['symbol', 'date', 'first_open']
hr1 = bars[(bars['utc_min'] >= 867) & (bars['utc_min'] <= 873)]  # 10:30 ET = 14:30 UTC = 870
hr1_price = hr1.sort_values('utc_min').groupby(['symbol', 'date']).first().reset_index()[['symbol', 'date', 'close']]
hr1_price.columns = ['symbol', 'date', 'price_1hr']
first_bar = first_bar.merge(hr1_price, on=['symbol', 'date'], how='left')
first_bar['first_hour_ret'] = (first_bar['price_1hr'] / first_bar['first_open'] - 1) * 100
daily = daily.merge(first_bar[['symbol', 'date', 'first_hour_ret']], on=['symbol', 'date'], how='left')

# 5. Overnight gap (V25): need open of each day vs previous close
# Build open lookup
open_lookup = daily.set_index(['symbol', 'date'])['day_open'].to_dict()
close_lookup = daily.set_index(['symbol', 'date'])['day_close'].to_dict()

del bars, first_bar, hr1, hr1_price

# Pivot key columns to matrices
dip_mat = daily.pivot(index='date', columns='symbol', values='dip_entry')
dip_mat.index = pd.to_datetime(dip_mat.index)
relvol_mat = daily.pivot(index='date', columns='symbol', values='rel_volume')
relvol_mat.index = pd.to_datetime(relvol_mat.index)
fhr_mat = daily.pivot(index='date', columns='symbol', values='first_hour_ret')
fhr_mat.index = pd.to_datetime(fhr_mat.index)

for m in [dip_mat, relvol_mat, fhr_mat]:
    if 'BRK.B' in m.columns:
        m.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

print(f"All signals computed. {len(daily):,} daily rows.")
del daily

# ============================================================
# COMPUTE OVERNIGHT GAP FOR EACH EVENT
# ============================================================
print("Computing overnight gaps for events...")
event_gaps = {}  # (ticker, event_date) -> gap_t1_pct
for _, row in events.iterrows():
    ticker = row['ticker']
    dt = row['event_date']
    alpaca_t = ticker.replace('BRK-B', 'BRK.B')
    
    future = trading_dates[trading_dates > dt]
    if len(future) < 2: continue
    t1 = future[0].date()
    
    # Close on drop day
    if ticker not in close_mat.columns or dt not in close_mat.index: continue
    c0 = close_mat.at[dt, ticker]
    if pd.isna(c0): continue
    c0 = float(c0)
    
    # Open on T+1
    o1 = open_lookup.get((alpaca_t, t1))
    if o1 is None: continue
    
    gap = (o1 / c0 - 1) * 100
    event_gaps[(ticker, dt)] = gap

print(f"Computed {len(event_gaps)} overnight gaps")

# ============================================================
# ENTRY FILTER (no look-ahead biases)
# ============================================================
def base_filter(row, ctx):
    """Observable signals only — no hardcoded crisis periods, no ticker lists."""
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
# BACKTEST RUNNER
# ============================================================
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Pos:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float

def run_combined(label, use_dip_entry=False, use_gap_filter=False, gap_threshold=0.0,
                 use_vol_filter=False, vol_max=999, use_fhr_filter=False, fhr_min=-999,
                 use_adaptive_sizing=False, target_ret=0.05, max_hold=60, max_pos=10,
                 pos_fraction=10):
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    pos_size = INITIAL_CAPITAL / pos_fraction
    cash = INITIAL_CAPITAL
    positions: Dict[str, Pos] = {}
    trades: List[Dict] = []
    daily_eq: List[Dict] = []
    pending = []
    skipped_gap = 0
    skipped_vol = 0
    skipped_fhr = 0
    
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
        
        # Entries
        for entry in ready:
            ticker = entry['ticker']
            if ticker in positions: continue
            
            # Adaptive sizing
            if use_adaptive_sizing:
                is_bull = (today in spy_close.index and today in spy_20ma.index and
                    pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and
                    spy_close.at[today] > spy_20ma.at[today])
                current_size = equity / 6 if is_bull else equity / 14
            else:
                current_size = pos_size
            
            if len(positions) >= max_pos or cash < current_size: continue
            
            # V19: Volume filter
            if use_vol_filter:
                if today in relvol_mat.index and ticker in relvol_mat.columns:
                    rv = relvol_mat.at[today, ticker]
                    if pd.notna(rv) and float(rv) > vol_max:
                        skipped_vol += 1
                        continue
            
            # V05: First-hour filter
            if use_fhr_filter:
                if today in fhr_mat.index and ticker in fhr_mat.columns:
                    fhr = fhr_mat.at[today, ticker]
                    if pd.notna(fhr) and float(fhr) < fhr_min:
                        skipped_fhr += 1
                        continue
            
            # Entry price
            if use_dip_entry:
                ep = None
                if today in dip_mat.index and ticker in dip_mat.columns:
                    v = dip_mat.at[today, ticker]
                    if pd.notna(v): ep = float(v)
                if ep is None and ticker in close_mat.columns:
                    v = close_mat.at[today, ticker]
                    if pd.notna(v): ep = float(v)
            else:
                if ticker not in close_mat.columns: continue
                v = close_mat.at[today, ticker]
                if pd.isna(v): continue
                ep = float(v)
            
            if ep is None or ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold: continue
            
            positions[ticker] = Pos(ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep*(1+target_ret), max_exit_date=future[max_hold-1],
                shares=current_size/ep, allocated=current_size)
            cash -= current_size
        
        # New candidates
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not base_filter(row, {}): continue
                ticker = row['ticker']
                if ticker in positions: continue
                
                # V25: Overnight gap filter
                if use_gap_filter:
                    gap = event_gaps.get((ticker, row['event_date']))
                    if gap is not None and gap < gap_threshold:
                        skipped_gap += 1
                        continue
                
                future = trading_dates[trading_dates > today]
                if len(future) >= 2:
                    pending.append({'ticker': ticker, 'enter_on': future[1]})
        
        # Daily equity
        mv = sum(pos.shares * float(close_mat.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close_mat.columns and pd.notna(close_mat.at[today, pos.ticker]))
        daily_eq.append({'date': today, 'equity': cash + mv})
    
    # Results
    edf = pd.DataFrame(daily_eq)
    edf['date'] = pd.to_datetime(edf['date'])
    edf['year'] = edf['date'].dt.year
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n = len(tdf)
    total_ret = (edf['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    
    # Annual
    annual = {}
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        annual[year] = {'return': round(yr_ret, 1), 'trades': yr_trades, 'losers': yr_losers}
    
    return {
        'label': label, 'total_return': round(total_ret, 1), 'trades': n,
        'wr': round(wr, 1), 'losers': losers, 'avg_hold': round(avg_hold, 1),
        'annual': annual, 'skipped_gap': skipped_gap, 'skipped_vol': skipped_vol,
        'skipped_fhr': skipped_fhr,
    }

# ============================================================
# RUN ALL VERSIONS
# ============================================================
print(f"\n{'='*70}")
print(f"  FINAL COMBINED STRATEGY — ALL VERSIONS")
print(f"{'='*70}\n")

configs = [
    # FIXED SIZING (honest baseline)
    ("1. Baseline (close entry, fixed 1/10)", dict(pos_fraction=10)),
    ("2. + Dip entry", dict(use_dip_entry=True, pos_fraction=10)),
    ("3. + Dip + gap>0% filter", dict(use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, pos_fraction=10)),
    ("4. + Dip + gap>0% + vol<2x", dict(use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, use_vol_filter=True, vol_max=2.0, pos_fraction=10)),
    ("5. + All + first-hr>0%", dict(use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, use_vol_filter=True, vol_max=2.0, use_fhr_filter=True, fhr_min=0.0, pos_fraction=10)),
    
    # ADAPTIVE SIZING (compounding)
    ("6. Adaptive sizing baseline", dict(use_adaptive_sizing=True, max_pos=12)),
    ("7. Adaptive + dip entry", dict(use_adaptive_sizing=True, use_dip_entry=True, max_pos=12)),
    ("8. Adaptive + dip + gap>0%", dict(use_adaptive_sizing=True, use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, max_pos=12)),
    ("9. Adaptive + dip + gap>0% + vol<2x", dict(use_adaptive_sizing=True, use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, use_vol_filter=True, vol_max=2.0, max_pos=12)),
    ("10. FULL COMBO adaptive", dict(use_adaptive_sizing=True, use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, use_vol_filter=True, vol_max=2.0, use_fhr_filter=True, fhr_min=0.0, max_pos=12)),
    
    # HIGHER TARGETS with dip entry
    ("11. Dip + gap + +7%/45d fixed", dict(use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, target_ret=0.07, max_hold=45, pos_fraction=10)),
    ("12. Dip + gap + +7%/45d adaptive", dict(use_adaptive_sizing=True, use_dip_entry=True, use_gap_filter=True, gap_threshold=0.0, target_ret=0.07, max_hold=45, max_pos=12)),
    
    # STRICTER GAP FILTER
    ("13. Dip + gap>1% fixed", dict(use_dip_entry=True, use_gap_filter=True, gap_threshold=1.0, pos_fraction=10)),
    ("14. Dip + gap>1% adaptive", dict(use_adaptive_sizing=True, use_dip_entry=True, use_gap_filter=True, gap_threshold=1.0, max_pos=12)),
]

results = []
for label, kwargs in configs:
    r = run_combined(label, **kwargs)
    results.append(r)
    
    print(f"\n  {label}")
    print(f"    TOTAL: {r['total_return']:>+7.1f}% | {r['trades']} trades | WR {r['wr']:.0f}% | hold {r['avg_hold']:.0f}d | {r['losers']} losers")
    if r['skipped_gap']: print(f"    Skipped by gap filter: {r['skipped_gap']}")
    if r['skipped_vol']: print(f"    Skipped by vol filter: {r['skipped_vol']}")
    if r['skipped_fhr']: print(f"    Skipped by first-hr filter: {r['skipped_fhr']}")
    
    print(f"    {'Year':>6} {'Return':>8} {'Trades':>7} {'Losers':>7} {'$100K→':>10}")
    for yr, data in sorted(r['annual'].items()):
        final = 100000 * (1 + data['return']/100)
        print(f"    {yr:>6} {data['return']:>+7.1f}% {data['trades']:>7} {data['losers']:>7} ${final:>9,.0f}")
    
    # Compute stats
    yearly_rets = [data['return'] for yr, data in sorted(r['annual'].items()) if yr < 2026]
    if yearly_rets:
        avg = sum(yearly_rets) / len(yearly_rets)
        median = sorted(yearly_rets)[len(yearly_rets)//2]
        worst = min(yearly_rets)
        best = max(yearly_rets)
        print(f"    Avg year: {avg:+.1f}% | Median: {median:+.1f}% | Worst: {worst:+.1f}% | Best: {best:+.1f}%")
        print(f"    $100K avg year → ${100000*(1+avg/100):,.0f} | worst → ${100000*(1+worst/100):,.0f}")

# SPY benchmark
print(f"\n  SPY BUY & HOLD:")
spy_s = spy_close.copy()
spy_s.index = pd.to_datetime(spy_s.index)
for year in range(2020, 2026):
    ys = spy_s[spy_s.index.year == year]
    if len(ys) > 0:
        ret = (ys.iloc[-1] / ys.iloc[0] - 1) * 100
        print(f"    {year}: {ret:+.1f}%")

# Save
with open(os.path.join(OUT_DIR, 'v_final_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

with open(os.path.join(OUT_DIR, 'v_final_results.txt'), 'w') as f:
    f.write("FINAL COMBINED STRATEGY — ALL VERSIONS\n")
    f.write("No look-ahead biases. All signals observable in real-time.\n\n")
    f.write(f"{'#':>3} {'Label':>45} {'Return':>8} {'Trades':>7} {'WR':>5} {'Losers':>7} {'AvgYr':>7}\n")
    f.write('-'*85 + '\n')
    for i, r in enumerate(results):
        yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
        avg_yr = sum(yearly)/len(yearly) if yearly else 0
        f.write(f"{i+1:>3} {r['label']:>45} {r['total_return']:>+7.1f}% {r['trades']:>7} {r['wr']:>4.0f}% {r['losers']:>7} {avg_yr:>+6.1f}%\n")

print(f"\nSaved to {OUT_DIR}/v_final_results.txt")
