#!/usr/bin/env python3
"""
OPENING RANGE BREAKOUT (ORB) on mean-reversion candidates.

Logic:
- Stock dropped 3%+ on day T
- On day T+1 (or T+2), wait for the first 30 minutes to establish a range
- If price breaks ABOVE the 30-min high → buy (upside breakout = recovery starting)
- If price breaks BELOW the 30-min low → skip (downside breakout = still selling)
- Sell at end of day (or when target hit intraday)
- Never hold overnight

This is a DAY TRADE — in and out same day. We use our drop signal as the
candidate filter, and ORB as the execution strategy.

Test multiple variants:
- 15-min vs 30-min vs 60-min opening range
- Different intraday targets (0.5%, 1%, 2%)
- With and without regime gate (SPY > 20MA)
- T+1 vs T+2 entry day
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

# Load intraday data and index by (symbol, date)
print("Loading intraday data...")
all_bars = []
for year in range(2020, 2026):
    path = os.path.join(INTRADAY_DIR, f'bars_1min_{year}_adjusted.parquet')
    if os.path.exists(path):
        all_bars.append(pd.read_parquet(path))
bars = pd.concat(all_bars, ignore_index=True)
bars['date'] = bars['timestamp'].dt.date
bars['utc_min'] = bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute
# 9:30 ET = 13:30 UTC = 810 min, 16:00 ET = 20:00 UTC = 1200 min
bars_index = {}
for (sym, date), group in bars.groupby(['symbol', 'date']):
    bars_index[(sym, date)] = group.sort_values('timestamp')
del bars
print(f"Indexed {len(bars_index):,} (symbol, date) groups")

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

def run_orb(range_minutes, intraday_target, entry_delay, use_regime_gate, bull_frac, max_pos, label):
    """
    range_minutes: 15, 30, or 60 — how long to wait for the opening range
    intraday_target: e.g., 0.01 = sell when up 1% from entry
    entry_delay: 1 = trade on T+1, 2 = trade on T+2
    """
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}
    cash = INITIAL_CAPITAL
    trades = []
    daily_eq = []
    pending = []
    
    # Opening range cutoff in UTC minutes
    # 9:30 ET = 13:30 UTC = 810
    range_end_utc = 810 + range_minutes  # e.g., 30 min → 840 (10:00 ET)
    
    for today in trading_dates:
        today_date = today.date()
        equity = cash  # no overnight positions in day trading
        
        is_bull = (today in spy_close.index and today in spy_20ma.index and
            pd.notna(spy_close.get(today)) and pd.notna(spy_20ma.get(today)) and
            spy_close.at[today] > spy_20ma.at[today])
        
        # Process pending entries for today
        ready = [p for p in pending if p['trade_on'] <= today]
        pending = [p for p in pending if p['trade_on'] > today]
        
        if use_regime_gate and not is_bull:
            # Skip trading but still record equity
            daily_eq.append({'date': today, 'equity': equity})
            # Still collect new candidates for future
            cands = events_by_date.get(today)
            if cands is not None:
                for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                    if not base_filter(row): continue
                    future = trading_dates[trading_dates > today]
                    if len(future) >= entry_delay:
                        pending.append({'ticker': row['ticker'], 'trade_on': future[entry_delay - 1]})
            continue
        
        # Execute ORB trades for today's ready candidates
        day_pnl = 0
        trades_today = 0
        
        for entry in ready:
            if trades_today >= max_pos:
                break
            
            ticker = entry['ticker']
            alpaca_t = ticker.replace('BRK-B', 'BRK.B')
            key = (alpaca_t, today_date)
            
            if key not in bars_index:
                continue
            
            day_bars = bars_index[key]
            if len(day_bars) < range_minutes + 30:  # need enough bars
                continue
            
            # Compute opening range
            range_bars = day_bars[day_bars['utc_min'] <= range_end_utc]
            if len(range_bars) < 5:
                continue
            
            range_high = float(range_bars['high'].max())
            range_low = float(range_bars['low'].min())
            
            if range_high <= 0 or range_low <= 0:
                continue
            
            # Look for upside breakout after the range period
            post_range = day_bars[day_bars['utc_min'] > range_end_utc]
            if len(post_range) == 0:
                continue
            
            entry_price = None
            entry_time = None
            
            for _, bar in post_range.iterrows():
                if float(bar['high']) > range_high:
                    # Upside breakout! Buy at range_high (limit fill)
                    entry_price = range_high
                    entry_time = bar['utc_min']
                    break
                elif float(bar['low']) < range_low:
                    # Downside breakout — skip this trade
                    break
            
            if entry_price is None:
                continue  # no breakout or downside breakout
            
            # Now manage the trade: look for target or sell at close
            position_size = equity / bull_frac
            if cash < position_size:
                continue
            
            shares = position_size / entry_price
            target_price = entry_price * (1 + intraday_target)
            exit_price = None
            exit_reason = None
            
            # Check remaining bars after entry
            remaining = post_range[post_range['utc_min'] > entry_time]
            
            for _, bar in remaining.iterrows():
                if float(bar['high']) >= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    break
            
            if exit_price is None:
                # Sell at close (last bar of the day)
                exit_price = float(day_bars.iloc[-1]['close'])
                exit_reason = 'eod_close'
            
            pnl = (exit_price - entry_price) * shares
            trades.append({
                'ticker': ticker, 'date': today, 'entry_price': entry_price,
                'exit_price': exit_price, 'pnl': pnl, 'exit_reason': exit_reason,
                'return_pct': (exit_price / entry_price - 1) * 100,
            })
            
            cash += pnl  # day trade: position opened and closed same day
            day_pnl += pnl
            trades_today += 1
        
        # New candidates
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not base_filter(row): continue
                ticker = row['ticker']
                future = trading_dates[trading_dates > today]
                if len(future) >= entry_delay:
                    pending.append({'ticker': ticker, 'trade_on': future[entry_delay - 1]})
        
        daily_eq.append({'date': today, 'equity': cash})
    
    # Results
    edf = pd.DataFrame(daily_eq)
    edf['date'] = pd.to_datetime(edf['date'])
    edf['year'] = edf['date'].dt.year
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    n = len(tdf)
    total_ret = (cash / INITIAL_CAPITAL - 1) * 100
    cagr = ((cash / INITIAL_CAPITAL) ** (1/6) - 1) * 100
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    targets = len(tdf[tdf['exit_reason'] == 'target']) if n > 0 else 0
    avg_ret = tdf['return_pct'].mean() if n > 0 else 0
    
    annual = {}
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        annual[year] = {'return': round(yr_ret, 1), 'trades': yr_trades, 'losers': yr_losers}
    
    return {'label': label, 'total': round(total_ret, 1), 'cagr': round(cagr, 1),
            'trades': n, 'wr': round(wr, 1), 'losers': losers, 'targets': targets,
            'avg_trade_ret': round(avg_ret, 3), 'annual': annual}

print(f"\n{'='*80}")
print(f"  OPENING RANGE BREAKOUT ON DROP CANDIDATES")
print(f"{'='*80}\n")

configs = [
    # (range_min, target, delay, regime_gate, bull_frac, max_pos, label)
    # 30-min range, various targets
    (30, 0.005, 1, True, 3, 5, "30min ORB, +0.5%, T+1, gated, 1/3"),
    (30, 0.01, 1, True, 3, 5, "30min ORB, +1%, T+1, gated, 1/3"),
    (30, 0.02, 1, True, 3, 5, "30min ORB, +2%, T+1, gated, 1/3"),
    (30, 0.03, 1, True, 3, 5, "30min ORB, +3%, T+1, gated, 1/3"),
    
    # 15-min range
    (15, 0.01, 1, True, 3, 5, "15min ORB, +1%, T+1, gated, 1/3"),
    (15, 0.02, 1, True, 3, 5, "15min ORB, +2%, T+1, gated, 1/3"),
    
    # 60-min range
    (60, 0.01, 1, True, 3, 5, "60min ORB, +1%, T+1, gated, 1/3"),
    (60, 0.02, 1, True, 3, 5, "60min ORB, +2%, T+1, gated, 1/3"),
    
    # T+2 entry
    (30, 0.01, 2, True, 3, 5, "30min ORB, +1%, T+2, gated, 1/3"),
    (30, 0.02, 2, True, 3, 5, "30min ORB, +2%, T+2, gated, 1/3"),
    
    # No regime gate
    (30, 0.01, 1, False, 3, 5, "30min ORB, +1%, T+1, NO gate, 1/3"),
    (30, 0.02, 1, False, 3, 5, "30min ORB, +2%, T+1, NO gate, 1/3"),
    
    # More positions
    (30, 0.01, 1, True, 4, 8, "30min ORB, +1%, T+1, gated, 1/4, max8"),
    (30, 0.02, 1, True, 4, 8, "30min ORB, +2%, T+1, gated, 1/4, max8"),
    
    # No intraday target (just sell at close)
    (30, 1.0, 1, True, 3, 5, "30min ORB, sell at close, T+1, gated"),
    (15, 1.0, 1, True, 3, 5, "15min ORB, sell at close, T+1, gated"),
]

results = []
for range_min, target, delay, gate, bf, mp, label in configs:
    r = run_orb(range_min, target, delay, gate, bf, mp, label)
    results.append(r)
    
    yearly = [d['return'] for yr, d in sorted(r['annual'].items()) if yr < 2026]
    worst = min(yearly) if yearly else 0
    
    print(f"  {label}")
    print(f"    Total: {r['total']:>+8.1f}% | CAGR: {r['cagr']:>+5.1f}% | WR: {r['wr']:.0f}% | {r['trades']} trades | targets: {r['targets']} | avg trade: {r['avg_trade_ret']:+.3f}%")
    for yr, d in sorted(r['annual'].items()):
        if yr >= 2026: continue
        print(f"      {yr}: {d['return']:>+7.1f}% ({d['trades']} trades, {d['losers']} losers)")
    print()

# Save
with open(os.path.join(OUT_DIR, 'v_orb_results.txt'), 'w') as f:
    f.write("OPENING RANGE BREAKOUT ON DROP CANDIDATES\n\n")
    f.write(f"{'Label':>45} {'Total':>8} {'CAGR':>6} {'WR':>5} {'Trades':>7} {'AvgTrade':>9}\n")
    f.write('-'*85 + '\n')
    for r in results:
        f.write(f"{r['label']:>45} {r['total']:>+7.1f}% {r['cagr']:>+5.1f}% {r['wr']:>4.0f}% {r['trades']:>7} {r['avg_trade_ret']:>+8.3f}%\n")

print(f"Saved to {OUT_DIR}/v_orb_results.txt")
