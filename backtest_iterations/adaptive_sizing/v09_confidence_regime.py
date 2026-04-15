#!/usr/bin/env python3
"""
SIZING V09: Combine bull/bear regime (1/6 bull, 1/12 bear equity-based) with
confidence multiplier based on event characteristics. High-confidence setups
get 1.3x size, low-confidence get 0.7x. Also test a "VIX regime" variant
that uses VIX level instead of SPY vs 50MA for bull/bear detection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import load_cached_data, run_backtest, save_results, print_results, INITIAL_CAPITAL
import pandas as pd
import numpy as np

close, high, low, spy, events = load_cached_data()
spy_close = spy['SPY']
spy_50ma = spy_close.rolling(50).mean()

# Load VIX for alternative regime detection
import yfinance as yf
vix = yf.download('^VIX', start='2019-01-01', end='2026-01-01', progress=False)['Close']
vix.index = vix.index.tz_localize(None)

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

# --- VARIANT A: Bull/bear + confidence multiplier ---
def sizing_confidence(row, ctx):
    today = ctx['today']
    equity = ctx.get('equity', INITIAL_CAPITAL)
    
    # Base: bull/bear regime
    is_bull = False
    if today in spy_close.index and today in spy_50ma.index:
        s = spy_close.at[today]
        m = spy_50ma.at[today]
        if pd.notna(s) and pd.notna(m) and s > m:
            is_bull = True
    base = equity / 6 if is_bull else equity / 12
    
    # Confidence multiplier based on event characteristics
    mult = 1.0
    t = row['ticker']
    d = row['drop_pct']
    sec = row.get('sector_etf', '')
    
    # High confidence signals (historically better bouncers)
    if t in BEST_TICKERS: mult += 0.15
    if d >= 0.07: mult += 0.15  # bigger drops bounce more
    if sec in BEST_SECTORS: mult += 0.1
    # Low confidence signals
    stress = row.get('liquidity_credit_stress_severity', 'none')
    if stress == 'high': mult -= 0.15  # extreme stress = risky
    if d < 0.04: mult -= 0.1  # small drops less reliable
    
    mult = max(0.6, min(1.4, mult))  # clamp
    return base * mult

# --- VARIANT B: VIX-based regime instead of SPY/50MA ---
def sizing_vix_regime(row, ctx):
    today = ctx['today']
    equity = ctx.get('equity', INITIAL_CAPITAL)
    
    # VIX-based regime: low VIX = bull, high VIX = bear
    v = vix.get(today, None) if today in vix.index else None
    if v is not None and not pd.isna(v):
        if float(v) < 20:
            return equity / 6   # calm market = bull sizing
        elif float(v) < 30:
            return equity / 10  # moderate fear
        else:
            return equity / 16  # high fear = small positions
    return equity / 12  # default

# --- VARIANT C: Dual regime (SPY trend + VIX level) ---
def sizing_dual_regime(row, ctx):
    today = ctx['today']
    equity = ctx.get('equity', INITIAL_CAPITAL)
    
    is_bull = False
    if today in spy_close.index and today in spy_50ma.index:
        s = spy_close.at[today]
        m = spy_50ma.at[today]
        if pd.notna(s) and pd.notna(m) and s > m:
            is_bull = True
    
    v = vix.get(today, None) if today in vix.index else None
    low_vix = v is not None and not pd.isna(v) and float(v) < 22
    
    if is_bull and low_vix:
        return equity / 5   # strong bull: aggressive
    elif is_bull:
        return equity / 7   # bull but elevated VIX: moderate
    elif low_vix:
        return equity / 10  # not trending up but calm
    else:
        return equity / 14  # bear + high VIX: conservative

configs = [
    (sizing_confidence, "V09a: bull/bear + confidence multiplier"),
    (sizing_vix_regime, "V09b: VIX-based regime"),
    (sizing_dual_regime, "V09c: dual regime (SPY trend + VIX)"),
]

for sfn, label in configs:
    results = run_backtest(events, close, high, low, entry_filter,
        target_return=0.05, max_hold_days=60, max_positions=10, entry_delay=2,
        sizing_fn=sfn)
    print_results(results, f"SIZING {label}")
    fname = label.split(':')[0].strip().lower().replace(' ', '_')
    save_results(results, os.path.join(os.path.dirname(__file__), f'{fname}_results.txt'))
