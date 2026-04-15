#!/usr/bin/env python3
"""
Thorough cross-check of Alpaca intraday data against yfinance daily data.
For every ticker in every year, aggregate 1-min bars to daily OHLC and
compare against yfinance unadjusted daily bars.
"""
import pandas as pd
import yfinance as yf
import os

print("="*70)
print("  COMPREHENSIVE DATA VERIFICATION: Alpaca vs yfinance")
print("="*70)

# We'll check 5 random trading days per year per ticker
# Aggregate Alpaca 1-min → daily, compare to yfinance daily

all_mismatches = []

for year in range(2020, 2026):
    path = f'intraday_data/bars_1min_{year}.parquet'
    if not os.path.exists(path):
        print(f"\n{year}: FILE MISSING")
        continue
    
    bars = pd.read_parquet(path)
    bars['date'] = bars['timestamp'].dt.date
    tickers = sorted(bars['symbol'].unique())
    
    # Aggregate to daily
    daily_alpaca = bars.groupby(['symbol', 'date']).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        bar_count=('close', 'count'),
    ).reset_index()
    
    # Pick check dates: first week of each quarter
    check_months = [1, 4, 7, 10]
    all_dates = sorted(daily_alpaca['date'].unique())
    check_dates = []
    for m in check_months:
        month_dates = [d for d in all_dates if d.month == m]
        if month_dates:
            check_dates.extend(month_dates[:3])  # first 3 trading days of that month
    
    if not check_dates:
        print(f"\n{year}: no check dates found")
        continue
    
    # Download yfinance data for all tickers for these date ranges
    yf_tickers = [t.replace('BRK.B', 'BRK-B') for t in tickers]
    start_date = str(min(check_dates))
    end_date = str(max(check_dates) + pd.Timedelta(days=1))
    
    print(f"\n{'='*70}")
    print(f"  {year}: checking {len(tickers)} tickers on {len(check_dates)} dates")
    print(f"{'='*70}")
    
    yf_data = yf.download(yf_tickers, start=start_date, end=end_date, 
                          auto_adjust=False, progress=False)
    
    if isinstance(yf_data.columns, pd.MultiIndex):
        # Multi-ticker format
        pass
    else:
        # Single ticker - shouldn't happen with our list
        continue
    
    matches = 0
    mismatches = 0
    big_mismatches = 0
    checked = 0
    
    for ticker in tickers:
        yf_ticker = ticker.replace('BRK.B', 'BRK-B')
        
        for check_date in check_dates:
            # Alpaca daily
            alpaca_row = daily_alpaca[(daily_alpaca['symbol'] == ticker) & 
                                      (daily_alpaca['date'] == check_date)]
            if len(alpaca_row) == 0:
                continue
            
            a_open = float(alpaca_row['open'].iloc[0])
            a_close = float(alpaca_row['close'].iloc[0])
            a_high = float(alpaca_row['high'].iloc[0])
            a_low = float(alpaca_row['low'].iloc[0])
            a_bars = int(alpaca_row['bar_count'].iloc[0])
            
            # yfinance daily
            ts = pd.Timestamp(check_date)
            if ts not in yf_data.index:
                continue
            
            try:
                y_open = float(yf_data.loc[ts, ('Open', yf_ticker)])
                y_close = float(yf_data.loc[ts, ('Close', yf_ticker)])
                y_high = float(yf_data.loc[ts, ('High', yf_ticker)])
                y_low = float(yf_data.loc[ts, ('Low', yf_ticker)])
            except (KeyError, TypeError):
                continue
            
            if pd.isna(y_close) or y_close == 0:
                continue
            
            checked += 1
            
            # Compare (allow 0.5% tolerance for open due to pre-market vs RTH)
            close_diff = abs(a_close - y_close) / y_close * 100
            open_diff = abs(a_open - y_open) / y_open * 100
            high_diff = abs(a_high - y_high) / y_high * 100
            low_diff = abs(a_low - y_low) / y_low * 100
            
            if close_diff > 1.0 or high_diff > 1.0 or low_diff > 1.0:
                big_mismatches += 1
                all_mismatches.append({
                    'year': year, 'ticker': ticker, 'date': check_date,
                    'a_open': a_open, 'y_open': y_open, 'open_diff': open_diff,
                    'a_close': a_close, 'y_close': y_close, 'close_diff': close_diff,
                    'a_high': a_high, 'y_high': y_high, 'high_diff': high_diff,
                    'a_low': a_low, 'y_low': y_low, 'low_diff': low_diff,
                    'bars': a_bars,
                })
                if big_mismatches <= 5:
                    print(f"  MISMATCH: {ticker} {check_date} | "
                          f"close: alpaca={a_close:.2f} yf={y_close:.2f} ({close_diff:.1f}%) | "
                          f"high: alpaca={a_high:.2f} yf={y_high:.2f} ({high_diff:.1f}%) | "
                          f"bars={a_bars}")
            elif close_diff > 0.1:
                mismatches += 1
            else:
                matches += 1
    
    print(f"  Checked: {checked} data points")
    print(f"  Perfect (<0.1% diff): {matches} ({matches/max(checked,1)*100:.0f}%)")
    print(f"  Small diff (0.1-1%):  {mismatches} ({mismatches/max(checked,1)*100:.0f}%)")
    print(f"  BIG diff (>1%):       {big_mismatches} ({big_mismatches/max(checked,1)*100:.0f}%)")

# Summary of all big mismatches
if all_mismatches:
    mdf = pd.DataFrame(all_mismatches)
    print(f"\n{'='*70}")
    print(f"  ALL BIG MISMATCHES (>1% difference)")
    print(f"{'='*70}")
    print(f"  Total: {len(mdf)}")
    
    # Check if they're all stock splits
    print(f"\n  By ticker:")
    for ticker, group in mdf.groupby('ticker'):
        avg_diff = group['close_diff'].mean()
        print(f"    {ticker}: {len(group)} mismatches, avg close diff {avg_diff:.1f}%")
    
    print(f"\n  These are likely stock splits (Alpaca=pre-split, yfinance=post-split adjusted)")
