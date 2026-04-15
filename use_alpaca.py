#!/usr/bin/env python3
"""
Test Alpaca data limits and plan the full download.
"""
from dotenv import load_dotenv
import os, time

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Test: How much data can we pull in one request?
# Try 1 ticker, 1 month of 1-min data
print("=== Test: AAPL 1-min, full month (Jan 2024) ===")
t0 = time.time()
req = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Minute,
    start="2024-01-02",
    end="2024-02-01",
)
bars = client.get_stock_bars(req).df
elapsed = time.time() - t0
print(f"Rows: {len(bars)} | Time: {elapsed:.1f}s")
print(f"Date range: {bars.index.get_level_values('timestamp').min().date()} to {bars.index.get_level_values('timestamp').max().date()}")
days = bars.index.get_level_values('timestamp').normalize().nunique()
print(f"Trading days: {days} | Bars/day: {len(bars)/days:.0f}")

# Test: 10 tickers, 1 month
print("\n=== Test: 10 tickers, 1-min, 1 month ===")
tickers = ["AAPL", "NVDA", "AMZN", "MSFT", "META", "GOOGL", "CVX", "XOM", "JPM", "V"]
t0 = time.time()
req2 = StockBarsRequest(
    symbol_or_symbols=tickers,
    timeframe=TimeFrame.Minute,
    start="2024-01-02",
    end="2024-02-01",
)
bars2 = client.get_stock_bars(req2).df
elapsed = time.time() - t0
print(f"Total rows: {len(bars2)} | Time: {elapsed:.1f}s")
for sym in tickers:
    if sym in bars2.index.get_level_values('symbol'):
        n = len(bars2.loc[sym])
        print(f"  {sym}: {n} bars")

# Estimate total download size
print("\n=== DOWNLOAD PLAN ===")
bars_per_ticker_per_day = 700  # approximate
tickers_total = 101  # S&P 100 + SPY
trading_days_per_year = 252
years = 6  # 2020-2025

total_bars = bars_per_ticker_per_day * tickers_total * trading_days_per_year * years
print(f"Estimated total bars: {total_bars:,.0f} ({total_bars/1e6:.1f}M)")
print(f"At ~8 bytes per value, 7 columns: {total_bars * 7 * 8 / 1e9:.1f} GB raw")
print(f"Parquet compression: ~{total_bars * 7 * 8 / 1e9 / 5:.1f} GB compressed")

# We only need regular trading hours (9:30 AM - 4:00 PM ET = 14:30-21:00 UTC)
rth_bars = 390  # 6.5 hours * 60 min
total_rth = rth_bars * tickers_total * trading_days_per_year * years
print(f"\nRegular hours only: {total_rth:,.0f} bars ({total_rth/1e6:.1f}M)")
print(f"Parquet: ~{total_rth * 7 * 8 / 1e9 / 5:.1f} GB compressed")
