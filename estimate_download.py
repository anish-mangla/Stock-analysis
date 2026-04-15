#!/usr/bin/env python3
"""
Benchmark Alpaca download speed and estimate total time.
Also cross-check a few data points against yfinance.
"""
from dotenv import load_dotenv
import os, time
import pandas as pd

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# === BENCHMARK: Time a realistic chunk ===
# 10 tickers, 3 months (one "unit" of our download plan)
print("=== BENCHMARKING ===")
tickers_10 = ["AAPL", "NVDA", "AMZN", "MSFT", "META", "GOOGL", "CVX", "XOM", "JPM", "V"]

t0 = time.time()
req = StockBarsRequest(
    symbol_or_symbols=tickers_10,
    timeframe=TimeFrame.Minute,
    start="2024-01-01",
    end="2024-04-01",
)
bars = client.get_stock_bars(req).df
elapsed = time.time() - t0
print(f"10 tickers x 3 months: {len(bars):,} bars in {elapsed:.1f}s")

# Our download plan: 101 tickers / 10 per chunk = 11 chunks per quarter
# 4 quarters per year, 6 years = 24 quarters
# Total requests: 11 chunks * 24 quarters = 264 requests
# Plus 0.3s delay between requests

chunks_per_quarter = 11  # ceil(101/10)
quarters = 24  # 6 years * 4
total_requests = chunks_per_quarter * quarters
time_per_request = elapsed  # ~3s based on benchmark
delay_per_request = 0.3

total_time = total_requests * (time_per_request + delay_per_request)
print(f"\nTotal requests needed: {total_requests}")
print(f"Time per request: ~{time_per_request:.1f}s + {delay_per_request}s delay")
print(f"Estimated total time: {total_time:.0f}s = {total_time/60:.1f} minutes")
print(f"\nWith 2 keys in parallel: ~{total_time/60/2:.1f} minutes")
print(f"With 3 keys in parallel: ~{total_time/60/3:.1f} minutes")

# === CROSS-CHECK: Verify Alpaca data against yfinance ===
print("\n\n=== CROSS-CHECK: Alpaca vs yfinance ===")
import yfinance as yf

# Get AAPL daily data from both sources for Jan 2024
# Alpaca: aggregate 1-min bars to daily OHLCV
alpaca_aapl = bars.loc["AAPL"].copy()
alpaca_aapl.index = alpaca_aapl.index.tz_localize(None)
alpaca_aapl['date'] = alpaca_aapl.index.date

daily_alpaca = alpaca_aapl.groupby('date').agg(
    open=('open', 'first'),
    high=('high', 'max'),
    low=('low', 'min'),
    close=('close', 'last'),
    volume=('volume', 'sum'),
).head(10)

# yfinance daily
yf_data = yf.download("AAPL", start="2024-01-02", end="2024-01-16", progress=False)
if isinstance(yf_data.columns, pd.MultiIndex):
    yf_data.columns = [c[0] for c in yf_data.columns]
yf_daily = yf_data[['Open', 'High', 'Low', 'Close', 'Volume']].head(10)
yf_daily.index = yf_daily.index.date

print(f"\n{'Date':>12} | {'Alpaca O':>10} {'yF O':>10} {'diff':>7} | {'Alpaca C':>10} {'yF C':>10} {'diff':>7} | {'Alpaca H':>10} {'yF H':>10} | {'Alpaca L':>10} {'yF L':>10}")
print("-" * 130)
for date in daily_alpaca.index:
    if date in yf_daily.index:
        ao, ac, ah, al = daily_alpaca.loc[date, ['open', 'high', 'low', 'close']]
        yo, yc, yh, yl = yf_daily.loc[date, ['Open', 'Close', 'High', 'Low']]
        do = (ao - yo) / yo * 100
        dc = (ac - yc) / yc * 100
        print(f"{str(date):>12} | {ao:>10.2f} {yo:>10.2f} {do:>+6.2f}% | {ac:>10.2f} {yc:>10.2f} {dc:>+6.2f}% | {ah:>10.2f} {yh:>10.2f} | {al:>10.2f} {yl:>10.2f}")

# Also check a high-dividend stock (CVX) to make sure adjustments match
print("\n--- CVX (high dividend stock) ---")
alpaca_cvx = bars.loc["CVX"].copy()
alpaca_cvx.index = alpaca_cvx.index.tz_localize(None)
alpaca_cvx['date'] = alpaca_cvx.index.date
daily_cvx = alpaca_cvx.groupby('date').agg(
    open=('open', 'first'), high=('high', 'max'),
    low=('low', 'min'), close=('close', 'last'),
).head(5)

yf_cvx = yf.download("CVX", start="2024-01-02", end="2024-01-10", auto_adjust=False, progress=False)
if isinstance(yf_cvx.columns, pd.MultiIndex):
    yf_cvx.columns = [c[0] for c in yf_cvx.columns]
yf_cvx.index = yf_cvx.index.date

print(f"{'Date':>12} | {'Alpaca C':>10} {'yF Close':>10} {'diff':>7}")
print("-" * 50)
for date in daily_cvx.index:
    if date in yf_cvx.index:
        ac = daily_cvx.loc[date, 'close']
        yc = yf_cvx.loc[date, 'Close']
        dc = (ac - yc) / yc * 100
        print(f"{str(date):>12} | {ac:>10.2f} {yc:>10.2f} {dc:>+6.2f}%")
