#!/usr/bin/env python3
"""
Download 1-minute intraday bars for all S&P 100 tickers from Alpaca.
Uses 4 API keys in parallel via ThreadPoolExecutor.
Saves one parquet file per year (2020-2025), regular trading hours only.
"""
from dotenv import load_dotenv
import os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Build clients from all 4 keys
CLIENTS = []
for i in range(1, 5):
    key = os.getenv(f"ALPACA_KEY_{i}")
    secret = os.getenv(f"ALPACA_SECRET_{i}")
    if key and secret:
        CLIENTS.append(StockHistoricalDataClient(key, secret))
        print(f"Key {i}: loaded ({key[:8]}...)")

print(f"Total clients: {len(CLIENTS)}")

# Tickers
events = pd.read_csv('outputs/events_fully_labeled.csv')
TICKERS = sorted(events['ticker'].unique().tolist())
# Alpaca uses BRK.B not BRK-B
TICKERS = [t.replace('BRK-B', 'BRK.B') for t in TICKERS]
print(f"Tickers: {len(TICKERS)}")

OUT_DIR = 'intraday_data'
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK_SIZE = 10  # tickers per request


def fetch_chunk(client_idx, tickers_chunk, start, end):
    """Fetch 1-min bars for a chunk of tickers. Returns DataFrame or None."""
    client = CLIENTS[client_idx % len(CLIENTS)]
    try:
        req = StockBarsRequest(
            symbol_or_symbols=tickers_chunk,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(req).df
        if len(bars) == 0:
            return None

        bars = bars.reset_index()
        # Filter to regular trading hours: 9:30-16:00 ET = 13:30-20:00 UTC
        h = bars['timestamp'].dt.hour
        m = bars['timestamp'].dt.minute
        mask = ((h == 13) & (m >= 30)) | ((h >= 14) & (h < 20))
        bars = bars[mask]
        return bars
    except Exception as e:
        print(f"  ERROR ({tickers_chunk[0]}...): {e}")
        return None


def download_year(year):
    outpath = os.path.join(OUT_DIR, f'bars_1min_{year}.parquet')
    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        print(f"  Already exists: {len(existing):,} bars — skipping")
        return

    quarters = [
        (f"{year}-01-01", f"{year}-04-01"),
        (f"{year}-04-01", f"{year}-07-01"),
        (f"{year}-07-01", f"{year}-10-01"),
        (f"{year}-10-01", f"{year+1}-01-01"),
    ]

    # Build all jobs: (quarter, ticker_chunk)
    jobs = []
    for q_start, q_end in quarters:
        for i in range(0, len(TICKERS), CHUNK_SIZE):
            chunk = TICKERS[i:i + CHUNK_SIZE]
            jobs.append((chunk, q_start, q_end))

    print(f"  {len(jobs)} requests to make across {len(CLIENTS)} keys...")

    all_bars = []
    completed = 0

    # Run in parallel with 4 threads (one per key)
    with ThreadPoolExecutor(max_workers=len(CLIENTS)) as executor:
        futures = {}
        for idx, (chunk, q_start, q_end) in enumerate(jobs):
            f = executor.submit(fetch_chunk, idx, chunk, q_start, q_end)
            futures[f] = (chunk, q_start)

        for future in as_completed(futures):
            chunk, q_start = futures[future]
            result = future.result()
            completed += 1
            if result is not None and len(result) > 0:
                all_bars.append(result)
            if completed % 20 == 0 or completed == len(jobs):
                print(f"  Progress: {completed}/{len(jobs)} requests done")

    if all_bars:
        df = pd.concat(all_bars, ignore_index=True)
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        df.to_parquet(outpath)
        syms = df['symbol'].nunique()
        days = df['timestamp'].dt.date.nunique()
        print(f"  SAVED: {len(df):,} bars | {syms} tickers | {days} days | {os.path.getsize(outpath)/1e6:.0f} MB")
    else:
        print(f"  NO DATA for {year}")


t_total = time.time()
for year in range(2020, 2026):
    print(f"\n{'='*50}")
    print(f"  {year}")
    print(f"{'='*50}")
    t0 = time.time()
    download_year(year)
    print(f"  Done in {time.time()-t0:.0f}s")

print(f"\n{'='*50}")
print(f"  ALL DONE in {(time.time()-t_total)/60:.1f} minutes")
print(f"{'='*50}")
total = 0
for year in range(2020, 2026):
    path = os.path.join(OUT_DIR, f'bars_1min_{year}.parquet')
    if os.path.exists(path):
        sz = os.path.getsize(path) / 1e6
        n = len(pd.read_parquet(path, columns=['symbol']))
        total += n
        print(f"  {year}: {n:>12,} bars | {sz:.0f} MB")
print(f"  TOTAL: {total:>12,} bars")
