"""
download_alpaca_intraday.py

Purpose:
    Download historical stock bars from Alpaca for a user-defined list of tickers
    over a user-defined date range, then save one file per ticker locally.

Important Alpaca notes for local AI / future maintenance:
1) Free stock historical data feed:
   - Use feed="iex" on free/no-subscription access.
   - Alpaca docs say IEX is the only stock feed usable without a subscription.
   - SIP is broader/full-market data, but generally needs paid access.

2) Timeframes supported by the historical stock bars endpoint include:
   - 1Min to 59Min
   - 1Hour to 23Hour
   - 1Day, 1Week, etc.

3) API result sizing:
   - The historical bars endpoint has a per-request limit of up to 10,000 bars.
   - Large responses are paginated with next_page_token.
   - The alpaca-py SDK generally handles pagination when fetching bars, but very large
     jobs can still be slow and memory-heavy.

4) Practical warning:
   - 5 years of 1-minute data for many tickers is a LOT of data.
   - Save locally and reuse instead of redownloading repeatedly.

5) Auth:
   - Requires APCA API key + secret.
   - Easiest setup is environment variables:
       export APCA_API_KEY_ID="..."
       export APCA_API_SECRET_KEY="..."
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# =========================
# USER CONFIG
# =========================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
]

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

# Choose one:
#   TimeFrame(1, TimeFrameUnit.Minute)
#   TimeFrame(5, TimeFrameUnit.Minute)
#   TimeFrame(15, TimeFrameUnit.Minute)
#   TimeFrame(1, TimeFrameUnit.Hour)
TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)

# Free users should generally use "iex"
FEED = "iex"

# Output folder
OUTPUT_DIR = Path("alpaca_data")

# Save format: "csv" or "parquet"
SAVE_FORMAT = "parquet"


def get_client() -> StockHistoricalDataClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials.\n"
            "Set environment variables first:\n"
            '  export APCA_API_KEY_ID="your_key"\n'
            '  export APCA_API_SECRET_KEY="your_secret"'
        )

    return StockHistoricalDataClient(api_key, api_secret)


def fetch_bars_for_ticker(
    client: StockHistoricalDataClient,
    ticker: str,
    start_date: str,
    end_date: str,
    timeframe: TimeFrame,
    feed: str = "iex",
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=timeframe,
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        feed=feed,
    )

    bars = client.get_stock_bars(request).df

    if bars.empty:
        return bars

    # alpaca-py often returns a MultiIndex: (symbol, timestamp)
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
    else:
        bars = bars.reset_index().rename(columns={"index": "timestamp"})

    # Standardize column order if present
    preferred_cols = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    existing_cols = [c for c in preferred_cols if c in bars.columns]
    other_cols = [c for c in bars.columns if c not in existing_cols]
    bars = bars[existing_cols + other_cols]

    return bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def save_df(df: pd.DataFrame, path: Path, save_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "csv":
        df.to_csv(path.with_suffix(".csv"), index=False)
    elif save_format == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        raise ValueError("SAVE_FORMAT must be either 'csv' or 'parquet'")


def main() -> None:
    client = get_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"Downloading {ticker} ...")
        df = fetch_bars_for_ticker(
            client=client,
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
            timeframe=TIMEFRAME,
            feed=FEED,
        )

        if df.empty:
            print(f"  No data returned for {ticker}")
            continue

        safe_tf = str(TIMEFRAME).replace(" ", "_")
        out_path = OUTPUT_DIR / f"{ticker}_{START_DATE}_{END_DATE}_{safe_tf}_{FEED}"
        save_df(df, out_path, SAVE_FORMAT)

        print(f"  Saved {len(df):,} rows for {ticker}")

    print("Done.")


if __name__ == "__main__":
    main()
