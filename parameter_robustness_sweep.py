#!/usr/bin/env python3
"""
parameter_robustness_sweep.py

Quest 5 of 5:
Stress test the strongest numeric rules by sweeping thresholds.

This script:
1. Reads outputs/events.csv
2. Learns GOOD/BAD combos from TRAIN data only
3. Labels TEST events using those learned combos
4. Adds sector-relative and volatility-normalized features
5. Sweeps threshold combinations:
   - stock underperformance vs sector: 1%, 2%, 3%
   - drop vs vol20: 1.25x, 1.5x, 1.75x, 2.0x
6. Runs portfolio simulations for each split + parameter combo

Inputs:
- outputs/events.csv

Outputs:
- outputs/parameter_sweep_split_summary.csv
- outputs/parameter_sweep_all_trades.csv
- outputs/parameter_sweep_all_equity.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd
import yfinance as yf


OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")

SWEEP_SPLIT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "parameter_sweep_split_summary.csv")
SWEEP_ALL_TRADES_CSV = os.path.join(OUTPUT_DIR, "parameter_sweep_all_trades.csv")
SWEEP_ALL_EQUITY_CSV = os.path.join(OUTPUT_DIR, "parameter_sweep_all_equity.csv")

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 10_000.0
MAX_POSITIONS = 10
ALLOW_DUPLICATE_TICKERS = False

GOOD_COMBO_MIN_EVENTS = 10
GOOD_COMBO_MIN_SUCCESS_RATE = 0.85
GOOD_COMBO_MIN_AVG_FINAL_RETURN = 0.0

BAD_COMBO_MIN_EVENTS = 8
BAD_COMBO_MAX_SUCCESS_RATE = 0.65
BAD_COMBO_MAX_AVG_FINAL_RETURN = 0.0

SORT_SAME_DAY_CANDIDATES_BY = [
    "combo_score",
    "success_rate",
    "avg_final_return",
    "total_events",
]

DOWNLOAD_AUTO_ADJUST = False
YF_GROUP_BY = "ticker"

SPLITS = [
    {
        "split_name": "train_2020_2021_test_2022",
        "train_start": "2020-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    {
        "split_name": "train_2020_2022_test_2023",
        "train_start": "2020-01-01",
        "train_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
    },
    {
        "split_name": "train_2020_2023_test_2024",
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "split_name": "train_2020_2024_test_2025",
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
    {
        "split_name": "train_2021_2023_test_2024_rolling",
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "split_name": "train_2022_2024_test_2025_rolling",
        "train_start": "2022-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
]

UNDERPERF_THRESHOLDS = [0.01, 0.02, 0.03]
DROP_VS_VOL_THRESHOLDS = [1.25, 1.50, 1.75, 2.00]

TICKER_TO_SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH",
    "AMAT": "SMH", "LRCX": "SMH", "QCOM": "SMH", "TXN": "SMH", "MU": "SMH",
    "INTC": "SMH", "NOW": "XLK", "INTU": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "IBM": "XLK", "ACN": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "AMZN": "XLY", "META": "XLC", "NFLX": "XLC", "CMCSA": "XLC", "TMUS": "XLC",
    "T": "XLC", "VZ": "XLC", "DIS": "XLC", "UBER": "XLY", "TSLA": "XLY",

    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF",
    "MS": "XLF", "AXP": "XLF", "BLK": "XLF", "USB": "XLF", "SCHW": "XLF",
    "SPGI": "XLF", "MCO": "XLF", "CB": "XLF", "PGR": "XLF", "BRK-B": "XLF",

    "LLY": "XLV", "PFE": "XLV", "BMY": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "TMO": "XLV", "DHR": "XLV", "ABT": "XLV", "MDT": "XLV", "GILD": "XLV",
    "ISRG": "XLV", "UNH": "XLV", "CI": "XLV", "CVS": "XLV", "AMGN": "XLV",

    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",

    "CAT": "XLI", "DE": "XLI", "HON": "XLI", "GE": "XLI", "GD": "XLI",
    "LMT": "XLI", "UNP": "XLI", "UPS": "XLI", "RTX": "XLI", "BA": "XLI",

    "HD": "XLY", "LOW": "XLY", "MCD": "XLY", "NKE": "XLY", "SBUX": "XLY",
    "BKNG": "XLY", "TJX": "XLY", "COST": "XLP", "WMT": "XLP", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "PM": "XLP", "MDLZ": "XLP", "CL": "XLP",

    "DUK": "XLU", "NEE": "XLU", "SO": "XLU", "AMT": "XLRE", "PLD": "XLRE",
    "SPG": "XLRE", "LIN": "XLB", "SHW": "XLB", "APD": "XLB",
    "GEV": "XLI", "PLTR": "XLK",
}
DEFAULT_SECTOR_ETF = "SPY"


@dataclass
class Position:
    event_id: int
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    allocated_capital: float
    drop_bucket: str
    combo_score: float
    split_name: str
    filter_name: str


def load_events() -> pd.DataFrame:
    if not os.path.exists(EVENTS_CSV):
        raise FileNotFoundError(f"Missing required file: {EVENTS_CSV}")

    df = pd.read_csv(EVENTS_CSV)
    if df.empty:
        raise ValueError("events.csv is empty.")

    df["event_date"] = pd.to_datetime(df["event_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["drop_bucket"] = df["drop_bucket"].astype(str)

    df = df.dropna(subset=["event_date", "exit_date", "entry_price", "exit_price"])
    df = df.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    df["event_id"] = df.index.astype(int)
    return df


def summarize_train_by_ticker_and_bucket(train_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty:
        return pd.DataFrame()

    def _avg_days_success_only(series: pd.Series) -> float:
        valid = series.dropna()
        return float(valid.mean()) if not valid.empty else float("nan")

    grouped = (
        train_df.groupby(["ticker", "drop_bucket"], dropna=False)
        .agg(
            total_events=("ticker", "count"),
            success_rate=("success", "mean"),
            avg_days_to_hit=("days_to_hit", _avg_days_success_only),
            avg_final_return=("final_return", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
        )
        .reset_index()
    )

    grouped["failure_rate"] = 1.0 - grouped["success_rate"]
    grouped["combo_score"] = (
        grouped["success_rate"] * 100.0
        + grouped["avg_final_return"] * 100.0
        + grouped["avg_max_drawdown"] * 10.0
    )
    return grouped


def classify_combos_from_train(
    combo_summary_df: pd.DataFrame,
) -> tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    if combo_summary_df.empty:
        return set(), set()

    df = combo_summary_df.copy()
    df["drop_bucket"] = df["drop_bucket"].astype(str)

    good_mask = (
        (df["total_events"] >= GOOD_COMBO_MIN_EVENTS)
        & (df["success_rate"] >= GOOD_COMBO_MIN_SUCCESS_RATE)
        & (df["avg_final_return"] > GOOD_COMBO_MIN_AVG_FINAL_RETURN)
    )

    bad_mask = (
        (df["total_events"] >= BAD_COMBO_MIN_EVENTS)
        & (df["success_rate"] <= BAD_COMBO_MAX_SUCCESS_RATE)
        & (df["avg_final_return"] < BAD_COMBO_MAX_AVG_FINAL_RETURN)
    )

    good_combos = set(zip(df.loc[good_mask, "ticker"], df.loc[good_mask, "drop_bucket"]))
    bad_combos = set(zip(df.loc[bad_mask, "ticker"], df.loc[bad_mask, "drop_bucket"]))
    return good_combos, bad_combos


def label_test_events(
    test_df: pd.DataFrame,
    train_combo_stats_df: pd.DataFrame,
    good_combos: Set[Tuple[str, str]],
    bad_combos: Set[Tuple[str, str]],
) -> pd.DataFrame:
    if test_df.empty:
        return test_df.copy()

    combo_stats_cols = [
        "ticker",
        "drop_bucket",
        "total_events",
        "success_rate",
        "avg_final_return",
        "avg_max_drawdown",
        "combo_score",
    ]

    stats = train_combo_stats_df[combo_stats_cols].copy()

    labeled = test_df.merge(
        stats,
        on=["ticker", "drop_bucket"],
        how="left",
        suffixes=("", "_train"),
    )

    combo_keys = list(zip(labeled["ticker"], labeled["drop_bucket"]))
    labeled["is_good_combo"] = [key in good_combos for key in combo_keys]
    labeled["is_bad_combo"] = [key in bad_combos for key in combo_keys]

    def _label(is_good: bool, is_bad: bool) -> str:
        if is_bad:
            return "bad"
        if is_good:
            return "good"
        return "neutral"

    labeled["combo_label"] = [
        _label(g, b) for g, b in zip(labeled["is_good_combo"], labeled["is_bad_combo"])
    ]
    return labeled


def download_single_symbol_close(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    start_str = (start_date - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    end_str = (end_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"Downloading {symbol} data from {start_str} to {end_str}...")

    df = yf.download(
        tickers=symbol,
        start=start_str,
        end=end_str,
        auto_adjust=DOWNLOAD_AUTO_ADJUST,
        progress=True,
        threads=True,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol].copy()
        elif symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1).copy()
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    if "Close" not in df.columns:
        raise RuntimeError(f"{symbol} missing Close column.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.astype(float)
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    return close


def build_feature_maps(events_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp):
    sector_etfs = sorted(set(TICKER_TO_SECTOR_ETF.values()) | {DEFAULT_SECTOR_ETF})
    sector_close_map: Dict[str, pd.Series] = {}
    for etf in sector_etfs:
        sector_close_map[etf] = download_single_symbol_close(etf, start_date, end_date)

    tickers = sorted(events_df["ticker"].unique().tolist())
    stock_feature_map: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        close = download_single_symbol_close(ticker, start_date, end_date)

        feat = pd.DataFrame(index=close.index)
        feat["stock_close"] = close
        feat["stock_daily_return"] = feat["stock_close"].pct_change()
        feat["stock_vol20"] = feat["stock_daily_return"].rolling(20).std()
        stock_feature_map[ticker] = feat

    return sector_close_map, stock_feature_map


def add_features(
    events_df: pd.DataFrame,
    sector_close_map: Dict[str, pd.Series],
    stock_feature_map: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = events_df.copy()
    out["sector_etf"] = out["ticker"].map(TICKER_TO_SECTOR_ETF).fillna(DEFAULT_SECTOR_ETF)

    sector_parts = []
    for etf, group in out.groupby("sector_etf", sort=False):
        close = sector_close_map[etf]
        sector_df = pd.DataFrame(index=close.index)
        sector_df["sector_close"] = close
        sector_df["sector_daily_return"] = sector_df["sector_close"].pct_change()
        sector_df = sector_df.reset_index().rename(columns={"Date": "event_date"})
        if "event_date" not in sector_df.columns:
            sector_df = sector_df.rename(columns={sector_df.columns[0]: "event_date"})
        merged = group.merge(sector_df, on="event_date", how="left")
        sector_parts.append(merged)

    out = pd.concat(sector_parts, ignore_index=True)

    stock_parts = []
    for ticker, group in out.groupby("ticker", sort=False):
        feat = stock_feature_map[ticker].reset_index().rename(columns={"Date": "event_date"})
        if "event_date" not in feat.columns:
            feat = feat.rename(columns={feat.columns[0]: "event_date"})
        merged = group.merge(feat, on="event_date", how="left")
        stock_parts.append(merged)

    out = pd.concat(stock_parts, ignore_index=True)

    out["stock_minus_sector_return"] = out["daily_return"] - out["sector_daily_return"]
    out["drop_vs_vol20"] = out["drop_pct"] / out["stock_vol20"]
    return out


def make_filter_name(underperf: float, drop_vs_vol: float) -> str:
    return f"underperf_{int(underperf*100)}pct__drop_vs_vol_{drop_vs_vol:.2f}x"


def apply_sweep_filter(events_df: pd.DataFrame, underperf: float, drop_vs_vol: float) -> pd.DataFrame:
    mask = (
        (events_df["stock_minus_sector_return"] <= -underperf)
        & (events_df["drop_vs_vol20"] >= drop_vs_vol)
    )
    return events_df[mask].copy()


def download_close_matrix(
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("No tickers provided for price download.")

    start_str = (start_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_str = (end_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"Downloading close prices for {len(tickers)} tickers from {start_str} to {end_str}...")

    raw = yf.download(
        tickers=tickers,
        start=start_str,
        end=end_str,
        auto_adjust=DOWNLOAD_AUTO_ADJUST,
        progress=True,
        group_by=YF_GROUP_BY,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No price data returned from yfinance.")

    close_map: Dict[str, pd.Series] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            ticker_df = raw[ticker].copy()
            if "Close" not in ticker_df.columns:
                continue
            s = ticker_df["Close"].copy().dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index)
                close_map[ticker] = s.sort_index()
    else:
        if len(tickers) != 1:
            raise RuntimeError("Unexpected flat-column format for multiple tickers.")
        s = raw["Close"].copy().dropna()
        s.index = pd.to_datetime(s.index)
        close_map[tickers[0]] = s.sort_index()

    all_dates = sorted(set().union(*[set(s.index) for s in close_map.values()]))
    close_df = pd.DataFrame(index=pd.DatetimeIndex(all_dates))

    for ticker, s in close_map.items():
        close_df[ticker] = s

    close_df = close_df.sort_index().ffill()
    return close_df


def simulate_portfolio_for_test_split(
    split_name: str,
    filter_name: str,
    tradable_test_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(close_matrix.index.unique())

    events_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        dt: g.copy() for dt, g in tradable_test_df.groupby("event_date", sort=True)
    }

    cash = INITIAL_CAPITAL
    open_positions: Dict[int, Position] = {}

    trade_log_rows: List[dict] = []
    daily_rows: List[dict] = []

    cumulative_realized_pnl = 0.0
    max_equity_seen = INITIAL_CAPITAL

    for today in all_dates:
        exiting_ids = [pid for pid, pos in open_positions.items() if pos.exit_date == today]

        for pid in exiting_ids:
            pos = open_positions.pop(pid)

            exit_value = pos.shares * pos.exit_price
            realized_pnl = exit_value - pos.allocated_capital
            cumulative_realized_pnl += realized_pnl
            cash += exit_value

            trade_log_rows.append(
                {
                    "split_name": split_name,
                    "filter_name": filter_name,
                    "date": today,
                    "action": "SELL",
                    "event_id": pos.event_id,
                    "ticker": pos.ticker,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price,
                    "exit_date": pos.exit_date,
                    "exit_price": pos.exit_price,
                    "shares": pos.shares,
                    "allocated_capital": pos.allocated_capital,
                    "cash_after": cash,
                    "realized_pnl": realized_pnl,
                    "realized_return": realized_pnl / pos.allocated_capital,
                    "drop_bucket": pos.drop_bucket,
                    "combo_score": pos.combo_score,
                }
            )

        todays_candidates = events_by_date.get(today)
        if todays_candidates is not None and not todays_candidates.empty:
            candidates = todays_candidates.copy()
            candidates = candidates[candidates["combo_label"] == "good"].copy()

            sort_cols = [c for c in SORT_SAME_DAY_CANDIDATES_BY if c in candidates.columns]
            if sort_cols:
                candidates = candidates.sort_values(sort_cols, ascending=False)

            for _, row in candidates.iterrows():
                if len(open_positions) >= MAX_POSITIONS:
                    break
                if cash < POSITION_SIZE:
                    break

                ticker = str(row["ticker"])

                if not ALLOW_DUPLICATE_TICKERS:
                    already_open = any(pos.ticker == ticker for pos in open_positions.values())
                    if already_open:
                        continue

                entry_price = float(row["entry_price"])
                exit_price = float(row["exit_price"])
                exit_date = pd.Timestamp(row["exit_date"])

                if entry_price <= 0:
                    continue

                shares = POSITION_SIZE / entry_price

                pos = Position(
                    event_id=int(row["event_id"]),
                    ticker=ticker,
                    entry_date=pd.Timestamp(row["event_date"]),
                    entry_price=entry_price,
                    exit_date=exit_date,
                    exit_price=exit_price,
                    shares=shares,
                    allocated_capital=POSITION_SIZE,
                    drop_bucket=str(row["drop_bucket"]),
                    combo_score=float(row["combo_score"]) if pd.notna(row["combo_score"]) else 0.0,
                    split_name=split_name,
                    filter_name=filter_name,
                )

                open_positions[pos.event_id] = pos
                cash -= POSITION_SIZE

                trade_log_rows.append(
                    {
                        "split_name": split_name,
                        "filter_name": filter_name,
                        "date": today,
                        "action": "BUY",
                        "event_id": pos.event_id,
                        "ticker": pos.ticker,
                        "entry_date": pos.entry_date,
                        "entry_price": pos.entry_price,
                        "exit_date": pos.exit_date,
                        "exit_price": pos.exit_price,
                        "shares": pos.shares,
                        "allocated_capital": pos.allocated_capital,
                        "cash_after": cash,
                        "realized_pnl": None,
                        "realized_return": None,
                        "drop_bucket": pos.drop_bucket,
                        "combo_score": pos.combo_score,
                    }
                )

        market_value = 0.0
        for pos in open_positions.values():
            if pos.ticker not in close_matrix.columns:
                continue
            current_close = close_matrix.at[today, pos.ticker]
            if pd.isna(current_close):
                continue
            market_value += pos.shares * float(current_close)

        equity = cash + market_value
        max_equity_seen = max(max_equity_seen, equity)
        drawdown = (equity / max_equity_seen) - 1.0 if max_equity_seen > 0 else 0.0

        daily_rows.append(
            {
                "split_name": split_name,
                "filter_name": filter_name,
                "date": today,
                "cash": cash,
                "market_value": market_value,
                "equity": equity,
                "open_positions": len(open_positions),
                "drawdown": drawdown,
                "cumulative_realized_pnl": cumulative_realized_pnl,
            }
        )

    trade_log_df = pd.DataFrame(trade_log_rows).sort_values(["date", "action", "ticker"]).reset_index(drop=True)
    daily_equity_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    summary_df = build_split_summary(split_name, filter_name, trade_log_df, daily_equity_df)

    return trade_log_df, daily_equity_df, summary_df


def build_split_summary(
    split_name: str,
    filter_name: str,
    trade_log_df: pd.DataFrame,
    daily_equity_df: pd.DataFrame,
) -> pd.DataFrame:
    sells = trade_log_df[trade_log_df["action"] == "SELL"].copy()
    buys = trade_log_df[trade_log_df["action"] == "BUY"].copy()

    ending_equity = float(daily_equity_df["equity"].iloc[-1]) if not daily_equity_df.empty else INITIAL_CAPITAL
    max_drawdown = float(daily_equity_df["drawdown"].min()) if not daily_equity_df.empty else 0.0
    avg_open_positions = float(daily_equity_df["open_positions"].mean()) if not daily_equity_df.empty else 0.0

    total_closed_trades = int(len(sells))
    win_rate = float((sells["realized_pnl"] > 0).mean()) if total_closed_trades > 0 else None
    avg_realized_return = float(sells["realized_return"].mean()) if total_closed_trades > 0 else None
    avg_realized_pnl = float(sells["realized_pnl"].mean()) if total_closed_trades > 0 else None

    if total_closed_trades > 0:
        holding_days = (pd.to_datetime(sells["exit_date"]) - pd.to_datetime(sells["entry_date"])).dt.days
        avg_holding_days = float(holding_days.mean())
    else:
        avg_holding_days = None

    return pd.DataFrame(
        [
            {
                "split_name": split_name,
                "filter_name": filter_name,
                "initial_capital": INITIAL_CAPITAL,
                "ending_equity": ending_equity,
                "total_return": (ending_equity / INITIAL_CAPITAL) - 1.0,
                "max_drawdown": max_drawdown,
                "total_buy_signals_taken": int(len(buys)),
                "total_closed_trades": total_closed_trades,
                "win_rate_closed_trades": win_rate,
                "avg_realized_return_closed_trades": avg_realized_return,
                "avg_realized_pnl_closed_trades": avg_realized_pnl,
                "avg_holding_days_closed_trades": avg_holding_days,
                "avg_open_positions": avg_open_positions,
            }
        ]
    )


def run_backtest(events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trade_logs = []
    all_equity_curves = []
    all_split_summaries = []

    start_date = events_df["event_date"].min()
    end_date = events_df["exit_date"].max()

    sector_close_map, stock_feature_map = build_feature_maps(events_df, start_date, end_date)

    for split in SPLITS:
        split_name = split["split_name"]
        train_start = pd.Timestamp(split["train_start"])
        train_end = pd.Timestamp(split["train_end"])
        test_start = pd.Timestamp(split["test_start"])
        test_end = pd.Timestamp(split["test_end"])

        print(f"\n{'=' * 90}")
        print(f"Running split: {split_name}")
        print(f"Train: {train_start.date()} -> {train_end.date()}")
        print(f"Test:  {test_start.date()} -> {test_end.date()}")

        train_df = events_df[(events_df["event_date"] >= train_start) & (events_df["event_date"] <= train_end)].copy()
        test_df = events_df[(events_df["event_date"] >= test_start) & (events_df["event_date"] <= test_end)].copy()

        if train_df.empty or test_df.empty:
            print("Skipping split because train or test set is empty.")
            continue

        train_combo_summary = summarize_train_by_ticker_and_bucket(train_df)
        good_combos, bad_combos = classify_combos_from_train(train_combo_summary)
        labeled_test_df = label_test_events(test_df, train_combo_summary, good_combos, bad_combos)
        labeled_test_df = add_features(labeled_test_df, sector_close_map, stock_feature_map)

        base_good_test_df = labeled_test_df[labeled_test_df["combo_label"] == "good"].copy()

        print(f"Train events: {len(train_df)}")
        print(f"Test events:  {len(test_df)}")
        print(f"Learned good combos: {len(good_combos)}")
        print(f"Learned bad combos:  {len(bad_combos)}")
        print(f"Tradable good-combo test events before sweep filter: {len(base_good_test_df)}")

        for underperf in UNDERPERF_THRESHOLDS:
            for drop_vs_vol in DROP_VS_VOL_THRESHOLDS:
                filter_name = make_filter_name(underperf, drop_vs_vol)
                filtered_test_df = apply_sweep_filter(base_good_test_df, underperf, drop_vs_vol)

                print(f"\n  Filter: {filter_name}")
                print(f"  Tradable events after filter: {len(filtered_test_df)}")

                if filtered_test_df.empty:
                    print("  No events remain after filter. Skipping.")
                    continue

                tickers = sorted(filtered_test_df["ticker"].unique().tolist())
                close_matrix = download_close_matrix(
                    tickers=tickers,
                    start_date=filtered_test_df["event_date"].min(),
                    end_date=filtered_test_df["exit_date"].max(),
                )

                trade_log_df, daily_equity_df, split_summary_df = simulate_portfolio_for_test_split(
                    split_name=split_name,
                    filter_name=filter_name,
                    tradable_test_df=filtered_test_df,
                    close_matrix=close_matrix,
                )

                print(split_summary_df.to_string(index=False))

                all_trade_logs.append(trade_log_df)
                all_equity_curves.append(daily_equity_df)
                all_split_summaries.append(split_summary_df)

    trades_df = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()
    equity_df = pd.concat(all_equity_curves, ignore_index=True) if all_equity_curves else pd.DataFrame()
    summary_df = pd.concat(all_split_summaries, ignore_index=True) if all_split_summaries else pd.DataFrame()

    return trades_df, equity_df, summary_df


def save_outputs(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades_df.to_csv(SWEEP_ALL_TRADES_CSV, index=False)
    equity_df.to_csv(SWEEP_ALL_EQUITY_CSV, index=False)
    summary_df.to_csv(SWEEP_SPLIT_SUMMARY_CSV, index=False)

    print(f"\nSaved split summary to: {SWEEP_SPLIT_SUMMARY_CSV}")
    print(f"Saved all trades to:    {SWEEP_ALL_TRADES_CSV}")
    print(f"Saved all equity to:    {SWEEP_ALL_EQUITY_CSV}")


def print_overall_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("\nNo split summaries to report.")
        return

    display = summary_df.copy()

    pct_cols = ["total_return", "max_drawdown", "win_rate_closed_trades", "avg_realized_return_closed_trades"]
    for col in pct_cols:
        if col in display.columns:
            display[col] = (display[col] * 100).round(2)

    if "avg_realized_pnl_closed_trades" in display.columns:
        display["avg_realized_pnl_closed_trades"] = display["avg_realized_pnl_closed_trades"].round(2)
    if "avg_holding_days_closed_trades" in display.columns:
        display["avg_holding_days_closed_trades"] = display["avg_holding_days_closed_trades"].round(2)
    if "avg_open_positions" in display.columns:
        display["avg_open_positions"] = display["avg_open_positions"].round(2)

    print("\n=== PARAMETER SWEEP SUMMARY ===")
    print(display.to_string(index=False))


def main() -> None:
    events_df = load_events()
    trades_df, equity_df, summary_df = run_backtest(events_df)
    print_overall_summary(summary_df)
    save_outputs(trades_df, equity_df, summary_df)


if __name__ == "__main__":
    main()