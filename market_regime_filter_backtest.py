#!/usr/bin/env python3
"""
market_regime_filter_backtest.py

Quest 1 of 5:
Test whether adding a broad market regime filter improves the strategy.

This script:
1. Reads outputs/events.csv
2. Learns GOOD combos from train periods
3. Labels test events using those learned combos
4. Adds SPY-based market regime features to test events
5. Runs portfolio simulations under multiple market filters
6. Compares which market filters improve out-of-sample results

Inputs:
- outputs/events.csv

Outputs:
- outputs/market_regime_split_summary.csv
- outputs/market_regime_all_trades.csv
- outputs/market_regime_all_equity.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd
import yfinance as yf


# ============================================================
# 1) CONFIG
# ============================================================

OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")

MARKET_REGIME_SPLIT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "market_regime_split_summary.csv")
MARKET_REGIME_ALL_TRADES_CSV = os.path.join(OUTPUT_DIR, "market_regime_all_trades.csv")
MARKET_REGIME_ALL_EQUITY_CSV = os.path.join(OUTPUT_DIR, "market_regime_all_equity.csv")

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

# Clean useful splits
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

# Market filters to test
MARKET_FILTERS = {
    "no_filter": lambda row: True,
    "spy_above_50dma": lambda row: bool(row["spy_close_above_50dma"]),
    "spy_above_200dma": lambda row: bool(row["spy_close_above_200dma"]),
    "spy_50dma_above_200dma": lambda row: bool(row["spy_50dma_above_200dma"]),
    "spy_not_red_big_today": lambda row: bool(row["spy_daily_return"] > -0.01),
    "spy_20d_return_not_too_bad": lambda row: bool(row["spy_20d_return"] > -0.05),
    "spy_above_50dma_and_not_red_big_today": (
        lambda row: bool(row["spy_close_above_50dma"]) and bool(row["spy_daily_return"] > -0.01)
    ),
    "spy_above_200dma_and_50_over_200": (
        lambda row: bool(row["spy_close_above_200dma"]) and bool(row["spy_50dma_above_200dma"])
    ),
}


# ============================================================
# 2) DATA STRUCTURES
# ============================================================

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
    market_filter_name: str


# ============================================================
# 3) LOAD DATA
# ============================================================

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


# ============================================================
# 4) LEARN COMBOS FROM TRAIN DATA
# ============================================================

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
) -> tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], pd.DataFrame]:
    if combo_summary_df.empty:
        return set(), set(), combo_summary_df.copy()

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

    labeled = df.copy()
    labeled["combo_label"] = "neutral"
    labeled.loc[good_mask, "combo_label"] = "good"
    labeled.loc[bad_mask, "combo_label"] = "bad"

    overlap = good_mask & bad_mask
    labeled.loc[overlap, "combo_label"] = "bad"

    return good_combos, bad_combos, labeled


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


# ============================================================
# 5) MARKET FEATURES (SPY)
# ============================================================

def download_spy_features(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start_str = (start_date - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    end_str = (end_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"Downloading SPY data from {start_str} to {end_str}...")

    spy = yf.download(
        tickers="SPY",
        start=start_str,
        end=end_str,
        auto_adjust=DOWNLOAD_AUTO_ADJUST,
        progress=True,
        threads=True,
    )

    if spy.empty:
        raise RuntimeError("No SPY data returned from yfinance.")

    # Handle weird yfinance column shapes robustly
    if isinstance(spy.columns, pd.MultiIndex):
        if "SPY" in spy.columns.get_level_values(0):
            spy = spy["SPY"].copy()
        elif "SPY" in spy.columns.get_level_values(-1):
            spy = spy.xs("SPY", axis=1, level=-1).copy()
        else:
            # fallback: flatten if possible
            spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]

    # If duplicate columns somehow exist, keep first occurrence
    if spy.columns.duplicated().any():
        spy = spy.loc[:, ~spy.columns.duplicated()]

    spy = spy.copy()
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()

    if "Close" not in spy.columns:
        raise RuntimeError(f"SPY download is missing Close column. Columns seen: {list(spy.columns)}")

    # Force Close to be a Series
    close = spy["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)

    out = pd.DataFrame(index=spy.index)
    out["spy_close"] = close
    out["spy_daily_return"] = out["spy_close"].pct_change()
    out["spy_20d_return"] = out["spy_close"].pct_change(20)
    out["spy_50dma"] = out["spy_close"].rolling(50).mean()
    out["spy_200dma"] = out["spy_close"].rolling(200).mean()

    out["spy_close_above_50dma"] = out["spy_close"] > out["spy_50dma"]
    out["spy_close_above_200dma"] = out["spy_close"] > out["spy_200dma"]
    out["spy_50dma_above_200dma"] = out["spy_50dma"] > out["spy_200dma"]

    return out

def add_spy_features_to_events(events_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    out = events_df.copy()

    spy_features = spy_df.reset_index().rename(columns={"Date": "event_date"})
    if "event_date" not in spy_features.columns:
        spy_features = spy_features.rename(columns={spy_features.columns[0]: "event_date"})

    out = out.merge(spy_features, on="event_date", how="left")
    return out


# ============================================================
# 6) PRICE DOWNLOAD FOR PORTFOLIO MTM
# ============================================================

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


# ============================================================
# 7) PORTFOLIO SIMULATION
# ============================================================

def filter_events_by_market_regime(
    events_df: pd.DataFrame,
    filter_name: str,
) -> pd.DataFrame:
    fn = MARKET_FILTERS[filter_name]
    mask = events_df.apply(fn, axis=1)
    return events_df[mask].copy()


def simulate_portfolio_for_test_split(
    split_name: str,
    market_filter_name: str,
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
        exiting_ids = [
            pid for pid, pos in open_positions.items()
            if pos.exit_date == today
        ]

        for pid in exiting_ids:
            pos = open_positions.pop(pid)

            exit_value = pos.shares * pos.exit_price
            realized_pnl = exit_value - pos.allocated_capital
            cumulative_realized_pnl += realized_pnl
            cash += exit_value

            trade_log_rows.append(
                {
                    "split_name": split_name,
                    "market_filter_name": market_filter_name,
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
                    market_filter_name=market_filter_name,
                )

                open_positions[pos.event_id] = pos
                cash -= POSITION_SIZE

                trade_log_rows.append(
                    {
                        "split_name": split_name,
                        "market_filter_name": market_filter_name,
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
                "market_filter_name": market_filter_name,
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
    summary_df = build_split_summary(split_name, market_filter_name, trade_log_df, daily_equity_df)

    return trade_log_df, daily_equity_df, summary_df


# ============================================================
# 8) SUMMARY
# ============================================================

def build_split_summary(
    split_name: str,
    market_filter_name: str,
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
                "market_filter_name": market_filter_name,
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


# ============================================================
# 9) DRIVER
# ============================================================

def run_market_regime_backtest(events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trade_logs = []
    all_equity_curves = []
    all_split_summaries = []

    spy_df = download_spy_features(
        start_date=events_df["event_date"].min(),
        end_date=events_df["exit_date"].max(),
    )

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
        good_combos, bad_combos, _ = classify_combos_from_train(train_combo_summary)
        labeled_test_df = label_test_events(test_df, train_combo_summary, good_combos, bad_combos)
        labeled_test_df = add_spy_features_to_events(labeled_test_df, spy_df)

        base_good_test_df = labeled_test_df[labeled_test_df["combo_label"] == "good"].copy()

        print(f"Train events: {len(train_df)}")
        print(f"Test events:  {len(test_df)}")
        print(f"Learned good combos: {len(good_combos)}")
        print(f"Learned bad combos:  {len(bad_combos)}")
        print(f"Tradable good-combo test events before market filter: {len(base_good_test_df)}")

        for filter_name in MARKET_FILTERS.keys():
            filtered_test_df = filter_events_by_market_regime(base_good_test_df, filter_name)

            print(f"\n  Market filter: {filter_name}")
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
                market_filter_name=filter_name,
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


# ============================================================
# 10) SAVE / PRINT
# ============================================================

def save_outputs(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades_df.to_csv(MARKET_REGIME_ALL_TRADES_CSV, index=False)
    equity_df.to_csv(MARKET_REGIME_ALL_EQUITY_CSV, index=False)
    summary_df.to_csv(MARKET_REGIME_SPLIT_SUMMARY_CSV, index=False)

    print(f"\nSaved split summary to: {MARKET_REGIME_SPLIT_SUMMARY_CSV}")
    print(f"Saved all trades to:    {MARKET_REGIME_ALL_TRADES_CSV}")
    print(f"Saved all equity to:    {MARKET_REGIME_ALL_EQUITY_CSV}")


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

    print("\n=== MARKET REGIME FILTER SUMMARY ===")
    print(display.to_string(index=False))


# ============================================================
# 11) MAIN
# ============================================================

def main() -> None:
    events_df = load_events()
    trades_df, equity_df, summary_df = run_market_regime_backtest(events_df)
    print_overall_summary(summary_df)
    save_outputs(trades_df, equity_df, summary_df)


if __name__ == "__main__":
    main()