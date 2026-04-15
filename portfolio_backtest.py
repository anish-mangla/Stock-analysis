#!/usr/bin/env python3
"""
portfolio_backtest.py

Purpose:
Simulate a capital-constrained portfolio using the event-level outputs from:
- bounce_backtest.py
- analyze_combo_filters.py

This script is the first "real strategy" layer.

It:
- reads outputs/events_with_combo_labels.csv
- trades only events labeled as GOOD combos
- allocates fixed capital per trade
- respects a max number of simultaneous positions
- exits on the precomputed exit_date at exit_price
- marks open positions to market daily using downloaded close prices

Outputs:
- outputs/portfolio_trade_log.csv
- outputs/portfolio_daily_equity.csv
- outputs/portfolio_summary.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# ============================================================
# 1) CONFIG
# ============================================================

OUTPUT_DIR = "outputs"

EVENTS_WITH_LABELS_CSV = os.path.join(OUTPUT_DIR, "events_with_combo_labels.csv")
LEARNED_COMBO_LABELS_CSV = os.path.join(OUTPUT_DIR, "learned_combo_labels.csv")

TRADE_LOG_CSV = os.path.join(OUTPUT_DIR, "portfolio_trade_log.csv")
DAILY_EQUITY_CSV = os.path.join(OUTPUT_DIR, "portfolio_daily_equity.csv")
PORTFOLIO_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "portfolio_summary.csv")

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 10_000.0
MAX_POSITIONS = 10

TRADE_ONLY_GOOD_COMBOS = True
ALLOW_DUPLICATE_TICKERS = False

# If more candidates appear on the same day than you can take,
# rank them by learned combo strength first.
SORT_SAME_DAY_CANDIDATES_BY = [
    "combo_score",
    "success_rate",
    "avg_final_return",
    "total_events",
]

DOWNLOAD_AUTO_ADJUST = False
YF_GROUP_BY = "ticker"


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
    combo_label: str
    drop_bucket: str
    combo_score: float


# ============================================================
# 3) LOAD INPUTS
# ============================================================

def load_inputs() -> pd.DataFrame:
    if not os.path.exists(EVENTS_WITH_LABELS_CSV):
        raise FileNotFoundError(f"Missing required file: {EVENTS_WITH_LABELS_CSV}")

    if not os.path.exists(LEARNED_COMBO_LABELS_CSV):
        raise FileNotFoundError(f"Missing required file: {LEARNED_COMBO_LABELS_CSV}")

    events_df = pd.read_csv(EVENTS_WITH_LABELS_CSV)
    combos_df = pd.read_csv(LEARNED_COMBO_LABELS_CSV)

    if events_df.empty:
        raise ValueError("events_with_combo_labels.csv is empty.")

    if combos_df.empty:
        raise ValueError("learned_combo_labels.csv is empty.")

    # Parse dates
    events_df["event_date"] = pd.to_datetime(events_df["event_date"])
    events_df["exit_date"] = pd.to_datetime(events_df["exit_date"], errors="coerce")

    # Normalize types
    events_df["drop_bucket"] = events_df["drop_bucket"].astype(str)
    events_df["combo_label"] = events_df["combo_label"].astype(str)

    combos_df["drop_bucket"] = combos_df["drop_bucket"].astype(str)
    combos_df["combo_label"] = combos_df["combo_label"].astype(str)

    # Merge combo stats onto event rows for ranking
    merge_cols = [
        "ticker",
        "drop_bucket",
        "combo_label",
        "total_events",
        "success_rate",
        "avg_final_return",
        "avg_max_drawdown",
        "combo_score",
    ]

    combos_subset = combos_df[merge_cols].copy()

    merged = events_df.merge(
        combos_subset,
        on=["ticker", "drop_bucket"],
        how="left",
        suffixes=("", "_combo"),
    )

    # event_id for stable position tracking
    merged = merged.reset_index(drop=True)
    merged["event_id"] = merged.index.astype(int)

    # Basic cleanup
    merged = merged.dropna(subset=["entry_price", "exit_price", "event_date", "exit_date"])

    if TRADE_ONLY_GOOD_COMBOS:
        merged = merged[merged["combo_label"] == "good"].copy()

    merged = merged.sort_values(["event_date", "ticker"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError("No tradable events remain after filtering.")

    return merged


# ============================================================
# 4) PRICE DOWNLOAD FOR DAILY MARK-TO-MARKET
# ============================================================

def download_close_matrix(
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Download close prices and build a forward-filled daily close matrix.
    """
    if not tickers:
        raise ValueError("No tickers provided for price download.")

    # small buffer around dates
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
        raise RuntimeError("No price data returned from yfinance for mark-to-market.")

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
        # fallback for single ticker
        if len(tickers) != 1:
            raise RuntimeError("Unexpected flat-column format for multiple tickers.")
        if "Close" not in raw.columns:
            raise RuntimeError("Close column missing from downloaded data.")
        s = raw["Close"].copy().dropna()
        s.index = pd.to_datetime(s.index)
        close_map[tickers[0]] = s.sort_index()

    if not close_map:
        raise RuntimeError("No usable close price series were built.")

    all_dates = sorted(set().union(*[set(s.index) for s in close_map.values()]))
    close_df = pd.DataFrame(index=pd.DatetimeIndex(all_dates))

    for ticker, s in close_map.items():
        close_df[ticker] = s

    close_df = close_df.sort_index().ffill()

    return close_df


# ============================================================
# 5) PORTFOLIO BACKTEST
# ============================================================

def simulate_portfolio(
    tradable_events_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main portfolio simulator.

    Conventions:
    - Entries occur at event-day close
    - Exits occur at exit-day close
    - Capital from same-day exits is allowed to be reused for same-day entries
    """
    all_dates = sorted(close_matrix.index.unique())

    # Group events by event_date for fast access
    events_by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        dt: g.copy()
        for dt, g in tradable_events_df.groupby("event_date", sort=True)
    }

    cash = INITIAL_CAPITAL
    open_positions: Dict[int, Position] = {}

    trade_log_rows: List[dict] = []
    daily_rows: List[dict] = []

    cumulative_realized_pnl = 0.0
    max_equity_seen = INITIAL_CAPITAL

    for today in all_dates:
        # ----------------------------------------------------
        # 1) PROCESS EXITS FIRST
        # ----------------------------------------------------
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
                    "combo_label": pos.combo_label,
                    "drop_bucket": pos.drop_bucket,
                    "combo_score": pos.combo_score,
                }
            )

        # ----------------------------------------------------
        # 2) PROCESS ENTRIES
        # ----------------------------------------------------
        todays_candidates = events_by_date.get(today)

        if todays_candidates is not None and not todays_candidates.empty:
            candidates = todays_candidates.copy()

            # Rank candidates using learned combo strength
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
                    combo_label=str(row["combo_label"]),
                    drop_bucket=str(row["drop_bucket"]),
                    combo_score=float(row["combo_score"]) if pd.notna(row["combo_score"]) else 0.0,
                )

                open_positions[pos.event_id] = pos
                cash -= POSITION_SIZE

                trade_log_rows.append(
                    {
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
                        "combo_label": pos.combo_label,
                        "drop_bucket": pos.drop_bucket,
                        "combo_score": pos.combo_score,
                    }
                )

        # ----------------------------------------------------
        # 3) MARK TO MARKET DAILY EQUITY
        # ----------------------------------------------------
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

    summary_df = build_portfolio_summary(trade_log_df, daily_equity_df, open_positions)

    return trade_log_df, daily_equity_df, summary_df


# ============================================================
# 6) SUMMARY METRICS
# ============================================================

def build_portfolio_summary(
    trade_log_df: pd.DataFrame,
    daily_equity_df: pd.DataFrame,
    open_positions: Dict[int, Position],
) -> pd.DataFrame:
    sells = trade_log_df[trade_log_df["action"] == "SELL"].copy()
    buys = trade_log_df[trade_log_df["action"] == "BUY"].copy()

    ending_equity = float(daily_equity_df["equity"].iloc[-1]) if not daily_equity_df.empty else INITIAL_CAPITAL
    ending_cash = float(daily_equity_df["cash"].iloc[-1]) if not daily_equity_df.empty else INITIAL_CAPITAL
    max_drawdown = float(daily_equity_df["drawdown"].min()) if not daily_equity_df.empty else 0.0
    avg_open_positions = float(daily_equity_df["open_positions"].mean()) if not daily_equity_df.empty else 0.0

    total_closed_trades = int(len(sells))
    win_rate = float((sells["realized_pnl"] > 0).mean()) if total_closed_trades > 0 else None
    avg_realized_return = float(sells["realized_return"].mean()) if total_closed_trades > 0 else None
    median_realized_return = float(sells["realized_return"].median()) if total_closed_trades > 0 else None
    avg_realized_pnl = float(sells["realized_pnl"].mean()) if total_closed_trades > 0 else None

    if total_closed_trades > 0:
        holding_days = (pd.to_datetime(sells["exit_date"]) - pd.to_datetime(sells["entry_date"])).dt.days
        avg_holding_days = float(holding_days.mean())
        median_holding_days = float(holding_days.median())
    else:
        avg_holding_days = None
        median_holding_days = None

    summary = pd.DataFrame(
        [
            {
                "initial_capital": INITIAL_CAPITAL,
                "ending_equity": ending_equity,
                "ending_cash": ending_cash,
                "total_return": (ending_equity / INITIAL_CAPITAL) - 1.0,
                "max_drawdown": max_drawdown,
                "total_buy_signals_taken": int(len(buys)),
                "total_closed_trades": total_closed_trades,
                "win_rate_closed_trades": win_rate,
                "avg_realized_return_closed_trades": avg_realized_return,
                "median_realized_return_closed_trades": median_realized_return,
                "avg_realized_pnl_closed_trades": avg_realized_pnl,
                "avg_holding_days_closed_trades": avg_holding_days,
                "median_holding_days_closed_trades": median_holding_days,
                "avg_open_positions": avg_open_positions,
                "ending_open_positions": int(len(open_positions)),
                "position_size": POSITION_SIZE,
                "max_positions": MAX_POSITIONS,
                "trade_only_good_combos": TRADE_ONLY_GOOD_COMBOS,
                "allow_duplicate_tickers": ALLOW_DUPLICATE_TICKERS,
            }
        ]
    )

    return summary


# ============================================================
# 7) SAVE / PRINT
# ============================================================

def save_outputs(
    trade_log_df: pd.DataFrame,
    daily_equity_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trade_log_df.to_csv(TRADE_LOG_CSV, index=False)
    daily_equity_df.to_csv(DAILY_EQUITY_CSV, index=False)
    summary_df.to_csv(PORTFOLIO_SUMMARY_CSV, index=False)

    print(f"\nSaved trade log to:      {TRADE_LOG_CSV}")
    print(f"Saved daily equity to:   {DAILY_EQUITY_CSV}")
    print(f"Saved summary to:        {PORTFOLIO_SUMMARY_CSV}")


def print_summary(summary_df: pd.DataFrame) -> None:
    row = summary_df.iloc[0].to_dict()

    print("\n=== PORTFOLIO SUMMARY ===")
    print(f"Initial capital:                 ${row['initial_capital']:,.2f}")
    print(f"Ending equity:                  ${row['ending_equity']:,.2f}")
    print(f"Ending cash:                    ${row['ending_cash']:,.2f}")
    print(f"Total return:                   {row['total_return'] * 100:.2f}%")
    print(f"Max drawdown:                   {row['max_drawdown'] * 100:.2f}%")
    print(f"Total buy signals taken:        {int(row['total_buy_signals_taken'])}")
    print(f"Total closed trades:            {int(row['total_closed_trades'])}")

    if pd.notna(row["win_rate_closed_trades"]):
        print(f"Win rate (closed trades):       {row['win_rate_closed_trades'] * 100:.2f}%")
    if pd.notna(row["avg_realized_return_closed_trades"]):
        print(f"Avg realized return / trade:    {row['avg_realized_return_closed_trades'] * 100:.2f}%")
    if pd.notna(row["median_realized_return_closed_trades"]):
        print(f"Median realized return / trade: {row['median_realized_return_closed_trades'] * 100:.2f}%")
    if pd.notna(row["avg_realized_pnl_closed_trades"]):
        print(f"Avg realized PnL / trade:       ${row['avg_realized_pnl_closed_trades']:,.2f}")
    if pd.notna(row["avg_holding_days_closed_trades"]):
        print(f"Avg holding days:               {row['avg_holding_days_closed_trades']:.2f}")
    if pd.notna(row["median_holding_days_closed_trades"]):
        print(f"Median holding days:            {row['median_holding_days_closed_trades']:.2f}")

    print(f"Average open positions:         {row['avg_open_positions']:.2f}")
    print(f"Ending open positions:          {int(row['ending_open_positions'])}")


# ============================================================
# 8) MAIN
# ============================================================

def main() -> None:
    tradable_events_df = load_inputs()

    tickers = sorted(tradable_events_df["ticker"].unique().tolist())
    start_date = tradable_events_df["event_date"].min()
    end_date = tradable_events_df["exit_date"].max()

    close_matrix = download_close_matrix(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    trade_log_df, daily_equity_df, summary_df = simulate_portfolio(
        tradable_events_df=tradable_events_df,
        close_matrix=close_matrix,
    )

    print_summary(summary_df)
    save_outputs(trade_log_df, daily_equity_df, summary_df)


if __name__ == "__main__":
    main()