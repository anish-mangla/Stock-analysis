#!/usr/bin/env python3
"""
portfolio_simulation.py

Full portfolio simulation that tracks actual P&L across different
target/hold period strategies.

Simulates a $100K portfolio with:
- Position size: $10K per trade (1/10th of portfolio)
- Max 10 simultaneous positions
- No duplicate tickers
- Entry at event-day close
- Exit: target hit OR max hold reached OR hold indefinitely

Tests 6 strategy variants:
1. +1% target, 14-day max hold
2. +1% target, 60-day max hold (patient)
3. +2% target, 14-day max hold
4. +2% target, 60-day max hold (patient)
5. +3% target, 14-day max hold
6. +3% target, 60-day max hold (patient)

Reports actual annual returns including all losses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


OUTPUT_DIR = "outputs"
EVENTS_CSV = os.path.join(OUTPUT_DIR, "events.csv")

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 10_000.0
MAX_POSITIONS = 10

STRATEGIES = [
    {"name": "+1%_14d", "target": 0.01, "max_hold": 14},
    {"name": "+1%_60d", "target": 0.01, "max_hold": 60},
    {"name": "+2%_14d", "target": 0.02, "max_hold": 14},
    {"name": "+2%_60d", "target": 0.02, "max_hold": 60},
    {"name": "+3%_14d", "target": 0.03, "max_hold": 14},
    {"name": "+3%_60d", "target": 0.03, "max_hold": 60},
]


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    target_price: float
    max_exit_date: pd.Timestamp
    shares: float
    allocated: float


def load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_CSV)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.dropna(subset=["entry_price"])
    df = df.sort_values("event_date").reset_index(drop=True)
    return df


def download_all_prices(events: pd.DataFrame) -> pd.DataFrame:
    """Download a close price matrix for all tickers across the full date range."""
    tickers = sorted(events["ticker"].unique().tolist())
    start = (events["event_date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (events["event_date"].max() + pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"Downloading prices for {len(tickers)} tickers from {start} to {end}...")

    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=False, progress=True, group_by="ticker", threads=True,
    )

    close_map: Dict[str, pd.Series] = {}
    high_map: Dict[str, pd.Series] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            tdf = raw[ticker].copy()
            if "Close" in tdf.columns:
                s = tdf["Close"].dropna().astype(float)
                s.index = pd.to_datetime(s.index)
                close_map[ticker] = s
            if "High" in tdf.columns:
                s = tdf["High"].dropna().astype(float)
                s.index = pd.to_datetime(s.index)
                high_map[ticker] = s

    all_dates = sorted(set().union(*[set(s.index) for s in close_map.values()]))
    close_df = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    high_df = pd.DataFrame(index=pd.DatetimeIndex(all_dates))

    for ticker, s in close_map.items():
        close_df[ticker] = s
    for ticker, s in high_map.items():
        high_df[ticker] = s

    close_df = close_df.sort_index().ffill()
    high_df = high_df.sort_index().ffill()

    return close_df, high_df


def simulate_strategy(
    strategy: Dict,
    events: pd.DataFrame,
    close_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a single strategy variant.

    Returns (trade_log, daily_equity).
    """
    target_return = strategy["target"]
    max_hold_days = strategy["max_hold"]
    strategy_name = strategy["name"]

    all_dates = sorted(close_matrix.index)
    trading_dates = pd.DatetimeIndex(all_dates)

    # Pre-compute: for each event, what's the max exit date?
    events_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    for dt, g in events.groupby("event_date"):
        events_by_date[dt] = g.copy()

    cash = INITIAL_CAPITAL
    open_positions: Dict[str, Position] = {}  # keyed by ticker
    trade_log: List[Dict] = []
    daily_rows: List[Dict] = []

    for today in trading_dates:
        # Check exits: target hit or max hold reached
        to_close = []
        for ticker, pos in open_positions.items():
            if ticker not in high_matrix.columns or ticker not in close_matrix.columns:
                continue

            today_high = high_matrix.at[today, ticker]
            today_close = close_matrix.at[today, ticker]

            if pd.isna(today_high) or pd.isna(today_close):
                continue

            # Check if target hit (using intraday high)
            if today_high >= pos.target_price:
                exit_price = pos.target_price  # assume we exit at target
                pnl = (exit_price - pos.entry_price) * pos.shares
                trade_log.append({
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "entry_date": pos.entry_date,
                    "exit_date": today,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "pnl": pnl,
                    "return_pct": (exit_price / pos.entry_price) - 1.0,
                    "hold_days": (today - pos.entry_date).days,
                    "exit_reason": "target_hit",
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)

            # Check if max hold reached
            elif today >= pos.max_exit_date:
                exit_price = today_close
                pnl = (exit_price - pos.entry_price) * pos.shares
                trade_log.append({
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "entry_date": pos.entry_date,
                    "exit_date": today,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "pnl": pnl,
                    "return_pct": (exit_price / pos.entry_price) - 1.0,
                    "hold_days": (today - pos.entry_date).days,
                    "exit_reason": "max_hold",
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)

        for ticker in to_close:
            del open_positions[ticker]

        # Check entries
        candidates = events_by_date.get(today)
        if candidates is not None:
            # Sort by drop magnitude (worst first)
            candidates = candidates.sort_values("drop_pct", ascending=False)

            for _, row in candidates.iterrows():
                if len(open_positions) >= MAX_POSITIONS:
                    break
                if cash < POSITION_SIZE:
                    break

                ticker = row["ticker"]
                if ticker in open_positions:
                    continue

                entry_price = float(row["entry_price"])
                if entry_price <= 0:
                    continue

                # Compute max exit date
                future_dates = trading_dates[trading_dates > today]
                if len(future_dates) < max_hold_days:
                    continue
                max_exit_date = future_dates[max_hold_days - 1]

                shares = POSITION_SIZE / entry_price
                target_price = entry_price * (1.0 + target_return)

                pos = Position(
                    ticker=ticker,
                    entry_date=today,
                    entry_price=entry_price,
                    target_price=target_price,
                    max_exit_date=max_exit_date,
                    shares=shares,
                    allocated=POSITION_SIZE,
                )
                open_positions[ticker] = pos
                cash -= POSITION_SIZE

        # Daily MTM
        market_value = 0.0
        for ticker, pos in open_positions.items():
            if ticker in close_matrix.columns:
                price = close_matrix.at[today, ticker]
                if pd.notna(price):
                    market_value += pos.shares * float(price)

        equity = cash + market_value

        daily_rows.append({
            "strategy": strategy_name,
            "date": today,
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "open_positions": len(open_positions),
        })

    return pd.DataFrame(trade_log), pd.DataFrame(daily_rows)


def compute_annual_returns(daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Compute annual returns from daily equity curve."""
    df = daily_equity.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    rows = []
    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year]
        start_equity = year_data["equity"].iloc[0]
        end_equity = year_data["equity"].iloc[-1]
        annual_return = (end_equity / start_equity) - 1.0
        max_equity = year_data["equity"].cummax()
        drawdown = (year_data["equity"] / max_equity) - 1.0
        max_drawdown = drawdown.min()

        rows.append({
            "year": year,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
        })

    return pd.DataFrame(rows)


def compute_trade_stats(trade_log: pd.DataFrame) -> Dict:
    """Compute trade-level statistics."""
    if trade_log.empty:
        return {}

    winners = trade_log[trade_log["pnl"] > 0]
    losers = trade_log[trade_log["pnl"] <= 0]

    return {
        "total_trades": len(trade_log),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": len(winners) / len(trade_log),
        "avg_win": winners["pnl"].mean() if len(winners) > 0 else 0,
        "avg_loss": losers["pnl"].mean() if len(losers) > 0 else 0,
        "total_pnl": trade_log["pnl"].sum(),
        "avg_hold_days": trade_log["hold_days"].mean(),
        "avg_return_pct": trade_log["return_pct"].mean(),
        "target_hits": (trade_log["exit_reason"] == "target_hit").sum(),
        "max_hold_exits": (trade_log["exit_reason"] == "max_hold").sum(),
    }


def main():
    events = load_events()
    print(f"Loaded {len(events)} events")

    close_matrix, high_matrix = download_all_prices(events)
    print(f"Price matrix: {close_matrix.shape[0]} days × {close_matrix.shape[1]} tickers")

    all_trade_logs = []
    all_daily_equity = []

    for strategy in STRATEGIES:
        print(f"\nSimulating strategy: {strategy['name']}...")
        trade_log, daily_equity = simulate_strategy(strategy, events, close_matrix, high_matrix)
        all_trade_logs.append(trade_log)
        all_daily_equity.append(daily_equity)

        stats = compute_trade_stats(trade_log)
        annual = compute_annual_returns(daily_equity)

        print(f"  Trades: {stats.get('total_trades', 0)}")
        print(f"  Win rate: {stats.get('win_rate', 0)*100:.1f}%")
        print(f"  Total PnL: ${stats.get('total_pnl', 0):,.0f}")
        print(f"  Avg hold: {stats.get('avg_hold_days', 0):.1f} days")
        print(f"  Target hits: {stats.get('target_hits', 0)} | Max-hold exits: {stats.get('max_hold_exits', 0)}")

    # Combine results
    combined_trades = pd.concat(all_trade_logs, ignore_index=True)
    combined_equity = pd.concat(all_daily_equity, ignore_index=True)

    # Print comparison
    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON — ANNUAL RETURNS")
    print("=" * 90)

    for strategy in STRATEGIES:
        name = strategy["name"]
        eq = combined_equity[combined_equity["strategy"] == name]
        annual = compute_annual_returns(eq)
        trades = combined_trades[combined_trades["strategy"] == name]
        stats = compute_trade_stats(trades)

        print(f"\n  Strategy: {name}")
        print(f"  {'Year':>6} {'Start$':>12} {'End$':>12} {'Return':>8} {'MaxDD':>8}")
        print(f"  {'-'*50}")
        for _, row in annual.iterrows():
            print(f"  {int(row['year']):>6} ${row['start_equity']:>11,.0f} ${row['end_equity']:>11,.0f} "
                  f"{row['annual_return']*100:>+7.2f}% {row['max_drawdown']*100:>7.2f}%")

        total_return = (eq["equity"].iloc[-1] / INITIAL_CAPITAL) - 1.0
        print(f"  {'TOTAL':>6} ${INITIAL_CAPITAL:>11,.0f} ${eq['equity'].iloc[-1]:>11,.0f} "
              f"{total_return*100:>+7.2f}%")
        print(f"  Trades: {stats['total_trades']} | Win rate: {stats['win_rate']*100:.1f}% | "
              f"Avg hold: {stats['avg_hold_days']:.1f}d | Avg return/trade: {stats['avg_return_pct']*100:.2f}%")

    # Save
    combined_trades.to_csv(os.path.join(OUTPUT_DIR, "portfolio_sim_trades.csv"), index=False)
    combined_equity.to_csv(os.path.join(OUTPUT_DIR, "portfolio_sim_equity.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
