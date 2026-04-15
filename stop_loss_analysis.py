#!/usr/bin/env python3
"""
stop_loss_analysis.py

Tests different stop-loss and time-based exit rules to optimize
capital recycling.

The hypothesis: cutting losers early frees up capital for new
high-confidence trades, improving overall returns even though
each individual cut is a realized loss.

Tests:
- Stop losses: -3%, -5%, -7%, -10%, none
- Time stops: cut if not profitable after 10d, 20d, 30d, none
- Combined: stop loss + time stop

Uses the +3% target, 60-day max hold, 1/10 position size as the base.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from trade_scorer import TradeScorer


OUTPUT_DIR = "outputs"
EVENTS_LABELED_CSV = os.path.join(OUTPUT_DIR, "events_fully_labeled.csv")

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE_FRACTION = 10
MAX_POSITIONS = 10
TARGET_RETURN = 0.03
MAX_HOLD_DAYS = 60

# Stop loss levels to test
STOP_LOSSES = [None, -0.03, -0.05, -0.07, -0.10]

# Time stops: cut if position is still negative after N days
TIME_STOPS = [None, 10, 20, 30]


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    target_price: float
    max_exit_date: pd.Timestamp
    shares: float
    allocated: float


def load_data():
    events = pd.read_csv(EVENTS_LABELED_CSV)
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.dropna(subset=["entry_price"])
    events = events.sort_values("event_date").reset_index(drop=True)
    return events


def load_prices():
    """Load pre-downloaded price matrices."""
    # Check if we already have them from previous simulation
    equity_file = os.path.join(OUTPUT_DIR, "portfolio_sim_equity.csv")
    if not os.path.exists(equity_file):
        raise FileNotFoundError("Run portfolio_simulation.py first to download prices")

    # Re-download (we need close and high and low)
    import yfinance as yf
    events = load_data()
    tickers = sorted(events["ticker"].unique().tolist())
    start = (events["event_date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (events["event_date"].max() + pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"Downloading prices for {len(tickers)} tickers...")
    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=False, progress=True, group_by="ticker", threads=True,
    )

    close_data = {}
    high_data = {}
    low_data = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            tdf = raw[ticker].copy()
            for col, store in [("Close", close_data), ("High", high_data), ("Low", low_data)]:
                if col in tdf.columns:
                    s = tdf[col].dropna().astype(float)
                    s.index = pd.to_datetime(s.index)
                    store[ticker] = s

    all_dates = sorted(set().union(*[set(s.index) for s in close_data.values()]))
    idx = pd.DatetimeIndex(all_dates)

    close_df = pd.DataFrame({t: s for t, s in close_data.items()}, index=idx).sort_index().ffill()
    high_df = pd.DataFrame({t: s for t, s in high_data.items()}, index=idx).sort_index().ffill()
    low_df = pd.DataFrame({t: s for t, s in low_data.items()}, index=idx).sort_index().ffill()

    return close_df, high_df, low_df


def simulate(
    events: pd.DataFrame,
    scorer: TradeScorer,
    close_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame,
    low_matrix: pd.DataFrame,
    stop_loss: float | None,
    time_stop: int | None,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    position_size = INITIAL_CAPITAL / POSITION_SIZE_FRACTION
    trading_dates = pd.DatetimeIndex(sorted(close_matrix.index))

    events_by_date = {dt: g for dt, g in events.groupby("event_date")}

    cash = INITIAL_CAPITAL
    open_positions: Dict[str, Position] = {}
    trade_log: List[Dict] = []
    daily_rows: List[Dict] = []

    for today in trading_dates:
        to_close = []
        for ticker, pos in open_positions.items():
            if ticker not in close_matrix.columns:
                continue

            today_high = high_matrix.at[today, ticker] if ticker in high_matrix.columns else None
            today_low = low_matrix.at[today, ticker] if ticker in low_matrix.columns else None
            today_close = close_matrix.at[today, ticker]

            if pd.isna(today_close):
                continue

            days_held = len(trading_dates[(trading_dates > pos.entry_date) & (trading_dates <= today)])
            current_return = (float(today_close) / pos.entry_price) - 1.0

            exit_reason = None
            exit_price = None

            # Check target hit (using high)
            if today_high is not None and not pd.isna(today_high) and float(today_high) >= pos.target_price:
                exit_reason = "target_hit"
                exit_price = pos.target_price

            # Check stop loss (using low)
            elif stop_loss is not None and today_low is not None and not pd.isna(today_low):
                stop_price = pos.entry_price * (1.0 + stop_loss)
                if float(today_low) <= stop_price:
                    exit_reason = "stop_loss"
                    exit_price = stop_price

            # Check time stop: if position is still negative after N days, cut it
            elif time_stop is not None and days_held >= time_stop and current_return < 0:
                exit_reason = "time_stop"
                exit_price = float(today_close)

            # Check max hold
            elif today >= pos.max_exit_date:
                exit_reason = "max_hold"
                exit_price = float(today_close)

            if exit_reason:
                pnl = (exit_price - pos.entry_price) * pos.shares
                trade_log.append({
                    "label": label, "ticker": ticker,
                    "entry_date": pos.entry_date, "exit_date": today,
                    "entry_price": pos.entry_price, "exit_price": exit_price,
                    "pnl": pnl, "return_pct": (exit_price / pos.entry_price) - 1,
                    "hold_days": days_held, "exit_reason": exit_reason,
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)

        for t in to_close:
            del open_positions[t]

        # Entries — HIGH confidence only
        candidates = events_by_date.get(today)
        if candidates is not None:
            candidates = candidates.sort_values("drop_pct", ascending=False)

            for _, row in candidates.iterrows():
                if len(open_positions) >= MAX_POSITIONS:
                    break
                if cash < position_size:
                    break

                ticker = row["ticker"]
                if ticker in open_positions:
                    continue

                entry_price = float(row["entry_price"])
                if entry_price <= 0:
                    continue

                market_stressed = row.get("liquidity_credit_stress_severity", "none") in ["medium", "high"]
                event_type = row.get("stock_event_type", "unlabeled")
                event_severity = row.get("stock_event_severity", "medium")
                if event_type == "unlabeled":
                    event_type = "no_clear_catalyst"
                    event_severity = "medium"

                score = scorer.score_candidate(
                    drop_bucket=row["drop_bucket"],
                    stock_event_type=event_type,
                    stock_event_severity=event_severity,
                    market_stressed=market_stressed,
                )

                if score["confidence"] != "high":
                    continue

                future_dates = trading_dates[trading_dates > today]
                if len(future_dates) < MAX_HOLD_DAYS:
                    continue

                open_positions[ticker] = Position(
                    ticker=ticker, entry_date=today, entry_price=entry_price,
                    target_price=entry_price * (1.0 + TARGET_RETURN),
                    max_exit_date=future_dates[MAX_HOLD_DAYS - 1],
                    shares=position_size / entry_price, allocated=position_size,
                )
                cash -= position_size

        mv = sum(
            pos.shares * float(close_matrix.at[today, pos.ticker])
            for pos in open_positions.values()
            if pos.ticker in close_matrix.columns and pd.notna(close_matrix.at[today, pos.ticker])
        )
        daily_rows.append({
            "label": label, "date": today, "equity": cash + mv,
            "open_positions": len(open_positions), "cash": cash,
        })

    return pd.DataFrame(trade_log), pd.DataFrame(daily_rows)


def main():
    events = load_data()
    scorer = TradeScorer()
    close_matrix, high_matrix, low_matrix = load_prices()

    print(f"Events: {len(events)}, Price matrix: {close_matrix.shape}")

    all_results = []

    for stop_loss in STOP_LOSSES:
        for time_stop in TIME_STOPS:
            sl_label = f"SL={stop_loss*100:.0f}%" if stop_loss else "no_SL"
            ts_label = f"TS={time_stop}d" if time_stop else "no_TS"
            label = f"{sl_label}_{ts_label}"

            print(f"\nSimulating: {label}...")
            trades, equity, = simulate(
                events, scorer, close_matrix, high_matrix, low_matrix,
                stop_loss, time_stop, label,
            )

            if trades.empty:
                continue

            total_return = (equity["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
            n_trades = len(trades)
            win_rate = (trades["pnl"] > 0).mean() * 100
            avg_ret = trades["return_pct"].mean() * 100
            avg_hold = trades["hold_days"].mean()
            total_pnl = trades["pnl"].sum()

            by_reason = trades.groupby("exit_reason").agg(
                count=("pnl", "count"),
                avg_pnl=("pnl", "mean"),
            ).to_dict("index")

            target_hits = by_reason.get("target_hit", {}).get("count", 0)
            stop_hits = by_reason.get("stop_loss", {}).get("count", 0)
            time_hits = by_reason.get("time_stop", {}).get("count", 0)
            max_holds = by_reason.get("max_hold", {}).get("count", 0)

            # Annual breakdown
            eq = equity.copy()
            eq["year"] = pd.to_datetime(eq["date"]).dt.year
            annual = {}
            for year in sorted(eq["year"].unique()):
                ydata = eq[eq["year"] == year]
                annual[year] = (ydata["equity"].iloc[-1] / ydata["equity"].iloc[0] - 1) * 100

            print(f"  Return: {total_return:+.1f}% | Trades: {n_trades} | WR: {win_rate:.0f}% | "
                  f"Targets: {target_hits} | Stops: {stop_hits} | TimeStops: {time_hits} | MaxHold: {max_holds}")

            all_results.append({
                "config": label,
                "stop_loss": stop_loss,
                "time_stop": time_stop,
                "total_return": total_return,
                "trades": n_trades,
                "win_rate": win_rate,
                "avg_return": avg_ret,
                "avg_hold": avg_hold,
                "target_hits": target_hits,
                "stop_hits": stop_hits,
                "time_hits": time_hits,
                "max_holds": max_holds,
                **{f"y{y}": r for y, r in annual.items()},
            })

    # Summary
    print("\n" + "=" * 120)
    print("STOP LOSS / TIME STOP COMPARISON (+3% target, 60d max, 1/10 position, HIGH confidence)")
    print("=" * 120)

    df = pd.DataFrame(all_results).sort_values("total_return", ascending=False)
    print(f"{'Config':>22} {'Return':>8} {'Trades':>7} {'WR':>5} {'AvgHold':>8} {'Targets':>8} {'Stops':>6} {'TStops':>7} {'MaxH':>5} ", end="")
    years = [c for c in df.columns if c.startswith("y")]
    for y in sorted(years):
        print(f"| {y:>7}", end="")
    print()
    print("-" * 120)

    for _, r in df.iterrows():
        print(f"{r['config']:>22} {r['total_return']:>+7.1f}% {r['trades']:>7} {r['win_rate']:>4.0f}% "
              f"{r['avg_hold']:>7.1f}d {r['target_hits']:>8} {r['stop_hits']:>6} {r['time_hits']:>7} {r['max_holds']:>5}", end="")
        for y in sorted(years):
            print(f"| {r[y]:>+6.1f}%", end="")
        print()

    df.to_csv(os.path.join(OUTPUT_DIR, "stop_loss_analysis.csv"), index=False)
    print(f"\nSaved to {OUTPUT_DIR}/stop_loss_analysis.csv")


if __name__ == "__main__":
    main()
