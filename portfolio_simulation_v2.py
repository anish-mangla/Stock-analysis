#!/usr/bin/env python3
"""
portfolio_simulation_v2.py

Full portfolio simulation WITH the classification system.

Uses the trade scorer to decide which events to trade (HIGH confidence only)
and which to skip. Tests multiple configurations:

Targets: +1%, +2%, +3%
Hold periods: 14d, 60d
Position sizes: 1/10, 1/20, 1/50, 1/100 of portfolio
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

from trade_scorer import TradeScorer


OUTPUT_DIR = "outputs"
EVENTS_LABELED_CSV = os.path.join(OUTPUT_DIR, "events_fully_labeled.csv")

INITIAL_CAPITAL = 100_000.0

# Configurations to test
CONFIGS = [
    # Target, max_hold, position_fraction, max_positions, label
    (0.03, 60, 10, 10, "+3%_60d_1/10"),
    (0.03, 60, 20, 20, "+3%_60d_1/20"),
    (0.03, 60, 50, 50, "+3%_60d_1/50"),
    (0.03, 60, 100, 100, "+3%_60d_1/100"),
    (0.02, 60, 10, 10, "+2%_60d_1/10"),
    (0.02, 60, 20, 20, "+2%_60d_1/20"),
    (0.02, 60, 50, 50, "+2%_60d_1/50"),
    (0.01, 14, 10, 10, "+1%_14d_1/10"),
    (0.01, 60, 10, 10, "+1%_60d_1/10"),
    (0.03, 14, 10, 10, "+3%_14d_1/10"),
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
    confidence: str
    success_rate: float


def load_data():
    events = pd.read_csv(EVENTS_LABELED_CSV)
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.dropna(subset=["entry_price"])
    events = events.sort_values("event_date").reset_index(drop=True)
    return events


def download_prices(events: pd.DataFrame):
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

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            tdf = raw[ticker].copy()
            if "Close" in tdf.columns:
                close_data[ticker] = tdf["Close"].dropna().astype(float)
                close_data[ticker].index = pd.to_datetime(close_data[ticker].index)
            if "High" in tdf.columns:
                high_data[ticker] = tdf["High"].dropna().astype(float)
                high_data[ticker].index = pd.to_datetime(high_data[ticker].index)

    all_dates = sorted(set().union(*[set(s.index) for s in close_data.values()]))
    close_df = pd.DataFrame({t: s for t, s in close_data.items()}, index=pd.DatetimeIndex(all_dates)).sort_index().ffill()
    high_df = pd.DataFrame({t: s for t, s in high_data.items()}, index=pd.DatetimeIndex(all_dates)).sort_index().ffill()

    return close_df, high_df


def simulate(
    events: pd.DataFrame,
    scorer: TradeScorer,
    close_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame,
    target_return: float,
    max_hold_days: int,
    position_fraction: int,
    max_positions: int,
    label: str,
    min_confidence: str = "high",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate with the classification system filtering trades.

    Only trades events where the scorer returns confidence >= min_confidence.
    """
    position_size = INITIAL_CAPITAL / position_fraction
    trading_dates = pd.DatetimeIndex(sorted(close_matrix.index))

    events_by_date = {}
    for dt, g in events.groupby("event_date"):
        events_by_date[dt] = g

    cash = INITIAL_CAPITAL
    open_positions: Dict[str, Position] = {}
    trade_log: List[Dict] = []
    daily_rows: List[Dict] = []
    skipped_count = 0

    for today in trading_dates:
        # Exits
        to_close = []
        for ticker, pos in open_positions.items():
            if ticker not in high_matrix.columns or ticker not in close_matrix.columns:
                continue
            today_high = high_matrix.at[today, ticker]
            today_close = close_matrix.at[today, ticker]
            if pd.isna(today_high) or pd.isna(today_close):
                continue

            if today_high >= pos.target_price:
                pnl = (pos.target_price - pos.entry_price) * pos.shares
                trade_log.append({
                    "label": label, "ticker": ticker,
                    "entry_date": pos.entry_date, "exit_date": today,
                    "entry_price": pos.entry_price, "exit_price": pos.target_price,
                    "pnl": pnl, "return_pct": (pos.target_price / pos.entry_price) - 1,
                    "hold_days": (today - pos.entry_date).days,
                    "exit_reason": "target_hit", "confidence": pos.confidence,
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)
            elif today >= pos.max_exit_date:
                pnl = (today_close - pos.entry_price) * pos.shares
                trade_log.append({
                    "label": label, "ticker": ticker,
                    "entry_date": pos.entry_date, "exit_date": today,
                    "entry_price": pos.entry_price, "exit_price": today_close,
                    "pnl": pnl, "return_pct": (today_close / pos.entry_price) - 1,
                    "hold_days": (today - pos.entry_date).days,
                    "exit_reason": "max_hold", "confidence": pos.confidence,
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)

        for t in to_close:
            del open_positions[t]

        # Entries — USE THE SCORER
        candidates = events_by_date.get(today)
        if candidates is not None:
            candidates = candidates.sort_values("drop_pct", ascending=False)

            for _, row in candidates.iterrows():
                if len(open_positions) >= max_positions:
                    break
                if cash < position_size:
                    break

                ticker = row["ticker"]
                if ticker in open_positions:
                    continue

                entry_price = float(row["entry_price"])
                if entry_price <= 0:
                    continue

                # Score this candidate
                market_stressed = row.get("liquidity_credit_stress_severity", "none") in ["medium", "high"]
                event_type = row.get("stock_event_type", "unlabeled")
                event_severity = row.get("stock_event_severity", "medium")

                # For unlabeled events, use drop_bucket only
                if event_type == "unlabeled":
                    event_type = "no_clear_catalyst"
                    event_severity = "medium"

                score = scorer.score_candidate(
                    drop_bucket=row["drop_bucket"],
                    stock_event_type=event_type,
                    stock_event_severity=event_severity,
                    market_stressed=market_stressed,
                )

                # Filter by confidence
                conf = score["confidence"]
                if min_confidence == "high" and conf != "high":
                    skipped_count += 1
                    continue
                elif min_confidence == "medium" and conf == "skip":
                    skipped_count += 1
                    continue
                elif min_confidence == "low" and conf == "skip":
                    skipped_count += 1
                    continue

                future_dates = trading_dates[trading_dates > today]
                if len(future_dates) < max_hold_days:
                    continue
                max_exit_date = future_dates[max_hold_days - 1]

                shares = position_size / entry_price
                target_price = entry_price * (1.0 + target_return)

                open_positions[ticker] = Position(
                    ticker=ticker, entry_date=today, entry_price=entry_price,
                    target_price=target_price, max_exit_date=max_exit_date,
                    shares=shares, allocated=position_size,
                    confidence=conf, success_rate=score["success_rate"],
                )
                cash -= position_size

        # Daily MTM
        mv = sum(
            pos.shares * float(close_matrix.at[today, pos.ticker])
            for pos in open_positions.values()
            if pos.ticker in close_matrix.columns and pd.notna(close_matrix.at[today, pos.ticker])
        )
        equity = cash + mv
        daily_rows.append({
            "label": label, "date": today, "cash": cash,
            "market_value": mv, "equity": equity,
            "open_positions": len(open_positions),
        })

    return pd.DataFrame(trade_log), pd.DataFrame(daily_rows), skipped_count


def main():
    events = load_data()
    print(f"Loaded {len(events)} events")

    scorer = TradeScorer()

    close_matrix, high_matrix = download_prices(events)
    print(f"Price matrix: {close_matrix.shape}")

    results = []

    for target, max_hold, pos_frac, max_pos, label in CONFIGS:
        print(f"\nSimulating: {label} (HIGH confidence only)...")
        trades, equity, skipped = simulate(
            events, scorer, close_matrix, high_matrix,
            target, max_hold, pos_frac, max_pos, label,
            min_confidence="high",
        )

        if trades.empty:
            print("  No trades taken.")
            continue

        # Annual returns
        eq = equity.copy()
        eq["year"] = pd.to_datetime(eq["date"]).dt.year

        total_return = (eq["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
        total_trades = len(trades)
        win_rate = (trades["pnl"] > 0).mean() * 100
        avg_return = trades["return_pct"].mean() * 100
        avg_hold = trades["hold_days"].mean()
        total_pnl = trades["pnl"].sum()

        winners = trades[trades["pnl"] > 0]
        losers = trades[trades["pnl"] <= 0]
        avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
        avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0

        print(f"  Trades: {total_trades} | Skipped: {skipped}")
        print(f"  Win rate: {win_rate:.1f}% | Avg return/trade: {avg_return:.2f}%")
        print(f"  Total PnL: ${total_pnl:,.0f} | Total return: {total_return:+.1f}%")

        # Per year
        annual_rows = []
        for year in sorted(eq["year"].unique()):
            ydata = eq[eq["year"] == year]
            yr_start = ydata["equity"].iloc[0]
            yr_end = ydata["equity"].iloc[-1]
            yr_ret = (yr_end / yr_start - 1) * 100
            yr_trades = trades[pd.to_datetime(trades["entry_date"]).dt.year == year]
            yr_wr = (yr_trades["pnl"] > 0).mean() * 100 if len(yr_trades) > 0 else 0
            annual_rows.append({
                "year": year, "return": yr_ret, "trades": len(yr_trades), "win_rate": yr_wr,
            })
            print(f"    {year}: {yr_ret:+.1f}% ({len(yr_trades)} trades, {yr_wr:.0f}% WR)")

        results.append({
            "config": label,
            "total_return": total_return,
            "total_trades": total_trades,
            "skipped": skipped,
            "win_rate": win_rate,
            "avg_return_per_trade": avg_return,
            "avg_hold_days": avg_hold,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        })

    # Summary comparison
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON (HIGH CONFIDENCE ONLY)")
    print("=" * 100)
    print(f"{'Config':>20} {'TotalRet':>10} {'Trades':>7} {'Skipped':>8} {'WinRate':>8} {'AvgRet':>8} {'AvgHold':>8} {'AvgWin$':>9} {'AvgLoss$':>9}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["total_return"]):
        print(f"{r['config']:>20} {r['total_return']:>+9.1f}% {r['total_trades']:>7} {r['skipped']:>8} "
              f"{r['win_rate']:>7.1f}% {r['avg_return_per_trade']:>7.2f}% {r['avg_hold_days']:>7.1f}d "
              f"${r['avg_win']:>8.0f} ${r['avg_loss']:>8.0f}")


if __name__ == "__main__":
    main()
