"""
Shared backtest engine used by all iteration scripts.
Handles portfolio simulation, trade tracking, and result reporting.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any
import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INITIAL_CAPITAL = 100_000.0

@dataclass
class Position:
    ticker: str; entry_date: pd.Timestamp; entry_price: float
    target_price: float; max_exit_date: pd.Timestamp
    shares: float; allocated: float; stop_price: float = 0.0

def load_cached_data():
    close = pd.read_parquet(os.path.join(BASE_DIR, 'close_matrix.parquet'))
    high = pd.read_parquet(os.path.join(BASE_DIR, 'high_matrix.parquet'))
    low = pd.read_parquet(os.path.join(BASE_DIR, 'low_matrix.parquet'))
    spy = pd.read_parquet(os.path.join(BASE_DIR, 'spy_benchmark.parquet'))
    events = pd.read_csv(os.path.join(BASE_DIR, '..', 'outputs', 'events_fully_labeled.csv'))
    events['event_date'] = pd.to_datetime(events['event_date'])
    events = events.dropna(subset=['entry_price']).sort_values('event_date').reset_index(drop=True)
    return close, high, low, spy, events

def run_backtest(
    events: pd.DataFrame,
    close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
    entry_filter: Callable,  # (row, context) -> bool
    target_return: float = 0.03,
    max_hold_days: int = 60,
    stop_loss: Optional[float] = None,  # e.g., -0.07
    position_fraction: int = 10,
    max_positions: int = 10,
    entry_delay: int = 0,  # 0=same day close, 1=next day close, etc.
    exit_loser_fn: Optional[Callable] = None,  # (pos, today, days_held, current_return) -> bool
    sizing_fn: Optional[Callable] = None,  # (row, context) -> float (position size in $)
) -> dict:
    default_position_size = INITIAL_CAPITAL / position_fraction
    trading_dates = pd.DatetimeIndex(sorted(close.index))
    events_by_date = {dt: g for dt, g in events.groupby('event_date')}

    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    trades: List[Dict] = []
    daily: List[Dict] = []
    pending_entries: List[Dict] = []  # for delayed entry

    for today in trading_dates:
        # Process delayed entries that are ready
        ready = [p for p in pending_entries if p['enter_on'] <= today]
        pending_entries = [p for p in pending_entries if p['enter_on'] > today]

        # Exits first
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in close.columns: continue
            tc = close.at[today, ticker]
            th = high.at[today, ticker] if ticker in high.columns else None
            tl = low.at[today, ticker] if ticker in low.columns else None
            if pd.isna(tc): continue

            days_held = len(trading_dates[(trading_dates > pos.entry_date) & (trading_dates <= today)])
            current_return = (float(tc) / pos.entry_price) - 1.0
            er, ep = None, None

            # Stop loss
            if stop_loss and tl is not None and not pd.isna(tl) and float(tl) <= pos.stop_price:
                er, ep = 'stop_loss', pos.stop_price
            # Custom exit logic for losers
            elif exit_loser_fn and exit_loser_fn(pos, today, days_held, current_return):
                er, ep = 'early_exit', float(tc)
            # Target hit
            elif th is not None and not pd.isna(th) and float(th) >= pos.target_price:
                er, ep = 'target', pos.target_price
            # Max hold
            elif today >= pos.max_exit_date:
                er, ep = 'max_hold', float(tc)

            if er:
                pnl = (ep - pos.entry_price) * pos.shares
                trades.append({
                    'ticker': ticker, 'entry_date': pos.entry_date, 'exit_date': today,
                    'entry_price': pos.entry_price, 'exit_price': ep,
                    'pnl': pnl, 'return_pct': (ep / pos.entry_price) - 1,
                    'hold_days': days_held, 'exit_reason': er,
                })
                cash += pos.allocated + pnl
                to_close.append(ticker)
        for t in to_close: del positions[t]

        # Process ready delayed entries
        for entry in ready:
            ticker = entry['ticker']
            position_size = entry.get('size', default_position_size)
            if ticker in positions or len(positions) >= max_positions or cash < position_size:
                continue
            if ticker not in close.columns: continue
            ep = close.at[today, ticker]
            if pd.isna(ep): continue
            ep = float(ep)
            if ep <= 0: continue
            future = trading_dates[trading_dates > today]
            if len(future) < max_hold_days: continue
            sp = ep * (1 + stop_loss) if stop_loss else 0
            positions[ticker] = Position(
                ticker=ticker, entry_date=today, entry_price=ep,
                target_price=ep * (1 + target_return), max_exit_date=future[max_hold_days - 1],
                shares=position_size / ep, allocated=position_size, stop_price=sp,
            )
            cash -= position_size

        # New candidates
        cands = events_by_date.get(today)
        if cands is not None:
            for _, row in cands.sort_values('drop_pct', ascending=False).iterrows():
                if not entry_filter(row, {'today': today, 'positions': positions, 'cash': cash}):
                    continue
                ticker = row['ticker']
                if ticker in positions: continue

                # Compute position size
                if sizing_fn:
                    position_size = sizing_fn(row, {'today': today, 'positions': positions, 'cash': cash, 'equity': cash + sum(pos.allocated for pos in positions.values())})
                else:
                    position_size = default_position_size

                if entry_delay > 0:
                    future = trading_dates[trading_dates > today]
                    if len(future) >= entry_delay:
                        pending_entries.append({'ticker': ticker, 'enter_on': future[entry_delay - 1], 'size': position_size})
                    continue

                if len(positions) >= max_positions or cash < position_size: continue
                ep = float(row['entry_price'])
                if ep <= 0: continue
                future = trading_dates[trading_dates > today]
                if len(future) < max_hold_days: continue
                sp = ep * (1 + stop_loss) if stop_loss else 0
                positions[ticker] = Position(
                    ticker=ticker, entry_date=today, entry_price=ep,
                    target_price=ep * (1 + target_return), max_exit_date=future[max_hold_days - 1],
                    shares=position_size / ep, allocated=position_size, stop_price=sp,
                )
                cash -= position_size

        # Daily equity
        mv = sum(
            pos.shares * float(close.at[today, pos.ticker])
            for pos in positions.values()
            if pos.ticker in close.columns and pd.notna(close.at[today, pos.ticker])
        )
        daily.append({
            'date': today, 'equity': cash + mv, 'cash': cash,
            'open_positions': len(positions),
            'capital_utilization': 1.0 - (cash / (cash + mv)) if (cash + mv) > 0 else 0,
        })

    return _compile_results(trades, daily)


def _compile_results(trades: List[Dict], daily: List[Dict]) -> dict:
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    edf = pd.DataFrame(daily)
    edf['date'] = pd.to_datetime(edf['date'])
    edf['year'] = edf['date'].dt.year

    total_return = (edf['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n = len(tdf)
    wr = (tdf['pnl'] > 0).mean() * 100 if n > 0 else 0
    losers = len(tdf[tdf['pnl'] <= 0]) if n > 0 else 0
    avg_hold = tdf['hold_days'].mean() if n > 0 else 0
    avg_util = edf['capital_utilization'].mean() * 100

    # Annual breakdown
    annual = {}
    for year in sorted(edf['year'].unique()):
        yd = edf[edf['year'] == year]
        yr_ret = (yd['equity'].iloc[-1] / yd['equity'].iloc[0] - 1) * 100
        yr_util = yd['capital_utilization'].mean() * 100
        yr_trades = len(tdf[pd.to_datetime(tdf['entry_date']).dt.year == year]) if n > 0 else 0
        yr_losers = len(tdf[(pd.to_datetime(tdf['entry_date']).dt.year == year) & (tdf['pnl'] <= 0)]) if n > 0 else 0
        annual[year] = {'return': yr_ret, 'trades': yr_trades, 'losers': yr_losers, 'utilization': yr_util}

    # Exit reason breakdown
    exit_reasons = {}
    if n > 0:
        for reason in tdf['exit_reason'].unique():
            subset = tdf[tdf['exit_reason'] == reason]
            exit_reasons[reason] = {'count': len(subset), 'avg_pnl': subset['pnl'].mean()}

    winners = tdf[tdf['pnl'] > 0] if n > 0 else pd.DataFrame()
    losers_df = tdf[tdf['pnl'] <= 0] if n > 0 else pd.DataFrame()

    return {
        'total_return': round(total_return, 2),
        'trades': n,
        'win_rate': round(wr, 1),
        'losers': losers,
        'avg_hold_days': round(avg_hold, 1),
        'avg_capital_utilization': round(avg_util, 1),
        'avg_win': round(winners['pnl'].mean(), 0) if len(winners) > 0 else 0,
        'avg_loss': round(losers_df['pnl'].mean(), 0) if len(losers_df) > 0 else 0,
        'annual': annual,
        'exit_reasons': exit_reasons,
        'equity_curve': edf[['date', 'equity', 'cash', 'open_positions', 'capital_utilization']].to_dict('records'),
    }


def save_results(results: dict, filepath: str):
    """Save results to a text file."""
    with open(filepath, 'w') as f:
        f.write(f"TOTAL RETURN: {results['total_return']:+.1f}%\n")
        f.write(f"Trades: {results['trades']} | Win rate: {results['win_rate']:.0f}% | Losers: {results['losers']}\n")
        f.write(f"Avg hold: {results['avg_hold_days']:.1f}d | Avg capital utilization: {results['avg_capital_utilization']:.0f}%\n")
        f.write(f"Avg win: ${results['avg_win']:.0f} | Avg loss: ${results['avg_loss']:.0f}\n\n")

        f.write("ANNUAL BREAKDOWN:\n")
        f.write(f"{'Year':>6} {'Return':>8} {'Trades':>7} {'Losers':>7} {'Util%':>6}\n")
        f.write("-" * 40 + "\n")
        for year, data in sorted(results['annual'].items()):
            f.write(f"{year:>6} {data['return']:>+7.1f}% {data['trades']:>7} {data['losers']:>7} {data['utilization']:>5.0f}%\n")

        f.write(f"\nEXIT REASONS:\n")
        for reason, data in results['exit_reasons'].items():
            f.write(f"  {reason}: {data['count']} trades, avg PnL ${data['avg_pnl']:.0f}\n")

    # Also save as JSON for programmatic access
    json_path = filepath.replace('.txt', '.json')
    json_safe = {k: v for k, v in results.items() if k != 'equity_curve'}
    # Convert numpy int keys to regular int for JSON
    if 'annual' in json_safe:
        json_safe['annual'] = {str(k): v for k, v in json_safe['annual'].items()}
    with open(json_path, 'w') as f:
        json.dump(json_safe, f, indent=2, default=str)

    print(f"Results saved to {filepath}")


def print_results(results: dict, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  TOTAL RETURN: {results['total_return']:+.1f}%")
    print(f"  Trades: {results['trades']} | WR: {results['win_rate']:.0f}% | Losers: {results['losers']}")
    print(f"  Avg hold: {results['avg_hold_days']:.1f}d | Capital util: {results['avg_capital_utilization']:.0f}%")
    print(f"  Avg win: ${results['avg_win']:.0f} | Avg loss: ${results['avg_loss']:.0f}")
    for year, data in sorted(results['annual'].items()):
        print(f"    {year}: {data['return']:+.1f}% ({data['trades']} trades, {data['losers']} losers, {data['utilization']:.0f}% util)")
