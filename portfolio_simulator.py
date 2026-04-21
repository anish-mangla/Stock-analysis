"""
Portfolio-Level Capital Simulator
==================================
Simulates deploying the screener with finite capital, overlapping positions,
and realistic constraints.

Tests:
- CONFIDENT_YES only
- CONFIDENT + SMALLER_YES
- Various max position caps (5, 10, 20)
- Equal weight vs score-weighted
"""

import pandas as pd
import numpy as np
from screener_v1 import (
    score_event, size_multiplier_from_tier, get_day_path,
    compute_good_trade, compute_ugly_loser, boolish, to_decimal,
    COL_SUCCESS, COL_FINAL_RETURN, COL_MAX_DRAWDOWN,
    DATE_COL, ensure_datetime
)
import os

os.makedirs('phase1_artifacts', exist_ok=True)

# ─── Load & Score ───
print("Loading data...")
df = pd.read_parquet('outputs/events_with_forward_path.parquet')
df = ensure_datetime(df, DATE_COL)
df['success_label'] = df[COL_SUCCESS].apply(boolish)
df['good_trade'] = df.apply(lambda r: compute_good_trade(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['ugly_loser'] = df.apply(lambda r: compute_ugly_loser(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['year'] = df[DATE_COL].dt.year

print("Scoring events...")
scores = []
for idx, row in df.iterrows():
    sr = score_event(row)
    scores.append({'score': sr.score, 'tier': sr.tier})
df = df.join(pd.DataFrame(scores, index=df.index))

# Get trading dates
close = pd.read_parquet('backtest_iterations/close_matrix.parquet')
trading_dates = sorted(close.index)
date_to_idx = {d: i for i, d in enumerate(trading_dates)}
del close  # free memory


# ─── Portfolio Simulator ───
class Position:
    def __init__(self, ticker, entry_date, entry_idx, tier, score, size_pct):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_idx = entry_idx  # index in trading_dates where entry happened (D+1)
        self.tier = tier
        self.score = score
        self.size_pct = size_pct  # fraction of capital allocated
        self.days_held = 0
        self.current_return = 0.0
        self.max_dd = 0.0
        self.exited = False
        self.exit_reason = None
        self.exit_return = 0.0


def simulate_portfolio(events_df, trading_dates, date_to_idx,
                       tiers_allowed={'CONFIDENT_YES'},
                       max_positions=10,
                       position_size_pct=0.10,  # fraction of capital per position
                       score_weighted=False,
                       profit_target=0.05,
                       stop_loss=-0.08,
                       max_hold_days=60,
                       initial_capital=100000.0,
                       label=""):
    """
    Day-by-day portfolio simulation.
    Entry: after D+1 close (so signal date = event_date, entry = next trading day close).
    """
    
    # Prepare signals: group by the date they become actionable (D+1)
    signals = events_df[events_df['tier'].isin(tiers_allowed)].copy()
    signals['entry_date_idx'] = signals[DATE_COL].map(lambda d: date_to_idx.get(d))
    signals = signals.dropna(subset=['entry_date_idx'])
    signals['entry_date_idx'] = signals['entry_date_idx'].astype(int) + 1  # entry is D+1
    signals = signals[signals['entry_date_idx'] < len(trading_dates)]
    
    # Group signals by entry date
    signals_by_entry = {}
    for _, row in signals.iterrows():
        eidx = row['entry_date_idx']
        if eidx not in signals_by_entry:
            signals_by_entry[eidx] = []
        signals_by_entry[eidx].append(row)
    
    # Simulation
    capital = initial_capital
    positions = []
    daily_equity = []
    trades_log = []
    signals_skipped = 0
    signals_taken = 0
    
    for day_idx in range(len(trading_dates)):
        date = trading_dates[day_idx]
        
        # Check exits for existing positions
        for pos in positions:
            if pos.exited:
                continue
            pos.days_held += 1
            
            # Get return for this day relative to entry
            day_offset = day_idx - pos.entry_idx
            if day_offset < 1:
                continue
            
            # We need the event row to get path data
            # Use pre-computed path columns
            row = pos.event_row
            path = get_day_path(row, day_offset + 1)  # +1 because d2 is first day after entry
            high_ret = path["high_ret"]
            low_ret = path["low_ret"]
            close_ret = path["close_ret"]
            
            if not pd.isna(close_ret):
                pos.current_return = close_ret
            if not pd.isna(low_ret):
                pos.max_dd = min(pos.max_dd, low_ret)
            
            # Exit checks
            if not pd.isna(low_ret) and low_ret <= stop_loss:
                pos.exited = True
                pos.exit_reason = "stop_loss"
                pos.exit_return = stop_loss
            elif not pd.isna(high_ret) and high_ret >= profit_target:
                pos.exited = True
                pos.exit_reason = "profit_target"
                pos.exit_return = profit_target
            elif pos.days_held >= max_hold_days:
                pos.exited = True
                pos.exit_reason = "max_hold"
                pos.exit_return = close_ret if not pd.isna(close_ret) else 0.0
        
        # Remove exited positions, log trades
        still_open = []
        for pos in positions:
            if pos.exited:
                pnl = pos.exit_return * pos.size_pct * capital
                capital += pnl
                trades_log.append({
                    'ticker': pos.ticker,
                    'entry_date': pos.entry_date,
                    'exit_date': date,
                    'tier': pos.tier,
                    'score': pos.score,
                    'days_held': pos.days_held,
                    'exit_reason': pos.exit_reason,
                    'return': pos.exit_return,
                    'pnl': pnl,
                    'size_pct': pos.size_pct,
                })
            else:
                still_open.append(pos)
        positions = still_open
        
        # New entries
        if day_idx in signals_by_entry:
            new_signals = signals_by_entry[day_idx]
            # Sort by score descending (best first)
            new_signals = sorted(new_signals, key=lambda r: -r['score'])
            
            for sig in new_signals:
                if len(positions) >= max_positions:
                    signals_skipped += 1
                    continue
                
                # Determine size
                if score_weighted:
                    base_size = position_size_pct
                    mult = size_multiplier_from_tier(sig['tier'], sig['score'])
                    size = base_size * mult
                else:
                    size = position_size_pct
                
                pos = Position(
                    ticker=sig['ticker'],
                    entry_date=trading_dates[day_idx],
                    entry_idx=day_idx,
                    tier=sig['tier'],
                    score=sig['score'],
                    size_pct=size,
                )
                pos.event_row = sig  # store for path lookup
                positions.append(pos)
                signals_taken += 1
        
        # Daily equity mark
        unrealized = sum(pos.current_return * pos.size_pct * capital for pos in positions)
        daily_equity.append({
            'date': date,
            'capital': capital,
            'unrealized': unrealized,
            'total_equity': capital + unrealized,
            'n_positions': len(positions),
        })
    
    # Close any remaining positions at last available price
    for pos in positions:
        row = pos.event_row
        day_offset = len(trading_dates) - 1 - pos.entry_idx
        path = get_day_path(row, min(day_offset + 1, 60))
        close_ret = path["close_ret"]
        pos.exit_return = close_ret if not pd.isna(close_ret) else pos.current_return
        pnl = pos.exit_return * pos.size_pct * capital
        capital += pnl
        trades_log.append({
            'ticker': pos.ticker, 'entry_date': pos.entry_date,
            'exit_date': trading_dates[-1], 'tier': pos.tier, 'score': pos.score,
            'days_held': pos.days_held, 'exit_reason': 'end_of_data',
            'return': pos.exit_return, 'pnl': pnl, 'size_pct': pos.size_pct,
        })
    
    equity_df = pd.DataFrame(daily_equity)
    trades_df = pd.DataFrame(trades_log)
    
    # Metrics
    total_return = (capital - initial_capital) / initial_capital
    n_years = (trading_dates[-1] - trading_dates[0]).days / 365.25
    cagr = (capital / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Max drawdown from equity curve
    equity_df['peak'] = equity_df['total_equity'].cummax()
    equity_df['dd'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
    max_dd = equity_df['dd'].min()
    
    # Daily returns for Sharpe
    equity_df['daily_ret'] = equity_df['total_equity'].pct_change()
    sharpe = equity_df['daily_ret'].mean() / equity_df['daily_ret'].std() * np.sqrt(252) if equity_df['daily_ret'].std() > 0 else 0
    
    avg_positions = equity_df['n_positions'].mean()
    max_positions_used = equity_df['n_positions'].max()
    pct_time_invested = (equity_df['n_positions'] > 0).mean()
    
    metrics = {
        'label': label,
        'tiers': str(tiers_allowed),
        'max_pos': max_positions,
        'score_weighted': score_weighted,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'n_trades': len(trades_df),
        'signals_taken': signals_taken,
        'signals_skipped': signals_skipped,
        'skip_rate': signals_skipped / (signals_taken + signals_skipped) if (signals_taken + signals_skipped) > 0 else 0,
        'avg_positions': avg_positions,
        'max_positions_used': max_positions_used,
        'pct_time_invested': pct_time_invested,
        'avg_hold_days': trades_df['days_held'].mean() if len(trades_df) > 0 else 0,
        'win_rate': (trades_df['return'] > 0).mean() if len(trades_df) > 0 else 0,
        'avg_trade_return': trades_df['return'].mean() if len(trades_df) > 0 else 0,
    }
    
    return metrics, equity_df, trades_df


# ─── Run Simulations ───
results = []

configs = [
    # (label, tiers, max_pos, score_weighted, pos_size)
    ("CONF_only_5pos_equal", {'CONFIDENT_YES'}, 5, False, 0.20),
    ("CONF_only_10pos_equal", {'CONFIDENT_YES'}, 10, False, 0.10),
    ("CONF_only_20pos_equal", {'CONFIDENT_YES'}, 20, False, 0.05),
    ("CONF_only_10pos_scored", {'CONFIDENT_YES'}, 10, True, 0.10),
    ("CONF+SMALL_10pos_equal", {'CONFIDENT_YES', 'SMALLER_YES'}, 10, False, 0.10),
    ("CONF+SMALL_20pos_equal", {'CONFIDENT_YES', 'SMALLER_YES'}, 20, False, 0.05),
    ("CONF+SMALL_10pos_scored", {'CONFIDENT_YES', 'SMALLER_YES'}, 10, True, 0.10),
]

for label, tiers, max_pos, score_w, pos_size in configs:
    print(f"  Running: {label}...")
    m, eq, tr = simulate_portfolio(
        df, trading_dates, date_to_idx,
        tiers_allowed=tiers,
        max_positions=max_pos,
        position_size_pct=pos_size,
        score_weighted=score_w,
        label=label,
    )
    results.append(m)

# ─── Display Results ───
print("\n" + "=" * 120)
print("PORTFOLIO SIMULATION RESULTS")
print("=" * 120)

res_df = pd.DataFrame(results)
display_cols = ['label', 'total_return', 'cagr', 'max_drawdown', 'sharpe',
                'n_trades', 'signals_skipped', 'skip_rate', 'avg_positions',
                'pct_time_invested', 'avg_hold_days', 'win_rate', 'avg_trade_return']
display = res_df[display_cols].copy()
display['total_return'] = display['total_return'].map(lambda x: f"{x*100:.1f}%")
display['cagr'] = display['cagr'].map(lambda x: f"{x*100:.2f}%")
display['max_drawdown'] = display['max_drawdown'].map(lambda x: f"{x*100:.1f}%")
display['sharpe'] = display['sharpe'].map(lambda x: f"{x:.2f}")
display['skip_rate'] = display['skip_rate'].map(lambda x: f"{x*100:.1f}%")
display['avg_positions'] = display['avg_positions'].map(lambda x: f"{x:.1f}")
display['pct_time_invested'] = display['pct_time_invested'].map(lambda x: f"{x*100:.1f}%")
display['avg_hold_days'] = display['avg_hold_days'].map(lambda x: f"{x:.1f}")
display['win_rate'] = display['win_rate'].map(lambda x: f"{x*100:.1f}%")
display['avg_trade_return'] = display['avg_trade_return'].map(lambda x: f"{x*100:.2f}%")

print(display.to_string(index=False))

res_df.to_csv('phase1_artifacts/portfolio_simulation_results.csv', index=False)
print(f"\nSaved to phase1_artifacts/portfolio_simulation_results.csv")
