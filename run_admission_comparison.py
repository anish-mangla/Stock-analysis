"""
Admission Logic Comparison
============================
Tests 3 variants against the baseline:
1. admission_function (score-first + sector caps + replacement)
2. admission_function_with_sector_penalty (sector-penalized ordering + replacement)
3. No replacement (same admission, but skip when full)
Plus baseline: raw score-weighted (no sector caps, no replacement)
"""

import pandas as pd
import numpy as np
from screener_v1 import (
    score_event, get_day_path, compute_good_trade, compute_ugly_loser,
    boolish, to_decimal, COL_SUCCESS, COL_FINAL_RETURN, COL_MAX_DRAWDOWN,
    DATE_COL, ensure_datetime
)
from admission_logic import (
    Signal, OpenPosition, admission_function, admission_function_with_sector_penalty,
    score_to_size_multiplier
)
import os

os.makedirs('phase1_artifacts', exist_ok=True)

# Load
print("Loading data...")
df = pd.read_parquet('outputs/events_with_forward_path.parquet')
df = ensure_datetime(df, DATE_COL)
df['success_label'] = df[COL_SUCCESS].apply(boolish)
df['good_trade'] = df.apply(lambda r: compute_good_trade(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['ugly_loser'] = df.apply(lambda r: compute_ugly_loser(r[COL_SUCCESS], r[COL_FINAL_RETURN], r[COL_MAX_DRAWDOWN]), axis=1)
df['year'] = df[DATE_COL].dt.year

print("Scoring events...")
scores_list = []
for idx, row in df.iterrows():
    sr = score_event(row)
    scores_list.append({'score': sr.score, 'tier': sr.tier})
df = df.join(pd.DataFrame(scores_list, index=df.index))

close = pd.read_parquet('backtest_iterations/close_matrix.parquet')
trading_dates = sorted(close.index)
date_to_idx = {d: i for i, d in enumerate(trading_dates)}
del close

# Filter to tradable signals
signals_df = df[df['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])].copy()
signals_df['entry_idx'] = signals_df[DATE_COL].map(lambda d: date_to_idx.get(d, -1)) + 1
signals_df = signals_df[signals_df['entry_idx'] > 0]
signals_df = signals_df[signals_df['entry_idx'] < len(trading_dates)]

# Group signals by entry date
signals_by_entry = {}
for idx, row in signals_df.iterrows():
    eidx = row['entry_idx']
    if eidx not in signals_by_entry:
        signals_by_entry[eidx] = []
    signals_by_entry[eidx].append(row)


def run_portfolio_with_admission(admission_mode='score_first', max_positions=10, max_per_sector=2,
                                  allow_replacement=True, label=""):
    """
    admission_mode: 'score_first', 'sector_penalty', 'baseline_no_caps'
    """
    capital = 100000.0
    initial_capital = capital
    positions = []  # list of dicts with position info
    trades_log = []
    signals_taken = 0
    signals_skipped = 0
    replacements_made = 0
    daily_equity = []

    for day_idx in range(len(trading_dates)):
        date = trading_dates[day_idx]

        # Update existing positions
        for pos in positions:
            if pos['exited']:
                continue
            pos['days_held'] += 1
            day_offset = day_idx - pos['entry_idx']
            if day_offset < 1:
                continue

            row = pos['event_row']
            path = get_day_path(row, day_offset + 1)
            high_ret = path["high_ret"]
            low_ret = path["low_ret"]
            close_ret = path["close_ret"]

            if not pd.isna(close_ret):
                pos['unrealized'] = close_ret
            if not pd.isna(low_ret) and low_ret < pos.get('max_dd', 0):
                pos['max_dd'] = low_ret

            # Exits: +5% target, -8% stop, 60d max hold
            if not pd.isna(low_ret) and low_ret <= -0.08:
                pos['exited'] = True
                pos['exit_reason'] = 'stop_loss'
                pos['exit_ret'] = -0.08
            elif not pd.isna(high_ret) and high_ret >= 0.05:
                pos['exited'] = True
                pos['exit_reason'] = 'profit_target'
                pos['exit_ret'] = 0.05
            elif pos['days_held'] >= 60:
                pos['exited'] = True
                pos['exit_reason'] = 'max_hold'
                pos['exit_ret'] = close_ret if not pd.isna(close_ret) else 0.0

        # Process exits
        still_open = []
        for pos in positions:
            if pos['exited']:
                pnl = pos['exit_ret'] * pos['size_pct'] * capital
                capital += pnl
                trades_log.append({
                    'ticker': pos['ticker'], 'tier': pos['tier'], 'score': pos['score'],
                    'sector_etf': pos['sector_etf'], 'days_held': pos['days_held'],
                    'exit_reason': pos['exit_reason'], 'return': pos['exit_ret'],
                    'pnl': pnl, 'size_pct': pos['size_pct'],
                    'success_label': pos['success_label'], 'good_trade': pos['good_trade'],
                    'ugly_loser': pos['ugly_loser'], 'year': pos['year'],
                })
            else:
                still_open.append(pos)
        positions = still_open

        # New signals
        if day_idx in signals_by_entry:
            new_sigs_raw = signals_by_entry[day_idx]

            if admission_mode == 'baseline_no_caps':
                # Simple: sort by score, fill up to max_positions, no sector caps
                sorted_sigs = sorted(new_sigs_raw, key=lambda r: -r['score'])
                for sig in sorted_sigs:
                    if len(positions) >= max_positions:
                        signals_skipped += 1
                        continue
                    # No ticker/sector check
                    size = 0.10 * score_to_size_multiplier(sig['score'], sig['tier'])
                    positions.append({
                        'ticker': sig['ticker'], 'sector_etf': sig.get('sector_etf'),
                        'score': sig['score'], 'tier': sig['tier'],
                        'entry_idx': day_idx, 'days_held': 0, 'unrealized': 0.0,
                        'exited': False, 'exit_reason': None, 'exit_ret': None,
                        'size_pct': size, 'event_row': sig, 'max_dd': 0.0,
                        'success_label': sig['success_label'], 'good_trade': sig['good_trade'],
                        'ugly_loser': sig['ugly_loser'], 'year': sig['year'],
                    })
                    signals_taken += 1
            else:
                # Build Signal objects
                new_signals = [
                    Signal(ticker=r['ticker'], event_date=r[DATE_COL],
                           score=int(r['score']), tier=r['tier'],
                           sector_etf=r.get('sector_etf'))
                    for r in new_sigs_raw
                ]
                # Build OpenPosition objects from current positions
                open_pos = [
                    OpenPosition(ticker=p['ticker'], sector_etf=p['sector_etf'],
                                 score=p['score'], days_held=p['days_held'],
                                 unrealized_return=p['unrealized'], tier=p['tier'])
                    for p in positions
                ]

                if admission_mode == 'score_first':
                    decision = admission_function(
                        new_signals, open_pos, max_positions=max_positions,
                        max_per_sector=max_per_sector, allow_replacement=allow_replacement)
                elif admission_mode == 'sector_penalty':
                    decision = admission_function_with_sector_penalty(
                        new_signals, open_pos, max_positions=max_positions,
                        max_per_sector=max_per_sector, allow_replacement=allow_replacement)
                else:
                    decision = admission_function(
                        new_signals, open_pos, max_positions=max_positions,
                        max_per_sector=max_per_sector, allow_replacement=allow_replacement)

                # Process replacements: exit replaced positions
                for incoming_sig, replaced_pos in decision.replacements:
                    for p in positions:
                        if p['ticker'] == replaced_pos.ticker and not p['exited']:
                            p['exited'] = True
                            p['exit_reason'] = 'replaced'
                            p['exit_ret'] = p['unrealized']
                            pnl = p['exit_ret'] * p['size_pct'] * capital
                            capital += pnl
                            trades_log.append({
                                'ticker': p['ticker'], 'tier': p['tier'], 'score': p['score'],
                                'sector_etf': p['sector_etf'], 'days_held': p['days_held'],
                                'exit_reason': 'replaced', 'return': p['exit_ret'],
                                'pnl': pnl, 'size_pct': p['size_pct'],
                                'success_label': p['success_label'], 'good_trade': p['good_trade'],
                                'ugly_loser': p['ugly_loser'], 'year': p['year'],
                            })
                            replacements_made += 1
                            break
                    positions = [p for p in positions if not p['exited']]

                # Enter accepted signals
                sig_map = {r['ticker']: r for r in new_sigs_raw}
                for accepted_sig in decision.acceptance_order:
                    if accepted_sig.ticker in sig_map:
                        sig = sig_map[accepted_sig.ticker]
                        size = 0.10 * score_to_size_multiplier(accepted_sig.score, accepted_sig.tier)
                        positions.append({
                            'ticker': sig['ticker'], 'sector_etf': sig.get('sector_etf'),
                            'score': sig['score'], 'tier': sig['tier'],
                            'entry_idx': day_idx, 'days_held': 0, 'unrealized': 0.0,
                            'exited': False, 'exit_reason': None, 'exit_ret': None,
                            'size_pct': size, 'event_row': sig, 'max_dd': 0.0,
                            'success_label': sig['success_label'], 'good_trade': sig['good_trade'],
                            'ugly_loser': sig['ugly_loser'], 'year': sig['year'],
                        })
                        signals_taken += 1

                signals_skipped += len(decision.skipped_signals)

        # Daily equity
        unrealized = sum(p['unrealized'] * p['size_pct'] * capital for p in positions if not p['exited'])
        daily_equity.append({'date': date, 'equity': capital + unrealized, 'n_pos': len(positions)})

    # Close remaining
    for pos in positions:
        if not pos['exited']:
            capital += pos['unrealized'] * pos['size_pct'] * capital

    eq_df = pd.DataFrame(daily_equity)
    trades_df = pd.DataFrame(trades_log)

    # Metrics
    total_ret = (capital - initial_capital) / initial_capital
    n_years = (trading_dates[-1] - trading_dates[0]).days / 365.25
    cagr = (capital / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
    eq_df['peak'] = eq_df['equity'].cummax()
    eq_df['dd'] = (eq_df['equity'] - eq_df['peak']) / eq_df['peak']
    max_dd = eq_df['dd'].min()
    eq_df['daily_ret'] = eq_df['equity'].pct_change()
    sharpe = eq_df['daily_ret'].mean() / eq_df['daily_ret'].std() * np.sqrt(252) if eq_df['daily_ret'].std() > 0 else 0

    win_rate = (trades_df['return'] > 0).mean() if len(trades_df) > 0 else 0
    avg_ret = trades_df['return'].mean() if len(trades_df) > 0 else 0

    return {
        'label': label, 'total_return': total_ret, 'cagr': cagr, 'max_dd': max_dd,
        'sharpe': sharpe, 'n_trades': len(trades_df), 'signals_taken': signals_taken,
        'signals_skipped': signals_skipped, 'replacements': replacements_made,
        'win_rate': win_rate, 'avg_trade_ret': avg_ret,
        'avg_positions': eq_df['n_pos'].mean(),
    }


# ─── Run Comparisons ───
results = []

configs = [
    ("baseline_score_weighted", 'baseline_no_caps', True),
    ("score_first_sector_caps_replace", 'score_first', True),
    ("sector_penalty_replace", 'sector_penalty', True),
    ("score_first_no_replace", 'score_first', False),
]

for label, mode, allow_repl in configs:
    print(f"  Running: {label}...")
    m = run_portfolio_with_admission(
        admission_mode=mode, max_positions=10, max_per_sector=2,
        allow_replacement=allow_repl, label=label)
    results.append(m)

# ─── Display ───
print("\n" + "=" * 110)
print("ADMISSION LOGIC COMPARISON (CONF+SMALL, 10 pos, score-weighted sizing)")
print("=" * 110)

res_df = pd.DataFrame(results)
display = res_df[['label', 'total_return', 'cagr', 'max_dd', 'sharpe', 'n_trades',
                   'signals_skipped', 'replacements', 'win_rate', 'avg_trade_ret', 'avg_positions']].copy()
display['total_return'] = display['total_return'].map(lambda x: f"{x*100:.1f}%")
display['cagr'] = display['cagr'].map(lambda x: f"{x*100:.2f}%")
display['max_dd'] = display['max_dd'].map(lambda x: f"{x*100:.1f}%")
display['sharpe'] = display['sharpe'].map(lambda x: f"{x:.2f}")
display['win_rate'] = display['win_rate'].map(lambda x: f"{x*100:.1f}%")
display['avg_trade_ret'] = display['avg_trade_ret'].map(lambda x: f"{x*100:.2f}%")
display['avg_positions'] = display['avg_positions'].map(lambda x: f"{x:.1f}")

print(display.to_string(index=False))
res_df.to_csv('phase1_artifacts/admission_comparison.csv', index=False)
print(f"\nSaved to phase1_artifacts/admission_comparison.csv")
