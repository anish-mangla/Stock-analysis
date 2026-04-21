"""
Regime Overlay Comparison
==========================
Tests HYG-based regime filters on the baseline portfolio.
Variants A, B, C as expert requested, plus additional HYG thresholds.
"""

import pandas as pd
import numpy as np
from screener_v1 import (
    score_event, get_day_path, compute_good_trade, compute_ugly_loser,
    boolish, to_decimal, COL_SUCCESS, COL_FINAL_RETURN, COL_MAX_DRAWDOWN,
    DATE_COL, ensure_datetime
)
from admission_logic import score_to_size_multiplier
import os

os.makedirs('phase1_artifacts', exist_ok=True)

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

signals_df = df[df['tier'].isin(['CONFIDENT_YES', 'SMALLER_YES'])].copy()
signals_df['entry_idx'] = signals_df[DATE_COL].map(lambda d: date_to_idx.get(d, -1)) + 1
signals_df = signals_df[(signals_df['entry_idx'] > 0) & (signals_df['entry_idx'] < len(trading_dates))]

signals_by_entry = {}
for idx, row in signals_df.iterrows():
    eidx = row['entry_idx']
    if eidx not in signals_by_entry:
        signals_by_entry[eidx] = []
    signals_by_entry[eidx].append(row)


def run_regime_variant(label, regime_fn):
    """
    regime_fn(sig_row) -> (allow: bool, size_mult_override: float or None)
    If allow=False, signal is skipped.
    If size_mult_override is not None, it overrides the normal size.
    """
    capital = 100000.0
    initial_capital = capital
    positions = []
    trades_log = []
    signals_taken = 0
    signals_skipped = 0
    daily_equity = []

    for day_idx in range(len(trading_dates)):
        date = trading_dates[day_idx]

        for pos in positions:
            if pos['exited']:
                continue
            pos['days_held'] += 1
            day_offset = day_idx - pos['entry_idx']
            if day_offset < 1:
                continue
            row = pos['event_row']
            path = get_day_path(row, day_offset + 1)
            high_ret, low_ret, close_ret = path["high_ret"], path["low_ret"], path["close_ret"]
            if not pd.isna(close_ret):
                pos['unrealized'] = close_ret

            if not pd.isna(low_ret) and low_ret <= -0.08:
                pos['exited'], pos['exit_reason'], pos['exit_ret'] = True, 'stop_loss', -0.08
            elif not pd.isna(high_ret) and high_ret >= 0.05:
                pos['exited'], pos['exit_reason'], pos['exit_ret'] = True, 'profit_target', 0.05
            elif pos['days_held'] >= 60:
                pos['exited'], pos['exit_reason'], pos['exit_ret'] = True, 'max_hold', close_ret if not pd.isna(close_ret) else 0.0

        still_open = []
        for pos in positions:
            if pos['exited']:
                pnl = pos['exit_ret'] * pos['size_pct'] * capital
                capital += pnl
                trades_log.append({
                    'ticker': pos['ticker'], 'tier': pos['tier'], 'score': pos['score'],
                    'days_held': pos['days_held'], 'exit_reason': pos['exit_reason'],
                    'return': pos['exit_ret'], 'pnl': pnl, 'size_pct': pos['size_pct'],
                    'success_label': pos['success_label'], 'good_trade': pos['good_trade'],
                    'ugly_loser': pos['ugly_loser'], 'year': pos['year'],
                })
            else:
                still_open.append(pos)
        positions = still_open

        if day_idx in signals_by_entry:
            new_sigs = sorted(signals_by_entry[day_idx], key=lambda r: -r['score'])
            for sig in new_sigs:
                if len(positions) >= 10:
                    signals_skipped += 1
                    continue

                allow, size_override = regime_fn(sig)
                if not allow:
                    signals_skipped += 1
                    continue

                base_size = 0.10 * score_to_size_multiplier(int(sig['score']), sig['tier'])
                size = size_override if size_override is not None else base_size

                positions.append({
                    'ticker': sig['ticker'], 'tier': sig['tier'], 'score': sig['score'],
                    'entry_idx': day_idx, 'days_held': 0, 'unrealized': 0.0,
                    'exited': False, 'exit_reason': None, 'exit_ret': None,
                    'size_pct': size, 'event_row': sig,
                    'success_label': sig['success_label'], 'good_trade': sig['good_trade'],
                    'ugly_loser': sig['ugly_loser'], 'year': sig['year'],
                })
                signals_taken += 1

        unrealized = sum(p['unrealized'] * p['size_pct'] * capital for p in positions if not p['exited'])
        daily_equity.append({'date': date, 'equity': capital + unrealized, 'n_pos': len(positions)})

    for pos in positions:
        if not pos['exited']:
            capital += pos['unrealized'] * pos['size_pct'] * capital

    eq_df = pd.DataFrame(daily_equity)
    trades_df = pd.DataFrame(trades_log)

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
    pct_invested = (eq_df['n_pos'] > 0).mean()

    # 2022 specific
    trades_2022 = trades_df[trades_df['year'] == 2022] if len(trades_df) > 0 else pd.DataFrame()
    dd_2022 = eq_df[(eq_df['date'] >= '2022-01-01') & (eq_df['date'] <= '2022-12-31')]['dd'].min() if len(eq_df) > 0 else 0

    return {
        'label': label, 'total_return': total_ret, 'cagr': cagr, 'max_dd': max_dd,
        'sharpe': sharpe, 'n_trades': len(trades_df), 'signals_skipped': signals_skipped,
        'win_rate': win_rate, 'avg_trade_ret': avg_ret, 'pct_invested': pct_invested,
        'dd_2022': dd_2022,
        'trades_2022': len(trades_2022),
    }


# ─── Define regime functions ───
def baseline(sig):
    return True, None

def variant_a(sig):
    """Skip SMALLER_YES when HYG 5d <= -1%"""
    hyg = to_decimal(sig.get('hyg_5d_return', np.nan))
    if sig['tier'] == 'SMALLER_YES' and not pd.isna(hyg) and hyg <= -0.01:
        return False, None
    return True, None

def variant_b(sig):
    """Skip ALL new trades when HYG 5d <= -1%"""
    hyg = to_decimal(sig.get('hyg_5d_return', np.nan))
    if not pd.isna(hyg) and hyg <= -0.01:
        return False, None
    return True, None

def variant_c(sig):
    """Regime-based sizing: HYG>0 normal, -1%<HYG<=0 0.75x, HYG<=-1% 0.5x CONF / 0x SMALL"""
    hyg = to_decimal(sig.get('hyg_5d_return', np.nan))
    base_size = 0.10 * score_to_size_multiplier(int(sig['score']), sig['tier'])
    if pd.isna(hyg):
        return True, base_size
    if hyg > 0:
        return True, base_size
    elif hyg > -0.01:
        return True, base_size * 0.75
    else:  # hyg <= -0.01
        if sig['tier'] == 'SMALLER_YES':
            return False, None
        return True, base_size * 0.50

def variant_d(sig):
    """Skip all when HYG 5d <= -1% AND SPY 5d <= -4%"""
    hyg = to_decimal(sig.get('hyg_5d_return', np.nan))
    spy = to_decimal(sig.get('spy_5d_return', np.nan))
    if not pd.isna(hyg) and not pd.isna(spy) and hyg <= -0.01 and spy <= -0.04:
        return False, None
    return True, None

def variant_e(sig):
    """Only trade when HYG 5d > 0"""
    hyg = to_decimal(sig.get('hyg_5d_return', np.nan))
    if not pd.isna(hyg) and hyg <= 0:
        return False, None
    return True, None


# ─── Run ───
results = []
variants = [
    ("Baseline (no regime filter)", baseline),
    ("A: Skip SMALL when HYG<=-1%", variant_a),
    ("B: Skip ALL when HYG<=-1%", variant_b),
    ("C: Regime sizing (HYG tiers)", variant_c),
    ("D: Skip ALL when HYG<=-1% AND SPY<=-4%", variant_d),
    ("E: Only trade when HYG>0", variant_e),
]

for label, fn in variants:
    print(f"  Running: {label}...")
    m = run_regime_variant(label, fn)
    results.append(m)

# ─── Display ───
print("\n" + "=" * 120)
print("REGIME OVERLAY COMPARISON")
print("=" * 120)

res_df = pd.DataFrame(results)
display = res_df[['label', 'total_return', 'cagr', 'max_dd', 'sharpe', 'n_trades',
                   'signals_skipped', 'win_rate', 'avg_trade_ret', 'pct_invested', 'dd_2022', 'trades_2022']].copy()
display['total_return'] = display['total_return'].map(lambda x: f"{x*100:.1f}%")
display['cagr'] = display['cagr'].map(lambda x: f"{x*100:.2f}%")
display['max_dd'] = display['max_dd'].map(lambda x: f"{x*100:.1f}%")
display['sharpe'] = display['sharpe'].map(lambda x: f"{x:.2f}")
display['win_rate'] = display['win_rate'].map(lambda x: f"{x*100:.1f}%")
display['avg_trade_ret'] = display['avg_trade_ret'].map(lambda x: f"{x*100:.2f}%")
display['pct_invested'] = display['pct_invested'].map(lambda x: f"{x*100:.1f}%")
display['dd_2022'] = display['dd_2022'].map(lambda x: f"{x*100:.1f}%")

print(display.to_string(index=False))
res_df.to_csv('phase1_artifacts/regime_overlay_comparison.csv', index=False)
print(f"\nSaved to phase1_artifacts/regime_overlay_comparison.csv")
