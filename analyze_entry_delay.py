"""
Entry Delay vs Profitability Analysis
=====================================
For each of our 7,831 drop events, check:
- If you enter at D+0 close, D+1 close, or D+2 close
- Whether the stock was still down or had already recovered by entry
- What % of trades hit various profit targets within the next 5 trading days (using intraday highs)
"""

import pandas as pd
import numpy as np

# Load data
close = pd.read_parquet('backtest_iterations/close_matrix.parquet')
high = pd.read_parquet('backtest_iterations/high_matrix.parquet')
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])

trading_dates = sorted(close.index)
date_to_idx = {d: i for i, d in enumerate(trading_dates)}

rows = []

for _, row in events.iterrows():
    ticker = row['ticker']
    event_date = row['event_date']
    
    if ticker not in close.columns or event_date not in date_to_idx:
        continue
    
    idx = date_to_idx[event_date]
    drop_close = close.loc[trading_dates[idx], ticker]
    if pd.isna(drop_close) or drop_close <= 0:
        continue

    for delay in [0, 1, 2]:
        entry_idx = idx + delay
        if entry_idx >= len(trading_dates):
            continue
        
        entry_price = close.loc[trading_dates[entry_idx], ticker]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Path status
        if delay == 0:
            path = 'same_day'
        elif entry_price < drop_close:
            path = 'still_down'
        else:
            path = 'recovered'

        # Max high over next 5 trading days from entry
        max_ret = 0.0
        for d in range(1, 6):
            look_idx = entry_idx + d
            if look_idx >= len(trading_dates):
                break
            h = high.loc[trading_dates[look_idx], ticker]
            if not pd.isna(h):
                ret = (h - entry_price) / entry_price
                if ret > max_ret:
                    max_ret = ret

        rows.append({
            'ticker': ticker,
            'event_date': event_date,
            'drop_pct': row['drop_pct'],
            'delay': delay,
            'path': path,
            'max_5d_high_ret': max_ret,
        })

df = pd.DataFrame(rows)

# Build pivot table
thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]

def make_table(sub, label):
    """Build a summary table for a subset."""
    table_rows = []
    for delay in [0, 1, 2]:
        for path in ['same_day', 'still_down', 'recovered']:
            s = sub[(sub['delay'] == delay) & (sub['path'] == path)]
            if len(s) == 0:
                continue
            n = len(s)
            entry_label = f"D+{delay} close"
            r = {'Entry': entry_label, 'Path': path, 'N': n}
            for t in thresholds:
                wins = (s['max_5d_high_ret'] >= t / 100).sum()
                r[f'>{t}%'] = f"{wins/n*100:.1f}%"
            table_rows.append(r)
    return pd.DataFrame(table_rows)


# ── MAIN TABLE ──
print("\n" + "=" * 90)
print("ALL EVENTS: Entry delay × path status × profit target (5-day window, intraday highs)")
print("=" * 90)
t = make_table(df, 'all')
print(t.to_string(index=False))

# ── BY DROP SIZE ──
for bucket, lo, hi in [('3-5% drops', 0.03, 0.05), ('5-7% drops', 0.05, 0.07), ('7%+ drops', 0.07, 1.0)]:
    sub = df[(df['drop_pct'] >= lo) & (df['drop_pct'] < hi)]
    print(f"\n{'=' * 90}")
    print(f"{bucket.upper()} (n={len(sub)//3:,} events)")
    print("=" * 90)
    t = make_table(sub, bucket)
    print(t.to_string(index=False))

print()
