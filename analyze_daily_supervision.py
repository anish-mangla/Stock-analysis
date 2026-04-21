"""
Daily trade supervision analysis.
For each day a position is open, ask: given where we are right now,
what's the expected value of HOLDING vs CLOSING?

This tells us the optimal daily decision at each price level.
"""
import pandas as pd, numpy as np

fp = pd.read_parquet('outputs/events_with_forward_path.parquet')
fp['event_date'] = pd.to_datetime(fp['event_date'])
sf = pd.read_parquet('outputs/speed_features.parquet')
sf['event_date'] = pd.to_datetime(sf['event_date'])
fp = fp.merge(sf[['ticker', 'event_date', 'target_distance_over_atr20']], on=['ticker', 'event_date'], how='left')

clusters = [('2020-02-13', '2020-03-18'), ('2022-04-20', '2022-05-18'),
    ('2022-09-13', '2022-10-07'), ('2025-02-18', '2025-03-13'),
    ('2022-06-01', '2022-06-17'), ('2022-03-29', '2022-04-13'),
    ('2022-01-05', '2022-01-21'), ('2025-03-21', '2025-04-08')]
mask = pd.Series(False, index=fp.index)
for s, e in clusters: mask |= (fp['event_date'] >= s) & (fp['event_date'] <= e)
clean = fp[~mask].copy()

print(f"Clean set: {len(clean)}")

# For each event, compute the BEST exit within 5 days
# (the max close return we could have gotten if we sold at the right time)
for day in range(2, 7):
    col = f'd{day}_close_ret'
    clean[col] = clean[col].apply(lambda x: float(x) if not pd.isna(x) else np.nan)

close_cols = [f'd{day}_close_ret' for day in range(2, 7)]
clean['best_close_5d'] = clean[close_cols].max(axis=1)
clean['worst_close_5d'] = clean[close_cols].min(axis=1)
clean['final_close_5d'] = clean['d6_close_ret']


# ═══════════════════════════════════════════════════════════
# PART 1: At each day's close, what's the EV of holding vs selling?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: HOLD vs SELL DECISION AT EACH DAY'S CLOSE")
print("For each D+N close level, what happens if you hold to D+6?")
print("=" * 70)

for check_day in [2, 3, 4]:
    check_col = f'd{check_day}_close_ret'
    remaining_cols = [f'd{d}_close_ret' for d in range(check_day + 1, 7)]
    remaining_high_cols = [f'd{d}_high_ret' for d in range(check_day + 1, 7)]
    
    print(f"\n--- Checking at D+{check_day} close ---")
    
    bins = [(-999, -0.05), (-0.05, -0.03), (-0.03, -0.02), (-0.02, -0.01),
            (-0.01, 0), (0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 999)]
    labels = ['<-5%', '-5 to -3%', '-3 to -2%', '-2 to -1%', '-1 to 0%',
              '0 to +0.5%', '+0.5 to +1%', '+1 to +2%', '+2 to +3%', '+3 to +5%', '+5%+']
    
    print(f"{'Current':>12} {'n':>5} {'Sell now':>10} {'Best if hold':>13} {'Worst if hold':>14} {'Final D+6':>10} {'Verdict':>10}")
    print('-' * 80)
    
    for (lo, hi), label in zip(bins, labels):
        sub = clean[(clean[check_col] >= lo) & (clean[check_col] < hi)]
        if len(sub) < 30: continue
        
        sell_now = sub[check_col].mean()
        
        # Best remaining close
        best_remaining = sub[remaining_cols].max(axis=1).mean()
        worst_remaining = sub[remaining_cols].min(axis=1).mean()
        final = sub['d6_close_ret'].mean()
        
        # Verdict: if selling now is better than the expected final, sell
        if sell_now > final + 0.002:
            verdict = 'SELL'
        elif final > sell_now + 0.005:
            verdict = 'HOLD'
        else:
            verdict = 'CLOSE'  # roughly equal, take the bird in hand
        
        print(f"{label:>12} {len(sub):>5} {sell_now:>+10.2%} {best_remaining:>+13.2%} {worst_remaining:>+14.2%} {final:>+10.2%} {verdict:>10}")

# ═══════════════════════════════════════════════════════════
# PART 2: Momentum-based supervision
# Is the trade trending up or down? Does direction predict outcome?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: D+2 to D+3 MOMENTUM")
print("If trade went up D+2, does it keep going? If down, does it recover?")
print("=" * 70)

clean['d2_to_d3'] = clean['d3_close_ret'] - clean['d2_close_ret']

# D+2 positive, D+3 still going up vs turning down
d2_pos = clean[clean['d2_close_ret'] > 0]
d2_neg = clean[clean['d2_close_ret'] < 0]

print(f"\nD+2 positive (n={len(d2_pos)}):")
d2p_up = d2_pos[d2_pos['d2_to_d3'] > 0]
d2p_down = d2_pos[d2_pos['d2_to_d3'] <= 0]
print(f"  Keeps going up D+3: n={len(d2p_up)}, final D+6 avg={d2p_up['d6_close_ret'].mean():+.2%}")
print(f"  Turns down D+3:     n={len(d2p_down)}, final D+6 avg={d2p_down['d6_close_ret'].mean():+.2%}")

print(f"\nD+2 negative (n={len(d2_neg)}):")
d2n_up = d2_neg[d2_neg['d2_to_d3'] > 0]
d2n_down = d2_neg[d2_neg['d2_to_d3'] <= 0]
print(f"  Recovers D+3:       n={len(d2n_up)}, final D+6 avg={d2n_up['d6_close_ret'].mean():+.2%}")
print(f"  Keeps falling D+3:  n={len(d2n_down)}, final D+6 avg={d2n_down['d6_close_ret'].mean():+.2%}")

# ═══════════════════════════════════════════════════════════
# PART 3: Simulate daily supervision strategies
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: DAILY SUPERVISION STRATEGIES")
print("=" * 70)

def get_target(atr):
    if pd.isna(atr): return 0.03
    if atr < 0.5: return 0.1216
    if atr < 0.8: return 0.0542
    if atr < 1.3: return 0.0358
    if atr < 2.0: return 0.0231
    return 0.0158

def run_strategy(clean, name, decide_fn):
    """
    decide_fn(day, close_ret, high_ret, low_ret, target, prev_close) -> 'hold' or 'sell'
    If 'sell', exit at close_ret.
    Target/stop still checked intraday before the decision.
    """
    rets = []; days_list = []
    for _, row in clean.iterrows():
        target = get_target(row['target_distance_over_atr20'])
        prev_close = 0
        exited = False
        for day in range(2, 7):
            h = row.get(f'd{day}_high_ret', np.nan)
            l = row.get(f'd{day}_low_ret', np.nan)
            c = row.get(f'd{day}_close_ret', np.nan)
            h = float(h) if not pd.isna(h) else np.nan
            l = float(l) if not pd.isna(l) else np.nan
            c = float(c) if not pd.isna(c) else np.nan
            
            if pd.isna(h) and pd.isna(l) and pd.isna(c): continue
            
            # Intraday: stop loss
            if not pd.isna(l) and l <= -0.08:
                rets.append(-0.08); days_list.append(day - 1); exited = True; break
            # Intraday: target hit
            if not pd.isna(h) and h >= target:
                rets.append(target); days_list.append(day - 1); exited = True; break
            
            # End of day: supervision decision
            if not pd.isna(c):
                decision = decide_fn(day, c, h, l, target, prev_close)
                if decision == 'sell':
                    rets.append(c); days_list.append(day - 1); exited = True; break
                prev_close = c
            
            # Time stop
            if day == 6:
                rets.append(c if not pd.isna(c) else 0); days_list.append(day - 1); exited = True; break
        
        if not exited:
            rets.append(0); days_list.append(5)
    
    rets = np.array(rets); days_arr = np.array(days_list)
    wr = (rets > 0).mean()
    avg_ret = rets.mean()
    avg_days = days_arr.mean()
    rpd = avg_ret / avg_days if avg_days > 0 else 0
    total = rets.sum() * 100
    print(f"  {name:>50}: wr={wr:.1%} avg={avg_ret:+.2%} days={avg_days:.1f} rpd={rpd:+.4%} total={total:+.0f}%")
    return rets, days_arr

# Strategy 1: Baseline (no supervision)
run_strategy(clean, "Baseline (target/stop/5d only)",
    lambda day, c, h, l, t, pc: 'hold')

# Strategy 2: Take profit if close > 75% of target
run_strategy(clean, "Take profit if close >= 75% of target",
    lambda day, c, h, l, t, pc: 'sell' if c >= t * 0.75 else 'hold')

# Strategy 3: Take profit if close >= 50% of target
run_strategy(clean, "Take profit if close >= 50% of target",
    lambda day, c, h, l, t, pc: 'sell' if c >= t * 0.50 else 'hold')

# Strategy 4: Sell if D+2 close < -2%, else hold
run_strategy(clean, "Sell if D+2 close < -2%",
    lambda day, c, h, l, t, pc: 'sell' if day == 2 and c < -0.02 else 'hold')

# Strategy 5: Sell if close drops from previous day by > 1.5%
run_strategy(clean, "Sell if day-over-day drop > 1.5%",
    lambda day, c, h, l, t, pc: 'sell' if day > 2 and c < pc - 0.015 else 'hold')

# Strategy 6: Sell if was green yesterday, red today (momentum reversal)
run_strategy(clean, "Sell on green-to-red reversal",
    lambda day, c, h, l, t, pc: 'sell' if day > 2 and pc > 0.005 and c < pc - 0.005 else 'hold')

# Strategy 7: Take profit if green + sell if red after D+2
run_strategy(clean, "D+2: take green / cut red after D+3",
    lambda day, c, h, l, t, pc: 'sell' if (c >= t * 0.5) or (day >= 3 and c < -0.01) else 'hold')

# Strategy 8: Aggressive — take any green close, cut any red after D+3
run_strategy(clean, "Take any green close, cut red after D+3",
    lambda day, c, h, l, t, pc: 'sell' if c > 0.005 or (day >= 3 and c < -0.005) else 'hold')

# Strategy 9: Patient — hold D+2, then take profit or cut loss
run_strategy(clean, "Hold D+2, then: take 50%tgt or cut -1%",
    lambda day, c, h, l, t, pc: 'hold' if day == 2 else ('sell' if c >= t * 0.5 or c < -0.01 else 'hold'))

# Strategy 10: Trailing from peak close
def trailing_decide(day, c, h, l, t, pc, state={'peak': 0}):
    if day == 2:
        state['peak'] = max(0, c)
        return 'hold'
    state['peak'] = max(state['peak'], c)
    # If we were up > 1% and now dropped 1.5% from peak, sell
    if state['peak'] > 0.01 and c < state['peak'] - 0.015:
        return 'sell'
    # If never went green and now D+3+, cut at -1%
    if day >= 3 and state['peak'] < 0.005 and c < -0.01:
        return 'sell'
    return 'hold'

# Can't use closure easily, do it manually
rets_t = []; days_t = []
for _, row in clean.iterrows():
    target = get_target(row['target_distance_over_atr20'])
    peak = 0; exited = False
    for day in range(2, 7):
        h = row.get(f'd{day}_high_ret', np.nan)
        l = row.get(f'd{day}_low_ret', np.nan)
        c = row.get(f'd{day}_close_ret', np.nan)
        h = float(h) if not pd.isna(h) else np.nan
        l = float(l) if not pd.isna(l) else np.nan
        c = float(c) if not pd.isna(c) else np.nan
        if pd.isna(h) and pd.isna(l) and pd.isna(c): continue
        if not pd.isna(l) and l <= -0.08:
            rets_t.append(-0.08); days_t.append(day-1); exited = True; break
        if not pd.isna(h) and h >= target:
            rets_t.append(target); days_t.append(day-1); exited = True; break
        if not pd.isna(c):
            peak = max(peak, c)
            # Trailing: was up > 1%, now dropped 1.5% from peak
            if peak > 0.01 and c < peak - 0.015:
                rets_t.append(c); days_t.append(day-1); exited = True; break
            # Never went green, D+3+, cut at -1%
            if day >= 3 and peak < 0.005 and c < -0.01:
                rets_t.append(c); days_t.append(day-1); exited = True; break
        if day == 6:
            rets_t.append(c if not pd.isna(c) else 0); days_t.append(day-1); exited = True; break
    if not exited:
        rets_t.append(0); days_t.append(5)

rets_t = np.array(rets_t); days_t = np.array(days_t)
print(f"  {'Trailing peak + cut never-green after D+3':>50}: wr={(rets_t>0).mean():.1%} avg={rets_t.mean():+.2%} days={days_t.mean():.1f} rpd={rets_t.mean()/days_t.mean():+.4%} total={rets_t.sum()*100:+.0f}%")
