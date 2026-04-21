#!/usr/bin/env python3
"""
Interactive recovery curve dashboard.
3 panels: recovery paths by drop size, D+1 fork, outcome heatmap.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

df = pd.read_csv('backtest_iterations/intraday_deep/recovery_curve_data.csv')
df['event_date'] = pd.to_datetime(df['event_date'])

# ============================================================
# CHART 1: Recovery paths by drop size (median + 25-75 band)
# ============================================================
fig1 = go.Figure()

colors = {'3-5%': '#2196F3', '5-7%': '#FF9800', '7-10%': '#E91E63', '10%+': '#9C27B0'}
days = [0, 1, 2, 3, 4, 5]
day_labels = ['Drop Day', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5']

for bucket in ['3-5%', '5-7%', '7-10%', '10%+']:
    sub = df[df['drop_bucket'] == bucket]
    if len(sub) < 20:
        continue
    
    medians = [0]  # D+0 = 0 (we bought here)
    p25s = [0]
    p75s = [0]
    p10s = [0]
    p90s = [0]
    
    for d in range(1, 6):
        col = f'close_d{d}'
        medians.append(sub[col].median())
        p25s.append(sub[col].quantile(0.25))
        p75s.append(sub[col].quantile(0.75))
        p10s.append(sub[col].quantile(0.10))
        p90s.append(sub[col].quantile(0.90))
    
    color = colors[bucket]
    
    # 10-90 band (light)
    fig1.add_trace(go.Scatter(
        x=day_labels + day_labels[::-1],
        y=p90s + p10s[::-1],
        fill='toself', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
        line=dict(width=0), showlegend=False, name=f'{bucket} 10-90',
        hoverinfo='skip',
        legendgroup=bucket,
    ))
    
    # 25-75 band (medium)
    fig1.add_trace(go.Scatter(
        x=day_labels + day_labels[::-1],
        y=p75s + p25s[::-1],
        fill='toself', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)',
        line=dict(width=0), showlegend=False, name=f'{bucket} 25-75',
        hoverinfo='skip',
        legendgroup=bucket,
    ))
    
    # Median line
    fig1.add_trace(go.Scatter(
        x=day_labels, y=medians,
        name=f'{bucket} drops (n={len(sub)})',
        line=dict(color=color, width=3),
        mode='lines+markers',
        hovertemplate=f'{bucket} drops<br>%{{x}}: %{{y:+.2f}}%<extra></extra>',
        legendgroup=bucket,
    ))

fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig1.add_hline(y=5, line_dash="dot", line_color="green", opacity=0.3, 
               annotation_text="+5% target", annotation_position="right")
fig1.update_layout(
    title='Recovery Path by Drop Size (median line + 25-75% band)',
    xaxis_title='Days after drop',
    yaxis_title='Return from buy price (%)',
    height=500, template='plotly_white',
    legend=dict(x=0.01, y=0.99),
)

# ============================================================
# CHART 2: D+1 fork — diverging paths based on D+1 return
# ============================================================
fig2 = go.Figure()

d1_buckets = [
    ('D+1 < -3%', df['close_d1'] < -3, '#D32F2F'),
    ('D+1 -3% to -1%', (df['close_d1'] >= -3) & (df['close_d1'] < -1), '#FF5722'),
    ('D+1 -1% to 0%', (df['close_d1'] >= -1) & (df['close_d1'] < 0), '#FF9800'),
    ('D+1 0% to +1%', (df['close_d1'] >= 0) & (df['close_d1'] < 1), '#8BC34A'),
    ('D+1 +1% to +3%', (df['close_d1'] >= 1) & (df['close_d1'] < 3), '#4CAF50'),
    ('D+1 > +3%', df['close_d1'] >= 3, '#1B5E20'),
]

for label, mask, color in d1_buckets:
    sub = df[mask]
    if len(sub) < 20:
        continue
    
    medians = [0]
    for d in range(1, 6):
        medians.append(sub[f'close_d{d}'].median())
    
    wr = sub['success'].mean() * 100
    
    fig2.add_trace(go.Scatter(
        x=day_labels, y=medians,
        name=f'{label} (n={len(sub)}, WR={wr:.0f}%)',
        line=dict(color=color, width=3),
        mode='lines+markers',
        hovertemplate=f'{label}<br>%{{x}}: %{{y:+.2f}}%<extra></extra>',
    ))

fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig2.add_hline(y=5, line_dash="dot", line_color="green", opacity=0.3,
               annotation_text="+5% target", annotation_position="right")
fig2.update_layout(
    title='The D+1 Fork: What happens on Day 1 determines everything',
    xaxis_title='Days after drop',
    yaxis_title='Return from buy price (%)',
    height=500, template='plotly_white',
    legend=dict(x=0.01, y=0.99),
)

# ============================================================
# CHART 3: Heatmap of outcomes
# ============================================================

# Build heatmap data: rows = conditions, columns = D+1 to D+5
heatmap_rows = []

# By drop bucket
for bucket in ['3-5%', '5-7%', '7-10%', '10%+']:
    sub = df[df['drop_bucket'] == bucket]
    if len(sub) < 20: continue
    row = {'condition': f'Drop: {bucket} (n={len(sub)})'}
    for d in range(1, 6):
        row[f'D+{d}'] = round(sub[f'close_d{d}'].mean(), 2)
    row['WR'] = round(sub['success'].mean() * 100, 0)
    heatmap_rows.append(row)

# By month
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
for m in range(1, 13):
    sub = df[df['month'] == m]
    if len(sub) < 20: continue
    row = {'condition': f'Month: {month_names[m]} (n={len(sub)})'}
    for d in range(1, 6):
        row[f'D+{d}'] = round(sub[f'close_d{d}'].mean(), 2)
    row['WR'] = round(sub['success'].mean() * 100, 0)
    heatmap_rows.append(row)

# By stress
for stress in ['none', 'low', 'medium', 'high']:
    sub = df[df['stress'] == stress]
    if len(sub) < 30: continue
    row = {'condition': f'Stress: {stress} (n={len(sub)})'}
    for d in range(1, 6):
        row[f'D+{d}'] = round(sub[f'close_d{d}'].mean(), 2)
    row['WR'] = round(sub['success'].mean() * 100, 0)
    heatmap_rows.append(row)

# By D+1 direction
for label, mask in [('D+1 positive', df['close_d1'] > 0), ('D+1 negative', df['close_d1'] <= 0)]:
    sub = df[mask]
    row = {'condition': f'{label} (n={len(sub)})'}
    for d in range(1, 6):
        row[f'D+{d}'] = round(sub[f'close_d{d}'].mean(), 2)
    row['WR'] = round(sub['success'].mean() * 100, 0)
    heatmap_rows.append(row)

# By day of week
dow_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
for dow in range(5):
    sub = df[df['day_of_week'] == dow]
    if len(sub) < 50: continue
    row = {'condition': f'Drop on {dow_names[dow]} (n={len(sub)})'}
    for d in range(1, 6):
        row[f'D+{d}'] = round(sub[f'close_d{d}'].mean(), 2)
    row['WR'] = round(sub['success'].mean() * 100, 0)
    heatmap_rows.append(row)

hdf = pd.DataFrame(heatmap_rows)
z_cols = ['D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'WR']
z_data = hdf[z_cols].values

fig3 = go.Figure(data=go.Heatmap(
    z=z_data,
    x=z_cols,
    y=hdf['condition'].tolist(),
    colorscale='RdYlGn',
    zmid=0,
    text=[[f'{v:.1f}%' if c != 'WR' else f'{v:.0f}%' for v, c in zip(row, z_cols)] for row in z_data],
    texttemplate='%{text}',
    textfont=dict(size=11),
    hovertemplate='%{y}<br>%{x}: %{z:.2f}%<extra></extra>',
))

fig3.update_layout(
    title='Outcome Heatmap: Average return by condition (green=positive, red=negative)',
    height=800, template='plotly_white',
    yaxis=dict(autorange='reversed'),
    xaxis=dict(side='top'),
)

# ============================================================
# CHART 4: Scatter — drop size vs D+5 return (correlation)
# ============================================================
fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=df['drop_pct'] * 100,
    y=df['close_d5'],
    mode='markers',
    marker=dict(
        size=4, opacity=0.3,
        color=df['success'].map({True: '#4CAF50', False: '#F44336'}),
    ),
    hovertemplate='Drop: %{x:.1f}%<br>D+5 return: %{y:+.1f}%<br><extra></extra>',
    showlegend=False,
))

# Add trend line
from numpy.polynomial import polynomial as P
mask = df['close_d5'].notna() & df['drop_pct'].notna()
x_clean = df.loc[mask, 'drop_pct'].values * 100
y_clean = df.loc[mask, 'close_d5'].values
coeffs = np.polyfit(x_clean, y_clean, 1)
x_line = np.linspace(3, 25, 100)
y_line = np.polyval(coeffs, x_line)
fig4.add_trace(go.Scatter(
    x=x_line, y=y_line, mode='lines',
    line=dict(color='black', width=2, dash='dash'),
    name=f'Trend (slope={coeffs[0]:.3f})',
))

fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
fig4.add_hline(y=5, line_dash="dot", line_color="green", opacity=0.3)
fig4.update_layout(
    title='Drop Size vs D+5 Return (green=winner, red=loser)',
    xaxis_title='Drop size (%)',
    yaxis_title='Return by D+5 (%)',
    height=500, template='plotly_white',
)

# ============================================================
# Combine into single HTML
# ============================================================
from plotly.io import to_html

html_parts = [
    '<html><head><title>Recovery Curve Dashboard</title></head><body>',
    '<h1 style="font-family:sans-serif">Recovery Curve Dashboard</h1>',
    '<p style="font-family:sans-serif;color:#666">7,831 drop events, 2020-2025. Zoom by dragging, hover for details.</p>',
    to_html(fig1, full_html=False, include_plotlyjs='cdn'),
    '<hr>',
    to_html(fig2, full_html=False, include_plotlyjs=False),
    '<hr>',
    to_html(fig4, full_html=False, include_plotlyjs=False),
    '<hr>',
    to_html(fig3, full_html=False, include_plotlyjs=False),
    '</body></html>',
]

with open('recovery_dashboard.html', 'w') as f:
    f.write('\n'.join(html_parts))

print('Dashboard saved to recovery_dashboard.html')

# Auto-open
import webbrowser
webbrowser.open('recovery_dashboard.html')
print('Opened in browser.')
