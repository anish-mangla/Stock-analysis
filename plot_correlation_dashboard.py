#!/usr/bin/env python3
"""Correlation dashboard for intraday indicators."""
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

df = pd.read_csv('backtest_iterations/intraday_deep/intraday_indicators_features.csv')

# Correlation with success
indicator_cols = [c for c in df.columns if c.startswith('d0_') or c.startswith('d1_') or c.startswith('d2_') or c == 'drop_pct']
corr_success = {col: df[col].corr(df['success']) for col in indicator_cols if df[col].notna().sum() > 100}
corr_speed = {col: df[col].corr(df['days_to_hit']) for col in indicator_cols if df[[col, 'days_to_hit']].dropna().shape[0] > 100}

# Sort
sorted_success = sorted(corr_success.items(), key=lambda x: abs(x[1]), reverse=True)
names = [x[0] for x in sorted_success]
vals = [x[1] for x in sorted_success]
colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]

fig1 = go.Figure(go.Bar(
    y=names, x=vals, orientation='h',
    marker_color=colors,
    text=[f'{v:+.3f}' for v in vals], textposition='outside',
))
fig1.update_layout(
    title='Correlation with Trade Success (higher = more predictive)',
    xaxis_title='Correlation coefficient',
    height=700, template='plotly_white',
    yaxis=dict(autorange='reversed'),
    margin=dict(l=200),
)

# Cross-correlation heatmap of top indicators
top = names[:12]
cross = df[top].corr()
fig2 = go.Figure(go.Heatmap(
    z=cross.values, x=top, y=top,
    colorscale='RdBu', zmid=0,
    text=[[f'{v:.2f}' for v in row] for row in cross.values],
    texttemplate='%{text}', textfont=dict(size=10),
))
fig2.update_layout(
    title='Cross-correlation between top indicators (are they redundant?)',
    height=600, template='plotly_white',
    yaxis=dict(autorange='reversed'),
)

html = [
    '<html><head><title>Indicator Correlations</title></head><body>',
    '<h1 style="font-family:sans-serif">Intraday Indicator Correlations</h1>',
    to_html(fig1, full_html=False, include_plotlyjs='cdn'),
    '<hr>',
    to_html(fig2, full_html=False, include_plotlyjs=False),
    '</body></html>',
]
with open('correlation_dashboard.html', 'w') as f:
    f.write('\n'.join(html))

import webbrowser
webbrowser.open('correlation_dashboard.html')
print('Saved to correlation_dashboard.html')
