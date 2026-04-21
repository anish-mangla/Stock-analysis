#!/usr/bin/env python3
"""
Interactive equity curve dashboard using Plotly.
Opens in your browser. You can zoom, hover, pan.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Load our equity curve
eq = pd.read_csv('outputs/equity_curve_401pct.csv')
eq['date'] = pd.to_datetime(eq['date'])

# Load SPY for comparison
spy = yf.download('SPY', start='2020-01-01', end='2026-04-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = [c[0] for c in spy.columns]
spy = spy[['Close']].copy()
spy.index = spy.index.tz_localize(None)
# Normalize SPY to start at $100K
spy_start = float(spy['Close'].iloc[0])
spy['equity'] = spy['Close'] / spy_start * 100000

# Build the dashboard
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=('Portfolio Equity ($)', 'Open Positions', 'Capital Utilization (%)')
)

# Panel 1: Equity curve + SPY
fig.add_trace(go.Scatter(
    x=eq['date'], y=eq['equity'],
    name='Our System', line=dict(color='#2196F3', width=2),
    hovertemplate='%{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=spy.index, y=spy['equity'],
    name='SPY Buy & Hold', line=dict(color='#FF9800', width=1.5, dash='dash'),
    hovertemplate='%{x|%Y-%m-%d}<br>SPY: $%{y:,.0f}<extra></extra>'
), row=1, col=1)

# Add annotations for key events
annotations = [
    ('2020-03-23', 'COVID bottom'),
    ('2022-01-03', '2022 bear starts\n(we stop trading)'),
    ('2022-11-01', 'Bear ends\n(we resume)'),
    ('2025-03-01', 'Tariff chaos\n(we stop)'),
    ('2025-05-01', 'Tariff ends\n(we resume)'),
]
for date_str, label in annotations:
    dt = pd.Timestamp(date_str)
    nearby = eq[eq['date'] <= dt]
    if len(nearby) > 0:
        y_val = nearby.iloc[-1]['equity']
        fig.add_annotation(
            x=dt, y=y_val, text=label,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            ax=0, ay=-40, font=dict(size=10),
            row=1, col=1
        )

# Panel 2: Open positions
fig.add_trace(go.Scatter(
    x=eq['date'], y=eq['open_positions'],
    name='Positions', line=dict(color='#4CAF50', width=1),
    fill='tozeroy', fillcolor='rgba(76,175,80,0.2)',
    hovertemplate='%{x|%Y-%m-%d}<br>Positions: %{y}<extra></extra>'
), row=2, col=1)

# Panel 3: Capital utilization
fig.add_trace(go.Scatter(
    x=eq['date'], y=eq['capital_utilization'] * 100,
    name='Utilization %', line=dict(color='#9C27B0', width=1),
    fill='tozeroy', fillcolor='rgba(156,39,176,0.15)',
    hovertemplate='%{x|%Y-%m-%d}<br>Utilization: %{y:.0f}%<extra></extra>'
), row=3, col=1)

fig.update_layout(
    title='Mean Reversion System — Equity Curve (2020-2025)',
    height=800,
    showlegend=True,
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified',
    template='plotly_white',
)

fig.update_yaxes(title_text='$', row=1, col=1, tickformat='$,.0f')
fig.update_yaxes(title_text='Count', row=2, col=1)
fig.update_yaxes(title_text='%', row=3, col=1)

# Save as HTML (opens in browser, fully interactive)
fig.write_html('equity_dashboard.html', auto_open=True)
print('Dashboard saved to equity_dashboard.html and opened in browser.')
print('You can zoom by dragging, hover for details, double-click to reset.')
