#!/usr/bin/env python3
"""
market_context.py

Computes the current market-level context for the trade scorer.

Downloads today's VIX, SPY, credit spread proxies, and sector ETF data,
then determines the market stress level and any active conditions.

This is the live equivalent of build_market_level_labels.py.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import yfinance as yf


def _download_close(symbol: str, period: str = "30d") -> pd.Series:
    """Download close prices for a symbol."""
    df = yf.download(symbol, period=period, auto_adjust=False, progress=False, threads=True)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(0):
            df = df[symbol].copy()
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float).dropna()


def get_market_context() -> Dict[str, Any]:
    """
    Compute current market context.

    Returns dict with:
    - vix_level: current VIX
    - spy_daily_return: SPY's return today
    - stress_level: "none", "low", "medium", "high"
    - market_stressed: bool (True if medium or high)
    - hyg_5d_return: high-yield bond ETF 5-day return (credit stress proxy)
    - sector_returns: dict of sector ETF daily returns
    """
    print("Computing market context...")

    context: Dict[str, Any] = {
        "vix_level": None,
        "spy_daily_return": None,
        "spy_close": None,
        "stress_level": "none",
        "market_stressed": False,
        "hyg_5d_return": None,
        "sector_returns": {},
    }

    # VIX
    try:
        vix = _download_close("^VIX", "5d")
        if len(vix) > 0:
            context["vix_level"] = float(vix.iloc[-1])
    except Exception as e:
        print(f"  Warning: VIX download failed: {e}")

    # SPY
    try:
        spy = _download_close("SPY", "5d")
        if len(spy) >= 2:
            context["spy_close"] = float(spy.iloc[-1])
            context["spy_daily_return"] = float((spy.iloc[-1] / spy.iloc[-2]) - 1.0)
    except Exception as e:
        print(f"  Warning: SPY download failed: {e}")

    # HYG (credit stress proxy)
    try:
        hyg = _download_close("HYG", "10d")
        if len(hyg) >= 6:
            context["hyg_5d_return"] = float((hyg.iloc[-1] / hyg.iloc[-6]) - 1.0)
    except Exception as e:
        print(f"  Warning: HYG download failed: {e}")

    # Determine stress level
    vix = context["vix_level"]
    hyg_5d = context["hyg_5d_return"]

    if vix is not None:
        if vix >= 35:
            base_stress = 3
        elif vix >= 25:
            base_stress = 2
        elif vix >= 20:
            base_stress = 1
        else:
            base_stress = 0

        # Credit stress bump
        credit_bump = 0
        if hyg_5d is not None:
            if hyg_5d <= -0.03:
                credit_bump = 2
            elif hyg_5d <= -0.01:
                credit_bump = 1

        final_stress = min(base_stress + credit_bump, 3)
        stress_map = {0: "none", 1: "low", 2: "medium", 3: "high"}
        context["stress_level"] = stress_map[final_stress]
        context["market_stressed"] = final_stress >= 2

    # Sector ETF returns
    sector_etfs = ["XLK", "SMH", "XLC", "XLY", "XLP", "XLF", "XLV", "XLE", "XLI", "XLU", "XLRE", "XLB"]
    for etf in sector_etfs:
        try:
            close = _download_close(etf, "5d")
            if len(close) >= 2:
                context["sector_returns"][etf] = float((close.iloc[-1] / close.iloc[-2]) - 1.0)
        except Exception:
            pass

    print(f"  VIX: {context['vix_level']}")
    print(f"  SPY return: {context['spy_daily_return']}")
    print(f"  Stress level: {context['stress_level']}")
    print(f"  Market stressed: {context['market_stressed']}")

    return context


if __name__ == "__main__":
    ctx = get_market_context()
    print()
    print("Market Context:")
    for k, v in ctx.items():
        if k != "sector_returns":
            print(f"  {k}: {v}")
    print("  Sector returns:")
    for etf, ret in ctx.get("sector_returns", {}).items():
        print(f"    {etf}: {ret*100:+.2f}%")
