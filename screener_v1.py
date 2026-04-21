"""
Mean-Reversion Screener v1
===========================
Implementation spec from expert. Hybrid rules + score framework.
Hard vetoes + whitelist rules + confidence scoring + position sizing + exit rules.

Usage:
    python run_screener_backtest.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
DATE_COL = "event_date"

# Core screener columns
COL_D1_RETURN = "d1_return"
COL_D1_VWAP = "d1_close_vs_vwap"
COL_HYG_5D = "hyg_5d_return"
COL_D2_FH = "d2_first_hour_ret"
COL_COMPANY_SPECIFIC = "company_specific_factor"
COL_EVENT_TYPE = "stock_event_type"
COL_EVENT_SEVERITY = "stock_event_severity"

# Outcome columns
COL_SUCCESS = "success"
COL_DAYS_TO_HIT = "days_to_hit"
COL_FINAL_RETURN = "final_return"
COL_MAX_DRAWDOWN = "max_drawdown"

PATH_DAY_START = 1
PATH_DAY_END = 60
RETURN_IS_DECIMAL = True


# ============================================================
# HELPERS
# ============================================================
def to_decimal(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    x = float(x)
    return x if RETURN_IS_DECIMAL else x / 100.0


def norm_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def boolish(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return None
    s = norm_str(x)
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float)) and not pd.isna(x):
        return bool(int(x))
    return None


def safe_get(row: pd.Series, col: str, default=np.nan):
    return row[col] if col in row.index else default


def get_close_ret_col(day: int) -> str:
    return f"d{day}_close_ret"


def get_low_ret_col(day: int) -> str:
    return f"d{day}_low_ret"


def get_high_ret_col(day: int) -> str:
    return f"d{day}_high_ret"


def get_day_path(row: pd.Series, day: int) -> Dict[str, float]:
    return {
        "close_ret": to_decimal(safe_get(row, get_close_ret_col(day))),
        "low_ret": to_decimal(safe_get(row, get_low_ret_col(day))),
        "high_ret": to_decimal(safe_get(row, get_high_ret_col(day))),
    }


def compute_good_trade(success, final_return, max_drawdown,
                       dd_cutoff=-0.08, final_cutoff=0.0) -> bool:
    s = boolish(success)
    fr = to_decimal(final_return)
    dd = to_decimal(max_drawdown)
    return (bool(s) and
            (not pd.isna(dd) and dd > dd_cutoff) and
            (not pd.isna(fr) and fr >= final_cutoff))


def compute_ugly_loser(success, final_return, max_drawdown,
                       final_cutoff=-0.10, dd_cutoff=-0.12) -> bool:
    fr = to_decimal(final_return)
    dd = to_decimal(max_drawdown)
    return ((not pd.isna(fr) and fr < final_cutoff) or
            (not pd.isna(dd) and dd < dd_cutoff))


def ensure_datetime(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


# ============================================================
# SCREENER
# ============================================================
BAD_EVENT_TYPES = {
    "product_service_failure", "product failure", "product/service failure",
    "management_governance", "management/governance", "management_change", "management change",
    "insider_selling",
}

# Data-driven: only product_service_failure and management types deserve hard veto
# demand_weakness (91.4% success) and regulatory_legal (92% success) should NOT be vetoed
SEVERE_DAMAGE_EVENT_TYPES = {
    "product_service_failure", "product failure",
    "management_governance", "management/governance", "management_change",
    "insider_selling",
}


@dataclass
class ScreenerResult:
    score: int
    tier: str
    veto_reason: Optional[str]
    rule_flags: Dict[str, Any]


def score_event(row: pd.Series) -> ScreenerResult:
    d1 = to_decimal(safe_get(row, COL_D1_RETURN))
    d1_vwap = to_decimal(safe_get(row, COL_D1_VWAP))
    hyg5 = to_decimal(safe_get(row, COL_HYG_5D))
    d2fh = to_decimal(safe_get(row, COL_D2_FH))
    company_specific = boolish(safe_get(row, COL_COMPANY_SPECIFIC))
    event_type = norm_str(safe_get(row, COL_EVENT_TYPE))
    severity = norm_str(safe_get(row, COL_EVENT_SEVERITY))

    flags: Dict[str, Any] = {
        "d1_return": d1, "d1_close_vs_vwap": d1_vwap,
        "hyg_5d_return": hyg5, "d2_first_hour_ret": d2fh,
        "company_specific": company_specific,
        "event_type": event_type, "severity": severity,
    }

    # Hard vetoes
    if pd.isna(d1):
        return ScreenerResult(-999, "VETO", "missing_d1_return", flags)
    if d1 < 0.0:
        return ScreenerResult(-100, "VETO", "d1_return_below_zero", flags)
    if (not pd.isna(d1_vwap)) and d1 < 0.02 and d1_vwap < 0.01:
        return ScreenerResult(-90, "VETO", "weak_confirmation_d1_and_vwap", flags)
    if event_type in SEVERE_DAMAGE_EVENT_TYPES and severity == "high":
        return ScreenerResult(-80, "VETO", "severe_damage_event_type", flags)
    if (not pd.isna(hyg5)) and hyg5 <= -0.01 and d1 < 0.04 and (pd.isna(d1_vwap) or d1_vwap < 0.01):
        return ScreenerResult(-70, "VETO", "credit_stress_without_strong_confirmation", flags)

    # Score
    score = 0

    # D+1 return
    if d1 >= 0.04:
        score += 3
    elif d1 >= 0.02:
        score += 2
    elif d1 >= 0.0:
        score += 1

    # D+1 vs VWAP
    if not pd.isna(d1_vwap):
        if d1_vwap >= 0.01:
            score += 2
        elif d1_vwap >= 0.0:
            score += 1
        else:
            score -= 2

    # HYG 5d regime
    if not pd.isna(hyg5):
        if hyg5 > 0.005:
            score += 2
        elif hyg5 > 0.0:
            score += 1
        elif hyg5 >= -0.01:
            score += 0
        else:
            score -= 2

    # D+2 first-hour
    if not pd.isna(d2fh):
        if d2fh >= 0.02:
            score += 2
        elif d2fh >= 0.0:
            score += 1
        else:
            score -= 1

    # Catalyst quality
    # Data-driven findings from 2513 labeled events:
    # - Only product_service_failure and management types are genuinely bad (veto above)
    # - company_specific=no bonus and severity penalties are noise
    # - Keep scoring simple: only penalize BAD_EVENT_TYPES
    if event_type in BAD_EVENT_TYPES:
        score -= 2

    # Tier mapping
    if d1 >= 0.04 and (not pd.isna(d1_vwap) and d1_vwap >= 0.01) and (not pd.isna(hyg5) and hyg5 > 0.0):
        tier = "CONFIDENT_YES"
    elif score >= 8:
        tier = "CONFIDENT_YES"
    elif score >= 5:
        tier = "SMALLER_YES"
    else:
        tier = "NO"

    flags["final_score"] = score
    flags["final_tier"] = tier
    return ScreenerResult(score, tier, None, flags)


# ============================================================
# POSITION SIZING
# ============================================================
def size_multiplier_from_tier(tier: str, score: int) -> float:
    if tier == "CONFIDENT_YES":
        return 1.00 if score >= 10 else 0.85
    if tier == "SMALLER_YES":
        return 0.60 if score >= 7 else 0.40
    return 0.0


# ============================================================
# EXIT RULES
# ============================================================
@dataclass
class ExitDecision:
    exit_now: bool
    reason: Optional[str]
    exit_ret: Optional[float]


@dataclass
class BacktestConfig:
    profit_target: float = 0.05
    partial_profit_target: float = 0.03
    stop_loss: float = -0.08
    ugly_loss_stop: float = -0.12
    max_hold_days: int = 60
    use_partial_profit: bool = True
    partial_fraction: float = 0.50
    exit_on_d2_failure: bool = True
    exit_on_lower_low_after_entry: bool = True


def should_exit_position(row, day, tier, config, running_state) -> ExitDecision:
    path = get_day_path(row, day)
    close_ret = path["close_ret"]
    low_ret = path["low_ret"]
    high_ret = path["high_ret"]

    if pd.isna(close_ret) and pd.isna(low_ret) and pd.isna(high_ret):
        return ExitDecision(False, None, None)

    # Hard stop on intraday low
    if not pd.isna(low_ret):
        if low_ret <= config.ugly_loss_stop:
            return ExitDecision(True, "ugly_loss_stop", config.ugly_loss_stop)
        if low_ret <= config.stop_loss:
            return ExitDecision(True, "stop_loss", config.stop_loss)

    # Partial profit
    if (not running_state.get("partial_taken", False) and
            config.use_partial_profit and not pd.isna(high_ret)):
        if high_ret >= config.partial_profit_target:
            running_state["partial_taken"] = True
            running_state["partial_day"] = day
            running_state["partial_ret"] = config.partial_profit_target

    # Full profit target
    if not pd.isna(high_ret) and high_ret >= config.profit_target:
        return ExitDecision(True, "profit_target", config.profit_target)

    # D+2 failure (first full day after entry)
    if config.exit_on_d2_failure and day == 2:
        d2fh = to_decimal(safe_get(row, COL_D2_FH))
        if (not pd.isna(close_ret) and close_ret < -0.02) or (not pd.isna(d2fh) and d2fh < 0.0):
            return ExitDecision(True, "d2_failure_after_entry", close_ret if not pd.isna(close_ret) else -0.02)

    # Lower-low rule for confident setups
    if config.exit_on_lower_low_after_entry and tier == "CONFIDENT_YES":
        if day <= 5 and not pd.isna(low_ret) and low_ret < -0.03:
            return ExitDecision(True, "lower_low_after_entry", low_ret)

    # Time stop
    if day >= config.max_hold_days:
        exit_ret = close_ret if not pd.isna(close_ret) else 0.0
        return ExitDecision(True, "max_hold", exit_ret)

    return ExitDecision(False, None, None)
