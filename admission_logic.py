"""
Portfolio Admission & Replacement Logic
========================================
From expert. Handles signal prioritization, sector caps, ticker caps,
and weak-position replacement when portfolio is full.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter, defaultdict


@dataclass(frozen=True)
class Signal:
    ticker: str
    event_date: Any
    score: int
    tier: str
    sector_etf: Optional[str] = None


@dataclass
class OpenPosition:
    ticker: str
    sector_etf: Optional[str]
    score: int
    days_held: int
    unrealized_return: float
    tier: Optional[str] = None
    entry_date: Any = None


@dataclass
class AdmissionDecision:
    accepted_signals: List[Signal]
    replacements: List[Tuple[Signal, OpenPosition]]
    skipped_signals: List[Tuple[Signal, str]]
    acceptance_order: List[Signal]


MAX_POSITIONS_DEFAULT = 10
MAX_PER_SECTOR_DEFAULT = 2
MAX_PER_TICKER_DEFAULT = 1

TIER_RANK = {
    "CONFIDENT_YES": 0,
    "SMALLER_YES": 1,
    "NO": 2,
    "VETO": 3,
}


def score_to_size_multiplier(score: int, tier: str) -> float:
    if tier == "CONFIDENT_YES":
        if score >= 10:
            return 1.25
        if score >= 8:
            return 1.10
        return 1.00
    if tier == "SMALLER_YES":
        if score >= 8:
            return 0.85
        if score >= 6:
            return 0.70
        return 0.50
    return 0.0


def signal_sort_key(signal: Signal) -> Tuple:
    tier_rank = TIER_RANK.get(signal.tier, 999)
    return (-signal.score, tier_rank, str(signal.ticker), str(signal.event_date))


def position_weakness_key(pos: OpenPosition) -> Tuple:
    return (pos.score, pos.unrealized_return, -pos.days_held, str(pos.ticker))


def normalize_sector(sector_etf: Optional[str]) -> str:
    return str(sector_etf) if sector_etf is not None else "UNKNOWN"


def portfolio_sector_counts(open_positions: List[OpenPosition]) -> Counter:
    counts = Counter()
    for p in open_positions:
        counts[normalize_sector(p.sector_etf)] += 1
    return counts


def can_add_signal_without_replacement(signal, open_positions, max_positions, max_per_sector, max_per_ticker=1):
    if len(open_positions) >= max_positions:
        return False, "portfolio_full"
    ticker_count = sum(1 for p in open_positions if p.ticker == signal.ticker)
    if ticker_count >= max_per_ticker:
        return False, "ticker_already_held"
    sector = normalize_sector(signal.sector_etf)
    sector_count = sum(1 for p in open_positions if normalize_sector(p.sector_etf) == sector)
    if sector_count >= max_per_sector:
        return False, "sector_cap_reached"
    return True, "ok"


def is_position_replaceable(existing, incoming, score_gap_required=2, min_days_held=5, max_unrealized_return=0.01):
    if existing.ticker == incoming.ticker:
        return False, "same_ticker"
    if existing.days_held <= min_days_held:
        return False, "held_too_short"
    if existing.unrealized_return > max_unrealized_return:
        return False, "existing_not_weak_enough"
    if (incoming.score - existing.score) < score_gap_required:
        return False, "score_gap_too_small"
    return True, "ok"


def choose_replacement_candidate(incoming, open_positions, max_per_sector=2,
                                  score_gap_required=2, min_days_held=5, max_unrealized_return=0.01):
    if not open_positions:
        return None, "no_open_positions"

    incoming_sector = normalize_sector(incoming.sector_etf)
    sector_counts = portfolio_sector_counts(open_positions)
    eligible = []

    for pos in open_positions:
        ok, reason = is_position_replaceable(pos, incoming, score_gap_required, min_days_held, max_unrealized_return)
        if not ok:
            continue
        # Simulate replacement
        simulated = [p for p in open_positions if p is not pos]
        simulated.append(OpenPosition(incoming.ticker, incoming.sector_etf, incoming.score, 0, 0.0, incoming.tier, incoming.event_date))
        sim_counts = portfolio_sector_counts(simulated)
        if sim_counts[incoming_sector] > max_per_sector:
            continue
        eligible.append(pos)

    if not eligible:
        return None, "no_eligible_replacement"

    # Prefer replacing from crowded other sectors
    crowded_other = [p for p in eligible
                     if normalize_sector(p.sector_etf) != incoming_sector and sector_counts[normalize_sector(p.sector_etf)] > 1]
    if crowded_other:
        crowded_other.sort(key=position_weakness_key)
        return crowded_other[0], "replace_crowded_other_sector"

    non_incoming = [p for p in eligible if normalize_sector(p.sector_etf) != incoming_sector]
    if non_incoming:
        non_incoming.sort(key=position_weakness_key)
        return non_incoming[0], "replace_non_incoming_sector"

    eligible.sort(key=position_weakness_key)
    return eligible[0], "replace_weakest_overall"


def admission_function(new_signals, open_positions, max_positions=10, max_per_sector=2,
                       max_per_ticker=1, score_gap_required_for_replacement=2,
                       replacement_min_days_held=5, replacement_max_unrealized=0.01,
                       allow_replacement=True):
    working_positions = list(open_positions)
    # Dedupe
    best_by_ticker = {}
    for s in new_signals:
        prev = best_by_ticker.get(s.ticker)
        if prev is None or signal_sort_key(s) < signal_sort_key(prev):
            best_by_ticker[s.ticker] = s
    ordered_signals = sorted(best_by_ticker.values(), key=signal_sort_key)

    accepted = []
    replacements = []
    skipped = []
    acceptance_order = []

    for signal in ordered_signals:
        if signal.tier not in {"CONFIDENT_YES", "SMALLER_YES"}:
            skipped.append((signal, "non_tradable_tier"))
            continue

        ok, reason = can_add_signal_without_replacement(signal, working_positions, max_positions, max_per_sector, max_per_ticker)
        if ok:
            accepted.append(signal)
            acceptance_order.append(signal)
            working_positions.append(OpenPosition(signal.ticker, signal.sector_etf, signal.score, 0, 0.0, signal.tier, signal.event_date))
            continue

        if not allow_replacement:
            skipped.append((signal, reason))
            continue

        candidate, repl_reason = choose_replacement_candidate(
            signal, working_positions, max_per_sector,
            score_gap_required_for_replacement, replacement_min_days_held, replacement_max_unrealized)

        if candidate is None:
            skipped.append((signal, reason if reason != "portfolio_full" else repl_reason))
            continue

        replacements.append((signal, candidate))
        accepted.append(signal)
        acceptance_order.append(signal)
        idx = next((i for i, p in enumerate(working_positions) if p is candidate), None)
        if idx is not None:
            working_positions.pop(idx)
        working_positions.append(OpenPosition(signal.ticker, signal.sector_etf, signal.score, 0, 0.0, signal.tier, signal.event_date))

    return AdmissionDecision(accepted, replacements, skipped, acceptance_order)


def score_with_concentration_penalty(signal, open_positions, max_per_sector=2):
    sector = normalize_sector(signal.sector_etf)
    sector_count = sum(1 for p in open_positions if normalize_sector(p.sector_etf) == sector)
    penalty = 0.0
    if sector_count == 1:
        penalty = 0.75
    elif sector_count >= max_per_sector:
        penalty = 2.00
    return signal.score - penalty


def admission_function_with_sector_penalty(new_signals, open_positions, max_positions=10,
                                            max_per_sector=2, max_per_ticker=1,
                                            score_gap_required_for_replacement=2,
                                            replacement_min_days_held=5,
                                            replacement_max_unrealized=0.01,
                                            allow_replacement=True):
    working_positions = list(open_positions)
    best_by_ticker = {}
    for s in new_signals:
        prev = best_by_ticker.get(s.ticker)
        if prev is None or signal_sort_key(s) < signal_sort_key(prev):
            best_by_ticker[s.ticker] = s

    accepted = []
    replacements = []
    skipped = []
    acceptance_order = []
    pending = list(best_by_ticker.values())

    while pending:
        pending = sorted(pending, key=lambda s: (
            -score_with_concentration_penalty(s, working_positions, max_per_sector),
            TIER_RANK.get(s.tier, 999), str(s.ticker), str(s.event_date)))
        signal = pending.pop(0)

        if signal.tier not in {"CONFIDENT_YES", "SMALLER_YES"}:
            skipped.append((signal, "non_tradable_tier"))
            continue

        ok, reason = can_add_signal_without_replacement(signal, working_positions, max_positions, max_per_sector, max_per_ticker)
        if ok:
            accepted.append(signal)
            acceptance_order.append(signal)
            working_positions.append(OpenPosition(signal.ticker, signal.sector_etf, signal.score, 0, 0.0, signal.tier, signal.event_date))
            continue

        if not allow_replacement:
            skipped.append((signal, reason))
            continue

        candidate, repl_reason = choose_replacement_candidate(
            signal, working_positions, max_per_sector,
            score_gap_required_for_replacement, replacement_min_days_held, replacement_max_unrealized)

        if candidate is None:
            skipped.append((signal, reason if reason != "portfolio_full" else repl_reason))
            continue

        replacements.append((signal, candidate))
        accepted.append(signal)
        acceptance_order.append(signal)
        idx = next((i for i, p in enumerate(working_positions) if p is candidate), None)
        if idx is not None:
            working_positions.pop(idx)
        working_positions.append(OpenPosition(signal.ticker, signal.sector_etf, signal.score, 0, 0.0, signal.tier, signal.event_date))

    return AdmissionDecision(accepted, replacements, skipped, acceptance_order)
