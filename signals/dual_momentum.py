"""Dual Momentum on Leveraged ETFs — V4 (6m filter, tight CB, 2-month re-entry).

Selected variant from backtest comparison (2010-2024):
  CAGR 24.9%  Sharpe 0.74  MaxDD -67.8%  2022 DD -29.5%  2022 protective 100%

# WARNING: TQQQ/UPRO/SOXL can lose 90%+ in bear markets
# WARNING: This strategy is for small capital + high risk tolerance ONLY
# WARNING: 3x leverage decays in choppy markets (volatility drag)
# WARNING: Past leveraged ETF performance does not predict future returns
# WARNING: Paper trade for minimum 30 days before risking real money
#
# KNOWN FAILURE MODE: choppy multi-correction regimes (e.g. 2011-2012)
# can produce -60 to -70% drawdowns via CB whipsaw. Strategy recovers
# but requires 1-2 years and psychological fortitude to hold through.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class V4Config:
    abs_lookback: int = 126        # 6-month absolute momentum on SPY
    rel_lookback: int = 63         # 3-month relative momentum within candidates
    skip: int = 21                 # skip last 1 month of momentum window
    risk_on: tuple[str, ...] = ("TQQQ", "UPRO", "SOXL")
    risk_off_candidates: tuple[str, ...] = ("TLT",)
    benchmark_filter: str = "SPY"
    cb_threshold: float = 0.25     # liquidate when DD >= 25% from peak
    reentry_confirmation_months: int = 2  # require N consecutive risk-on months after CB


@dataclass
class V4State:
    """Persistent state across runs. Stored as JSON in logs/momentum_state.json."""
    peak: float = 0.0
    cash_value: float = 0.0
    in_cb: bool = False
    cb_confirm_count: int = 0
    last_target: Optional[str] = None
    last_eval_date: Optional[str] = None  # ISO date string


@dataclass
class V4Signal:
    target: Optional[str]                   # "TQQQ" / "UPRO" / "SOXL" / "TLT" / None (CASH)
    regime: str                             # "risk_on" / "risk_off"
    spy_lookback_return: float              # SPY total return over abs_lookback (skipped)
    candidate_scores: dict[str, float] = field(default_factory=dict)
    cb_status: str = "normal"               # "normal" / "triggered" / "awaiting_confirmation" / "lifted" / "still_risk_off"
    drawdown: float = 0.0
    peak: float = 0.0
    decision_reason: str = ""


def total_return_skip(series: pd.Series, lookback: int, skip: int) -> float:
    """Total return over `lookback` trading days, ending `skip` days before the latest bar."""
    s = series.dropna()
    if len(s) < lookback + skip + 1:
        return float("nan")
    end = s.iloc[-(skip + 1)]
    start = s.iloc[-(lookback + skip + 1)]
    if start <= 0 or pd.isna(start) or pd.isna(end):
        return float("nan")
    return float(end / start - 1.0)


def compute_signal(
    prices: pd.DataFrame,
    state: V4State,
    current_portfolio_value: float,
    cfg: V4Config = V4Config(),
) -> tuple[V4Signal, V4State]:
    """Compute today's target position and updated state.

    Args:
        prices: DataFrame with one column per ticker (must include all of
            cfg.risk_on, cfg.risk_off_candidates, and cfg.benchmark_filter).
            Index must be sorted ascending dates. Latest row = today.
        state: prior persistent state (peak, CB flags).
        current_portfolio_value: today's mark-to-market portfolio value.
        cfg: strategy config (default = V4 production config).

    Returns:
        (signal, new_state). new_state is intended to be persisted after acting on signal.
    """
    new_state = V4State(
        peak=max(state.peak, current_portfolio_value),
        cash_value=state.cash_value,
        in_cb=state.in_cb,
        cb_confirm_count=state.cb_confirm_count,
        last_target=state.last_target,
    )

    # Drawdown from peak
    dd = 0.0
    if new_state.peak > 0:
        dd = max(0.0, (new_state.peak - current_portfolio_value) / new_state.peak)

    # Intraday-style CB trigger (liquidate immediately if DD >= threshold)
    cb_status = "normal"
    if dd >= cfg.cb_threshold and not new_state.in_cb:
        new_state.in_cb = True
        new_state.cb_confirm_count = 0
        cb_status = "triggered"

    # Absolute momentum on benchmark
    spy_ret = total_return_skip(prices[cfg.benchmark_filter], cfg.abs_lookback, cfg.skip)
    if pd.isna(spy_ret):
        signal = V4Signal(
            target=None, regime="insufficient_history",
            spy_lookback_return=float("nan"),
            cb_status=cb_status, drawdown=dd, peak=new_state.peak,
            decision_reason=f"Insufficient SPY history: need {cfg.abs_lookback + cfg.skip + 1} bars",
        )
        return signal, new_state

    # Determine candidate universe
    if spy_ret > 0:
        regime = "risk_on"
        candidates = list(cfg.risk_on)
    else:
        regime = "risk_off"
        tlt_3m = total_return_skip(prices["TLT"], cfg.rel_lookback, cfg.skip)
        candidates = ["TLT"] if (pd.notna(tlt_3m) and tlt_3m > 0) else []

    # Score candidates by relative momentum (3m skipped)
    scores: dict[str, float] = {}
    for t in candidates:
        if t in prices.columns:
            r = total_return_skip(prices[t], cfg.rel_lookback, cfg.skip)
            if pd.notna(r):
                scores[t] = r

    target: Optional[str] = max(scores, key=scores.get) if scores else None
    decision_reason = (
        f"SPY {cfg.abs_lookback//21}m return {spy_ret:+.1%} → {regime}. "
        + (f"Top {target} by 3m mom {scores.get(target, 0):+.1%}" if target else "No candidate (cash)")
    )

    # CB re-entry / continuation logic — overrides target if CB is active
    if new_state.in_cb:
        if regime == "risk_on":
            new_state.cb_confirm_count += 1
            if new_state.cb_confirm_count >= cfg.reentry_confirmation_months + 1:
                # Lift CB: reset peak (prevents stuck-in-cash trap), allow re-entry.
                new_state.in_cb = False
                new_state.peak = current_portfolio_value
                new_state.cb_confirm_count = 0
                cb_status = "lifted"
                decision_reason += f" | CB lifted after {cfg.reentry_confirmation_months + 1} consecutive risk-on months"
            else:
                target = None
                cb_status = "awaiting_confirmation" if cb_status == "normal" else cb_status
                decision_reason = (
                    f"CB active, awaiting confirmation "
                    f"({new_state.cb_confirm_count}/{cfg.reentry_confirmation_months + 1} risk-on months). "
                    f"Forced cash."
                )
        else:
            new_state.cb_confirm_count = 0
            target = None
            cb_status = "still_risk_off" if cb_status == "normal" else cb_status
            decision_reason = (
                f"CB active and regime is risk_off ({decision_reason}). "
                f"Confirmation count reset to 0. Forced cash."
            )

    new_state.last_target = target

    signal = V4Signal(
        target=target,
        regime=regime,
        spy_lookback_return=spy_ret,
        candidate_scores=scores,
        cb_status=cb_status,
        drawdown=dd,
        peak=new_state.peak,
        decision_reason=decision_reason,
    )
    return signal, new_state


def state_to_dict(state: V4State) -> dict:
    return asdict(state)


def state_from_dict(data: dict) -> V4State:
    return V4State(
        peak=float(data.get("peak", 0.0)),
        cash_value=float(data.get("cash_value", 0.0)),
        in_cb=bool(data.get("in_cb", False)),
        cb_confirm_count=int(data.get("cb_confirm_count", 0)),
        last_target=data.get("last_target"),
        last_eval_date=data.get("last_eval_date"),
    )
