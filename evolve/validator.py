"""Candidate validation: backtest gates, out-of-sample holdout, rising Sharpe bar.

Selection discipline:
- The selection backtest runs on data through T-12 months and must pass the
  family's existing gate set (reused from backtester/ — not reimplemented).
- The trailing 12 months are the holdout: the candidate must keep a positive
  Sharpe and stay above a per-family drawdown floor out-of-sample.
- Multiple-testing guard: the candidate's selection Sharpe must clear the
  incumbent's by a margin that RISES with the cumulative number of trials
  ever run for the family:  required = incumbent + 0.10 + 0.02*ln(1+n).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from loguru import logger

from evolve.families import get_family

SHARPE_BAR_BASE = 0.10
SHARPE_BAR_SLOPE = 0.02
HOLDOUT_DAYS = 365


@dataclass(frozen=True)
class FamilyRuntime:
    fetch: Callable[[], pd.DataFrame]
    run: Callable[[pd.DataFrame, Any, str, str], Any]  # -> backtest result with .equity/.gates
    default_start: str
    holdout_dd_floor: float
    periods_per_year: int


def _etf_runtime() -> FamilyRuntime:
    from backtester.dual_momentum import fetch_etf_prices, run_dual_momentum_backtest

    return FamilyRuntime(
        fetch=lambda: fetch_etf_prices(start="2010-03-01", end=_today()),
        run=lambda prices, cfg, start, end: run_dual_momentum_backtest(
            prices, cfg, start=start, end=end
        ),
        default_start="2010-03-01",
        holdout_dd_floor=-0.40,
        periods_per_year=252,
    )


def _crypto_runtime() -> FamilyRuntime:
    from backtester.crypto_momentum import fetch_crypto_prices, run_crypto_backtest

    return FamilyRuntime(
        fetch=lambda: fetch_crypto_prices(start="2018-01-01", end=_today()),
        run=lambda prices, cfg, start, end: run_crypto_backtest(prices, cfg, start=start, end=end),
        default_start="2018-01-01",
        holdout_dd_floor=-0.50,
        periods_per_year=365,
    )


RUNTIMES: dict[str, Callable[[], FamilyRuntime]] = {
    "dual_momentum_etf": _etf_runtime,
    "crypto_momentum": _crypto_runtime,
}


def _today() -> str:
    return pd.Timestamp.now(tz="UTC").date().isoformat()


@dataclass
class ValidationResult:
    ok_backtest: bool = False
    ok_holdout: bool = False
    ok_sharpe_bar: bool = False
    selection_metrics: dict = field(default_factory=dict)
    holdout_metrics: dict = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    candidate_sharpe: float = float("nan")
    incumbent_sharpe: float = float("nan")
    required_sharpe: float = float("nan")
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.ok_backtest and self.ok_holdout and self.ok_sharpe_bar


def required_sharpe(incumbent_sharpe: float, n_trials: int) -> float:
    """Deflated-Sharpe proxy: the bar rises with every trial ever run."""
    return incumbent_sharpe + SHARPE_BAR_BASE + SHARPE_BAR_SLOPE * math.log(1 + n_trials)


def equity_metrics(equity: pd.Series, periods_per_year: int) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 3:
        return {}
    returns = equity.pct_change().dropna()
    years = len(returns) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = returns.std() * math.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else float("nan")
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {"cagr": float(cagr), "sharpe": float(sharpe), "max_drawdown": max_dd}


def validate_candidate(
    family_id: str,
    params: dict,
    incumbent_params: dict,
    n_trials: int,
    *,
    prices: pd.DataFrame | None = None,
    overrides: dict | None = None,
) -> ValidationResult:
    """Full validation of one candidate parameter set (no state is written here)."""
    family = get_family(family_id)
    runtime = RUNTIMES[family_id]()
    result = ValidationResult()

    cfg = family.build_config(params, overrides=overrides)
    incumbent_cfg = family.build_config(incumbent_params, overrides=overrides)

    if prices is None:
        prices = runtime.fetch()
    end = prices.index[-1]
    cutoff = (end - pd.Timedelta(days=HOLDOUT_DAYS)).date().isoformat()
    start = runtime.default_start

    # 1. Selection window: reuse the family's own gate set
    selection = runtime.run(prices, cfg, start, cutoff)
    result.gates = [
        {"gate": str(idx), "value": float(row["value"]), "passed": bool(row["passed"])}
        for idx, row in selection.gates.iterrows()
    ]
    result.ok_backtest = bool(selection.passed)
    result.selection_metrics = equity_metrics(selection.equity, runtime.periods_per_year)
    result.candidate_sharpe = result.selection_metrics.get("sharpe", float("nan"))

    # 2. Rising Sharpe bar vs the incumbent on the same selection window
    incumbent = runtime.run(prices, incumbent_cfg, start, cutoff)
    inc_metrics = equity_metrics(incumbent.equity, runtime.periods_per_year)
    result.incumbent_sharpe = inc_metrics.get("sharpe", float("nan"))
    result.required_sharpe = required_sharpe(result.incumbent_sharpe, n_trials)
    result.ok_sharpe_bar = bool(result.candidate_sharpe >= result.required_sharpe)

    # 3. Holdout: trailing 12 months of a continuous full-period run
    full = runtime.run(prices, cfg, start, end.date().isoformat())
    holdout_equity = full.equity.loc[cutoff:]
    result.holdout_metrics = equity_metrics(holdout_equity, runtime.periods_per_year)
    h_sharpe = result.holdout_metrics.get("sharpe", float("nan"))
    h_dd = result.holdout_metrics.get("max_drawdown", float("nan"))
    result.ok_holdout = bool(
        h_sharpe == h_sharpe and h_sharpe > 0 and h_dd > runtime.holdout_dd_floor
    )

    result.detail = (
        f"selection sharpe {result.candidate_sharpe:.2f} vs required {result.required_sharpe:.2f} "
        f"(incumbent {result.incumbent_sharpe:.2f}, trials {n_trials}); "
        f"gates {'PASS' if result.ok_backtest else 'FAIL'}; "
        f"holdout sharpe {h_sharpe:.2f} dd {h_dd:.1%} "
        f"{'PASS' if result.ok_holdout else 'FAIL'}"
    )
    logger.info(f"Evolve validation [{family_id}]: {result.detail}")
    return result
