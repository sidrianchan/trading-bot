"""Strategy families: the bounded parameter spaces the evolution agent may search.

A strategy is (family_id, params). Each family binds to an existing frozen
config dataclass + pure signal function and declares coarse-stepped bounds
for every evolvable parameter. Proposals outside the bounds (or off the
step grid) are REJECTED, never clamped — a rejection is logged evidence
for the next evolution cycle.

The runtime never executes AI-authored code: adding a new family is a
human act (a ~30-line entry here, reviewed and committed like any code).

Non-evolvable constructor fields (universe tickers, benchmark) live in
Family.fixed. Note: risk_off_candidates is fixed because
signals/dual_momentum.py hardcodes TLT in its risk-off branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evolve.guardrails import check_hard_limits
from signals.crypto_momentum import CryptoMomentumConfig
from signals.dual_momentum import V4Config


class ParamValidationError(ValueError):
    """Raised when proposed params fall outside a family's bounded space."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "int" | "float"
    lo: float
    hi: float
    step: float

    def check(self, value: Any) -> str | None:
        """Return a violation message, or None if the value is on the grid."""
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{self.name}={value!r} is not an int"
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{self.name}={value!r} is not a number"
        v = float(value)
        if v < self.lo - 1e-9 or v > self.hi + 1e-9:
            return f"{self.name}={value} outside bounds [{self.lo}, {self.hi}]"
        k = round((v - self.lo) / self.step)
        if abs(self.lo + k * self.step - v) > 1e-6:
            return f"{self.name}={value} not on step grid (lo={self.lo}, step={self.step})"
        return None


@dataclass(frozen=True)
class Family:
    family_id: str
    config_cls: type
    params: tuple[ParamSpec, ...]
    fixed: dict[str, Any]
    description: str = ""

    def param_names(self) -> set[str]:
        return {p.name for p in self.params}

    def default_params(self) -> dict[str, Any]:
        """Evolvable params at their production defaults (from the config dataclass)."""
        defaults = self.config_cls()
        return {p.name: getattr(defaults, p.name) for p in self.params}

    def validate_params(self, params: dict) -> list[str]:
        """All violations: unknown/missing keys, off-grid values, hard-limit breaches."""
        violations: list[str] = []
        unknown = set(params) - self.param_names()
        if unknown:
            violations.append(f"unknown params: {sorted(unknown)}")
        missing = self.param_names() - set(params)
        if missing:
            violations.append(f"missing params: {sorted(missing)}")
        for spec in self.params:
            if spec.name in params:
                msg = spec.check(params[spec.name])
                if msg:
                    violations.append(msg)
        violations.extend(check_hard_limits(self.family_id, params))
        return violations

    def build_config(self, params: dict, overrides: dict | None = None) -> Any:
        """Validate params and return the frozen config dataclass.

        overrides supplies runtime-scoped fields (e.g. crypto capital from
        config.yaml) that are neither evolvable nor family-fixed.
        """
        violations = self.validate_params(params)
        if violations:
            raise ParamValidationError(
                f"{self.family_id}: " + "; ".join(violations)
            )
        return self.config_cls(**{**self.fixed, **(overrides or {}), **params})


FAMILIES: dict[str, Family] = {
    "dual_momentum_etf": Family(
        family_id="dual_momentum_etf",
        config_cls=V4Config,
        params=(
            ParamSpec("abs_lookback", "int", 63, 252, 21),
            ParamSpec("rel_lookback", "int", 21, 126, 21),
            ParamSpec("skip", "int", 0, 21, 7),
            ParamSpec("cb_threshold", "float", 0.15, 0.35, 0.05),
            ParamSpec("reentry_confirmation_months", "int", 1, 3, 1),
        ),
        fixed={
            "risk_on": ("TQQQ", "UPRO", "SOXL"),
            "risk_off_candidates": ("TLT",),
            "benchmark_filter": "SPY",
        },
        description=(
            "V4 dual momentum on 3x leveraged ETFs (TQQQ/UPRO/SOXL), TLT risk-off, "
            "monthly rebalance, drawdown circuit breaker."
        ),
    ),
    "crypto_momentum": Family(
        family_id="crypto_momentum",
        config_cls=CryptoMomentumConfig,
        params=(
            ParamSpec("abs_lookback", "int", 28, 168, 14),
            ParamSpec("abs_skip", "int", 0, 21, 7),
            ParamSpec("rel_lookback", "int", 7, 28, 7),
            ParamSpec("rel_skip", "int", 0, 21, 7),
            ParamSpec("cb_threshold", "float", -0.50, -0.25, 0.05),
        ),
        fixed={
            "universe": ("BTC/USD", "ETH/USD"),
            "stable": "USDC/USD",
        },
        description=(
            "BTC/ETH absolute + relative momentum vs USDC, weekly Monday rebalance, "
            "drawdown circuit breaker."
        ),
    ),
}


def get_family(family_id: str) -> Family:
    try:
        return FAMILIES[family_id]
    except KeyError:
        raise ParamValidationError(f"unknown strategy family: {family_id!r}") from None
