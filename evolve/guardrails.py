"""AI-immutable hard safety limits for the strategy-evolution system.

These constants are deliberately hard-coded rather than read from
config.yaml or any file that an LLM proposal could influence. Loosening
them requires a human editing this module.

Enforced at three chokepoints: proposal intake (lifecycle), promotion
(registry write), and live config resolution (registry.resolve_config).
The existing risk stack (risk/kill_switch.py, risk/drawdown.py,
risk/limits.py) is never modified by evolution and continues to wrap
every strategy, including newly promoted ones.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardLimits:
    # Proposal / search discipline
    max_candidates_per_cycle: int = 3
    max_concurrent_shadows: int = 5

    # Shadow-trading minimums before a promotion can even be proposed
    min_shadow_days: int = 21
    min_shadow_rebalances_etf: int = 1     # ETF strategy rebalances monthly
    min_shadow_rebalances_crypto: int = 3  # crypto strategy rebalances weekly

    # Capital ramp for newly promoted strategies
    new_strategy_capital_fraction: float = 0.20
    ramp_days: int = 30

    # Automatic demotion envelope
    demote_dd_multiplier: float = 1.25  # live DD > 1.25x backtest MaxDD -> demote
    demote_dd_hard: float = 0.30        # absolute live-DD ceiling for promoted strategies

    # Circuit-breaker bounds no proposal may loosen.
    # V4Config uses a positive drawdown convention (0.25 = liquidate at -25%);
    # CryptoMomentumConfig uses a negative one (-0.40 = liquidate at -40%).
    etf_cb_threshold_max: float = 0.40
    crypto_cb_threshold_min: float = -0.50


HARD = HardLimits()


def check_hard_limits(family_id: str, params: dict) -> list[str]:
    """Return violation messages for params that would weaken safety floors.

    ParamSpec bounds in evolve/families.py are the first line of defense;
    this is the second, independent one — it must hold even if a family's
    declared bounds are ever widened by mistake.
    """
    violations: list[str] = []
    cb = params.get("cb_threshold")
    if cb is None:
        return violations
    try:
        cb = float(cb)
    except (TypeError, ValueError):
        return [f"cb_threshold is not numeric: {cb!r}"]
    if family_id == "dual_momentum_etf" and cb > HARD.etf_cb_threshold_max:
        violations.append(
            f"cb_threshold {cb} exceeds hard maximum {HARD.etf_cb_threshold_max} "
            f"(would loosen the ETF circuit breaker)"
        )
    if family_id == "crypto_momentum" and cb < HARD.crypto_cb_threshold_min:
        violations.append(
            f"cb_threshold {cb} below hard minimum {HARD.crypto_cb_threshold_min} "
            f"(would loosen the crypto circuit breaker)"
        )
    return violations


def evolution_allowed() -> tuple[bool, str]:
    """The evolution system refuses to run or promote while the kill switch is active."""
    from risk.kill_switch import KillSwitch

    if KillSwitch().is_triggered():
        return False, "kill switch is ACTIVE"
    return True, ""
