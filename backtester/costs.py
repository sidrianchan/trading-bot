"""Trading cost model shared by every backtest engine.

Added 2026-09-05 after an audit found that neither live strategy charged anything
to trade. The crypto engine was switching ~52 times a year for free; at Alpaca's
real taker fee that is 12.2% a year, and it was the difference between a reported
+91.1% CAGR and an actual -10.0%.

Two rules that follow from that:

1. **A "leg" is one buy or one sell.** Switching BTC -> ETH is two legs. Going
   BTC -> cash is one. Costs are charged per leg on the full notional moved.
2. **The honest cost is the default.** No engine takes ``cost_bps=0`` implicitly.
   A free backtest has to be asked for by name, and the only legitimate reason to
   ask is to measure how much the costs are worth.
"""
from __future__ import annotations

# Alpaca crypto spot, lowest 30-day volume tier (<$100k), taker side.
# 0.15% maker / 0.25% taker — https://docs.alpaca.markets/us/docs/crypto-fees
# Taker is the right assumption: the bot sends marketable orders on a schedule.
ALPACA_CRYPTO_TAKER_BPS = 25.0

# Alpaca US equities and ETFs are genuinely commission-free, so this is spread
# plus slippage only. 5 bps is deliberately conservative for TQQQ/UPRO/SOXL,
# which are liquid; it is not conservative for a stressed market, when the
# circuit breaker is exactly what fires.
ETF_SPREAD_SLIPPAGE_BPS = 5.0

# Explicit opt-out for cost-sensitivity studies. Never a default.
NO_COSTS_BPS = 0.0


def cost_multiplier(legs: int, cost_bps: float) -> float:
    """Fraction of portfolio value surviving ``legs`` trades at ``cost_bps`` each.

    >>> round(cost_multiplier(2, 25.0), 6)
    0.995006
    """
    if legs <= 0 or cost_bps <= 0:
        return 1.0
    return (1.0 - cost_bps / 1e4) ** legs


def legs_for_switch(held: object | None, target: object | None) -> int:
    """Number of legs to move from ``held`` to ``target``. Cash is ``None``."""
    if held == target:
        return 0
    return (1 if held is not None else 0) + (1 if target is not None else 0)


def annual_cost_drag(legs_per_year: float, cost_bps: float) -> float:
    """Fraction of capital lost per year to fees at this turnover.

    The number that makes turnover legible. 52 legs/year at 25 bps is 12.2%,
    which no momentum signal on two assets is going to out-earn.
    """
    if legs_per_year <= 0 or cost_bps <= 0:
        return 0.0
    return 1.0 - (1.0 - cost_bps / 1e4) ** legs_per_year


def turnover_summary(legs: int, years: float, cost_bps: float) -> dict[str, float]:
    """Turnover block reported alongside every backtest result."""
    legs_per_year = legs / years if years > 0 else float("nan")
    return {
        "legs": float(legs),
        "years": float(years),
        "legs_per_year": float(legs_per_year),
        "cost_bps_per_leg": float(cost_bps),
        "annual_cost_drag": float(annual_cost_drag(legs_per_year, cost_bps)),
    }
