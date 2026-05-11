from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from signals.composite import CompositeSignal
from portfolio.construction import PortfolioConstructor
from risk.limits import RiskLimits


@dataclass
class BacktestConfig:
    initial_capital: float = 5000.0
    transaction_cost_bps: float = 10.0  # round-trip
    momentum_lookback: int = 252
    skip_days: int = 21
    vol_lookback: int = 63
    top_n: int = 25
    momentum_weight: float = 0.60
    quality_weight: float = 0.30
    low_vol_weight: float = 0.10
    max_position_size: float = 0.10
    min_position_size: float = 0.01
    max_sector_exposure: float = 0.30
    target_volatility: float = 0.15
    drawdown_limit: float = 0.15
    drawdown_reset: float = 0.07
    trend_filter_days: int = 200


class WalkForwardBacktester:
    """Daily-resolution walk-forward backtester with no lookahead bias.

    Rebalances on the last trading day of each month. Uses only price
    history available up to each rebalance date — no future data leaks.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        fundamentals: pd.DataFrame | None = None,
        signal=None,  # optional pre-built BaseSignal (e.g. CompositeSignal with ML)
    ):
        self.config = config or BacktestConfig()
        self.fundamentals = fundamentals if fundamentals is not None else pd.DataFrame()
        self._external_signal = signal  # if set, used instead of creating fresh signal each month

    def run(self, prices: pd.DataFrame, benchmark_ticker: str = "SPY") -> pd.Series:
        """Run the full backtest and return a daily portfolio-value Series."""
        cfg = self.config
        close = prices.copy()

        if benchmark_ticker not in close.columns:
            raise ValueError(f"Benchmark {benchmark_ticker} not in price data")

        shares: dict[str, float] = {}
        cash = cfg.initial_capital
        portfolio_peak = cfg.initial_capital
        circuit_breaker = False

        rebalance_dates = _last_trading_days_of_month(close.index)
        min_history = cfg.momentum_lookback + cfg.skip_days + 5

        daily_values: dict[pd.Timestamp, float] = {}

        for i, date in enumerate(close.index):
            prices_today = close.loc[date]

            # Mark to market
            equity = sum(shares.get(t, 0) * prices_today.get(t, 0) for t in shares)
            portfolio_value = cash + equity
            portfolio_peak = max(portfolio_peak, portfolio_value)

            dd = (portfolio_peak - portfolio_value) / portfolio_peak if portfolio_peak > 0 else 0
            if dd >= cfg.drawdown_limit:
                circuit_breaker = True
                logger.debug(f"{date.date()} — Circuit breaker ON (dd={dd:.1%})")
            elif circuit_breaker and dd <= cfg.drawdown_reset:
                circuit_breaker = False
                logger.debug(f"{date.date()} — Circuit breaker OFF (dd recovered to {dd:.1%})")

            if date in rebalance_dates and i >= min_history:
                # Trend filter (SPY < 200d MA → cash) inside _compute_target is now
                # the single source of truth for cash vs invested. The drawdown
                # circuit breaker is informational only; previously its `if not
                # circuit_breaker:` guard skipped rebalance entirely, locking the
                # strategy into losing positions through every multi-month bear.
                available = close.iloc[: i + 1]
                target_weights = self._compute_target(available, prices_today)
                shares, cost = self._rebalance(
                    shares, target_weights, portfolio_value, prices_today, cfg
                )
                cash = portfolio_value - sum(
                    shares.get(t, 0) * prices_today.get(t, 0) for t in shares
                )
                portfolio_value -= cost
                cash -= cost

            daily_values[date] = cash + sum(
                shares.get(t, 0) * prices_today.get(t, 0) for t in shares
            )

        result = pd.Series(daily_values, name="portfolio_value")
        logger.info(
            f"Backtest complete: {len(result)} days, "
            f"final value=${result.iloc[-1]:,.2f}, "
            f"total return={result.iloc[-1]/cfg.initial_capital - 1:.1%}"
        )
        return result

    def _compute_target(
        self, available_prices: pd.DataFrame, prices_today: pd.Series
    ) -> pd.Series:
        cfg = self.config

        if self._external_signal is not None:
            signal = self._external_signal
        else:
            signal = CompositeSignal(
                momentum_weight=cfg.momentum_weight,
                quality_weight=cfg.quality_weight,
                low_vol_weight=cfg.low_vol_weight,
                lookback_days=cfg.momentum_lookback,
                skip_days=cfg.skip_days,
                vol_lookback_days=cfg.vol_lookback,
                top_n=cfg.top_n,
            )

        scores = signal.compute(available_prices, self.fundamentals)

        # Trend filter: go to cash if SPY < 200-day MA
        _eval_date = available_prices.index[-1].date()
        _spy_present = "SPY" in available_prices.columns
        _enough_lookback = len(available_prices) >= cfg.trend_filter_days
        if _spy_present and _enough_lookback:
            spy = available_prices["SPY"]
            spy_ma = spy.rolling(cfg.trend_filter_days).mean().iloc[-1]
            spy_last = spy.iloc[-1]
            _below = spy_last < spy_ma
            logger.warning(
                f"TREND_DIAG {_eval_date} spy_present=YES rows={len(available_prices)} "
                f"spy_last={spy_last:.2f} spy_ma={spy_ma:.2f} below={_below} "
                f"decision={'CASH' if _below else 'INVEST'}"
            )
            if _below:
                logger.info(f"{_eval_date} — Trend filter active: going to cash")
                return pd.Series(dtype=float)
        else:
            _reason = []
            if not _spy_present:
                _reason.append("SPY_MISSING")
            if not _enough_lookback:
                _reason.append(f"INSUFFICIENT_LOOKBACK({len(available_prices)}<{cfg.trend_filter_days})")
            logger.warning(
                f"TREND_DIAG {_eval_date} spy_present={_spy_present} "
                f"reason={','.join(_reason)} decision=INVEST(filter_skipped)"
            )

        if scores.empty:
            return pd.Series(dtype=float)

        constructor = PortfolioConstructor(
            target_volatility=cfg.target_volatility,
            max_position_size=cfg.max_position_size,
            min_position_size=cfg.min_position_size,
            max_sector_exposure=cfg.max_sector_exposure,
            vol_lookback_days=cfg.vol_lookback,
        )

        fund = self.fundamentals if not self.fundamentals.empty else None
        weights = constructor.construct(scores, available_prices, fund)

        risk = RiskLimits(max_position_size=cfg.max_position_size)
        return risk.apply(weights)

    @staticmethod
    def _rebalance(
        shares: dict[str, float],
        target_weights: pd.Series,
        portfolio_value: float,
        prices_today: pd.Series,
        cfg: BacktestConfig,
    ) -> tuple[dict[str, float], float]:
        cost_rate = cfg.transaction_cost_bps / 10_000

        new_shares: dict[str, float] = {}
        total_traded = 0.0

        for ticker, weight in target_weights.items():
            price = prices_today.get(ticker, 0)
            if price <= 0:
                continue
            target_notional = portfolio_value * weight
            current_shares = shares.get(ticker, 0.0)
            new_share_count = target_notional / price
            trade_notional = abs(new_share_count - current_shares) * price
            total_traded += trade_notional
            new_shares[ticker] = new_share_count

        # Account for positions being fully closed
        for ticker in set(shares) - set(target_weights.index):
            price = prices_today.get(ticker, 0)
            total_traded += shares.get(ticker, 0) * price

        transaction_cost = total_traded * cost_rate
        return new_shares, transaction_cost


def _last_trading_days_of_month(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Return the set of dates that are the last trading day in their month."""
    df = pd.DataFrame({"date": index})
    df["ym"] = df["date"].dt.to_period("M")
    last_days = df.groupby("ym")["date"].max()
    return set(last_days)
