"""BTC/ETH crypto momentum signal logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CryptoMomentumConfig:
    capital: float = 30_000.0
    universe: tuple[str, str] = ("BTC/USD", "ETH/USD")
    stable: str = "USDC/USD"
    abs_lookback: int = 84
    abs_skip: int = 14
    rel_lookback: int = 7
    rel_skip: int = 14
    cb_threshold: float = -0.40


@dataclass
class CryptoMomentumState:
    """Persistent crypto bot state. Stored as JSON under logs/."""

    peak: float = 0.0
    cash_value: float = 0.0
    last_target: Optional[str] = None          # signal target (written by daily summary)
    last_executed_target: Optional[str] = None  # actual position (written only after order fills)
    last_eval_date: Optional[str] = None
    last_btc_abs: Optional[float] = None
    last_btc_rel: Optional[float] = None
    last_eth_rel: Optional[float] = None
    start_btc_price: Optional[float] = None


@dataclass(frozen=True)
class CryptoMomentumSignal:
    target: Optional[str]
    regime: str
    btc_abs_return: float
    relative_scores: dict[str, float] = field(default_factory=dict)
    drawdown: float = 0.0
    peak: float = 0.0
    cb_status: str = "normal"
    decision_reason: str = ""


def yahoo_to_alpaca_symbol(symbol: str) -> str:
    mapping = {"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "USDC-USD": "USDC/USD"}
    return mapping.get(symbol, symbol)


def alpaca_to_yahoo_symbol(symbol: str) -> str:
    mapping = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "USDC/USD": "USDC-USD"}
    return mapping.get(symbol, symbol)


def normalize_crypto_symbol(symbol: str) -> str:
    """Normalize common Alpaca/Yahoo crypto symbol shapes to Alpaca pair form."""
    symbol = yahoo_to_alpaca_symbol(symbol.upper())
    if "/" in symbol:
        return symbol
    if symbol.endswith("USD") and len(symbol) > 3:
        return f"{symbol[:-3]}/USD"
    if symbol.endswith("USDC") and len(symbol) > 4:
        return f"{symbol[:-4]}/USDC"
    return symbol


def total_return_skip(series: pd.Series, lookback: int, skip: int) -> float:
    """Return over `lookback` rows ending `skip` rows before the latest row."""
    s = series.dropna()
    if len(s) < lookback + skip + 1:
        return float("nan")
    end = s.iloc[-(skip + 1)]
    start = s.iloc[-(lookback + skip + 1)]
    if pd.isna(start) or pd.isna(end) or start <= 0:
        return float("nan")
    return float(end / start - 1.0)


def compute_crypto_signal(
    prices: pd.DataFrame,
    state: CryptoMomentumState,
    current_value: float,
    cfg: CryptoMomentumConfig,
) -> tuple[CryptoMomentumSignal, CryptoMomentumState]:
    """Compute the BTC/ETH/USDC target and updated peak state."""
    prices = prices.rename(columns={c: normalize_crypto_symbol(str(c)) for c in prices.columns})
    btc, eth = cfg.universe

    new_state = CryptoMomentumState(
        peak=max(state.peak, current_value),
        cash_value=state.cash_value,
        last_target=state.last_target,
        last_executed_target=state.last_executed_target,
        last_eval_date=state.last_eval_date,
        start_btc_price=state.start_btc_price,
    )

    drawdown = 0.0
    if new_state.peak > 0:
        drawdown = current_value / new_state.peak - 1.0

    if drawdown <= cfg.cb_threshold and state.last_target in cfg.universe:
        new_state.last_target = None
        signal = CryptoMomentumSignal(
            target=None,
            regime="circuit_breaker",
            btc_abs_return=float("nan"),
            drawdown=drawdown,
            peak=new_state.peak,
            cb_status="triggered",
            decision_reason=(
                f"Crypto circuit breaker: drawdown {drawdown:.1%} <= "
                f"{cfg.cb_threshold:.0%}; liquidate to USDC/cash"
            ),
        )
        return signal, new_state

    btc_abs = total_return_skip(prices[btc], cfg.abs_lookback, cfg.abs_skip)
    btc_rel = total_return_skip(prices[btc], cfg.rel_lookback, cfg.rel_skip)
    eth_rel = total_return_skip(prices[eth], cfg.rel_lookback, cfg.rel_skip)
    new_state.last_btc_abs = btc_abs
    new_state.last_btc_rel = btc_rel
    new_state.last_eth_rel = eth_rel

    if pd.isna(btc_abs):
        signal = CryptoMomentumSignal(
            target=None,
            regime="insufficient_history",
            btc_abs_return=float("nan"),
            drawdown=drawdown,
            peak=new_state.peak,
            decision_reason=(
                f"Insufficient BTC history: need "
                f"{cfg.abs_lookback + cfg.abs_skip + 1} daily bars"
            ),
        )
        return signal, new_state

    scores = {btc: btc_rel, eth: eth_rel}
    if btc_abs > 0:
        valid_scores = {k: v for k, v in scores.items() if pd.notna(v)}
        target = max(valid_scores, key=valid_scores.get) if valid_scores else None
        regime = "risk_on"
        reason = (
            f"BTC {cfg.abs_lookback}d return {btc_abs:+.1%} > 0; "
            f"target {target or 'USDC'} by {cfg.rel_lookback}d relative momentum"
        )
    else:
        target = None
        regime = "risk_off"
        reason = f"BTC {cfg.abs_lookback}d return {btc_abs:+.1%} <= 0; hold USDC/cash"

    new_state.last_target = target
    signal = CryptoMomentumSignal(
        target=target,
        regime=regime,
        btc_abs_return=btc_abs,
        relative_scores=scores,
        drawdown=drawdown,
        peak=new_state.peak,
        decision_reason=reason,
    )
    return signal, new_state


def state_to_dict(state: CryptoMomentumState) -> dict:
    return asdict(state)


def state_from_dict(data: dict) -> CryptoMomentumState:
    return CryptoMomentumState(
        peak=float(data.get("peak", 0.0)),
        cash_value=float(data.get("cash_value", 0.0)),
        last_target=data.get("last_target"),
        last_executed_target=data.get("last_executed_target"),
        last_eval_date=data.get("last_eval_date"),
        last_btc_abs=data.get("last_btc_abs"),
        last_btc_rel=data.get("last_btc_rel"),
        last_eth_rel=data.get("last_eth_rel"),
        start_btc_price=data.get("start_btc_price"),
    )
