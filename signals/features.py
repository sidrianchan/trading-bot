"""Feature engineering for the XGBoost ranker signal.

All features are computed using only data available up to a given date.
Forward returns (used as training labels) are computed separately and
only used to build historical training datasets — never for prediction.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from signals.momentum import MomentumSignal
from signals.quality import QualitySignal
from signals.volatility import LowVolatilitySignal

FEATURE_COLS = [
    "momentum",       # 6-month price momentum rank
    "quality",        # composite quality rank (ROE, D/E, rev growth)
    "low_vol",        # inverse-volatility rank
    "prox_52w_high",  # (price / 52-week high) - 1, closer to 0 = near high
    "reversal_1m",    # negated 21-day return (fade short-term overcrowding)
    "earnings_growth",# YoY earnings growth (EPS revision proxy)
]


def build_features_at_date(
    date: pd.Timestamp,
    available_prices: pd.DataFrame,  # prices up to and including `date` ONLY
    fundamentals: pd.DataFrame,
    momentum_signal: MomentumSignal,
    quality_signal: QualitySignal,
    low_vol_signal: LowVolatilitySignal,
) -> pd.DataFrame:
    """Compute all features for every ticker at a single date.

    Returns a DataFrame indexed by ticker, columns = FEATURE_COLS.
    Only uses data available up to `date` — no lookahead.
    """
    non_spy = [c for c in available_prices.columns if c != "SPY"]

    # Core factor ranks — each returns a percentile-rank Series
    mom = momentum_signal.rank(available_prices, fundamentals).reindex(non_spy)
    qual = quality_signal.rank(pd.DataFrame(), fundamentals).reindex(non_spy)
    vol = low_vol_signal.rank(available_prices, fundamentals).reindex(non_spy)

    # 52-week high proximity: (current_price / max_over_252d) - 1
    if len(available_prices) >= 252:
        high_52w = available_prices.iloc[-252:][non_spy].max()
        cur = available_prices.iloc[-1][non_spy]
        prox = (cur / high_52w.replace(0, np.nan)) - 1.0
    else:
        prox = pd.Series(0.0, index=non_spy)

    # 1-month reversal: negate 21-day return so higher score = recent laggard
    if len(available_prices) >= 22:
        ret_21d = (available_prices.iloc[-1][non_spy] / available_prices.iloc[-22][non_spy]) - 1.0
        reversal = -ret_21d
    else:
        reversal = pd.Series(0.0, index=non_spy)

    # Earnings growth proxy from fundamentals (same for all dates — yfinance limitation)
    if not fundamentals.empty and "earnings_growth" in fundamentals.columns:
        earn = fundamentals["earnings_growth"].reindex(non_spy)
    else:
        earn = pd.Series(np.nan, index=non_spy)

    df = pd.DataFrame({
        "momentum": mom,
        "quality": qual,
        "low_vol": vol,
        "prox_52w_high": prox,
        "reversal_1m": reversal,
        "earnings_growth": earn,
    }, index=pd.Index(non_spy, name="ticker"))

    return df[FEATURE_COLS]


def build_feature_history(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    momentum_signal: MomentumSignal,
    quality_signal: QualitySignal,
    low_vol_signal: LowVolatilitySignal,
    min_history_days: int = 160,
) -> pd.DataFrame:
    """Pre-compute features at every monthly rebalance date.

    Returns a DataFrame with MultiIndex (date, ticker).
    Only uses data available up to each date.
    """
    rebalance_dates = _monthly_last_days(prices.index)
    frames = []

    for i, date in enumerate(rebalance_dates):
        avail = prices.loc[:date]
        if len(avail) < min_history_days:
            continue
        feat = build_features_at_date(
            date, avail, fundamentals, momentum_signal, quality_signal, low_vol_signal
        )
        if feat.empty:
            continue
        feat.index = pd.MultiIndex.from_tuples(
            [(date, t) for t in feat.index], names=["date", "ticker"]
        )
        frames.append(feat)

        if (i + 1) % 12 == 0:
            logger.debug(f"Feature history: processed {i + 1}/{len(rebalance_dates)} dates")

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS)

    logger.info(f"Built feature history: {len(frames)} dates × ~{len(frames[0])} tickers each")
    return pd.concat(frames)


def build_target_history(
    prices: pd.DataFrame,
    forward_days: int = 21,
) -> pd.DataFrame:
    """Compute cross-sectional forward-return rank for every rebalance date.

    Forward returns ARE future data, but are used only as historical training
    labels — never for predicting current scores. No lookahead in inference.

    Returns DataFrame with MultiIndex (date, ticker), column 'rank' in [0, 1].
    """
    rebalance_dates = _monthly_last_days(prices.index)
    rows = []

    for date in rebalance_dates:
        future_idx = prices.index[prices.index > date]
        if len(future_idx) < forward_days:
            continue
        future_date = future_idx[forward_days - 1]

        curr = prices.loc[date].drop("SPY", errors="ignore")
        fut = prices.loc[future_date].drop("SPY", errors="ignore")
        common = curr.index.intersection(fut.index)

        fwd_ret = (fut[common] / curr[common].replace(0, np.nan)) - 1.0
        ranks = fwd_ret.rank(pct=True)

        for ticker, rank in ranks.items():
            rows.append({"date": date, "ticker": ticker, "rank": rank})

    if not rows:
        return pd.DataFrame(columns=["rank"])

    return pd.DataFrame(rows).set_index(["date", "ticker"])


def _monthly_last_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    df = pd.DataFrame({"date": index})
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym")["date"].max().tolist()
