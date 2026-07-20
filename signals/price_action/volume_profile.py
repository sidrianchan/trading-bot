"""Volume-at-price profile: where volume actually traded, not just when.

Time-based volume tells you a bar was busy; a volume profile tells you at which
*prices* that participation happened. High-volume prices are where positions were
built and are the levels participants defend — which is the information the
previous support/resistance work was structurally missing.

This module needs real daily volume. It could not have been written against
`data/market.py::fetch_prices` (adjusted closes only) or against
`backtester/ta_engine.py::_close_only_to_ohlc`, which hard-codes volume to zero.
Use `data/market.py::fetch_daily_ohlcv`.

KNOWN LIMITATION — uniform intra-bar distribution
-------------------------------------------------
Each bar's volume is spread evenly across its high-low range because daily bars
carry no intra-bar detail. That is wrong in a specific, non-random way: on
gap-and-go days and capitulation wicks, volume clusters hard at one extreme of
the range, so the POC and HVN anchors get smeared by up to roughly half a bar
range on those days. Acceptable for level detection at ATR-band resolution;
`scripts/poc_displacement.py` measures the actual error against intraday bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("high", "low", "volume")


def volume_profile(bars: pd.DataFrame, bins: int = 50) -> pd.Series:
    """Volume traded at each price level.

    Returns a Series indexed by bin mid-price, valued by volume, ascending by
    price. Empty Series if the frame is empty or degenerate.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"volume_profile requires columns {missing}")
    if bars.empty or bins < 1:
        return pd.Series(dtype=float)

    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float)

    valid = np.isfinite(low) & np.isfinite(high) & np.isfinite(vol) & (high >= low)
    low, high, vol = low[valid], high[valid], vol[valid]
    if len(low) == 0:
        return pd.Series(dtype=float)

    price_min, price_max = float(low.min()), float(high.max())
    if not np.isfinite(price_min) or price_max <= price_min:
        return pd.Series(dtype=float)

    edges = np.linspace(price_min, price_max, bins + 1)
    bin_lo, bin_hi = edges[:-1], edges[1:]
    mids = (bin_lo + bin_hi) / 2.0

    # Overlap of each bar's [low, high] with each bin, as a (n_bars, n_bins) matrix.
    overlap = np.clip(
        np.minimum(high[:, None], bin_hi[None, :]) - np.maximum(low[:, None], bin_lo[None, :]),
        0.0,
        None,
    )
    span = (high - low)[:, None]

    # Zero-range bars (limit moves, illiquid sessions) get their volume placed
    # whole into the single bin containing that price rather than discarded.
    flat = (span[:, 0] <= 0.0)
    if flat.any():
        idx = np.clip(np.searchsorted(edges, high[flat], side="right") - 1, 0, bins - 1)
        overlap[flat] = 0.0
        overlap[np.where(flat)[0], idx] = 1.0
        span[flat] = 1.0

    weights = overlap / span
    row_sums = weights.sum(axis=1, keepdims=True)
    np.divide(weights, row_sums, out=weights, where=row_sums > 0)

    return pd.Series(weights.T @ vol, index=mids, name="volume").sort_index()


def poc(profile: pd.Series) -> float | None:
    """Point of control — the single price bin with the most traded volume."""
    if profile.empty:
        return None
    return float(profile.idxmax())


def high_volume_nodes(profile: pd.Series, pct_of_poc: float = 0.70) -> list[float]:
    """Prices of high-volume nodes, one per contiguous run above the threshold.

    `pct_of_poc` is a fraction of the POC bin's volume, NOT a quantile over the
    bins. A quantile is the wrong threshold here: most bins in a trending name
    hold little or no volume, which drags any percentile down until thin
    price regions qualify as "high volume" — the exact opposite of the intent.

    Adjacent qualifying bins describe a single shelf, so each run collapses to
    its volume-weighted centre rather than emitting several near-identical
    levels that would then have to be de-duplicated downstream.
    """
    if profile.empty:
        return []
    peak = float(profile.max())
    if peak <= 0:
        return []
    threshold = pct_of_poc * peak
    qualifying = profile >= threshold
    if not qualifying.any():
        return []

    nodes: list[float] = []
    run_prices: list[float] = []
    run_volumes: list[float] = []
    for price, is_node, vol in zip(profile.index, qualifying.to_numpy(), profile.to_numpy()):
        if is_node:
            run_prices.append(float(price))
            run_volumes.append(float(vol))
            continue
        if run_prices:
            nodes.append(_weighted_centre(run_prices, run_volumes))
            run_prices, run_volumes = [], []
    if run_prices:
        nodes.append(_weighted_centre(run_prices, run_volumes))
    return nodes


def _weighted_centre(prices: list[float], volumes: list[float]) -> float:
    total = sum(volumes)
    if total <= 0:
        return float(np.mean(prices))
    return float(np.average(prices, weights=volumes))
