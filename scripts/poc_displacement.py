"""How wrong is the uniform intra-bar volume assumption?

`signals/price_action/volume_profile.py` spreads each daily bar's volume evenly
across its high-low range, because daily bars carry no intra-bar detail. That is
wrong in a specific direction: on gap-and-go days and capitulation wicks, volume
clusters at one extreme.

This measures the error rather than assuming it is small. For tickers present in
both the daily and the 1-minute cache, it builds each month's profile both ways
and reports how far the point of control moves, in ATR units — the unit that
matters, since zone bands are ATR-scaled (+/-0.35 ATR).

Read the result as: displacement well under 0.35 ATR means the uniform
assumption is invisible at band resolution and can stay. Displacement at or
above it means levels are being placed in the wrong band, and daily profiles
should be rebuilt from intraday bars.

Run: PYTHONPATH=. .venv/bin/python3 scripts/poc_displacement.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from data.market import fetch_daily_ohlcv
from signals.indicators.volatility import atr
from signals.price_action.volume_profile import poc, volume_profile

INTRADAY_CACHE = Path("data/cache/intraday")
MAX_TICKERS = int(os.environ.get("POC_MAX_TICKERS", "40"))
BINS = 60


def _intraday_profile(bars_1m: pd.DataFrame, bins: int, lo: float, hi: float) -> pd.Series:
    """Reference profile: 1-minute bars are narrow enough to treat as points."""
    price = (bars_1m["high"] + bars_1m["low"] + bars_1m["close"]) / 3.0
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.searchsorted(edges, price.to_numpy(), side="right") - 1, 0, bins - 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    totals = np.zeros(bins)
    np.add.at(totals, idx, bars_1m["volume"].to_numpy(dtype=float))
    return pd.Series(totals, index=mids)


def main() -> None:
    if not INTRADAY_CACHE.exists():
        print("No intraday cache found — nothing to validate against.")
        return

    tickers = sorted({p.name.split("_")[0] for p in INTRADAY_CACHE.glob("*.parquet")})[:MAX_TICKERS]
    logger.info(f"POC displacement: {len(tickers)} tickers with intraday coverage")

    daily = fetch_daily_ohlcv(tickers, "2023-01-01", "2024-12-31")

    rows = []
    for ticker in tickers:
        files = sorted(INTRADAY_CACHE.glob(f"{ticker}_*.parquet"))
        if not files or ticker not in daily:
            continue
        try:
            m1 = pd.read_parquet(files[-1])
        except Exception:
            continue
        if m1.empty or "volume" not in m1.columns:
            continue

        m1.index = pd.to_datetime(m1.index)
        if m1.index.tz is not None:
            m1.index = m1.index.tz_convert("UTC").tz_localize(None)

        d = daily[ticker]
        atr_series = atr(d, period=20)

        for period, d_month in d.groupby(pd.Grouper(freq="ME")):
            if len(d_month) < 15:
                continue
            mask = (m1.index.year == period.year) & (m1.index.month == period.month)
            m1_month = m1[mask]
            if len(m1_month) < 500:
                continue   # intraday cache doesn't cover this month

            # The two sources are on DIFFERENT PRICE SCALES: yfinance daily bars
            # are dividend/split-adjusted, Alpaca 1-minute bars are raw. Compared
            # as absolute prices this measures the adjustment offset, not the
            # distribution error. Comparing each POC's *relative position* within
            # its own month's range is immune to any such level shift.
            d_lo, d_hi = float(d_month["low"].min()), float(d_month["high"].max())
            m_lo, m_hi = float(m1_month["low"].min()), float(m1_month["high"].max())
            if d_hi <= d_lo or m_hi <= m_lo:
                continue

            p_a = poc(volume_profile(d_month, bins=BINS))
            p_t = poc(_intraday_profile(m1_month, BINS, m_lo, m_hi))
            a = float(atr_series.loc[:d_month.index[-1]].iloc[-1])
            if p_a is None or p_t is None or not np.isfinite(a) or a <= 0:
                continue

            pos_a = (p_a - d_lo) / (d_hi - d_lo)
            pos_t = (p_t - m_lo) / (m_hi - m_lo)
            rows.append({
                "ticker": ticker, "month": str(period)[:7],
                # Re-express the relative disagreement in ATR on the daily scale,
                # since that is the unit zone bands are measured in.
                "displacement_atr": abs(pos_a - pos_t) * (d_hi - d_lo) / a,
            })

    if not rows:
        print("No overlapping months with enough data.")
        return

    df = pd.DataFrame(rows)
    band = 0.35
    print("\n" + "=" * 70)
    print("  POC DISPLACEMENT — uniform daily assumption vs 1-minute truth")
    print("=" * 70)
    print(f"  months compared : {len(df)} across {df.ticker.nunique()} tickers")
    print(f"  median          : {df.displacement_atr.median():.3f} ATR")
    print(f"  75th percentile : {df.displacement_atr.quantile(0.75):.3f} ATR")
    print(f"  90th percentile : {df.displacement_atr.quantile(0.90):.3f} ATR")
    print(f"  share beyond the {band} ATR band half-width: "
          f"{(df.displacement_atr > band).mean():.1%}")
    verdict = (
        "uniform assumption is fine at band resolution"
        if df.displacement_atr.median() < band
        else "levels land in the wrong band too often — rebuild profiles from intraday bars"
    )
    print(f"\n  Verdict: {verdict}")


if __name__ == "__main__":
    main()
