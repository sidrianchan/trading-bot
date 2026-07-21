"""Screen liquid US equities for high volatility + high volume (high risk/reward).

Analysis only. Ranks a candidate set by annualized realized volatility and
median daily dollar volume, then selects a tradeable high-vol basket (must be
BOTH volatile AND liquid — high vol without liquidity is untradeable).

Run: .venv/bin/python scripts/stock_screener.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# Well-known liquid, high-beta / high-volatility US names (archetypal HRHR).
CANDIDATES = [
    "TSLA", "NVDA", "AMD", "COIN", "MSTR", "PLTR", "SMCI", "MARA", "RIOT",
    "SOFI", "AFRM", "RIVN", "NIO", "GME", "AMC", "SHOP", "NET", "SNOW",
    "CRWD", "DKNG", "ROKU", "UPST", "MU", "AVGO", "META", "AMZN", "NFLX",
    "ARM", "DELL", "CVNA", "HOOD", "UBER", "ON", "ENPH",
]

START = "2022-01-01"
END = "2025-12-31"


def screen() -> pd.DataFrame:
    raw = yf.download(CANDIDATES, start=START, end=END, auto_adjust=True, progress=False, threads=False)
    close = raw["Close"]
    volume = raw["Volume"]

    rows = []
    for t in CANDIDATES:
        c = close[t].dropna()
        v = volume[t].reindex(c.index)
        if len(c) < 250:
            continue
        rets = c.pct_change().dropna()
        ann_vol = float(rets.std() * np.sqrt(252))
        dollar_vol = float((c * v).median())  # median daily $ traded
        total_ret = float(c.iloc[-1] / c.iloc[0] - 1.0)
        max_dd = float((c / c.cummax() - 1.0).min())
        rows.append({
            "ticker": t,
            "ann_vol": ann_vol,
            "median_$vol_M": dollar_vol / 1e6,
            "total_ret": total_ret,
            "max_dd": max_dd,
            "n_days": len(c),
            "start": c.index[0].date().isoformat(),
        })

    df = pd.DataFrame(rows).set_index("ticker")
    # liquidity floor: median daily dollar volume >= $300M (easily tradeable)
    liquid = df[df["median_$vol_M"] >= 300].copy()
    liquid = liquid.sort_values("ann_vol", ascending=False)
    return df, liquid


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    full, liquid = screen()
    print("\n=== all candidates (sorted by annualized vol) ===\n")
    print(full.sort_values("ann_vol", ascending=False).to_string())
    print("\n=== LIQUID high-vol basket (median $vol >= $300M/day, top by vol) ===\n")
    print(liquid.to_string())
    basket = list(liquid.head(8).index)
    print(f"\nSELECTED BASKET (top 8 high-vol & liquid): {basket}")


if __name__ == "__main__":
    main()
