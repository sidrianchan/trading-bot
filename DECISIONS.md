# Strategy Decisions

## What's live

- **ETF bot** (`momentum-paper`): V4 dual momentum on TQQQ/UPRO/SOXL. Backtest 24.5% CAGR, -67.8% MDD.
- **Crypto bot** (`crypto-paper`): BTC/ETH absolute + relative momentum. Backtest 68.9% CAGR, -55.2% MDD.

## What was tried and failed (do not rebuild)

- **Intraday ORB/gap strategy**: failed gate across 3 attempts. Win rate 43–46%, R:R 1.04–1.17. Root cause: large-cap S&P 500 stocks don't move enough intraday to hit 2:1 targets.
- **Intraday VWAP mean reversion**: failed gate across 4 attempts. Win rate 20–43%, R:R 0.94–1.39. Root cause: high-beta names trend rather than revert; large-caps don't deviate enough from VWAP.
- **Intraday TA engine (RSI/MACD/S&R)**: failed gate across 6 attempts. Long win rate 29–43%, R:R 1.17–1.39. Root cause: couldn't distinguish "support holding" from "support breaking."
- **Monthly factor model (momentum/quality/low-vol)**: underperforms SPY on raw returns. Trend filter whipsaws on monthly rebalance. Abandoned in favor of leveraged ETF momentum.

- **Swing S/R v7 — killed at the measurement stage, 2026-07-21.** Before building a
  seventh S/R strategy, we measured the premise directly: does level strength predict
  whether support holds? Answer: **a real but economically insufficient relationship.**

  Study: 32,475 zone touches, 135 tickers, 2010–2019, zones built causally from a
  trailing 500-bar window (`scripts/level_calibration.py`). Pre-registered kill criteria,
  2 of 3 failed:
  - Tercile spread (D8-10 minus D1-3): **+3.4pp** (needed ≥6pp) — FAIL, though p<0.0001
  - Spearman rho(decile, hold rate): **+0.673** (needed ≥0.60) — PASS
  - Top decile vs random-band placebo: **+2.4pp** (needed ≥5pp) — FAIL

  Hold rate runs 60.5% (D1) → 65.4% (D8) against a **59.3% random-band baseline**. The
  entire usable signal is ~5pp on a near-coin-flip event, before costs.

  Three findings worth keeping:
  1. **Decile 10 inverts** — the very strongest zones (61.7%) underperform deciles 7–9
     (~65%). Consistent with the idea that a level tested repeatedly and recently is
     being ground down rather than defended. This argues against *both* naive touch
     counting and the reaction-weighted refinement, at the extreme.
  2. **Time-to-target does not vary with strength** — median 2 bars to 1R in every
     decile. That independently kills the "expected R per day" ranking/exit idea, which
     needs `days_to_target` to differ across setups.
  3. **Regime moves the number about as much as strength does** (uptrend 64.7% vs
     downtrend 61.0%), and it is far cheaper to compute.

  Prior attempts also contained three measurement bugs, now documented in
  `signals/price_action/support_resistance.py` and `signals/setup.py`: the `$1`
  round-number ladder diluted and evicted real pivots; `t2 = resistances[0]` placed
  target-2 inside target-1; and daily levels were built from `_close_only_to_ohlc`,
  which synthesizes H/L from closes with volume=0. Those bugs mean the old
  win-rate/R:R numbers measured something other than the intended strategy — but fixing
  them does not rescue the premise, which is what this study establishes.

  **Do not build attempt #8 on support/resistance level strength.** Reusable output:
  `data/market.py::fetch_daily_ohlcv` (real daily OHLCV+volume),
  `signals/price_action/volume_profile.py`, `signals/price_action/level_quality.py`,
  and the calibration harness. 2020–2025 was deliberately left unexamined.

## Next review date

June 30, 2026 — 30-day paper trading results. Live capital discussion only after this date.

## Intraday bot status

Config exists in codebase but NO agent loop deployed. Do not build the agent loop until a strategy passes the backtest gate. Current VWAP config FAILED the gate.
