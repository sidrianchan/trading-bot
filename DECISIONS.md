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

- **Swing S/R breakdown-short — killed at the measurement stage, 2026-07-21.** The v7
  study surfaced one anomaly worth chasing: the strongest zones (decile 10) *held less*
  than moderate ones, suggesting obvious support is swept for liquidity. We probed the
  short-side mirror directly: when a strong support breaks down cleanly, does price flush?
  Answer: **no relationship at all.** (`scripts/breakdown_calibration.py`.)

  Study: 8,143 clean breakdowns, 135 tickers, 2010–2019. Observation moved from the touch
  to the break; strength sealed at the last rebuild, ATR/volume sealed at b-1, entry at the
  open after the break. Outcome = flush (hit 1R down before a close back above hi+0.2·ATR)
  vs sweep (that reclaim within 5 bars) vs neither. Matched control = a structureless ghost
  band of identical width placed 0.5–5 ATR below, broken under identical rules. Kill
  criteria, 2 of 3 failed:
  - Tercile spread flush(D8-10) − flush(D1-3): **+0.0pp**, p=0.98 — FAIL (dead flat)
  - D10 flush (47.0%) − ghost-band flush (43.1%): **+3.8pp** (needed ≥5pp) — FAIL
  - D10 flush (47.0%) ≥ D10 sweep (16.3%): PASS (breaks aren't *primarily* bear traps)

  Spearman rho(decile, flush) = **−0.152**: flush rate, if anything, *falls* slightly with
  strength. The base rate of a flush after *any* clean break is ~43%; zone strength adds at
  most ~4pp and is non-monotone. The v7 hold-side anomaly does **not** invert into a
  tradeable breakdown short — strong support breaking flushes no more reliably than a random
  band breaking. Regime is again nearly flat (uptrend flush 45.1% vs downtrend 43.5%).

  **Do not build a breakdown-short on level strength either.** Same reusable harness;
  2020–2025 remains sealed. Two S/R premises (hold, break) now measured and killed before a
  single line of strategy code — the point of probing first.

- **Trend-pullback / dynamic mean-reversion — killed at the measurement stage, 2026-07-21.**
  Pivoted off *static* horizontal levels to a *dynamic* mean: in a strict structural uptrend
  (20>50>200 EMA held ≥10 bars at t-1), does a pullback that touches the rising 20-EMA bounce
  a full symmetric 1R (±1·ATR20), and does it bounce *more* the more "stretched" the trend was
  first (trend elasticity)? Answer: **no edge, and no gradient.** (`scripts/trend_pullback_calibration.py`.)

  Study: 37,906 touches, 135 tickers, 2010–2019. Trigger line = 20-EMA as of t-1 (touch can't
  move the line it touched); stretch = max ATR-distance above the mean over the leg since the
  last touch, sealed at t-1; ATR sized at t-1; entry at the open after the touch; intrabar
  target+stop ties scored as losses (conservative). Matched control = a **Ghost Mean**: the
  20-EMA shifted a random 0.5–1.5 ATR (either side, ±0.2 dead-zone excluded), touched and
  resolved under identical regime/1R rules — the base rate of buying any random dip in the same
  name at the same time. Pre-registered kill criteria, **all 3 failed:**
  - Precision Lift: real bounce 53.5% − ghost 54.1% = **−0.6pp** (needed ≥5pp) — FAIL (negative)
  - Elasticity Gradient: High − Low tercile = **−0.8pp**, p=0.20 (needed ≥5pp, p<0.05) — FAIL
  - Momentum: median 3 bars to 1R vs ghost 2 bars — FAIL (slower, not faster)

  The real bounce rate sits *below* the drift-neutral ghost: the apparent mean-reversion is
  entirely bull-market drift, and the most-stretched pullbacks bounce slightly *less*, not more.
  Methodological note: the literal touch rule (low ≤ EMA20[t-1] ≤ high) fires on frequent daily
  straddles of the EMA, so ~2/3 of touches have ~0 stretch — the Stretch axis is degenerate for
  most of the sample. But the conclusion is robust to this: even the genuinely-stretched top of
  the High tercile (up to ~13 ATR) shows no lift. Reviving the Stretch hypothesis would require
  a *definition* change (de-bounce consecutive touches / require a minimum prior excursion), not
  a code fix — not pursued, because the overall placebo-controlled lift is already negative.

  **Do not build a 20-EMA pullback long on trend stretch.** Same reusable harness; 2020–2025
  remains sealed. Three premises (static hold, static break, dynamic pullback) now measured and
  killed before any strategy code.

- **TA confluence ("does stacking indicators raise the win rate?") — killed at the measurement
  stage, 2026-07-21.** The recurring question after three single-variable kills: does *combining*
  TA conditions rescue the pullback entry? Probed directly — and note the *raw* win rate is a
  trap, because in a drifting-up market more filters = more selectivity = higher raw rate *and*
  higher base rate together. So we measured lift over a **locally-matched placebo** (a random
  same-year bar in the same name). (`scripts/confluence_calibration.py`.)

  Study: 59,081 pullback touches, 126 tickers, **widened to 2000–2019** (adds dot-com + GFC bear
  regimes; survivorship caveat: current S&P 500 membership applied back to 2000, so the cohort is
  upward-biased — read the cross-bucket gradient, not the absolute rate). Each touch scored 0–5 by
  pre-registered sealed-at-t-1 conditions: trend maturity (stack ≥30 bars), RSI≤55, MACD hist ≥0,
  S/R-zone confluence (strength ≥0.4), controlled pullback (stretch ≤3 ATR). Kill criteria, 2 of 3
  failed (3rd is a selectivity guard):
  - Top-bucket (4–5 conds) lift vs matched ghost: **−1.5pp, p=0.02** (needed ≥+5pp) — FAIL, and
    *significantly negative*: the most-confluent setups slightly **underperform** a random
    same-year entry.
  - Lift rises with confluence: **rho=−0.50, gradient −0.0pp** (needed ≥0.6, ≥5pp) — FAIL.
  - Selectivity mirage guard (raw up but lift flat): PASS — raw rate is *also* flat/down, so it
    isn't even the usual selectivity illusion; there is simply no signal.

  **Every** confluence level has negative placebo-adjusted lift (−0.3pp to −2.9pp). A 20-ticker
  smoke slice showed a spurious monotone gradient (rho=+1.0) that **vanished at full n** — a
  textbook small-sample artefact, and the reason the full run is the decider. Combining these TA
  conditions does not create edge; it slightly destroys it. **Do not build a confluence/multi-
  factor TA entry.** Four premises now measured and killed before a line of strategy code. Same
  harness; 2020→now still sealed for a one-shot final validation.

- **Cross-sectional relative strength (the "Beach Ball" thesis) — killed at the measurement
  stage, 2026-07-21.** Officially pivoted off single-stock price-action geometry (four kills) to
  an *orthogonal, cross-sectional* signal: a stock's relative-strength rank against its peers.
  Thesis: when SPY makes a local pullback and reverses, do top-decile-RS (D10) names snap back
  (win a symmetric 1R) more than market-beta (D5) and laggards (D1)? **Verdict: STOP — all four
  gates failed.** (`scripts/relative_strength_calibration.py`.)

  Study: 108,597 events, 126 tickers, 2000–2019. SPY trigger = a ≥1.5·ATR20 drawdown off the
  10-bar high (at t-1) followed by a reversal bar on day t (green close OR close > prior high).
  On each trigger, every active name is ranked cross-sectionally by trailing-10-bar log excess
  return vs SPY (all at t-1) into deciles, entered at the open of t+1, resolved to a symmetric
  ±1·ATR20(t-1) 1R with gap-aware forward resolution (gap-open checked before intrabar high/low;
  same-bar collision = conservative loss). Kill criteria, **4 of 4 failed:**
  - Gradient win(D10)−win(D1): **+0.8pp**, p=0.21 (needed ≥8pp, p<0.01) — FAIL.
  - Lift over beta win(D10)−win(D5): **−0.2pp** (needed ≥5pp) — FAIL.
  - Speed: median 3 bars (D10) vs 2 (D5) — FAIL (slower).
  - Monotonicity Spearman rho(decile, win): **+0.36** (needed ≥0.70) — FAIL.

  Win rates are **dead flat across all ten deciles (52.4–53.5%)**; RS rank carries no reversal
  edge, and the uniform ~53% is just upward drift on a symmetric long. Statistical caveat recorded
  in the script and honoured here: events are heavily cross-sectionally clustered (one SPY reversal
  fires hundreds of simultaneous events; reversals cluster in time), so the z p-values are
  anti-conservative — but that only makes a *passing* gate suspect, and nothing passed. **Do not
  build a relative-strength reversal setup.** Five premises now measured and killed before any
  strategy code — the first cross-sectional one included. Same seal; 2020→now still reserved for a
  single final validation.

## Next review date

June 30, 2026 — 30-day paper trading results. Live capital discussion only after this date.

## Intraday bot status

Config exists in codebase but NO agent loop deployed. Do not build the agent loop until a strategy passes the backtest gate. Current VWAP config FAILED the gate.
