# Automated Trading Bot & Agent System

Autonomous Python trading system implementing dual-momentum strategies across crypto and leveraged
ETF markets, with full production infrastructure for live paper-trading deployment — and an audit
trail showing where its own results turned out to be wrong.

## Status

**Paper trading only. Not run with real capital, and not intended to be.**
This is a research and engineering project. The headline backtest figures this README used to
advertise did not survive validation; what follows is what actually held up.

## Findings

Reproduce with `python scripts/walk_forward_validation.py`. Every figure below is stitched
out-of-sample, charged real trading costs, and rebalance timing is tranched rather than chosen.

**ETF strategy** — walk-forward OOS 2016-03 → 2024-12, 5 bps/leg:

| | CAGR | Sharpe | Vol | Max drawdown |
|---|---|---|---|---|
| **V4 dual momentum** | **+27.7%** | **0.75** | 48.5% | **−52.5%** |
| 74% TQQQ, never traded (same vol) | +35.1% | 0.87 | 48.5% | −68.8% |
| TQQQ buy & hold | +41.8% | 0.87 | 65.4% | −81.7% |
| SPY buy & hold | +15.1% | 0.88 | 17.8% | −33.7% |

The strategy does **not** beat a static position in the thing it holds. Against a 74% TQQQ position
held at the same volatility with no trading at all, it gives up **7.4 points of annual return** and
buys **16.2 points less drawdown**. That trade may be worth making. It is not alpha, and it should
not be described as alpha.

**Crypto strategy — RETIRED 2026-09-05** — walk-forward OOS 2021 → 2024, 25 bps/leg:

| | CAGR | Sharpe | Vol | Max drawdown |
|---|---|---|---|---|
| **BTC/ETH momentum** | **+12.4%** | **0.62** | 23.2% | **−40.0%** |
| 38% BTC, never traded (same vol) | +16.6% | 0.78 | 23.2% | −38.9% |
| BTC buy & hold | +33.6% | 0.78 | 61.9% | −76.6% |

Lower return, no better drawdown, 34 legs a year in fees. **This strategy does not work.** It is
retired: `crypto.enabled: false` in `config.yaml`, its registry record is marked `retired`, and
`python main.py crypto-paper` refuses to start and prints the table above. The backtest still runs —
measuring a dead strategy is how it stays dead.

## What the earlier numbers were, and why they were wrong

Previous versions of this README claimed 24.5% and 68.9% CAGR, and a walk-forward out-of-sample
figure of 91.1%. Those numbers reproduce exactly from the code as it stood. They were wrong for
three reasons, found by auditing my own repository on 2026-09-05:

**1. Rebalance timing was a fitted parameter nobody had noticed.** The crypto strategy evaluated on
Mondays. Changing only that:

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| **91.1%** | 86.5% | 59.9% | 11.6% | 14.8% | 1.2% | 19.3% |

Monday was the best of seven, and the ETF strategy's month-end rebalance ranked 1 of 11 day-offsets.
A walk-forward only protects the parameters it re-selects; timing was never one of them.

**2. Trading was free.** `config.yaml` assumed Alpaca charges no commission. True for equities,
false for crypto — the real schedule is 0.15% maker / 0.25% taker. At ~52 legs a year that is
**12.2% annually**. The crypto strategy loses money on every weekday once it is charged.

**3. No benchmark was ever printed.** A 91.1% CAGR is not a result until BTC's 33.6% over the same
window sits next to it. Across all 525 parameter combinations of the crypto strategy at real costs,
the **median is −9.4% a year** and only **3.4% beat BTC buy-and-hold**. The reported figure was
close to the maximum of that distribution.

## What changed in the code

- `backtester/costs.py` — one cost model, used by both engines. **Real costs are the default**; a
  free backtest has to be requested by name.
- `run_crypto_tranched` / `run_dual_momentum_tranched` — run one sub-account per rebalance day at
  1/N capital and combine. Removes timing luck instead of harvesting it. Directly implementable
  live.
- `scripts/walk_forward_validation.py` — re-selects **every** free parameter inside each training
  window, charges costs, tranches timing, and prints each benchmark plus a **volatility-matched
  static position in that benchmark**. That last row is the one that matters: any strategy can beat
  a benchmark by taking more risk, or beat it on risk by holding cash. The real question is whether
  it beats the same asset held at the same volatility with no trading at all.
- Validation gates are now benchmark-relative (`Sharpe > TQQQ buy & hold`,
  `CAGR > BTC buy & hold`), not absolute thresholds that a levered ETF's beta clears on its own.
- `tests/test_costs.py` — 17 tests covering the two bugs that caused all of this.

## Research method

The part of this repo worth reading is `DECISIONS.md`. It records **eleven strategy premises tested
and rejected** across ~246,000 measured events, each against kill criteria written down before
measurement, with matched placebo controls (random-band and random-entry benchmarks) and a sealed
2020–2025 holdout. Six were killed before any strategy code was written.

That method was applied rigorously to the strategies that failed and, until this audit, not to the
two that passed.

## Features

- Live execution via Alpaca API, deployed on DigitalOcean (24/7 paper trading)
- Automated rebalancing with position sizing, circuit breakers and kill-switch logic
- Telegram alerts for trades and system health
- Walk-forward validation with cost modelling, timing tranches and benchmark-relative gates
- Strategy-evolution agent with human-gated promotion (`evolve/`) — **currently disabled by
  judgment**: an LLM proposing candidates into an automated validator multiplies trials faster than
  they can be audited, which is the exact failure documented above. Do not enable until the trial
  counter is global and persistent and the gate is a Deflated Sharpe Ratio.

## Tech stack

Python 3.11+, pandas, NumPy, SciPy, Alpaca API, yfinance, DigitalOcean, Telegram Bot API, pytest.

## Running it

```bash
pip install -e ".[dev]"
pytest                                        # 215 tests
python scripts/walk_forward_validation.py     # the numbers above, ~90s
```
