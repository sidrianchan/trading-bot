# Strategy Decisions

## What's live

- **ETF bot** (`momentum-paper`): V4 dual momentum on TQQQ/UPRO/SOXL. Backtest 24.5% CAGR, -67.8% MDD.
- **Crypto bot** (`crypto-paper`): BTC/ETH absolute + relative momentum. Backtest 68.9% CAGR, -55.2% MDD.

## What was tried and failed (do not rebuild)

- **Intraday ORB/gap strategy**: failed gate across 3 attempts. Win rate 43–46%, R:R 1.04–1.17. Root cause: large-cap S&P 500 stocks don't move enough intraday to hit 2:1 targets.
- **Intraday VWAP mean reversion**: failed gate across 4 attempts. Win rate 20–43%, R:R 0.94–1.39. Root cause: high-beta names trend rather than revert; large-caps don't deviate enough from VWAP.
- **Intraday TA engine (RSI/MACD/S&R)**: failed gate across 6 attempts. Long win rate 29–43%, R:R 1.17–1.39. Root cause: couldn't distinguish "support holding" from "support breaking."
- **Monthly factor model (momentum/quality/low-vol)**: underperforms SPY on raw returns. Trend filter whipsaws on monthly rebalance. Abandoned in favor of leveraged ETF momentum.

## Next review date

June 30, 2026 — 30-day paper trading results. Live capital discussion only after this date.

## Intraday bot status

Config exists in codebase but NO agent loop deployed. Do not build the agent loop until a strategy passes the backtest gate. Current VWAP config FAILED the gate.
