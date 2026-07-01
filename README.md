# Automated Trading Bot & Agent System

Autonomous Python trading system implementing dual-momentum strategies across crypto and leveraged ETF markets, with full production infrastructure for live paper-trading deployment.

## Performance (Backtested)
- **ETF Strategy (2010–2024):** 24.5% CAGR, 0.73 Sharpe
- **Crypto Strategy (2018–2024):** 68.9% CAGR, 1.33 Sharpe
- Validated using walk-forward testing across 14 years of historical data

## Features
- Live execution via Alpaca API, deployed on DigitalOcean (24/7 paper trading)
- Automated rebalancing (monthly/weekly) with position sizing and risk management
- Circuit breakers and kill-switch logic for risk control
- Telegram alerts for trade notifications and system monitoring
- Rigorous backtesting framework with win rate and risk/reward validation gates, tested across 6+ strategy configurations

## Tech Stack
Python, Alpaca API, pandas, NumPy, DigitalOcean, Telegram Bot API

## Status
Currently running in paper-trading mode. Not deployed with live capital.
