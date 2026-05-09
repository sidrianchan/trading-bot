"""Trading bot entry point.

Usage:
    python main.py intraday          # start intraday paper trading loop
    python main.py intraday --dry-run  # morning scan only (no orders)
    python main.py intraday-backtest # run 2023-2024 intraday backtest
    python main.py status            # show current portfolio status
    python main.py kill              # trigger emergency kill switch
    python main.py reset-kill        # reset kill switch
    python main.py save-model        # save XGBoost architecture to models/
    python main.py backtest          # run legacy walk-forward factor backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"config.yaml not found at {CONFIG_PATH.resolve()}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_dir = Path(config.get("monitor", {}).get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    level = config.get("monitor", {}).get("log_level", "INFO")
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(log_dir / "trading_bot.log", rotation="1 day", retention="30 days", level="DEBUG")


def cmd_backtest(config: dict) -> None:
    from data import get_sp500_tickers, fetch_prices, fetch_fundamentals
    from backtester.engine import WalkForwardBacktester, BacktestConfig
    from backtester.report import BacktestReport, plot_feature_importance
    from signals import (
        CompositeSignal, MomentumSignal, QualitySignal, LowVolatilitySignal,
        XGBoostRankerSignal, build_feature_history, build_target_history,
    )

    bt_cfg = config.get("backtest", {})
    strat = config.get("strategy", {})
    port = config.get("portfolio", {})
    risk = config.get("risk", {})

    mom_cfg   = strat.get("momentum", {})
    qual_cfg  = strat.get("quality", {})
    vol_cfg   = strat.get("low_volatility", {})

    engine_config = BacktestConfig(
        initial_capital=bt_cfg.get("initial_capital", 5000.0),
        transaction_cost_bps=bt_cfg.get("transaction_cost_bps", 10.0),
        momentum_lookback=mom_cfg.get("lookback_days", 126),
        skip_days=mom_cfg.get("skip_days", 21),
        top_n=strat.get("top_n", 25),
        momentum_weight=mom_cfg.get("weight", 0.70),
        quality_weight=qual_cfg.get("weight", 0.20),
        low_vol_weight=vol_cfg.get("weight", 0.10),
        max_position_size=port.get("max_position_size", 0.10),
        min_position_size=port.get("min_position_size", 0.01),
        target_volatility=port.get("target_volatility", 0.15),
        drawdown_limit=risk.get("portfolio_drawdown_limit", 0.15),
        drawdown_reset=risk.get("drawdown_reset_threshold", 0.07),
        trend_filter_days=strat.get("trend_filter", {}).get("ma_days", 200),
    )

    start     = bt_cfg.get("start_date", "2018-01-01")
    end       = bt_cfg.get("end_date", "2024-12-31")
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    top_n_mc  = config.get("universe", {}).get("top_n_market_cap")

    logger.info(f"Fetching universe for backtest ({start} → {end})")
    tickers = get_sp500_tickers()

    logger.info("Fetching fundamentals")
    fundamentals = fetch_fundamentals(tickers)

    if top_n_mc and "market_cap" in fundamentals.columns:
        top_tickers = (
            fundamentals["market_cap"].dropna().nlargest(top_n_mc).index.tolist()
        )
        logger.info(f"Filtered universe: {len(tickers)} → top {len(top_tickers)} by market cap")
        tickers = top_tickers
        fundamentals = fundamentals.loc[fundamentals.index.isin(tickers)]

    prices = fetch_prices(tickers, start=start, end=end, source="yfinance")

    # ── XGBoost ensemble ───────────────────────────────────────────────
    mom_sig = MomentumSignal(
        lookback_days=mom_cfg.get("lookback_days", 126),
        skip_days=mom_cfg.get("skip_days", 21),
    )
    qual_sig = QualitySignal()
    vol_sig  = LowVolatilitySignal(lookback_days=vol_cfg.get("lookback_days", 63))

    logger.info("Pre-computing feature history (no lookahead)…")
    feature_history = build_feature_history(prices, fundamentals, mom_sig, qual_sig, vol_sig)

    logger.info("Pre-computing forward-return targets…")
    target_history = build_target_history(prices, forward_days=21)

    ml_signal = XGBoostRankerSignal(
        feature_history=feature_history,
        target_history=target_history,
        train_window_months=24,
        gap_months=6,
    )

    # Factor composite — weights will be rescaled to 40% when ML is registered
    composite = CompositeSignal(
        momentum_weight=mom_cfg.get("weight", 0.70),
        quality_weight=qual_cfg.get("weight", 0.20),
        low_vol_weight=vol_cfg.get("weight", 0.10),
        lookback_days=mom_cfg.get("lookback_days", 126),
        skip_days=mom_cfg.get("skip_days", 21),
        vol_lookback_days=vol_cfg.get("lookback_days", 63),
        top_n=strat.get("top_n", 25),
    )
    composite.register_signal("ml", ml_signal, weight=0.60)  # 60% ML / 40% factors

    engine = WalkForwardBacktester(
        config=engine_config, fundamentals=fundamentals, signal=composite
    )
    # ──────────────────────────────────────────────────────────────────

    logger.info("Running backtest with ML ensemble…")
    portfolio_values = engine.run(prices, benchmark_ticker=benchmark)

    benchmark_values = prices[benchmark].reindex(portfolio_values.index).ffill()
    benchmark_values = (
        benchmark_values / benchmark_values.iloc[0] * engine_config.initial_capital
    )

    report = BacktestReport(portfolio_values, benchmark_values)
    report.print_summary()
    report.plot("backtest_results.png")

    # Feature importance chart
    fi = ml_signal.feature_importances
    if fi is not None:
        plot_feature_importance(fi, "feature_importance.png")
        logger.info(
            f"XGBoost: {ml_signal.train_count} retrains, "
            f"last train Spearman={ml_signal.last_spearman:.3f}"
        )
    else:
        logger.warning("XGBoost model was not trained — check xgboost installation")


def cmd_intraday(config: dict, dry_run: bool = False) -> None:
    import schedule
    import time
    from agent.loop import IntradayAgentLoop

    logger.info(f"Starting intraday loop {'(dry-run)' if dry_run else '(paper mode)'}")
    agent = IntradayAgentLoop(config)

    intra = config.get("intraday", {})
    exec_cfg    = intra.get("execution", {})
    hard_close  = exec_cfg.get("hard_close_time", "15:45")
    report_time = exec_cfg.get("report_time", "16:00")

    if dry_run:
        # Dry-run: just run the morning scan and print candidates, then exit
        agent.run_morning_scan()
        return

    schedule.every().day.at("09:25").do(agent.run_morning_scan)
    schedule.every().day.at("09:30").do(agent.start_streaming)
    schedule.every().day.at("09:45").do(agent.classify_day)
    schedule.every().day.at(hard_close).do(agent.hard_close)
    schedule.every().day.at(report_time).do(agent.run_report)

    logger.info(
        f"Schedule: scan=09:25  stream=09:30  entries=09:45  "
        f"hard_close={hard_close}  report={report_time}"
    )
    while True:
        schedule.run_pending()
        time.sleep(10)


def cmd_intraday_backtest(config: dict) -> None:
    """Run 2023-2024 intraday backtest using Alpaca historical 1-min bars."""
    from data import get_sp500_tickers, fetch_intraday_bars_range
    from backtester.intraday_engine import IntradayBacktester, IntradayBacktestConfig
    from backtester.metrics import compute_metrics
    import pandas as pd

    intra = config.get("intraday", {})
    bt = intra.get("backtest", {})

    sig    = intra.get("signals", {})
    rk     = intra.get("risk", {})
    execc  = intra.get("execution", {})

    cfg = IntradayBacktestConfig(
        initial_capital=config.get("backtest", {}).get("initial_capital", 100_000.0),
        slippage_bps=bt.get("slippage_bps", 2),
        spread_bps=bt.get("spread_bps", 1),
        risk_per_trade_pct=rk.get("risk_per_trade_pct", 0.005),
        max_concurrent=rk.get("max_concurrent_positions", 8),
        hard_close_time=execc.get("hard_close_time", "15:30"),
        entry_start_time=sig.get("entry_start_time", "10:00"),
        day_classify_time=sig.get("day_classify_time", "10:00"),
        gap_min_pct=sig.get("gap_min_pct", 1.5),
        volume_ratio_min=sig.get("premarket_volume_ratio", 2.0),
        atr_min_dollars=sig.get("atr_min_dollars", 0.50),
        orb_minutes=sig.get("orb_minutes", 15),
        vwap_std_threshold=sig.get("vwap_std_threshold", 1.5),
        vwap_stop_std=sig.get("vwap_stop_std", 2.5),
        vwap_min_dollar_deviation_pct=sig.get("vwap_min_dollar_deviation_pct", 0.015),
        vwap_confirm_reversal=sig.get("vwap_confirm_reversal", True),
        cooldown_bars_after_stop=rk.get("cooldown_bars_after_stop", 15),
        max_trades_per_ticker_per_day=rk.get("max_trades_per_ticker_per_day", 2),
        day_trend_threshold=sig.get("day_trend_threshold", 0.01),
        spy_range_threshold=sig.get("spy_range_threshold", 0.005),
        trend_day_size_mult=rk.get("trend_day_size_mult", 0.5),
        vix_range_override=sig.get("vix_range_override", 20.0),
        daily_pnl_halt_pct=rk.get("daily_pnl_halt_pct", -0.02),
        min_rr_ratio=rk.get("min_rr_ratio", 1.5),
        top_n_candidates=intra.get("candidates", {}).get("top_n", 15),
    )

    start_date = bt.get("start_date", "2023-01-01")
    end_date   = bt.get("end_date", "2024-12-31")

    # Universe: top 100 S&P 500 names + high-ATR/high-beta names guaranteed to gap
    HIGH_BETA = [
        "TSLA", "NVDA", "AMD", "META", "MSTR", "SMCI", "COIN",
        "PLTR", "RIVN", "SOFI", "HOOD", "LCID", "SNAP", "ROKU",
        "DKNG", "UPST", "AFRM", "MARA", "RIOT",
    ]
    all_sp500 = get_sp500_tickers()
    tickers = sorted(set(all_sp500[:100]) | set(HIGH_BETA) | {"SPY"})
    logger.info(f"Backtest universe: {len(tickers)} tickers ({start_date} → {end_date})")

    all_bars = fetch_intraday_bars_range(tickers, start_date, end_date)
    logger.info(f"Bars loaded for {len(all_bars)} tickers")

    engine = IntradayBacktester(cfg)
    portfolio_values, trade_log = engine.run(all_bars)

    wins  = [t for t in trade_log if t.pnl > 0]
    total = len(trade_log)
    win_rate = len(wins) / total if total else 0
    avg_win  = sum(t.pnl for t in wins) / len(wins) if wins else 0
    losses   = [t for t in trade_log if t.pnl <= 0]
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    print(f"\n{'='*54}")
    print(f"  INTRADAY BACKTEST  {start_date} → {end_date}")
    print(f"{'='*54}")
    print(f"  Total trades      : {total}")
    print(f"  Win rate          : {win_rate:.1%}")
    print(f"  Avg win           : ${avg_win:>+,.2f}")
    print(f"  Avg loss          : ${avg_loss:>,.2f}")
    print(f"  Reward:Risk       : {rr:.2f}:1")

    if total > 0:
        metrics = compute_metrics(portfolio_values)
        print(f"  CAGR              : {metrics['cagr']:.1%}")
        print(f"  Sharpe            : {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown      : {metrics['max_drawdown']:.1%}")

    if win_rate < 0.45 or rr < 1.5:
        print(f"\n  ⚠  BACKTEST GATE FAILED:")
        if win_rate < 0.45:
            print(f"     Win rate {win_rate:.1%} < 45% minimum")
        if rr < 1.5:
            print(f"     R:R {rr:.2f} < 1.5:1 minimum")
        print("     Do NOT proceed to paper trading.")
        print("     Recommend revisiting VWAP-primary strategy.")
    else:
        print(f"\n  Backtest gate PASSED. Safe to begin paper trading.")
    print(f"{'='*54}")


def cmd_paper(config: dict) -> None:
    """Legacy monthly factor paper loop — kept for reference."""
    import schedule
    import time
    from agent.loop import IntradayAgentLoop

    logger.warning("'paper' command runs the intraday loop. Use 'intraday' instead.")
    cmd_intraday(config)


def cmd_status(config: dict) -> None:
    from execution.broker import AlpacaBroker
    from risk.kill_switch import KillSwitch

    broker = AlpacaBroker()
    ks = KillSwitch()

    value = broker.get_portfolio_value()
    cash = broker.get_cash()
    positions = broker.get_positions()

    print(f"\nPortfolio Value : ${value:>10,.2f}")
    print(f"Cash            : ${cash:>10,.2f}")
    print(f"Positions       : {len(positions)}")
    print(f"Kill Switch     : {'ACTIVE ⚠' if ks.is_triggered() else 'inactive'}")
    if not positions.empty:
        print("\nPositions:")
        print(positions.to_string())


def cmd_kill() -> None:
    from risk.kill_switch import KillSwitch
    reason = input("Reason for kill switch (press Enter to confirm): ").strip() or "manual trigger"
    KillSwitch().trigger(reason)


def cmd_reset_kill() -> None:
    from risk.kill_switch import KillSwitch
    KillSwitch().reset()


def cmd_save_model(config: dict) -> None:
    """Save the latest XGBoost model snapshot to models/."""
    from pathlib import Path
    from data import get_sp500_tickers, fetch_prices, fetch_fundamentals
    from signals import (
        MomentumSignal, QualitySignal, LowVolatilitySignal,
        XGBoostRankerSignal, build_feature_history, build_target_history,
    )
    from datetime import date

    strat = config.get("strategy", {})
    mom_cfg = strat.get("momentum", {})
    vol_cfg = strat.get("low_volatility", {})
    bt_cfg = config.get("backtest", {})

    today = str(date.today())
    tickers = get_sp500_tickers()
    logger.info("Fetching data to rebuild features for model snapshot...")
    prices = fetch_prices(tickers, start=bt_cfg.get("start_date", "2018-01-01"), end=today, source="yfinance")
    fundamentals = fetch_fundamentals(tickers)

    mom_sig  = MomentumSignal(lookback_days=mom_cfg.get("lookback_days", 126), skip_days=mom_cfg.get("skip_days", 21))
    qual_sig = QualitySignal()
    vol_sig  = LowVolatilitySignal(lookback_days=vol_cfg.get("lookback_days", 63))

    feature_history = build_feature_history(prices, fundamentals, mom_sig, qual_sig, vol_sig)
    target_history  = build_target_history(prices, forward_days=21)

    ml_signal = XGBoostRankerSignal(
        feature_history=feature_history,
        target_history=target_history,
        train_window_months=bt_cfg.get("training_window_months", 24),
        gap_months=6,
    )
    # Force a train pass on current date
    ml_signal.compute(prices, fundamentals)
    ml_signal.save_architecture()
    logger.info("XGBoost model snapshot saved.")


def cli() -> None:
    config = load_config()
    setup_logging(config)

    dry_run = "--dry-run" in sys.argv

    commands = {
        "intraday":           lambda: cmd_intraday(config, dry_run=dry_run),
        "intraday-backtest":  lambda: cmd_intraday_backtest(config),
        "backtest":           lambda: cmd_backtest(config),
        "paper":              lambda: cmd_paper(config),
        "status":             lambda: cmd_status(config),
        "kill":               cmd_kill,
        "reset-kill":         cmd_reset_kill,
        "save-model":         lambda: cmd_save_model(config),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(0)

    commands[sys.argv[1]]()


if __name__ == "__main__":
    cli()
