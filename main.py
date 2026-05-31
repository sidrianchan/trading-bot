"""Trading bot entry point.

Usage:
    python main.py intraday          # start intraday paper trading loop
    python main.py intraday --dry-run  # morning scan only (no orders)
    python main.py intraday-backtest # run 2023-2024 intraday backtest
    python main.py crypto-backtest   # run BTC/ETH crypto momentum backtest
    python main.py crypto-paper --dry-run  # show current crypto signal without orders
    python main.py crypto-paper      # start BTC/ETH crypto paper loop
    python main.py status --bot all  # show current portfolio/bot status
    python main.py kill              # trigger emergency kill switch
    python main.py reset-kill        # reset kill switch
    python main.py save-model        # save XGBoost architecture to models/
    python main.py backtest          # run legacy walk-forward factor backtest
    python main.py momentum-backtest # run V4 leveraged ETF dual-momentum backtest
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
    daily_dir = log_dir / "daily"
    daily_dir.mkdir(exist_ok=True)
    level = config.get("monitor", {}).get("log_level", "INFO")
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
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

    import pandas as pd
    report = BacktestReport(portfolio_values, benchmark_values)
    report.print_summary()
    report.plot("backtest_results.png")
    pd.DataFrame({"strategy": portfolio_values, "spy": benchmark_values}).to_csv(
        "/tmp/backtest_equity.csv"
    )

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


def cmd_status(config: dict, bot: str = "all") -> None:
    from execution.broker import AlpacaBroker
    from risk.kill_switch import KillSwitch
    from signals.dual_momentum import V4Config

    broker = AlpacaBroker()
    ks = KillSwitch()
    positions = broker.get_positions()

    print(f"\nAccount Value   : ${broker.get_portfolio_value():>10,.2f}")
    print(f"Account Cash    : ${broker.get_cash():>10,.2f}")
    print(f"Kill Switch     : {'ACTIVE' if ks.is_triggered() else 'inactive'}")

    if bot in {"etf", "all"}:
        etf_cfg = V4Config()
        etf_symbols = list(etf_cfg.risk_on) + list(etf_cfg.risk_off_candidates)
        etf_positions = broker.get_positions_for_symbols(etf_symbols)
        etf_value = 0.0 if etf_positions.empty else float(etf_positions["market_value"].sum())
        print(f"\nETF Bot")
        print(f"Capital Base    : ${config.get('momentum_paper', {}).get('capital', 70000):>10,.2f}")
        print(f"Scoped Positions: {len(etf_positions)}")
        print(f"Scoped Value    : ${etf_value:>10,.2f}")
        if not etf_positions.empty:
            print(etf_positions[["raw_symbol", "qty", "market_value", "unrealized_pnl"]].to_string())

    if bot in {"crypto", "all"}:
        crypto_cfg = config.get("crypto", {})
        crypto_symbols = list(crypto_cfg.get("universe", ["BTC/USD", "ETH/USD"])) + [crypto_cfg.get("stable", "USDC/USD")]
        crypto_positions = broker.get_positions_for_symbols(crypto_symbols)
        crypto_value = 0.0 if crypto_positions.empty else float(crypto_positions["market_value"].sum())
        print(f"\nCrypto Bot")
        print(f"Capital Base    : ${crypto_cfg.get('capital', 30000):>10,.2f}")
        print(f"Scoped Positions: {len(crypto_positions)}")
        print(f"Scoped Value    : ${crypto_value:>10,.2f}")
        if not crypto_positions.empty:
            print(crypto_positions[["raw_symbol", "qty", "market_value", "unrealized_pnl"]].to_string())

    if bot not in {"etf", "crypto", "all"}:
        print("Unknown --bot value. Use etf, crypto, or all.")

    if bot == "all" and not positions.empty:
        print("\nAll Positions:")
        print(positions.to_string())


def cmd_kill() -> None:
    from risk.kill_switch import KillSwitch
    reason = input("Reason for kill switch (Enter to confirm): ").strip() or "manual trigger"
    KillSwitch().trigger(reason)


def cmd_reset_kill() -> None:
    from risk.kill_switch import KillSwitch
    KillSwitch().reset()


def cmd_intraday(config: dict, dry_run: bool = False) -> None:
    logger.error(
        "Live intraday loop is disabled until Phase F. "
        "Run `python main.py ta-backtest` first; the gate must pass before paper trading resumes."
    )
    sys.exit(2)


def cmd_ta_backtest(config: dict) -> None:
    try:
        from backtester.ta_engine import run_ta_backtest  # type: ignore
    except ImportError:
        logger.error("TA backtester not implemented yet — arrives in Phase D.")
        sys.exit(2)
    run_ta_backtest(config)


def _crypto_config_from_yaml(config: dict):
    from signals.crypto_momentum import CryptoMomentumConfig

    crypto = config.get("crypto", {})
    abs_cfg = crypto.get("absolute_momentum", {})
    rel_cfg = crypto.get("relative_momentum", {})
    cb_cfg = crypto.get("circuit_breaker", {})
    return CryptoMomentumConfig(
        capital=float(crypto.get("capital", 30000)),
        universe=tuple(crypto.get("universe", ["BTC/USD", "ETH/USD"])),
        stable=crypto.get("stable", "USDC/USD"),
        abs_lookback=int(abs_cfg.get("lookback_days", 84)),
        abs_skip=int(abs_cfg.get("skip_days", 14)),
        rel_lookback=int(rel_cfg.get("lookback_days", 7)),
        rel_skip=int(rel_cfg.get("skip_days", 14)),
        cb_threshold=float(cb_cfg.get("max_drawdown_from_peak", -0.40)),
    )


def _print_metric_table(title: str, df) -> None:
    formatters = {
        "ending_value": "${:,.0f}".format,
        "total_return": "{:.1%}".format,
        "cagr": "{:.1%}".format,
        "max_drawdown": "{:.1%}".format,
        "sharpe": "{:.2f}".format,
    }
    print(f"\n{title}")
    print(df.to_string(formatters={k: v for k, v in formatters.items() if k in df.columns}))


def cmd_crypto_backtest(config: dict) -> None:
    from backtester.crypto_momentum import fetch_crypto_prices, run_crypto_backtest

    cfg = _crypto_config_from_yaml(config)
    prices = fetch_crypto_prices(start="2018-01-01", end="2025-01-01")
    result = run_crypto_backtest(prices, cfg)
    print("\nBTC/ETH CRYPTO MOMENTUM BACKTEST")
    print(f"Rules: BTC abs {cfg.abs_lookback}d skip {cfg.abs_skip}d; "
          f"relative {cfg.rel_lookback}d skip {cfg.rel_skip}d; CB {cfg.cb_threshold:.0%}")
    _print_metric_table("Full period 2018-2024", result.summary)
    _print_metric_table("Walk-forward split", result.windows)
    gates = result.gates.copy()
    gates["value"] = [
        f"{value:.2f}" if gate == "Sharpe > 0.6" else f"{value:.1%}"
        for gate, value in gates["value"].items()
    ]
    gates["passed"] = gates["passed"].map(lambda x: "PASS" if x else "FAIL")
    print("\nBacktest gates")
    print(gates.to_string())
    if not result.passed:
        print("\nBacktest gate FAILED. Do NOT proceed to paper trading.")
        sys.exit(2)
    print("\nBacktest gate PASSED.")


def cmd_momentum_backtest(config: dict) -> None:
    from backtester.dual_momentum import fetch_etf_prices, run_dual_momentum_backtest
    from signals.dual_momentum import V4Config

    cfg = V4Config()
    capital = float(config.get("momentum_paper", {}).get("capital", 70_000.0))
    print("\nFetching ETF prices (TQQQ / UPRO / SOXL / TLT / SPY) …")
    prices = fetch_etf_prices(start="2010-03-01", end="2024-12-31")
    result = run_dual_momentum_backtest(prices, cfg, initial_capital=capital)
    print("\nV4 DUAL-MOMENTUM LEVERAGED ETF BACKTEST  (2010-2024)")
    print(f"Rules: SPY {cfg.abs_lookback}d abs filter skip {cfg.skip}d; "
          f"3m relative rank skip {cfg.skip}d; CB {cfg.cb_threshold:.0%}; "
          f"re-entry after {cfg.reentry_confirmation_months} risk-on months")
    _print_metric_table("Full period 2010-2024", result.summary)
    _print_metric_table("Sub-period windows", result.windows)
    gates = result.gates.copy()
    gates["value"] = [
        f"{v:.2f}" if "Sharpe" in gate else f"{v:.1%}"
        for gate, v in gates["value"].items()
    ]
    gates["passed"] = gates["passed"].map(lambda x: "PASS" if x else "FAIL")
    print("\nBacktest gates")
    print(gates.to_string())
    status = "PASSED" if result.passed else "FAILED"
    print(f"\nBacktest gate {status}.")


def cmd_crypto_paper(config: dict, dry_run: bool = False) -> None:
    from agent.crypto_loop import CryptoPaperLoop

    loop = CryptoPaperLoop(config)
    if dry_run:
        loop.dry_run()
        return
    loop.run()


def cmd_health(config: dict) -> None:
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from execution.broker import AlpacaBroker
    from risk.kill_switch import KillSwitch
    from signals.dual_momentum import V4Config

    broker = AlpacaBroker()
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    log_dir = Path("logs") / "health"
    log_dir.mkdir(parents=True, exist_ok=True)
    etf_symbols = list(V4Config().risk_on) + list(V4Config().risk_off_candidates)
    crypto_cfg = config.get("crypto", {})
    crypto_symbols = list(crypto_cfg.get("universe", ["BTC/USD", "ETH/USD"])) + [crypto_cfg.get("stable", "USDC/USD")]
    etf_positions = broker.get_positions_for_symbols(etf_symbols)
    crypto_positions = broker.get_positions_for_symbols(crypto_symbols)
    lines = [
        f"Health check {now.isoformat()}",
        f"Account value: ${broker.get_portfolio_value():,.2f}",
        f"Cash: ${broker.get_cash():,.2f}",
        f"Kill switch: {'ACTIVE' if KillSwitch().is_triggered() else 'inactive'}",
        f"ETF scoped positions: {len(etf_positions)}",
        f"Crypto scoped positions: {len(crypto_positions)}",
    ]
    path = log_dir / f"{now.date().isoformat()}.log"
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


def cmd_momentum_paper(config: dict, dry_run: bool = False) -> None:
    """Live monthly paper loop for V4 dual-momentum on leveraged ETFs.

    --dry-run : compute today's signal once, print details, no orders submitted.
    live      : schedule daily check at 15:30 ET; on the last trading day of
                each month, rebalance via MOC (TimeInForce.CLS) orders.

    State (peak / CB flags) persists at logs/momentum_state.json.

    See signals/dual_momentum.py for warnings + known failure modes.
    """
    import json
    import time
    from datetime import datetime, date
    from pathlib import Path
    from zoneinfo import ZoneInfo

    import pandas as pd
    import yfinance as yf

    from signals.dual_momentum import (
        V4Config, V4State, compute_signal, state_from_dict, state_to_dict,
    )
    from execution.broker import AlpacaBroker
    from execution.orders import Order, Side
    from risk.kill_switch import KillSwitch

    cfg = V4Config()
    capital = float(config.get("momentum_paper", {}).get("capital", 70_000.0))
    state_path = Path("logs") / "momentum_state.json"
    state_path.parent.mkdir(exist_ok=True)
    log_path = Path("logs") / "paper_loop.log"
    logger.add(log_path, rotation="1 day", retention="30 days", level="INFO")

    trade_universe = list(cfg.risk_on) + list(cfg.risk_off_candidates)
    universe = trade_universe + [cfg.benchmark_filter]

    def load_state() -> V4State:
        if state_path.exists():
            with open(state_path) as f:
                raw_state = json.load(f)
            state = state_from_dict(raw_state)
            if "cash_value" not in raw_state and state.last_target is None and state.peak >= capital:
                state.peak = capital
                state.cash_value = capital
            return state
        return V4State(peak=capital, cash_value=capital)

    def save_state(s: V4State) -> None:
        with open(state_path, "w") as f:
            json.dump(state_to_dict(s), f, indent=2)

    def fetch_history() -> pd.DataFrame:
        # Need ~9 months for 6m + skip + buffer; pull 1 year for safety
        end = pd.Timestamp.now(tz="UTC").normalize()
        start = end - pd.Timedelta(days=365)
        raw = yf.download(universe, start=start.date(), end=end.date(),
                          auto_adjust=True, progress=False)
        return raw["Close"].dropna(how="all").ffill()

    def is_last_trading_day_of_month(today: date, broker: AlpacaBroker) -> bool:
        """Use Alpaca calendar to check if today is the last trading day this month."""
        try:
            from alpaca.trading.requests import GetCalendarRequest
            req = GetCalendarRequest(
                start=date(today.year, today.month, 1),
                end=date(today.year + (today.month // 12), (today.month % 12) + 1, 1),
            )
            cal = broker._client.get_calendar(req)
            month_days = [c.date for c in cal if c.date.month == today.month]
            return bool(month_days) and today >= month_days[-1]
        except Exception as exc:
            logger.warning(f"Alpaca calendar fetch failed: {exc}; falling back to last-weekday heuristic")
            # Fallback: roughly last business day (won't catch holidays)
            from calendar import monthrange
            last_dom = monthrange(today.year, today.month)[1]
            for d in range(last_dom, 0, -1):
                if date(today.year, today.month, d).weekday() < 5:
                    return today.day == d
            return False

    def scoped_value(broker: AlpacaBroker, state: V4State, dry: bool) -> float:
        if dry:
            return capital
        positions = broker.get_positions_for_symbols(trade_universe)
        position_value = 0.0 if positions.empty else float(positions["market_value"].sum())
        if position_value > 0:
            return position_value
        return state.cash_value or capital

    def rebalance_once(broker: AlpacaBroker, dry: bool) -> None:
        ks = KillSwitch()
        if ks.is_triggered():
            logger.critical("Kill switch is ACTIVE. Skipping rebalance.")
            return

        state = load_state()
        prices = fetch_history()
        portfolio_value = scoped_value(broker, state, dry)
        # On first run, seed peak from initial capital so CB has something to compare to
        if state.peak == 0.0:
            state.peak = portfolio_value

        signal, new_state = compute_signal(prices, state, portfolio_value, cfg)

        # Print signal details (always — useful in both modes)
        print(f"\n{'='*72}")
        print(f"  V4 DUAL MOMENTUM — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {'(DRY RUN)' if dry else '(LIVE)'}")
        print(f"{'='*72}")
        print(f"  ETF scoped value    : ${portfolio_value:>12,.2f}")
        print(f"  Tracked peak        : ${signal.peak:>12,.2f}")
        print(f"  Drawdown from peak  : {signal.drawdown:>12.2%}")
        print(f"  CB threshold        : {cfg.cb_threshold:>12.0%}")
        print(f"  CB status           : {signal.cb_status:>12}")
        if new_state.in_cb:
            print(f"  CB confirm count    : {new_state.cb_confirm_count}/{cfg.reentry_confirmation_months + 1} risk-on months")
        print(f"  ── Signal ──")
        print(f"  SPY {cfg.abs_lookback//21}m return    : {signal.spy_lookback_return:>12.2%}")
        print(f"  Regime              : {signal.regime:>12}")
        print(f"  Candidate scores (3m relative momentum, skipped):")
        if signal.candidate_scores:
            for t, r in sorted(signal.candidate_scores.items(), key=lambda x: -x[1]):
                marker = " ← TARGET" if t == signal.target else ""
                print(f"     {t:<6} {r:>+8.2%}{marker}")
        else:
            print(f"     (no candidates — going to CASH)")
        print(f"  ── Decision ──")
        print(f"  Target              : {signal.target if signal.target else 'CASH':>12}")
        print(f"  Last held           : {state.last_target if state.last_target else 'CASH':>12}")
        print(f"  Reason              : {signal.decision_reason}")

        # Compare to current Alpaca position
        if not dry:
            current_positions = broker.get_positions_for_symbols(trade_universe)
            current_held = current_positions.index.tolist() if not current_positions.empty else []
            print(f"  Current positions   : {current_held if current_held else '[CASH]'}")

            target_set = {signal.target} if signal.target else set()
            current_set = set(current_held) - {"CASH"}

            if target_set == current_set:
                print(f"  -> No action needed (already holding {signal.target or 'CASH'})")
            else:
                # Liquidate any non-target positions
                for ticker in current_set - target_set:
                    pos = current_positions.loc[ticker]
                    notional = float(pos.get("market_value", 0))
                    if notional > 0:
                        order = Order(ticker=ticker, side=Side.SELL, notional=notional,
                                      reason=f"V4 rotate out ({signal.regime}, target={signal.target})")
                        oid = broker.submit_order(order)
                        print(f"  -> SELL {ticker} ${notional:,.2f}  order_id={oid}")
                        notify.trade("ETF", "SELL", ticker, notional, signal.decision_reason)
                        new_state.cash_value = portfolio_value
                # Buy target with cash (after sells settle, but for paper this is fine)
                if signal.target and signal.target not in current_set:
                    cash_after = broker.get_cash()
                    buy_notional = min(portfolio_value, cash_after)
                    if buy_notional > 100:
                        order = Order(ticker=signal.target, side=Side.BUY,
                                      notional=buy_notional,
                                      reason=f"V4 rotate in ({signal.regime})")
                        oid = broker.submit_order(order)
                        print(f"  -> BUY  {signal.target} ${buy_notional:,.2f}  order_id={oid}")
                        notify.trade("ETF", "BUY", signal.target, buy_notional, signal.decision_reason)
                        if oid is not None:
                            new_state.cash_value = 0.0
                elif not signal.target:
                    new_state.cash_value = portfolio_value

            new_state.last_eval_date = date.today().isoformat()
            save_state(new_state)
            logger.info(f"Rebalance complete. State saved: {state_to_dict(new_state)}")
        else:
            print(f"\n  [dry-run] No orders submitted, no state file modified.")
        print(f"{'='*72}\n")

    if dry_run:
        broker = AlpacaBroker() if any("ALPACA" in k for k in __import__("os").environ) else None
        if broker is None:
            print("No Alpaca creds in env - using configured ETF capital for dry-run.")
            class _MockBroker:
                def get_portfolio_value(self): return capital
                def get_cash(self): return capital
                def get_positions(self):
                    import pandas as pd
                    return pd.DataFrame()
                def get_positions_for_symbols(self, symbols):
                    import pandas as pd
                    return pd.DataFrame()
            broker = _MockBroker()
        rebalance_once(broker, dry=True)
        return

    # Live mode — schedule loop
    import schedule
    broker = AlpacaBroker()
    if not broker.is_paper:
        logger.critical("BROKER IS NOT IN PAPER MODE. ABORTING.")
        sys.exit(1)
    from monitor.notify import TelegramNotifier
    notify = TelegramNotifier()
    logger.info(f"Starting V4 paper loop. PID={__import__('os').getpid()}. Logging to {log_path}")
    notify.startup("ETF")

    et = ZoneInfo("America/New_York")

    def daily_check() -> None:
        today = datetime.now(tz=et).date()
        if is_last_trading_day_of_month(today, broker):
            logger.info(f"{today} is last trading day of month — running V4 rebalance")
            prev_state = load_state()
            rebalance_once(broker, dry=False)
            new_state = load_state()
            if new_state.last_target != prev_state.last_target:
                pv = scoped_value(broker, new_state, False)
                notify.rebalance(
                    "ETF", new_state.last_target or "CASH",
                    "risk_on" if new_state.last_target else "risk_off", pv
                )
        else:
            logger.debug(f"{today} not last trading day this month, skipping rebalance")
            # Still update peak / CB state daily so intraday CB triggers work
            state = load_state()
            pv = scoped_value(broker, state, False)
            if state.peak == 0.0:
                state.peak = pv
            new_peak = max(state.peak, pv)
            dd = (new_peak - pv) / new_peak if new_peak > 0 else 0.0
            if dd >= cfg.cb_threshold and not state.in_cb:
                logger.warning(f"INTRADAY CB TRIGGER: DD={dd:.1%} >= {cfg.cb_threshold:.0%}. Liquidating to cash.")
                broker.liquidate_symbols(trade_universe)
                state.cash_value = pv
                state.in_cb = True
                state.cb_confirm_count = 0
                notify.circuit_breaker("ETF", dd, pv)
            state.peak = new_peak
            save_state(state)

    # Initial check on startup — populates state.json and surfaces any errors
    # before entering the schedule loop.
    logger.info("Running initial daily_check on startup...")
    try:
        daily_check()
    except Exception as exc:
        logger.error(f"Initial check failed: {exc}", exc_info=True)

    # Schedule for 15:30 ET (Alpaca calendar uses NY time)
    schedule.every().day.at("15:30").do(daily_check)
    logger.info("Schedule: daily check at 15:30 ET. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


def cli() -> None:
    config = load_config()
    setup_logging(config)

    dry_run = "--dry-run" in sys.argv
    bot = "all"
    if "--bot" in sys.argv:
        try:
            bot = sys.argv[sys.argv.index("--bot") + 1]
        except IndexError:
            bot = "all"

    commands = {
        "intraday":           lambda: cmd_intraday(config, dry_run=dry_run),
        "intraday-backtest":  lambda: cmd_intraday_backtest(config),
        "backtest":           lambda: cmd_backtest(config),
        "crypto-backtest":    lambda: cmd_crypto_backtest(config),
        "crypto-paper":       lambda: cmd_crypto_paper(config, dry_run=dry_run),
        "paper":              lambda: cmd_paper(config),
        "momentum-paper":     lambda: cmd_momentum_paper(config, dry_run=dry_run),
        "momentum-backtest":  lambda: cmd_momentum_backtest(config),
        "status":             lambda: cmd_status(config, bot=bot),
        "health":             lambda: cmd_health(config),
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
