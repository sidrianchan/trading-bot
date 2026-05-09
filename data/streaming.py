from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Callable

from loguru import logger

from signals.gap import StockSnapshot

BarCallback = Callable[[str, dict], None]


class BarStreamer:
    """Alpaca 1-minute bar feed with automatic WebSocket → REST fallback.

    On startup, attempts to open an Alpaca WebSocket stream. If the account's
    subscription level does not support streaming (detected by connection error
    or SIP-feed unavailability), falls back to 30-second REST polling of
    `/v2/stocks/bars/latest`.

    Usage:
        streamer = BarStreamer()
        streamer.subscribe(["AAPL", "NVDA"], callback=my_handler)
        streamer.start()     # non-blocking — runs in background thread
        ...
        streamer.stop()
    """

    def __init__(self):
        self._api_key    = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._tickers: list[str] = []
        self._callbacks: list[BarCallback] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._mode: str = "unknown"   # "websocket" or "polling"

    def subscribe(self, tickers: list[str], callback: BarCallback) -> None:
        self._tickers = list(tickers)
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._try_websocket()
        except Exception as exc:
            logger.warning(f"WebSocket stream unavailable ({exc}); falling back to REST polling")
            self._mode = "polling"
            self._poll_loop()

    def _try_websocket(self) -> None:
        from alpaca.data.live import StockDataStream  # type: ignore

        stream = StockDataStream(self._api_key, self._secret_key)

        async def bar_handler(bar):
            data = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "timestamp": bar.timestamp,
            }
            for cb in self._callbacks:
                try:
                    cb(bar.symbol, data)
                except Exception as e:
                    logger.error(f"Bar callback error for {bar.symbol}: {e}")

        for ticker in self._tickers:
            stream.subscribe_bars(bar_handler, ticker)

        self._mode = "websocket"
        logger.info(f"WebSocket stream active for {len(self._tickers)} tickers")

        # Run the stream until stop is requested
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_with_stop(stream))
        finally:
            loop.close()

    async def _run_with_stop(self, stream) -> None:
        stream_task = asyncio.create_task(stream._run_forever())
        while not self._stop_event.is_set():
            await asyncio.sleep(1)
        stream_task.cancel()

    def _poll_loop(self) -> None:
        """REST polling fallback: fetches latest 1-min bar every 30 seconds."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
            from alpaca.data.requests import StockLatestBarRequest         # type: ignore
        except ImportError:
            logger.error("alpaca-py not installed — cannot poll bars")
            return

        client = StockHistoricalDataClient(self._api_key, self._secret_key)
        logger.info(f"REST polling active (30s interval) for {len(self._tickers)} tickers")

        seen_timestamps: dict[str, object] = {}

        while not self._stop_event.is_set():
            try:
                req = StockLatestBarRequest(symbol_or_symbols=self._tickers)
                latest = client.get_stock_latest_bar(req)

                for ticker, bar in latest.items():
                    ts = bar.timestamp
                    if seen_timestamps.get(ticker) == ts:
                        continue
                    seen_timestamps[ticker] = ts
                    data = {
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                        "timestamp": ts,
                    }
                    for cb in self._callbacks:
                        try:
                            cb(ticker, data)
                        except Exception as e:
                            logger.error(f"Bar callback error for {ticker}: {e}")

            except Exception as exc:
                logger.warning(f"REST poll error: {exc}")

            self._stop_event.wait(timeout=30)


def build_snapshots_from_alpaca(
    tickers: list[str],
    api_key: str,
    secret_key: str,
) -> list[StockSnapshot]:
    """Fetch pre-market snapshots for the morning scan (09:25 ET).

    Uses Alpaca's snapshot endpoint which returns latest trade, prev daily bar,
    and minute bar in a single API call.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
        from alpaca.data.requests import StockSnapshotRequest          # type: ignore
    except ImportError:
        logger.error("alpaca-py not installed — cannot fetch snapshots")
        return []

    client = StockHistoricalDataClient(api_key, secret_key)
    snapshots: list[StockSnapshot] = []

    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            req = StockSnapshotRequest(symbol_or_symbols=chunk)
            raw = client.get_stock_snapshot(req)

            for ticker, snap in raw.items():
                try:
                    prev = snap.prev_daily_bar
                    latest = snap.latest_trade or snap.minute_bar
                    if prev is None or latest is None:
                        continue

                    prev_close  = float(prev.close)
                    curr_price  = float(latest.price if hasattr(latest, "price") else latest.close)
                    prev_volume = int(prev.volume)
                    gap_pct     = (curr_price / prev_close) - 1.0 if prev_close > 0 else 0.0

                    # Approximate ATR from prev day's daily bar (rough but available pre-market)
                    atr = float(prev.high - prev.low) if hasattr(prev, "high") else 0.0

                    snapshots.append(StockSnapshot(
                        ticker=ticker,
                        prev_close=prev_close,
                        latest_price=curr_price,
                        pre_market_volume=prev_volume,
                        avg_volume_30d=float(prev_volume),   # single-day proxy; improve with 30d avg
                        atr=atr,
                        gap_pct=gap_pct,
                        volume_ratio=1.0,   # will be updated once 30d vol is available
                    ))
                except Exception as e:
                    logger.debug(f"Snapshot parse error for {ticker}: {e}")
        except Exception as exc:
            logger.warning(f"Snapshot fetch failed for chunk: {exc}")

        if i + chunk_size < len(tickers):
            time.sleep(0.2)

    logger.info(f"Built {len(snapshots)} snapshots from Alpaca")
    return snapshots
