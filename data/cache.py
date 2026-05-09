from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
from loguru import logger


class Cache:
    """Parquet-backed disk cache with TTL expiry."""

    def __init__(self, cache_dir: str = ".cache", ttl_hours: float = 24.0):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600

    def _path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{safe}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        path = self._path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._path(key))
        except Exception as exc:
            logger.warning(f"Cache write failed for {key!r}: {exc}")

    def invalidate(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
