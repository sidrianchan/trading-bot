"""XGBoost ranker signal: learns to rank stocks by forward returns.

Training regime:
  - Target: cross-sectional percentile rank of forward 21-day returns
  - Window: rolling 24-month training, 6-month gap before test date
  - Retrain: monthly (lazy — triggers on first compute() call each month)
  - Validation metric logged: Spearman rank correlation on training data

Architecture persistence:
  - Auto-saved to models/xgboost_ranker.joblib after each retrain
  - save_architecture() also writes a dated snapshot for the 6-month
    retraining cycle on proprietary paper-trading labels
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from signals.base import BaseSignal

_MODEL_DIR = Path("models")
_MODEL_PATH = _MODEL_DIR / "xgboost_ranker.joblib"


class XGBoostRankerSignal(BaseSignal):

    def __init__(
        self,
        feature_history: pd.DataFrame,   # MultiIndex (date, ticker), cols = FEATURE_COLS
        target_history: pd.DataFrame,    # MultiIndex (date, ticker), col = 'rank'
        train_window_months: int = 24,
        gap_months: int = 6,
        n_estimators: int = 200,
        max_depth: int = 3,
    ):
        self.feature_history = feature_history
        self.target_history = target_history
        self._train_offset = pd.DateOffset(months=train_window_months)
        self._gap_offset = pd.DateOffset(months=gap_months)
        self._n_estimators = n_estimators
        self._max_depth = max_depth

        self._model = None
        self._last_trained_month: tuple[int, int] | None = None
        self._train_count = 0
        self._last_spearman: float | None = None
        self._feature_names: list[str] = (
            list(feature_history.columns) if not feature_history.empty else []
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.Series:
        if self.feature_history.empty:
            return pd.Series(dtype=float)

        current_date = prices.index[-1]
        self._maybe_retrain(current_date)

        if self._model is None:
            return pd.Series(dtype=float)

        X = self._features_at(current_date)
        if X.empty:
            return pd.Series(dtype=float)

        preds = self._model.predict(X.values)
        return pd.Series(preds, index=X.index, name="ml_score")

    def save_architecture(self, path: Path | str | None = None, dated_snapshot: bool = True) -> Path:
        """Persist model weights, hyperparameters, and feature names to disk.

        Args:
            path: Output path. Defaults to models/xgboost_ranker.joblib.
            dated_snapshot: Also write a dated copy for the 6-month retrain cycle.

        Returns:
            Path where the primary model was written.
        """
        try:
            import joblib
        except ImportError:
            logger.error("joblib not installed — cannot save model. Run: pip install joblib")
            raise

        if self._model is None:
            raise RuntimeError("No trained model to save — call compute() first")

        out = Path(path) if path else _MODEL_PATH
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "hyperparameters": {
                "n_estimators": self._n_estimators,
                "max_depth": self._max_depth,
                "learning_rate": 0.03,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_lambda": 2.0,
                "min_child_weight": 10,
                "random_state": 42,
            },
            "train_window_months": int(self._train_offset.months),
            "gap_months": int(self._gap_offset.months),
            "saved_at": str(date.today()),
            "train_count": self._train_count,
            "last_spearman": self._last_spearman,
        }
        joblib.dump(payload, out)
        logger.info(f"XGBoost architecture saved → {out}")

        if dated_snapshot:
            snapshot = out.parent / f"xgboost_ranker_{date.today()}.joblib"
            joblib.dump(payload, snapshot)
            logger.info(f"Dated snapshot → {snapshot}")

        return out

    @property
    def feature_importances(self) -> pd.Series | None:
        if self._model is None or not self._feature_names:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    @property
    def train_count(self) -> int:
        return self._train_count

    @property
    def last_spearman(self) -> float | None:
        return self._last_spearman

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_retrain(self, current_date: pd.Timestamp) -> None:
        month_key = (current_date.year, current_date.month)
        if month_key == self._last_trained_month:
            return

        try:
            import xgboost as xgb
            from scipy.stats import spearmanr
        except ImportError:
            logger.warning("xgboost or scipy not installed — ML signal inactive")
            return

        X_train, y_train = self._training_slice(current_date)
        if len(X_train) < 100:
            logger.debug(
                f"XGBoost: insufficient training data at {current_date.date()} "
                f"({len(X_train)} rows, need ≥100)"
            )
            return

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=2.0,
            min_child_weight=10,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_train.values, y_train.values)

        preds = model.predict(X_train.values)
        rho, _ = spearmanr(preds, y_train.values)

        self._model = model
        self._last_trained_month = month_key
        self._last_spearman = float(rho)
        self._train_count += 1
        logger.debug(
            f"XGBoost retrain #{self._train_count} @ {current_date.date()}: "
            f"n={len(X_train)}, train Spearman={rho:.3f}"
        )
        try:
            # dated_snapshot=False — only save dated copies via save_architecture()
            self.save_architecture(dated_snapshot=False)
        except Exception as exc:
            logger.warning(f"Auto-save of XGBoost model failed: {exc}")

    def _training_slice(
        self, as_of_date: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) for dates in [as_of_date - window - gap, as_of_date - gap]."""
        train_end = as_of_date - self._gap_offset
        train_start = train_end - self._train_offset

        feat_dates = self.feature_history.index.get_level_values("date")
        X = self.feature_history.loc[
            (feat_dates >= train_start) & (feat_dates <= train_end)
        ].fillna(0.5)

        tgt_dates = self.target_history.index.get_level_values("date")
        y = self.target_history.loc[
            (tgt_dates >= train_start) & (tgt_dates <= train_end)
        ]["rank"]

        common = X.index.intersection(y.index)
        X_aligned = X.loc[common].fillna(0.5)
        y_aligned = y.loc[common].dropna()
        final_idx = X_aligned.index.intersection(y_aligned.index)
        return X_aligned.loc[final_idx], y_aligned.loc[final_idx]

    def _features_at(self, current_date: pd.Timestamp) -> pd.DataFrame:
        """Return features from the most recent rebalance date ≤ current_date."""
        feat_dates = self.feature_history.index.get_level_values("date").unique()
        past = feat_dates[feat_dates <= current_date]
        if past.empty:
            return pd.DataFrame()
        lookup = past[-1]
        return self.feature_history.loc[lookup].fillna(0.5)
