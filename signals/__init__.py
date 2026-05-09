from signals.base import BaseSignal
from signals.gap import GapSignal, StockSnapshot
from signals.orb import ORBSignal, OpeningRange
from signals.vwap import VWAPSignal
from signals.intraday_composite import IntradayComposite, IntradayCandidate
# Legacy factor signals — kept for backtest compatibility
from signals.momentum import MomentumSignal
from signals.quality import QualitySignal
from signals.volatility import LowVolatilitySignal
from signals.ml_signal import XGBoostRankerSignal
from signals.features import build_feature_history, build_target_history

__all__ = [
    # Intraday
    "GapSignal",
    "StockSnapshot",
    "ORBSignal",
    "OpeningRange",
    "VWAPSignal",
    "IntradayComposite",
    "IntradayCandidate",
    # Legacy factor (backtest only)
    "BaseSignal",
    "MomentumSignal",
    "QualitySignal",
    "LowVolatilitySignal",
    "XGBoostRankerSignal",
    "build_feature_history",
    "build_target_history",
]
