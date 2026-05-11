from signals.base import BaseSignal
from signals.gap import GapSignal, StockSnapshot
from signals.orb import ORBSignal, OpeningRange
from signals.vwap import VWAPSignal
from signals.intraday_composite import IntradayComposite, IntradayCandidate
# Legacy factor signals — kept for backtest compatibility
from signals.momentum import MomentumSignal
from signals.quality import QualitySignal
from signals.volatility import LowVolatilitySignal
from signals.composite import CompositeSignal
from signals.ml_signal import XGBoostRankerSignal
from signals.features import build_feature_history, build_target_history
from signals.crypto_momentum import CryptoMomentumConfig, CryptoMomentumState, compute_crypto_signal

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
    "CompositeSignal",
    "XGBoostRankerSignal",
    "build_feature_history",
    "build_target_history",
    "CryptoMomentumConfig",
    "CryptoMomentumState",
    "compute_crypto_signal",
]
