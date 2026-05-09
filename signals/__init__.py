"""Technical-analysis signal layer."""
from signals.scoring import (
    DEFAULT_WEIGHTS,
    ScoreCard,
    score_setup,
    passes_threshold,
)
from signals.setup import (
    Direction,
    HoldType,
    SetupEngine,
    SetupEngineConfig,
    TradeSetup,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "ScoreCard",
    "score_setup",
    "passes_threshold",
    "Direction",
    "HoldType",
    "SetupEngine",
    "SetupEngineConfig",
    "TradeSetup",
]
