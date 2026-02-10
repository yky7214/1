"""Trading subpackage.

This package contains the daily state machine and trading engine:

- :mod:`ftf.trading.atr` implements ATR/TR calculations.
- :mod:`ftf.trading.exits` implements the long-only ATR stop/timeout exit logic.
- :mod:`ftf.trading.engine` orchestrates signals → sizing → exits → execution lag → costs.
- :mod:`ftf.trading.logs` defines structured trade event logging.

The subpackage is intentionally dependency-light; most heavy work happens in
:func:`ftf.trading.engine.run_engine`.
"""

from .atr import ATRState, compute_atr, compute_atr_from_cfg, fit_atr_state, true_range
from .engine import EngineResult, run_engine
from .exits import ATRExitState, fit_atr_exit_state, generate_target_weights
from .logs import EventType, TradeEvent, TradingLog

__all__ = [
    "ATRState",
    "fit_atr_state",
    "true_range",
    "compute_atr",
    "compute_atr_from_cfg",
    "ATRExitState",
    "fit_atr_exit_state",
    "generate_target_weights",
    "EngineResult",
    "run_engine",
    "EventType",
    "TradeEvent",
    "TradingLog",
]
