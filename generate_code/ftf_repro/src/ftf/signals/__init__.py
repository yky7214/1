"""Signal subpackage.

This package implements the decision-time signal stack described in the
Forecast-to-Fill reproduction plan:

- EMA-trend probability from EMA of log-price slope, standardized to a
  training-window distribution.
- Simple K-day momentum confirmation.
- Regime composition that blends the two into p_bull/p_bear, an entry gate,
  and optional attribution labels (bull/bear/chop).

The modules are designed to support strict walk-forward training:
"fit" functions must only use the training slice and return immutable state
objects that are then reused unchanged during OOS evaluation.
"""

from .ema_trend import (
    EMATrendState,
    compute_p_trend,
    ema_log_price,
    ema_slope_from_ema,
    fit_ema_trend_state,
)
from .momentum import (
    MomentumState,
    compute_momentum,
    compute_momentum_indicator,
    fit_momentum_state,
)
from .regime import RegimeState, compute_regime_features, fit_regime_state, label_regime

__all__ = [
    # EMA-trend
    "EMATrendState",
    "ema_log_price",
    "ema_slope_from_ema",
    "fit_ema_trend_state",
    "compute_p_trend",
    # Momentum
    "MomentumState",
    "fit_momentum_state",
    "compute_momentum_indicator",
    "compute_momentum",
    # Regime composition
    "RegimeState",
    "fit_regime_state",
    "compute_regime_features",
    "label_regime",
]
