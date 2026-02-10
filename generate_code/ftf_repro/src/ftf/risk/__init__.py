"""Risk subpackage.

This package contains deterministic risk budgeting components used by the
Forecast-to-Fill pipeline.

Currently implemented:
- EWMA variance forecasting + volatility targeting (ewma_vol)
- Confidence shaping (confidence)

The package re-exports the small public surface so callers can do:

    from ftf.risk import fit_ewma_vol_state, vol_target_weight

"""

from .confidence import ConfidenceState, confidence_share, confidence_weight, fit_confidence_state
from .ewma_vol import EWMAVolState, ewma_variance_forecast, fit_ewma_vol_state, vol_target_weight

__all__ = [
    "ConfidenceState",
    "fit_confidence_state",
    "confidence_share",
    "confidence_weight",
    "EWMAVolState",
    "fit_ewma_vol_state",
    "ewma_variance_forecast",
    "vol_target_weight",
]
