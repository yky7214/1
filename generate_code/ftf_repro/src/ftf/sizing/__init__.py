"""Sizing subpackage.

This subpackage contains the position sizing components of the Forecast-to-Fill
pipeline:

- Kelly sizing with linear + concave impact penalty (train-only estimate; frozen per anchor)
- Policy weight construction that combines volatility targeting, confidence shaping,
  fractional Kelly scaling, and baseline-floor behavior.

The package exposes a small, stable API surface via re-exports.
"""

from .kelly import (
    KellyInputs,
    estimate_kelly_inputs,
    fractional_kelly,
    growth_proxy,
    solve_friction_adjusted_kelly,
)
from .policy_weight import PolicyWeightState, compute_w_raw, fit_policy_weight_state

__all__ = [
    "KellyInputs",
    "estimate_kelly_inputs",
    "growth_proxy",
    "solve_friction_adjusted_kelly",
    "fractional_kelly",
    "PolicyWeightState",
    "fit_policy_weight_state",
    "compute_w_raw",
]
