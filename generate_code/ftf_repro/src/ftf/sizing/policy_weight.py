"""ftf.sizing.policy_weight

Combine volatility targeting, confidence shaping, and (fractional) Kelly sizing
into a raw target weight series.

This module intentionally does *not* apply entry/exit gating, execution latency,
or transaction costs. Those are handled by the trading engine and execution
modules.

Definitions (decision-time t):
  conf_share(t) = max(0, (p_bull(t) - 0.5)/0.5) in [0,1]
  w_conf(t) = w_vol(t) * conf_share(t)
  w_raw(t) = clip(f_tilde * w_conf(t), 0, Wmax)

Baseline floor when f_tilde is effectively zero:
  - FLOOR_ON_WVOL: w_raw = baseline_floor * w_vol
  - FLOOR_ON_WCONF: w_raw = baseline_floor * w_conf

All operations preserve NaNs where upstream inputs are NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ftf.risk.confidence import confidence_weight
from ftf.utils.config import BaselineFloorMode, KellyConfig, RiskConfig

__all__ = [
    "PolicyWeightState",
    "fit_policy_weight_state",
    "compute_w_raw",
]


@dataclass(frozen=True)
class PolicyWeightState:
    """Frozen parameters for policy weight construction."""

    w_max: float = 2.0
    baseline_floor: float = 0.25
    baseline_floor_mode: BaselineFloorMode = "FLOOR_ON_WVOL"
    baseline_floor_eps: float = 1e-6


def fit_policy_weight_state(
    *,
    risk_cfg: Optional[RiskConfig] = None,
    kelly_cfg: Optional[KellyConfig] = None,
    w_max: Optional[float] = None,
) -> PolicyWeightState:
    """Create a PolicyWeightState from configs/overrides.

    This is a small convenience function to keep a uniform fit/freeze/apply API
    across the project.
    """

    risk_cfg = risk_cfg or RiskConfig()
    kelly_cfg = kelly_cfg or KellyConfig()

    w_max_v = float(w_max if w_max is not None else risk_cfg.w_max)
    if w_max_v <= 0:
        raise ValueError("w_max must be positive")

    if not (0.0 <= kelly_cfg.baseline_floor <= 1.0):
        raise ValueError("baseline_floor must be in [0,1]")
    if kelly_cfg.baseline_floor_eps < 0:
        raise ValueError("baseline_floor_eps must be >= 0")

    if kelly_cfg.baseline_floor_mode not in ("FLOOR_ON_WVOL", "FLOOR_ON_WCONF"):
        raise ValueError(f"Unknown baseline_floor_mode: {kelly_cfg.baseline_floor_mode}")

    return PolicyWeightState(
        w_max=w_max_v,
        baseline_floor=float(kelly_cfg.baseline_floor),
        baseline_floor_mode=kelly_cfg.baseline_floor_mode,
        baseline_floor_eps=float(kelly_cfg.baseline_floor_eps),
    )


def compute_w_raw(
    *,
    w_vol: pd.Series,
    p_bull: pd.Series,
    f_tilde: float,
    state: Optional[PolicyWeightState] = None,
) -> pd.Series:
    """Compute the raw target weight series (long-only, pre-gating).

    Parameters
    ----------
    w_vol:
        Volatility targeting weight series w_vol(t).
    p_bull:
        Blended bull probability at decision time t.
    f_tilde:
        Fractional Kelly scalar (f~). Typically frozen per walk-forward anchor.
    state:
        PolicyWeightState specifying caps/floor behavior.

    Returns
    -------
    pd.Series
        w_raw(t) series (nonnegative), aligned to the inner intersection of the
        input indices.
    """

    if not isinstance(w_vol, pd.Series) or not isinstance(p_bull, pd.Series):
        raise TypeError("w_vol and p_bull must be pandas Series")
    if not isinstance(w_vol.index, pd.DatetimeIndex) or not isinstance(p_bull.index, pd.DatetimeIndex):
        raise TypeError("w_vol and p_bull must have DatetimeIndex")

    state = state or PolicyWeightState()

    f_tilde = float(f_tilde)
    if not np.isfinite(f_tilde):
        raise ValueError("f_tilde must be finite")
    if f_tilde < 0:
        # Long-only: negative Kelly is treated as 0.
        f_tilde = 0.0

    # Align indices and compute confidence-shaped weight.
    w_conf = confidence_weight(w_vol, p_bull)

    # Raw weight.
    w_raw = f_tilde * w_conf

    # Baseline floor when f_tilde effectively zero.
    if f_tilde < state.baseline_floor_eps:
        if state.baseline_floor_mode == "FLOOR_ON_WVOL":
            w_vol_aligned = w_vol.reindex(w_raw.index)
            w_raw = state.baseline_floor * w_vol_aligned
        elif state.baseline_floor_mode == "FLOOR_ON_WCONF":
            w_raw = state.baseline_floor * w_conf
        else:
            raise ValueError(f"Unknown baseline_floor_mode: {state.baseline_floor_mode}")

    # Enforce long-only and cap.
    w_raw = w_raw.clip(lower=0.0, upper=state.w_max)

    # Preserve NaNs where either upstream input was NaN.
    nan_mask = w_vol.reindex(w_raw.index).isna() | p_bull.reindex(w_raw.index).isna()
    w_raw = w_raw.mask(nan_mask)

    # Avoid -0.0
    w_raw = w_raw.where(~np.isclose(w_raw.to_numpy(dtype=float), 0.0), 0.0)
    w_raw.name = "w_raw"
    return w_raw
