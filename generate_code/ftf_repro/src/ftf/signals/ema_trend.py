"""EMA-trend signal computation.

Implements the paper's EMA-slope -> z-score -> probability mapping:

- y_t = log(P_t)
- EMA: ỹ_t = λ ỹ_{t-1} + (1-λ) y_t
- slope proxy: Δỹ_t = ỹ_t - ỹ_{t-1}
- z_t = (Δỹ_t - μ_train)/σ_train, clipped to [-3,3]
- p_trend = (z_clip + 3)/6 ∈ [0,1]

Critical convention:
- p_trend(t) is a decision-time feature using information up to close of day t.

This module does *not* apply any trading latency; that is handled by execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ftf.utils.config import SignalConfig


@dataclass(frozen=True)
class EMATrendState:
    """Frozen EMA-trend parameters for walk-forward anchors."""

    ema_lambda: float
    slope_mu: float
    slope_sigma: float
    z_clip: Tuple[float, float] = (-3.0, 3.0)


def ema_log_price(close: pd.Series, *, ema_lambda: float) -> pd.Series:
    """Compute EMA of log-price with explicit recursion.

    We use a manual recursion (instead of pandas ewm) to avoid ambiguity in
    initialization and to make unit testing deterministic.
    """

    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by DatetimeIndex")
    y = np.log(close.astype(float).to_numpy())
    out = np.empty_like(y)
    out[:] = np.nan

    lam = float(ema_lambda)
    if not (0.0 < lam < 1.0):
        raise ValueError("ema_lambda must be in (0,1)")

    # Initialize with first observed value
    if len(y) == 0:
        return pd.Series([], index=close.index, dtype=float)

    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * y[i]

    return pd.Series(out, index=close.index, name="ema_log")


def ema_slope_from_ema(ema_log: pd.Series) -> pd.Series:
    """Slope proxy Δỹ_t = ỹ_t - ỹ_{t-1}."""

    slope = ema_log.diff()
    slope.name = "ema_slope"
    return slope


def fit_ema_trend_state(
    close_train: pd.Series,
    *,
    cfg: Optional[SignalConfig] = None,
    ema_lambda: Optional[float] = None,
) -> EMATrendState:
    """Fit EMA slope distribution parameters on a training window."""

    if cfg is None:
        cfg = SignalConfig()
    lam = float(cfg.ema_lambda if ema_lambda is None else ema_lambda)

    ema_log = ema_log_price(close_train, ema_lambda=lam)
    slope = ema_slope_from_ema(ema_log)

    # Training stats exclude the first NaN from diff
    slope_vals = slope.dropna().to_numpy(dtype=float)
    if slope_vals.size < 5:
        raise ValueError("Not enough training data to fit EMA slope stats")

    mu = float(np.mean(slope_vals))
    sigma = float(np.std(slope_vals, ddof=0))

    # Avoid divide-by-zero; a flat market should collapse p_trend to 0.5.
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-12

    return EMATrendState(ema_lambda=lam, slope_mu=mu, slope_sigma=sigma, z_clip=cfg.z_clip)


def compute_p_trend(
    close: pd.Series,
    *,
    state: EMATrendState,
) -> pd.DataFrame:
    """Compute EMA log, slope, z-score, clipped z, and p_trend.

    Returns a DataFrame with columns:
    - ema_log
    - slope
    - z
    - z_clipped
    - p_trend
    """

    ema_log = ema_log_price(close, ema_lambda=state.ema_lambda)
    slope = ema_slope_from_ema(ema_log)

    z = (slope - state.slope_mu) / state.slope_sigma
    z.name = "z"

    lo, hi = state.z_clip
    zc = z.clip(lower=lo, upper=hi)
    zc.name = "z_clipped"

    p_trend = (zc + 3.0) / 6.0
    p_trend = p_trend.clip(lower=0.0, upper=1.0)
    p_trend.name = "p_trend"

    out = pd.concat(
        [
            ema_log.rename("ema_log"),
            slope.rename("slope"),
            z,
            zc,
            p_trend,
        ],
        axis=1,
    )
    return out


__all__ = [
    "EMATrendState",
    "ema_log_price",
    "ema_slope_from_ema",
    "fit_ema_trend_state",
    "compute_p_trend",
]
