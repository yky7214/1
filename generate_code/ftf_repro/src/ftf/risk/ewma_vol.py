"""EWMA volatility model and volatility targeting.

Implements the paper's recursion (decision-time / no lookahead):

    sigma2_{t+1} = theta * sigma2_t + (1-theta) * r_t^2

Vol targeting uses the *next-day* forecast (sigma_{t+1}) to size the risk budget
weight at time t:

    w_vol(t) = min(Wmax, sigma_target_daily / sqrt(sigma2_{t+1}))

Initialization at the OOS anchor uses the training variance of returns.

This module is intentionally simple and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ftf.utils.config import RiskConfig


@dataclass(frozen=True)
class EWMAVolState:
    """Frozen EWMA volatility state (per walk-forward anchor)."""

    theta: float
    sigma2_init: float
    vol_target_annual: float = 0.15
    w_max: float = 2.0

    @property
    def sigma_target_daily(self) -> float:
        return float(self.vol_target_annual) / np.sqrt(252.0)


def fit_ewma_vol_state(
    r_train: pd.Series,
    *,
    cfg: Optional[RiskConfig] = None,
    theta: Optional[float] = None,
) -> EWMAVolState:
    """Fit (freeze) EWMA volatility state on training returns.

    Parameters
    ----------
    r_train:
        Close-to-close returns on training window.
    cfg:
        RiskConfig providing defaults.
    theta:
        Override EWMA decay.

    Returns
    -------
    EWMAVolState
        Frozen state including sigma2_init = Var_train(r).
    """

    if not isinstance(r_train, pd.Series) or not isinstance(r_train.index, pd.DatetimeIndex):
        raise TypeError("r_train must be a pandas Series indexed by DatetimeIndex")

    cfg = cfg or RiskConfig()
    th = float(theta if theta is not None else cfg.ewma_theta)
    if not (0.0 < th < 1.0):
        raise ValueError("theta must be in (0, 1)")

    r = r_train.dropna().astype(float)
    if len(r) < 2:
        raise ValueError("Need at least 2 non-NaN returns to estimate training variance")

    sigma2_init = float(np.var(r.to_numpy(), ddof=0))
    # Avoid zero-vol degenerate division.
    sigma2_init = max(sigma2_init, 1e-18)

    return EWMAVolState(
        theta=th,
        sigma2_init=sigma2_init,
        vol_target_annual=float(cfg.vol_target_annual),
        w_max=float(cfg.w_max),
    )


def ewma_variance_forecast(
    r: pd.Series,
    *,
    state: EWMAVolState,
) -> pd.Series:
    """Compute EWMA variance forecast series sigma2_{t+1}.

    Output is indexed like r, and at each date t stores the forecast for t+1
    based only on information up to t.
    """

    if not isinstance(r, pd.Series) or not isinstance(r.index, pd.DatetimeIndex):
        raise TypeError("r must be a pandas Series indexed by DatetimeIndex")

    th = float(state.theta)
    sigma2 = np.empty(len(r), dtype=float)
    sigma2_prev = float(state.sigma2_init)

    r2 = np.square(r.astype(float).to_numpy())
    # Treat NaN returns as missing: carry forward previous forecast.
    for i in range(len(r2)):
        if np.isfinite(r2[i]):
            sigma2_next = th * sigma2_prev + (1.0 - th) * r2[i]
        else:
            sigma2_next = sigma2_prev
        sigma2[i] = sigma2_next
        sigma2_prev = sigma2_next

    return pd.Series(sigma2, index=r.index, name="ewma_sigma2_next")


def vol_target_weight(
    r: pd.Series,
    *,
    state: EWMAVolState,
) -> pd.Series:
    """Compute volatility targeting weight w_vol(t).

    Uses sigma2_{t+1} forecast at time t to compute w_vol(t).
    """

    sigma2_next = ewma_variance_forecast(r, state=state)
    denom = np.sqrt(np.maximum(sigma2_next.to_numpy(), 1e-18))
    w = state.sigma_target_daily / denom
    w = np.minimum(w, float(state.w_max))
    w = np.maximum(w, 0.0)
    return pd.Series(w, index=r.index, name="w_vol")


__all__ = [
    "EWMAVolState",
    "fit_ewma_vol_state",
    "ewma_variance_forecast",
    "vol_target_weight",
]
