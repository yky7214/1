"""ftf.signals.regime

Signal composition for the Forecast-to-Fill reproduction.

Implements the paper-defined pipeline at *decision time* t:

- EMA log-price -> EMA slope -> z-score -> p_trend(t) in [0, 1]
- Momentum confirmation m(t) in {0,1}
- Blend into p_bull(t) = omega * p_trend + (1-omega) * m
- Entry gating (long-only): eligible_to_enter(t) = (p_bull>=pbull_th) and (slope>0)
- Optional regime labels for attribution: bull/bear/chop.

All computations are causal: each value at t only depends on prices up to and including t
(and, for ATR later, close[t-1] is allowed).

This module does not apply execution latency; that is handled by execution/latency.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ftf.signals.ema_trend import EMATrendState, compute_p_trend, fit_ema_trend_state
from ftf.signals.momentum import MomentumState, compute_momentum, fit_momentum_state
from ftf.utils.config import SignalConfig


@dataclass(frozen=True)
class RegimeState:
    """Frozen regime parameters for a walk-forward anchor."""

    ema_state: EMATrendState
    mom_state: MomentumState
    blend_omega: float
    pbull_threshold: float


def fit_regime_state(
    close_train: pd.Series,
    *,
    cfg: Optional[SignalConfig] = None,
    ema_lambda: Optional[float] = None,
    blend_omega: Optional[float] = None,
    pbull_threshold: Optional[float] = None,
    momentum_k: Optional[int] = None,
) -> RegimeState:
    """Fit/freeze regime parameters on a training slice.

    Notes
    -----
    - Fits EMA slope distribution parameters (mu/sigma) on train only.
    - Momentum has no distributional fitting; just freezes K.
    """

    if cfg is None:
        cfg = SignalConfig()

    omega = float(cfg.blend_omega if blend_omega is None else blend_omega)
    pb_th = float(cfg.pbull_threshold if pbull_threshold is None else pbull_threshold)
    if not (0.0 <= omega <= 1.0):
        raise ValueError("blend_omega must be in [0,1]")
    if not (0.0 <= pb_th <= 1.0):
        raise ValueError("pbull_threshold must be in [0,1]")

    ema_state = fit_ema_trend_state(close_train, cfg=cfg, ema_lambda=ema_lambda)
    mom_state = fit_momentum_state(cfg=cfg, k=momentum_k)

    return RegimeState(
        ema_state=ema_state,
        mom_state=mom_state,
        blend_omega=omega,
        pbull_threshold=pb_th,
    )


def label_regime(p_bull: pd.Series) -> pd.Series:
    """Regime label for attribution.

    bull if p_bull >= 0.55
    bear if p_bull <= 0.45
    else chop
    """

    if not isinstance(p_bull.index, pd.DatetimeIndex):
        raise TypeError("p_bull must be indexed by a DatetimeIndex")

    out = pd.Series(index=p_bull.index, dtype="object")
    out[p_bull >= 0.55] = "bull"
    out[p_bull <= 0.45] = "bear"
    out[(p_bull < 0.55) & (p_bull > 0.45)] = "chop"
    return out


def compute_regime_features(close: pd.Series, *, state: RegimeState) -> pd.DataFrame:
    """Compute p_trend, momentum, p_bull/p_bear and entry eligibility.

    Returns a DataFrame with columns:
      ema_log, slope, z, z_clipped, p_trend,
      momentum, p_bull, p_bear,
      eligible_to_enter, regime

    The first K days (momentum) and the first EMA-slope day will contain NaNs.
    """

    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a DatetimeIndex")

    tr = compute_p_trend(close, state=state.ema_state)
    m = compute_momentum(close, state=state.mom_state).rename("momentum")

    omega = state.blend_omega
    p_bull = omega * tr["p_trend"] + (1.0 - omega) * m
    # if momentum is NaN early, p_bull becomes NaN; that is fine.
    p_bear = 1.0 - p_bull

    eligible = (p_bull >= state.pbull_threshold) & (tr["slope"] > 0.0)
    eligible = eligible.astype("float")  # keep NaN where p_bull is NaN
    eligible[p_bull.isna() | tr["slope"].isna()] = np.nan
    eligible = eligible.rename("eligible_to_enter")

    df = pd.concat(
        [
            tr,
            m,
            p_bull.rename("p_bull"),
            p_bear.rename("p_bear"),
            eligible,
        ],
        axis=1,
    )
    df["regime"] = label_regime(df["p_bull"])
    return df


__all__ = [
    "RegimeState",
    "fit_regime_state",
    "compute_regime_features",
    "label_regime",
]
