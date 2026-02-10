"""ftf.signals.momentum

Momentum confirmation signal.

Implements the paper definition:
    m_t = 1{ P_t / P_{t-K} > 1 } else 0

Notes
-----
- This is computed at decision time t using prices up to close of day t.
- For t < K, the signal is NaN by default (not enough history). Callers may
  choose to fill NaN with 0 or 0.5 depending on desired behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ftf.utils.config import SignalConfig


@dataclass(frozen=True)
class MomentumState:
    """Frozen momentum parameters."""

    k: int = 50


def compute_momentum_indicator(close: pd.Series, *, k: int) -> pd.Series:
    """Compute binary momentum indicator m_t.

    Parameters
    ----------
    close:
        Close/settle price series.
    k:
        Lookback in business days.

    Returns
    -------
    pd.Series
        Series with values {0.0, 1.0} and NaN for the first k observations.
    """

    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a DatetimeIndex")
    if k <= 0:
        raise ValueError("k must be positive")

    ratio = close / close.shift(k)
    m = (ratio > 1.0).astype(float)
    m[ratio.isna()] = np.nan
    m.name = "momentum"
    return m


def fit_momentum_state(*, cfg: Optional[SignalConfig] = None, k: Optional[int] = None) -> MomentumState:
    """Create a frozen momentum state.

    Momentum has no trainable parameters besides the lookback K.
    """

    if cfg is None:
        cfg = SignalConfig()
    k_val = int(cfg.momentum_k if k is None else k)
    if k_val <= 0:
        raise ValueError("momentum_k must be positive")
    return MomentumState(k=k_val)


def compute_momentum(close: pd.Series, *, state: MomentumState) -> pd.Series:
    """Compute momentum signal using a frozen state."""

    return compute_momentum_indicator(close, k=state.k)


__all__ = [
    "MomentumState",
    "fit_momentum_state",
    "compute_momentum_indicator",
    "compute_momentum",
]
