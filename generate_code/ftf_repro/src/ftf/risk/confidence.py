"""ftf.risk.confidence

Confidence shaping layer.

Paper convention:
    conf_share(t) = max(0, (p_bull(t) - 0.5)/0.5) in [0,1]
    w_conf(t) = w_vol(t) * conf_share(t)

This module is intentionally simple and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfidenceState:
    """Frozen confidence shaping parameters.

    We keep this as a state object for symmetry with other fitted states and to
    allow future extensions (e.g., nonlinear shaping).
    """

    # Pivot and scale are fixed by the reproduction plan, but kept explicit.
    pivot: float = 0.5
    scale: float = 0.5


def fit_confidence_state() -> ConfidenceState:
    """Return the canonical confidence state.

    There is no training-dependent estimation for confidence shaping in the plan.
    """

    return ConfidenceState()


def confidence_share(p_bull: pd.Series, *, state: Optional[ConfidenceState] = None) -> pd.Series:
    """Compute confidence share from bull probability.

    Parameters
    ----------
    p_bull:
        Decision-time blended bull probability in [0,1].

    Returns
    -------
    pd.Series
        conf_share in [0,1] with NaNs preserved.
    """

    if not isinstance(p_bull, pd.Series) or not isinstance(p_bull.index, pd.DatetimeIndex):
        raise TypeError("p_bull must be a pandas Series indexed by DatetimeIndex")
    st = state or ConfidenceState()
    if st.scale <= 0:
        raise ValueError("state.scale must be positive")

    # Preserve NaNs.
    x = (p_bull - st.pivot) / st.scale
    out = x.clip(lower=0.0, upper=1.0)
    out.name = "conf_share"
    return out


def confidence_weight(w_vol: pd.Series, p_bull: pd.Series, *, state: Optional[ConfidenceState] = None) -> pd.Series:
    """Compute confidence-shaped weight: w_conf = w_vol * conf_share.

    Notes
    -----
    - Alignment: uses inner alignment on index intersection.
    - NaNs are propagated (if either input is NaN on a date).
    """

    if not isinstance(w_vol, pd.Series) or not isinstance(w_vol.index, pd.DatetimeIndex):
        raise TypeError("w_vol must be a pandas Series indexed by DatetimeIndex")

    conf = confidence_share(p_bull, state=state)
    wv, cf = w_vol.align(conf, join="inner")
    out = wv * cf
    out.name = "w_conf"

    # Ensure non-negative; if user passes negative w_vol, clip to 0.
    out = out.clip(lower=0.0)
    # Avoid -0.0
    out = out.where(~np.isclose(out, 0.0), 0.0)
    return out


__all__ = [
    "ConfidenceState",
    "fit_confidence_state",
    "confidence_share",
    "confidence_weight",
]
