"""ftf.execution.fills

Execution-related utilities beyond simple lagging.

The reproduction plan distinguishes between:
- *Decision time* at close of day t (information set F_t)
- *Execution* at close of day t+d (latency/forecast-to-fill)

In this codebase, the canonical representation of execution latency is via
`apply_exec_lag` in :mod:`ftf.execution.latency`, which shifts a daily target
weight series by an integer number of business days.

This module provides a small façade around that function and a helper to
compute daily P&L attribution consistent with the plan:

    gross_ret[t] = w_exec[t-1] * r[t]

where r[t] is the close-to-close return between (t-1 -> t).

We keep this module lightweight because fills are deterministic (daily close
fills). More elaborate intraday modeling is explicitly out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .latency import apply_exec_lag as _apply_exec_lag


@dataclass(frozen=True)
class FillResult:
    """Container for executed weights and attribution series."""

    w_exec: pd.Series
    gross_ret: pd.Series


def compute_gross_return(r: pd.Series, w_exec: pd.Series) -> pd.Series:
    """Compute gross strategy return series with the project convention.

    Parameters
    ----------
    r:
        Close-to-close simple returns series, indexed by business day.
    w_exec:
        Executed weights series.

    Returns
    -------
    pd.Series
        gross_ret[t] = w_exec[t-1] * r[t]

    Notes
    -----
    - This function does *not* subtract costs.
    - Both series are inner-aligned to avoid silent reindexing.
    """

    if not isinstance(r, pd.Series) or not isinstance(w_exec, pd.Series):
        raise TypeError("r and w_exec must be pandas Series")
    if not isinstance(r.index, pd.DatetimeIndex) or not isinstance(w_exec.index, pd.DatetimeIndex):
        raise TypeError("r and w_exec must be indexed by DatetimeIndex")

    r2, w2 = r.align(w_exec, join="inner")
    gross = w2.shift(1).fillna(0.0) * r2
    gross.name = "gross_ret"

    # Clean tiny negatives / -0.0
    gross = gross.where(~np.isclose(gross.to_numpy(dtype=float), 0.0, atol=1e-15), 0.0)
    return gross

def apply_exec_lag(
    w_target: pd.Series,
    *,
    lag: Optional[int] = None,
    exec_lag: Optional[int] = None,
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Compatibility wrapper.

    Accepts either:
      - lag= (used by engine.py)
      - exec_lag= (used by fill_from_targets / legacy)

    Delegates to latency.apply_exec_lag.
    """
    if exec_lag is None:
        exec_lag = 0 if lag is None else int(lag)
    return _apply_exec_lag(w_target, exec_lag=int(exec_lag), fill_value=float(fill_value))


def fill_from_targets(
    w_target: pd.Series,
    r: pd.Series,
    *,
    exec_lag: int,
    fill_value: float = 0.0,
) -> FillResult:
    """Apply execution lag to targets and compute gross returns.

    This is a convenience wrapper used by scripts/tests.
    The main engine calls :func:`apply_exec_lag` and computes gross returns
    directly for speed.
    """

    w_exec = apply_exec_lag(w_target, exec_lag=exec_lag, fill_value=fill_value)
    gross_ret = compute_gross_return(r, w_exec)
    return FillResult(w_exec=w_exec, gross_ret=gross_ret)


__all__ = ["FillResult", "compute_gross_return", "fill_from_targets", "apply_exec_lag"]
