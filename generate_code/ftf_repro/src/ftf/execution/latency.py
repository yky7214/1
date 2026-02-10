"""ftf.execution.latency

Latency / forecast-to-fill execution lag buffer.

The pipeline uses the following convention:
- Decision time uses information up to the close of day t (F_t).
- The trading engine produces a *target* weight series w_target[t] at decision time.
- Executed weight applies a latency buffer:
    w_exec[t] = w_target[t - d]
  where d is an integer number of business days (0, 1, 2).

This module implements only the deterministic lagging transformation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["apply_exec_lag"]


def apply_exec_lag(w_target: pd.Series, *, exec_lag: int, fill_value: float = 0.0) -> pd.Series:
    """Apply an integer execution lag to a target weight series.

    Parameters
    ----------
    w_target:
        Decision-time target weights indexed by business date.
    exec_lag:
        Integer lag (days). Baseline is 1 (T+1 close execution).
    fill_value:
        Value to use for the first `exec_lag` observations where prior targets
        are not available.

    Returns
    -------
    pd.Series
        Executed weights w_exec with the same index as w_target.
    """

    if not isinstance(w_target, pd.Series):
        raise TypeError("w_target must be a pandas Series")
    if not isinstance(w_target.index, pd.DatetimeIndex):
        raise TypeError("w_target must be indexed by a DatetimeIndex")
    if int(exec_lag) != exec_lag:
        raise ValueError("exec_lag must be an integer")
    exec_lag = int(exec_lag)
    if exec_lag < 0 or exec_lag > 10:
        raise ValueError("exec_lag must be nonnegative and reasonably small")

    w_exec = w_target.shift(exec_lag)
    if exec_lag > 0:
        w_exec = w_exec.fillna(fill_value)

    # Preserve NaNs where the target is NaN *and* would have affected the exec series.
    # For typical usage, w_target should have NaNs only during warmup; lagging then
    # causes NaNs to appear later. We do not forward-fill weights here.
    w_exec = w_exec.astype(float)

    # Clean negative zero.
    w_exec = w_exec.where(~np.isclose(w_exec.to_numpy(), 0.0), 0.0)
    w_exec.name = "w_exec"
    return w_exec
