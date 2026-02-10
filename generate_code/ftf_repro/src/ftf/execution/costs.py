"""ftf.execution.costs

Deterministic turnover-based transaction cost model.

This implements the reproduction plan's reduced-form daily cost accounting:

- turnover[t] = |w_exec[t] - w_exec[t-1]|
- linear cost:   k * turnover
- impact cost:   gamma * turnover^(3/2)

Costs are intended to be subtracted from daily returns (in *return* units),
where weights are unit-notional exposure scalars (e.g., 1.0 = 100% notional).

The trading engine uses the forecast-to-fill P&L convention:

    gross_ret[t] = w_exec[t-1] * r[t]
    net_ret[t]   = gross_ret[t] - cost_lin[t] - cost_imp[t]

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ftf.utils.config import CostImpactConfig


@dataclass(frozen=True)
class CostSeries:
    """Container for cost and turnover series."""

    turnover: pd.Series
    cost_linear: pd.Series
    cost_impact: pd.Series
    cost_total: pd.Series


def turnover_from_exec(w_exec: pd.Series, *, fill_first: bool = True) -> pd.Series:
    """Compute daily turnover from executed weights.

    Parameters
    ----------
    w_exec:
        Executed weight series indexed by business day.
    fill_first:
        If True, sets turnover on the first date to |w_exec[first] - 0|.
        If False, first turnover is NaN.

    Returns
    -------
    pd.Series
        Turnover series (nonnegative).
    """

    if not isinstance(w_exec, pd.Series):
        raise TypeError("w_exec must be a pandas Series")
    if not isinstance(w_exec.index, pd.DatetimeIndex):
        raise TypeError("w_exec must be indexed by a DatetimeIndex")

    tw = w_exec.astype(float)
    to = (tw - tw.shift(1)).abs()

    if fill_first and len(to) > 0:
        to.iloc[0] = 0.0
    to.name = "turnover"

    # tiny cleanups
    to = to.where(~np.isclose(to.to_numpy(), 0.0, atol=1e-15), 0.0)
    return to


def compute_costs(
    w_exec: pd.Series,
    *,
    costs_cfg: Optional[CostImpactConfig] = None,
    k_linear: Optional[float] = None,
    gamma_impact: Optional[float] = None,
    turnover: Optional[pd.Series] = None,
) -> CostSeries:
    """Compute turnover, linear costs, impact costs, and total costs.

    Costs are computed deterministically from executed weights.

    Parameters
    ----------
    w_exec:
        Executed weights.
    costs_cfg:
        Optional CostImpactConfig providing defaults.
    k_linear:
        Override for linear cost coefficient (return units per 1.0 turnover).
    gamma_impact:
        Override for impact coefficient.
    turnover:
        Optionally provide a precomputed turnover series. If provided, it must
        align exactly to w_exec.index.

    Returns
    -------
    CostSeries
        Turnover and cost components.
    """

    if costs_cfg is None:
        costs_cfg = CostImpactConfig()

    k = float(costs_cfg.k_linear if k_linear is None else k_linear)
    g = float(costs_cfg.gamma_impact if gamma_impact is None else gamma_impact)

    if k < 0 or g < 0:
        raise ValueError("k_linear and gamma_impact must be nonnegative")

    if turnover is None:
        to = turnover_from_exec(w_exec, fill_first=True)
    else:
        if not isinstance(turnover, pd.Series):
            raise TypeError("turnover must be a pandas Series")
        if not turnover.index.equals(w_exec.index):
            raise ValueError("turnover must have the same index as w_exec")
        to = turnover.astype(float)
        to.name = "turnover"

    # Costs
    cost_lin = k * to
    cost_imp = g * np.power(to, 1.5)

    cost_lin = cost_lin.astype(float)
    cost_imp = cost_imp.astype(float)

    # Preserve NaNs (if any) but do not allow negatives.
    cost_lin = cost_lin.clip(lower=0.0)
    cost_imp = cost_imp.clip(lower=0.0)

    cost_lin.name = "cost_linear"
    cost_imp.name = "cost_impact"

    cost_total = (cost_lin.fillna(0.0) + cost_imp.fillna(0.0)).reindex(w_exec.index)
    # Where either component is NaN, mark total as NaN (so caller can decide).
    nan_mask = cost_lin.isna() | cost_imp.isna()
    cost_total = cost_total.mask(nan_mask)
    cost_total.name = "cost_total"

    # clean near-zeros
    for s in (cost_lin, cost_imp, cost_total):
        arr = s.to_numpy()
        s.loc[np.isclose(arr, 0.0, atol=1e-15)] = 0.0

    return CostSeries(turnover=to, cost_linear=cost_lin, cost_impact=cost_imp, cost_total=cost_total)


def apply_costs_to_returns(
    gross_ret: pd.Series, *, cost_total: pd.Series
) -> pd.Series:
    """Subtract costs from a gross return series.

    This is a small helper mainly used in tests/diagnostics.

    Returns
    -------
    pd.Series
        net_ret = gross_ret - cost_total
    """

    if not isinstance(gross_ret, pd.Series) or not isinstance(cost_total, pd.Series):
        raise TypeError("gross_ret and cost_total must be pandas Series")
    gross_ret, cost_total = gross_ret.align(cost_total, join="inner")
    net = gross_ret - cost_total
    net.name = "net_ret"
    return net


__all__ = ["CostSeries", "turnover_from_exec", "compute_costs", "apply_costs_to_returns"]
