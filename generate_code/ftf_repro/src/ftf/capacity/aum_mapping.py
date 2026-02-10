"""ftf.capacity.aum_mapping

Map participation constraints to an AUM capacity estimate.

This module implements the paper-plan capacity mapping:

    q_t = |Δcontracts_t| / ADV_t

and with AUM A and executed weight change Δw_t:

    |Δcontracts_t| ≈ |Δw_t| * A / (P_t * M)

where:
    - P_t is the futures price in $/oz
    - M is the contract multiplier in ounces (GC: 100)

So a representative AUM limit for a participation cap q_cap can be estimated via:

    A_max ≈ q_cap * P_med * M * ADV_med / |Δw|_med

We also expose helper routines to compute a time-series of q_t for a given AUM.

The implementation is deterministic and uses inner-join alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.capacity.participation import (
    ParticipationSummary,
    participation_rate,
    representative_participation_inputs,
    summarize_participation,
)


@dataclass(frozen=True)
class AUMCapacityResult:
    """Outputs of AUM capacity estimation."""

    aum_max: float
    q_cap: float
    method: str
    rep_delta_w: float
    rep_price: float
    rep_adv: float
    contract_multiplier: float


def _check_series(x: pd.Series, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index must be a DatetimeIndex")
    if x.index.tz is not None:
        x = x.copy()
        x.index = x.index.tz_convert(None)
    if not x.index.is_monotonic_increasing:
        raise ValueError(f"{name}.index must be monotonic increasing")
    if x.index.has_duplicates:
        raise ValueError(f"{name}.index must not contain duplicates")
    return x


def participation_for_aum(
    w_exec: pd.Series,
    *,
    aum: float,
    price: pd.Series,
    adv: pd.Series,
    contract_multiplier: float = 100.0,
) -> pd.Series:
    """Compute participation series q_t for a given AUM.

    Uses executed weights and their daily changes.

    Notes
    -----
    The executed weight is a *notional* exposure fraction. We interpret weight
    changes as turnover in notional; absolute contract traded is proportional to
    |Δw| * AUM / (price * multiplier).
    """

    if aum <= 0:
        raise ValueError("aum must be > 0")
    w_exec = _check_series(w_exec, "w_exec").astype(float)
    price = _check_series(price, "price").astype(float)
    adv = _check_series(adv, "adv").astype(float)

    delta_w = w_exec.diff().fillna(0.0)
    q = participation_rate(
        delta_w,
        aum=aum,
        price=price,
        adv=adv,
        contract_multiplier=float(contract_multiplier),
    )
    return q


def estimate_aum_capacity(
    w_exec: pd.Series,
    *,
    price: pd.Series,
    adv: pd.Series,
    participation_cap: float = 0.01,
    contract_multiplier: float = 100.0,
    active_threshold: float = 1e-3,
) -> AUMCapacityResult:
    """Estimate AUM capacity from representative medians.

    Parameters
    ----------
    w_exec:
        Executed weights (notional). Capacity is based on the magnitude of
        executed weight changes.
    price:
        Futures price in $/oz.
    adv:
        Average daily volume in contracts/day.
    participation_cap:
        Participation cap, e.g. 0.01 = 1% of ADV.

    Returns
    -------
    AUMCapacityResult
        Includes AUM estimate and representative inputs used.
    """

    if not (0 < participation_cap < 1.0):
        raise ValueError("participation_cap must be in (0,1)")

    reps = representative_participation_inputs(
        _check_series(w_exec, "w_exec"),
        _check_series(price, "price"),
        _check_series(adv, "adv"),
        active_threshold=active_threshold,
    )

    rep_dw = float(reps["median_abs_delta_w_active"])
    rep_adv = float(reps["median_adv"])
    rep_price = float(reps["median_price"])

    if not np.isfinite(rep_dw) or rep_dw <= 0:
        # If no active days (or delta_w ~ 0), capacity is unbounded by this model.
        aum_max = float("inf")
        method = "median_abs_delta_w_active<=0"
    else:
        aum_max = float(participation_cap) * rep_price * float(contract_multiplier) * rep_adv / rep_dw
        method = "median_mapping"

    return AUMCapacityResult(
        aum_max=aum_max,
        q_cap=float(participation_cap),
        method=method,
        rep_delta_w=rep_dw,
        rep_price=rep_price,
        rep_adv=rep_adv,
        contract_multiplier=float(contract_multiplier),
    )


def aum_participation_summary(
    w_exec: pd.Series,
    *,
    aum: float,
    price: pd.Series,
    adv: pd.Series,
    contract_multiplier: float = 100.0,
) -> Tuple[pd.Series, ParticipationSummary]:
    """Convenience: compute q_t and summary stats for a given AUM."""

    q = participation_for_aum(
        w_exec,
        aum=aum,
        price=price,
        adv=adv,
        contract_multiplier=contract_multiplier,
    )
    summ = summarize_participation(q)
    return q, summ


def capacity_dict(res: AUMCapacityResult) -> Dict[str, float]:
    """Serialize AUMCapacityResult into JSON-friendly scalars."""

    return {
        "aum_max": float(res.aum_max) if np.isfinite(res.aum_max) else float("inf"),
        "q_cap": float(res.q_cap),
        "rep_delta_w": float(res.rep_delta_w),
        "rep_price": float(res.rep_price),
        "rep_adv": float(res.rep_adv),
        "contract_multiplier": float(res.contract_multiplier),
    }


__all__ = [
    "AUMCapacityResult",
    "participation_for_aum",
    "estimate_aum_capacity",
    "aum_participation_summary",
    "capacity_dict",
]
