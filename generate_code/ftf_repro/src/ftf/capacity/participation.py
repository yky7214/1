"""ftf.capacity.participation

Liquidity/participation primitives used for capacity mapping.

Implements the reproduction plan formulas:

- Convert weight/AUM changes to contracts traded:
    Δcontracts_t ≈ |Δw_t| * A / (P_contract_t * M)
  where:
    * Δw_t is change in *executed* weight (dimensionless notional fraction)
    * A is AUM in USD
    * P_contract_t is futures price in USD per ounce
    * M is contract multiplier (GC: 100 troy ounces)

- Participation rate:
    q_t = |Δcontracts_t| / ADV_t

All functions are deterministic and vectorized where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.utils.config import DataConfig


@dataclass(frozen=True)
class ParticipationSummary:
    """Summary stats for participation rate series."""

    n: int
    median: float
    p90: float
    p95: float
    p99: float
    mean: float
    max: float


def _check_series(x: pd.Series, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index must be a pandas DatetimeIndex")
    if x.index.has_duplicates:
        raise ValueError(f"{name}.index has duplicates")
    if not x.index.is_monotonic_increasing:
        raise ValueError(f"{name}.index must be monotonic increasing")
    return x


def contracts_delta(
    delta_w: pd.Series,
    *,
    aum: float,
    price: pd.Series,
    adv: Optional[pd.Series] = None,
    contract_multiplier: float = 100.0,
) -> pd.Series:
    """Convert executed weight changes into estimated contracts traded.

    Parameters
    ----------
    delta_w:
        Change in executed weight (e.g., `w_exec.diff()`), typically nonnegative
        if passed as absolute change.
    aum:
        Assets under management in USD.
    price:
        Futures price in USD per ounce.
    adv:
        Optional ADV series. Not required for contract conversion itself.
    contract_multiplier:
        GC multiplier (100 oz).

    Returns
    -------
    pd.Series
        Estimated absolute contracts traded (float), aligned on common dates.
    """

    if aum < 0:
        raise ValueError("aum must be nonnegative")
    if contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be positive")

    delta_w = _check_series(delta_w, "delta_w")
    price = _check_series(price, "price")
    if adv is not None:
        adv = _check_series(adv, "adv")

    # Align; adv is not used but include in alignment if supplied.
    if adv is None:
        dw, px = delta_w.align(price, join="inner")
    else:
        tmp = pd.concat({"dw": delta_w, "px": price, "adv": adv}, axis=1, join="inner")
        dw, px = tmp["dw"], tmp["px"]

    dw_abs = dw.abs()

    denom = px.astype(float) * float(contract_multiplier)
    denom = denom.replace(0.0, np.nan)
    contracts = (dw_abs.astype(float) * float(aum)) / denom

    contracts = contracts.replace([np.inf, -np.inf], np.nan)
    contracts.name = "delta_contracts"
    return contracts


def participation_rate(
    delta_w: pd.Series,
    *,
    aum: float,
    price: pd.Series,
    adv: pd.Series,
    contract_multiplier: float = 100.0,
) -> pd.Series:
    """Compute participation rate series q_t.

    q_t = |Δcontracts_t| / ADV_t

    ADV is expected in contracts/day.
    """

    adv = _check_series(adv, "adv")
    dc = contracts_delta(
        delta_w,
        aum=aum,
        price=price,
        adv=adv,
        contract_multiplier=contract_multiplier,
    )

    dc, adv_al = dc.align(adv, join="inner")
    adv_pos = adv_al.astype(float).replace(0.0, np.nan)
    q = (dc.astype(float) / adv_pos).replace([np.inf, -np.inf], np.nan)
    q.name = "participation"
    return q


def summarize_participation(q: pd.Series) -> ParticipationSummary:
    """Compute distribution summary for a participation series."""

    q = _check_series(q, "q")
    x = q.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) == 0:
        return ParticipationSummary(n=0, median=np.nan, p90=np.nan, p95=np.nan, p99=np.nan, mean=np.nan, max=np.nan)

    return ParticipationSummary(
        n=int(len(x)),
        median=float(x.quantile(0.50)),
        p90=float(x.quantile(0.90)),
        p95=float(x.quantile(0.95)),
        p99=float(x.quantile(0.99)),
        mean=float(x.mean()),
        max=float(x.max()),
    )


def representative_participation_inputs(
    w_exec: pd.Series,
    price: pd.Series,
    adv: pd.Series,
    *,
    active_threshold: float = 1e-3,
) -> Dict[str, float]:
    """Compute representative turnover/ADV/price statistics used for AUM mapping.

    The plan suggests using median |Δw| on active days and median ADV.

    Returns dict with:
      - median_abs_dw_active
      - median_adv
      - median_price
      - n_active_days
    """

    w_exec = _check_series(w_exec, "w_exec")
    price = _check_series(price, "price")
    adv = _check_series(adv, "adv")

    df = pd.concat({"w": w_exec, "px": price, "adv": adv}, axis=1, join="inner")

    held = df["w"].shift(1)
    active = held.abs() > float(active_threshold)
    abs_dw = df["w"].diff().abs()

    abs_dw_active = abs_dw.where(active)

    out = {
        "median_abs_dw_active": float(abs_dw_active.dropna().median()) if abs_dw_active.dropna().size else np.nan,
        "median_adv": float(df["adv"].dropna().median()) if df["adv"].dropna().size else np.nan,
        "median_price": float(df["px"].dropna().median()) if df["px"].dropna().size else np.nan,
        "n_active_days": int(active.sum()),
    }
    return out


__all__ = [
    "ParticipationSummary",
    "contracts_delta",
    "participation_rate",
    "summarize_participation",
    "representative_participation_inputs",
]
