"""ftf.capacity.growth_curve

Capacity analysis: leverage-space growth proxy.

Implements the reproduction plan's reduced-form growth curve in leverage space L:

    g(L) ≈ μ_u L - 0.5 (σ_u L)^2 - n k L - γ (nL)^{3/2}

where μ_u and σ_u are estimated from a *unit-notional sleeve* return series (as in
Kelly estimation), and costs/impact use the same coefficients as the backtest.

We also provide a solver for L_max (the positive root where g(L)=0 beyond L=0).

This module is deterministic and intentionally lightweight; it does not run the
trading engine itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.utils.config import CostImpactConfig


@dataclass(frozen=True)
class GrowthCurveResult:
    """Outputs for a computed growth curve."""

    L: np.ndarray
    g: np.ndarray
    mu_u: float
    sigma_u: float
    n: float
    k_linear: float
    gamma_impact: float


def estimate_unit_notional_stats(R: pd.Series) -> Tuple[float, float]:
    """Estimate μ_u and σ_u from a unit-notional sleeve return series.

    Parameters
    ----------
    R:
        Daily sleeve return series, consistent with the project's timing
        convention (i.e., R[t] already reflects held exposure over t-1→t).

    Returns
    -------
    (mu, sigma):
        Mean and standard deviation of daily returns (population std, ddof=0).
    """

    if not isinstance(R, pd.Series):
        raise TypeError("R must be a pandas Series")
    x = pd.to_numeric(R, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return float("nan"), float("nan")
    mu = float(x.mean())
    sigma = float(x.std(ddof=0))
    return mu, sigma


def growth_proxy_L(
    L: np.ndarray | float,
    *,
    mu_u: float,
    sigma_u: float,
    n: float = 1.0,
    costs: Optional[CostImpactConfig] = None,
    k_linear: Optional[float] = None,
    gamma_impact: Optional[float] = None,
) -> np.ndarray | float:
    """Compute growth proxy g(L) for leverage L.

    Uses the same k/gamma coefficients as backtest unless overridden.

    Notes
    -----
    This is a reduced-form proxy; it is not a full microstructure model.
    """

    if costs is None:
        costs = CostImpactConfig()
    k = float(costs.k_linear if k_linear is None else k_linear)
    gamma = float(costs.gamma_impact if gamma_impact is None else gamma_impact)

    L_arr = np.asarray(L, dtype=float)
    # allow negative for diagnostics, but treat impact term with abs
    term_mu = mu_u * L_arr
    term_var = 0.5 * (sigma_u * L_arr) ** 2
    term_lin = n * k * L_arr
    term_imp = gamma * np.power(np.maximum(0.0, n * np.abs(L_arr)), 1.5)
    g = term_mu - term_var - term_lin - term_imp
    return g if isinstance(L, np.ndarray) else float(np.asarray(g).item())


def growth_curve(
    *,
    mu_u: float,
    sigma_u: float,
    L_grid: Optional[Iterable[float]] = None,
    L_max: float = 5.0,
    n: float = 1.0,
    costs: Optional[CostImpactConfig] = None,
    k_linear: Optional[float] = None,
    gamma_impact: Optional[float] = None,
) -> GrowthCurveResult:
    """Compute g(L) over a leverage grid."""

    if sigma_u < 0 or not np.isfinite(sigma_u):
        raise ValueError("sigma_u must be finite and nonnegative")
    if not np.isfinite(mu_u):
        raise ValueError("mu_u must be finite")
    if n <= 0:
        raise ValueError("n must be > 0")

    if L_grid is None:
        L = np.linspace(0.0, float(L_max), 201)
    else:
        L = np.asarray(list(L_grid), dtype=float)
        if L.ndim != 1:
            raise ValueError("L_grid must be 1-dimensional")

    g = growth_proxy_L(
        L,
        mu_u=mu_u,
        sigma_u=sigma_u,
        n=n,
        costs=costs,
        k_linear=k_linear,
        gamma_impact=gamma_impact,
    )

    if costs is None:
        costs = CostImpactConfig()
    k = float(costs.k_linear if k_linear is None else k_linear)
    gamma = float(costs.gamma_impact if gamma_impact is None else gamma_impact)

    return GrowthCurveResult(
        L=L,
        g=np.asarray(g, dtype=float),
        mu_u=float(mu_u),
        sigma_u=float(sigma_u),
        n=float(n),
        k_linear=k,
        gamma_impact=gamma,
    )


def solve_L_max(
    *,
    mu_u: float,
    sigma_u: float,
    n: float = 1.0,
    costs: Optional[CostImpactConfig] = None,
    k_linear: Optional[float] = None,
    gamma_impact: Optional[float] = None,
    bracket: Tuple[float, float] = (0.0, 10.0),
    grid_points: int = 2001,
) -> float:
    """Solve for L_max where g(L)=0 (positive root beyond 0).

    We avoid adding a hard scipy dependency by using a dense grid search with
    linear interpolation on the first sign change.

    Parameters
    ----------
    bracket:
        Range to search for the positive root.

    Returns
    -------
    float
        Approximate L_max. Returns 0.0 if no positive root is found.
    """

    lo, hi = float(bracket[0]), float(bracket[1])
    if lo < 0 or hi <= lo:
        raise ValueError("bracket must satisfy 0 <= lo < hi")
    if grid_points < 50:
        raise ValueError("grid_points too small")

    L = np.linspace(lo, hi, int(grid_points))
    g = np.asarray(
        growth_proxy_L(
            L,
            mu_u=mu_u,
            sigma_u=sigma_u,
            n=n,
            costs=costs,
            k_linear=k_linear,
            gamma_impact=gamma_impact,
        ),
        dtype=float,
    )

    # Look for first sign change after L=0.
    # g(0) should be ~0; we want where it crosses downwards.
    s = np.sign(g)
    # Treat tiny values as zero
    s[np.isclose(g, 0.0, atol=1e-15)] = 0.0

    # Find indices where sign changes from positive to negative or to zero then negative.
    for i in range(1, len(L)):
        if (s[i - 1] > 0 and s[i] <= 0) or (s[i - 1] == 0 and s[i] < 0):
            # interpolate between i-1 and i
            g0, g1 = g[i - 1], g[i]
            L0, L1 = L[i - 1], L[i]
            if np.isclose(g0, g1):
                return float(L1)
            t = (0.0 - g0) / (g1 - g0)
            return float(L0 + t * (L1 - L0))

    # If never crosses below zero, treat as no finite L_max in bracket.
    return 0.0


__all__ = [
    "GrowthCurveResult",
    "estimate_unit_notional_stats",
    "growth_proxy_L",
    "growth_curve",
    "solve_L_max",
]
