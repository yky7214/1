"""ftf.sizing.kelly

Friction- and impact-adjusted Kelly sizing.

Implements the closed-form optimizer described in the reproduction plan.

We model a (unit-notional) sleeve return series R_t, and compute its mean/variance
on the *training* window only (handled by the trainer). Given (mu, sigma2) and
cost/impact parameters, the paper uses a reduced-form growth proxy:

    g(f) ≈ μ f - 0.5 σ^2 f^2 - n k f - γ (n f)^{3/2}

with n=1 by default.

The derivative in x = sqrt(f) gives a quadratic with the positive root used.

This module is deliberately independent of the trading engine; it only solves
for Kelly fraction and provides helpers to estimate μ, σ² from a return series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ftf.utils.config import CostImpactConfig, KellyConfig

__all__ = [
    "KellyInputs",
    "estimate_kelly_inputs",
    "growth_proxy",
    "solve_friction_adjusted_kelly",
    "fractional_kelly",
]


@dataclass(frozen=True)
class KellyInputs:
    """Frozen Kelly inputs, usually estimated on a training slice."""

    mu: float
    sigma2: float
    n: float = 1.0


def estimate_kelly_inputs(R_train: pd.Series, *, n: float = 1.0) -> KellyInputs:
    """Estimate mean/variance for the unit-notional sleeve.

    Parameters
    ----------
    R_train:
        Training sleeve returns. Can include zeros when flat.
    n:
        Unit notional / activity scaling (paper uses n=1).

    Returns
    -------
    KellyInputs
    """
    if not isinstance(R_train, pd.Series):
        raise TypeError("R_train must be a pandas Series")
    x = R_train.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return KellyInputs(mu=0.0, sigma2=0.0, n=float(n))
    mu = float(np.mean(x))
    sigma2 = float(np.var(x, ddof=0))
    sigma2 = max(sigma2, 0.0)
    return KellyInputs(mu=mu, sigma2=sigma2, n=float(n))


def growth_proxy(
    f: np.ndarray | float,
    *,
    inputs: KellyInputs,
    k_linear: float,
    gamma_impact: float,
) -> np.ndarray | float:
    """Compute the reduced-form growth proxy g(f).

    Accepts scalar or numpy array f.
    """
    mu, sigma2, n = inputs.mu, inputs.sigma2, inputs.n
    f_arr = np.asarray(f, dtype=float)
    # Ensure nonnegative f in proxy
    f_pos = np.maximum(f_arr, 0.0)
    g = mu * f_pos - 0.5 * sigma2 * f_pos**2 - n * k_linear * f_pos - gamma_impact * (n * f_pos) ** 1.5
    if np.isscalar(f):
        return float(np.asarray(g).item())
    return g


def solve_friction_adjusted_kelly(
    *,
    inputs: KellyInputs,
    costs: Optional[CostImpactConfig] = None,
    k_linear: Optional[float] = None,
    gamma_impact: Optional[float] = None,
) -> float:
    """Closed-form friction-adjusted Kelly optimum f*.

    Implements the plan exactly:

    If μ ≤ n k => f* = 0.

    Else let x = sqrt(f) and solve:
        2σ^2 x^2 + 3γ n^{3/2} x - 2(μ - n k) = 0

    with positive root:
        x* = (-3γ n^{3/2} + sqrt(9γ^2 n^3 + 16 σ^2 (μ - n k))) / (4σ^2)
        f* = x*^2

    Notes
    -----
    - When sigma2 is ~0, the quadratic degenerates; we fall back to a small
      numeric safeguard.
    """
    if costs is None:
        costs = CostImpactConfig()  # defaults
    if k_linear is None:
        k_linear = costs.k_linear
    if gamma_impact is None:
        gamma_impact = costs.gamma_impact

    mu = float(inputs.mu)
    sigma2 = float(inputs.sigma2)
    n = float(inputs.n)

    # Edge: no edge after linear costs
    if mu <= n * k_linear:
        return 0.0

    # Edge: vanishing variance -> objective dominated by cost/impact; treat as small variance
    sigma2_eff = max(sigma2, 1e-18)

    a = 2.0 * sigma2_eff
    b = 3.0 * gamma_impact * (n ** 1.5)
    c = -2.0 * (mu - n * k_linear)

    disc = b * b - 4.0 * a * c
    disc = max(disc, 0.0)

    # positive root for x
    x_star = (-b + np.sqrt(disc)) / (2.0 * a)
    x_star = max(float(x_star), 0.0)
    f_star = x_star * x_star

    # Guard against pathological huge values in ultra-low sigma2 settings.
    if not np.isfinite(f_star):
        return 0.0
    return float(f_star)


def fractional_kelly(
    f_star: float,
    *,
    kelly_cfg: Optional[KellyConfig] = None,
    lambda_kelly: Optional[float] = None,
) -> float:
    """Compute fractional Kelly f_tilde = lambda_kelly * f_star."""
    if kelly_cfg is None:
        kelly_cfg = KellyConfig()
    if lambda_kelly is None:
        lambda_kelly = kelly_cfg.lambda_kelly
    if lambda_kelly < 0:
        raise ValueError("lambda_kelly must be nonnegative")
    f_star = float(f_star)
    if f_star <= 0:
        return 0.0
    return float(lambda_kelly * f_star)
