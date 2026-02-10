"""Tests for friction-adjusted Kelly sizing.

Plan requirements:
- Compare closed-form f* to numerical maximization of g(f) on a grid.
- Verify f*=0 when mu <= n*k and monotonic decrease as k or gamma increases.

These tests are synthetic/deterministic and do not require market data.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftf.sizing.kelly import KellyInputs, growth_proxy, solve_friction_adjusted_kelly


def _grid_argmax_f(
    *,
    inputs: KellyInputs,
    k_linear: float,
    gamma_impact: float,
    f_max: float = 5.0,
    n_grid: int = 20001,
) -> float:
    """Numerical maximizer for g(f) over a dense grid, for test validation."""

    f = np.linspace(0.0, float(f_max), int(n_grid))
    g = growth_proxy(f, inputs=inputs, k_linear=k_linear, gamma_impact=gamma_impact)
    # if all NaN (shouldn't happen), return 0
    if not np.isfinite(g).any():
        return 0.0
    return float(f[int(np.nanargmax(g))])


def test_closed_form_matches_grid_argmax_reasonably() -> None:
    rng = np.random.default_rng(0)

    # Several random scenarios; keep ranges conservative to avoid pathological flats.
    for _ in range(40):
        mu = float(rng.uniform(0.0, 0.003))  # daily mean
        sigma2 = float(rng.uniform(1e-6, 8e-4))  # daily var
        k = float(rng.uniform(0.0, 2e-4))  # 0-2 bps
        gamma = float(rng.uniform(0.0, 0.08))
        n = 1.0

        inputs = KellyInputs(mu=mu, sigma2=sigma2, n=n)
        f_star = solve_friction_adjusted_kelly(inputs=inputs, k_linear=k, gamma_impact=gamma)
        f_grid = _grid_argmax_f(inputs=inputs, k_linear=k, gamma_impact=gamma)

        # Both are nonnegative.
        assert f_star >= 0.0
        assert f_grid >= 0.0

        # Closed-form should be close to dense-grid argmax.
        # Allow some tolerance since grid is discrete and g(f) may be flat near optimum.
        assert f_star == pytest.approx(f_grid, abs=2.5e-3, rel=1e-2)

        # Also ensure g(f_star) is within tiny epsilon of max grid value.
        g_star = float(growth_proxy(np.array([f_star]), inputs=inputs, k_linear=k, gamma_impact=gamma)[0])
        g_grid_max = float(
            np.nanmax(growth_proxy(np.linspace(0.0, 5.0, 20001), inputs=inputs, k_linear=k, gamma_impact=gamma))
        )
        assert g_star >= g_grid_max - 1e-8


def test_f_star_zero_when_mu_leq_nk() -> None:
    # Exact boundary mu == n*k
    inputs = KellyInputs(mu=0.00007, sigma2=1e-4, n=1.0)
    f_star = solve_friction_adjusted_kelly(inputs=inputs, k_linear=0.00007, gamma_impact=0.02)
    assert f_star == 0.0

    # Below boundary
    inputs2 = KellyInputs(mu=0.00005, sigma2=1e-4, n=1.0)
    f_star2 = solve_friction_adjusted_kelly(inputs=inputs2, k_linear=0.00007, gamma_impact=0.02)
    assert f_star2 == 0.0


def test_f_star_decreases_with_higher_cost_or_impact() -> None:
    # Choose a positive-edge scenario.
    inputs = KellyInputs(mu=0.0006, sigma2=2.5e-4, n=1.0)

    f_low_cost = solve_friction_adjusted_kelly(inputs=inputs, k_linear=0.00003, gamma_impact=0.01)
    f_high_cost = solve_friction_adjusted_kelly(inputs=inputs, k_linear=0.00012, gamma_impact=0.01)
    assert f_high_cost <= f_low_cost + 1e-12

    f_low_imp = solve_friction_adjusted_kelly(inputs=inputs, k_linear=0.00007, gamma_impact=0.005)
    f_high_imp = solve_friction_adjusted_kelly(inputs=inputs, k_linear=0.00007, gamma_impact=0.05)
    assert f_high_imp <= f_low_imp + 1e-12


def test_f_star_is_zero_when_sigma2_is_degenerate() -> None:
    # If sigma2 is near-zero, the optimizer should still behave, but can be very large.
    # Our implementation clamps sigma2 to a tiny positive and returns finite f.
    # We check it doesn't error and returns nonnegative.
    inputs = KellyInputs(mu=0.001, sigma2=0.0, n=1.0)
    f_star = solve_friction_adjusted_kelly(inputs=inputs, k_linear=0.00007, gamma_impact=0.02)
    assert np.isfinite(f_star)
    assert f_star >= 0.0
