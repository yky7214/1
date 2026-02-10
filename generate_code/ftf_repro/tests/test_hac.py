import numpy as np
import pandas as pd
import pytest

from ftf.stats.regression import hac_regression


def _mk_returns(n: int = 800, seed: int = 0, beta: float = 0.0, alpha: float = 0.0002):
    """Generate synthetic daily returns with controllable alpha/beta.

    We keep the benchmark i.i.d. normal and add noise; this is sufficient to test
    basic correctness (alignment, sign, and approximate magnitude).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0, 0.01, size=n), index=idx, name="bench")
    eps = rng.normal(0.0, 0.01, size=n)
    strat = pd.Series(alpha + beta * bench.to_numpy() + eps, index=idx, name="strat")
    return strat, bench


def test_hac_regression_beta_near_zero_when_constructed():
    strat, bench = _mk_returns(beta=0.0)
    res = hac_regression(strat, bench, nw_lags=10)

    assert np.isfinite(res.beta)
    assert abs(res.beta) < 0.10


def test_hac_regression_recovers_positive_beta():
    strat, bench = _mk_returns(beta=0.6)
    res = hac_regression(strat, bench, nw_lags=10)

    assert np.isfinite(res.beta)
    assert res.beta > 0.4
    assert res.beta < 0.8


def test_hac_regression_alpha_annualization_sanity():
    # alpha_daily=20 bps/year approx => 0.0002 daily ~ 5% annual
    strat, bench = _mk_returns(beta=0.0, alpha=0.0002)
    res = hac_regression(strat, bench, nw_lags=10)

    assert np.isfinite(res.alpha_ann)
    assert res.alpha_ann > 0.0
    assert res.alpha_ann == pytest.approx(252.0 * res.alpha_daily)


def test_hac_regression_alignment_inner_join():
    strat, bench = _mk_returns(n=200)

    # Remove a chunk from bench; regression should use intersection.
    bench2 = bench.drop(bench.index[10:30])
    res = hac_regression(strat, bench2, nw_lags=5)

    assert res.n_obs == len(strat.index.intersection(bench2.index))
    assert res.n_obs < len(strat)
