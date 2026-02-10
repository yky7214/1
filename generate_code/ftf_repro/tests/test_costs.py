import numpy as np
import pandas as pd
import pytest

from ftf.execution.costs import compute_costs, turnover_from_exec


def test_turnover_and_costs_match_definition():
    # Build deterministic weights with known turnover.
    idx = pd.bdate_range("2020-01-01", periods=6)
    w_exec = pd.Series([0.0, 0.5, 0.5, 1.0, 0.25, 0.25], index=idx, name="w_exec")

    turnover = turnover_from_exec(w_exec)
    expected_turnover = pd.Series([0.0, 0.5, 0.0, 0.5, 0.75, 0.0], index=idx, name="turnover")
    pd.testing.assert_series_equal(turnover, expected_turnover)

    k = 0.0001
    gamma = 0.02
    cs = compute_costs(w_exec, k_linear=k, gamma_impact=gamma)

    # Costs follow deterministic formulas.
    exp_lin = expected_turnover * k
    exp_imp = expected_turnover.pow(1.5) * gamma
    exp_tot = exp_lin + exp_imp

    pd.testing.assert_series_equal(cs.turnover, expected_turnover)
    pd.testing.assert_series_equal(cs.cost_linear, exp_lin.rename("cost_linear"))
    pd.testing.assert_series_equal(cs.cost_impact, exp_imp.rename("cost_impact"))
    pd.testing.assert_series_equal(cs.cost_total, exp_tot.rename("cost_total"))


def test_costs_zero_when_weights_constant():
    idx = pd.bdate_range("2020-01-01", periods=10)
    w_exec = pd.Series(np.ones(len(idx)) * 0.7, index=idx)
    cs = compute_costs(w_exec, k_linear=0.00007, gamma_impact=0.02)
    assert float(cs.turnover.sum()) == pytest.approx(0.0)
    assert float(cs.cost_total.sum()) == pytest.approx(0.0)


def test_costs_nonnegative_and_increase_with_turnover():
    idx = pd.bdate_range("2020-01-01", periods=4)
    w_low = pd.Series([0.0, 0.1, 0.1, 0.2], index=idx)
    w_high = pd.Series([0.0, 1.0, 1.0, 2.0], index=idx)

    cs_low = compute_costs(w_low, k_linear=0.00007, gamma_impact=0.02)
    cs_high = compute_costs(w_high, k_linear=0.00007, gamma_impact=0.02)

    assert (cs_low.cost_total >= 0).all()
    assert (cs_high.cost_total >= 0).all()

    # Higher turnover path should have strictly higher total cost.
    assert float(cs_high.cost_total.sum()) > float(cs_low.cost_total.sum())
