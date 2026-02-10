"""Statistical evaluation subpackage.

This package contains deterministic implementations of:
- Performance metrics (Sharpe/CAGR/MaxDD/active-day stats)
- HAC/Newey–West benchmark regression
- Time-series bootstraps (block + stationary)
- SPA / White Reality Check data-snooping tests

The goal is to provide a small, stable import surface for scripts and
reporting modules.
"""

from .bootstrap import (
    SharpeCI,
    block_bootstrap,
    block_bootstrap_indices,
    bootstrap_sharpe_ci,
    bootstrap_statistic,
    stationary_bootstrap,
    stationary_bootstrap_indices,
)
from .metrics import (
    ActiveDayStats,
    PerfStats,
    active_day_stats,
    annualized_sharpe,
    annualized_vol,
    cagr_from_returns,
    equity_curve,
    max_drawdown,
    perf_stats,
    summarize,
)
from .regression import (
    HACRegressionResult,
    align_returns,
    hac_regression,
    hac_regression_sensitivity,
    result_to_dict,
)
from .spa import (
    DiffMetric,
    SPAResult,
    TestKind,
    compute_differentials,
    spa_reality_check,
)

__all__ = [
    # metrics
    "PerfStats",
    "ActiveDayStats",
    "annualized_sharpe",
    "equity_curve",
    "max_drawdown",
    "cagr_from_returns",
    "annualized_vol",
    "perf_stats",
    "active_day_stats",
    "summarize",
    # regression
    "HACRegressionResult",
    "align_returns",
    "hac_regression",
    "hac_regression_sensitivity",
    "result_to_dict",
    # bootstrap
    "SharpeCI",
    "block_bootstrap_indices",
    "block_bootstrap",
    "stationary_bootstrap_indices",
    "stationary_bootstrap",
    "bootstrap_statistic",
    "bootstrap_sharpe_ci",
    # SPA / RC
    "SPAResult",
    "TestKind",
    "DiffMetric",
    "compute_differentials",
    "spa_reality_check",
]
