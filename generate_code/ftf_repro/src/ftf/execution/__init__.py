"""Execution subpackage.

This package contains deterministic, daily-frequency execution modeling helpers
used by the Forecast-to-Fill (FTF) pipeline:

- Latency/forecast-to-fill lag buffer (targets -> executed weights)
- Gross return attribution consistent with the project convention
- Turnover-based linear costs and concave impact costs

The subpackage is intentionally lightweight and purely deterministic.
"""

from .latency import apply_exec_lag
from .fills import FillResult, compute_gross_return, fill_from_targets
from .costs import CostSeries, apply_costs_to_returns, compute_costs, turnover_from_exec

__all__ = [
    "apply_exec_lag",
    "FillResult",
    "compute_gross_return",
    "fill_from_targets",
    "CostSeries",
    "turnover_from_exec",
    "compute_costs",
    "apply_costs_to_returns",
]
