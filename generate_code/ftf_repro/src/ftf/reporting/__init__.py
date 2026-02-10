"""Reporting helpers.

This subpackage contains thin wrappers around metrics/regression output to build
CSV-friendly tables and (optional) matplotlib figures.

The core backtest logic lives elsewhere; reporting is intentionally lightweight
and safe to import even when optional plotting dependencies are missing.
"""

from .tables import dict_table, performance_table, regression_table
from .figures import FigurePaths, plot_drawdown, plot_equity_curve, plot_growth_curve

__all__ = [
    "dict_table",
    "performance_table",
    "regression_table",
    "FigurePaths",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_growth_curve",
]
