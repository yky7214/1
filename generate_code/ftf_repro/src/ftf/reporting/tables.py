"""ftf.reporting.tables

Minimal tabular reporting helpers.

The reproduction plan calls for a reporting layer that can write summary tables
(performance stats, regression outputs, bootstrap CI, etc.). The project already
includes a pragmatic reporting script (scripts/07_report.py) that produces CSV/
JSON outputs directly. This module provides small, reusable helpers so other
scripts (e.g., capacity/latency/cost-impact) can emit consistent CSV tables.

These helpers are intentionally lightweight and dependency-free beyond pandas.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

from ftf.stats.metrics import summarize
from ftf.stats.regression import HACRegressionResult, result_to_dict

__all__ = [
    "performance_table",
    "regression_table",
    "dict_table",
]


def dict_table(rows: Iterable[Mapping[str, Any]], *, index_col: Optional[str] = None) -> pd.DataFrame:
    """Convert a list/iterable of dict-like rows into a DataFrame.

    Parameters
    ----------
    rows:
        Iterable of dict-like objects.
    index_col:
        If provided and present in each row, sets that key as index.

    Returns
    -------
    pd.DataFrame
    """

    df = pd.DataFrame(list(rows))
    if index_col is not None and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def performance_table(
    panel: Mapping[str, pd.Series],
    *,
    w_exec_panel: Optional[Mapping[str, pd.Series]] = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build a performance summary table for multiple strategies.

    Parameters
    ----------
    panel:
        Mapping name -> net return series.
    w_exec_panel:
        Optional mapping name -> executed weight series, used to compute
        active-day stats.

    Returns
    -------
    pd.DataFrame
        One row per strategy.
    """

    rows: list[Dict[str, Any]] = []
    for name, r in panel.items():
        w = None
        if w_exec_panel is not None:
            w = w_exec_panel.get(name)
        row = summarize(r, w_exec=w, periods_per_year=periods_per_year)
        row["name"] = name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("name").sort_index()
    return df


def regression_table(results: Mapping[int, HACRegressionResult]) -> pd.DataFrame:
    """Convert a NW-lag -> regression result mapping into a DataFrame."""

    rows = []
    for L, res in results.items():
        d = result_to_dict(res)
        d["nw_lags"] = L
        rows.append(d)

    df = pd.DataFrame(rows).set_index("nw_lags").sort_index()
    return df
