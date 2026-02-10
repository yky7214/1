"""ftf.reporting.figures

Lightweight plotting helpers.

The reproduction plan focuses on *functionality over presentation*.  This module
therefore implements a minimal set of utilities used by robustness/capacity
scripts and the report script.

All functions are optional dependencies: if matplotlib is not available, the
functions raise a clear ImportError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FigurePaths:
    equity_curve: Optional[str] = None
    drawdown: Optional[str] = None


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install matplotlib>=3.8"
        ) from e


def plot_equity_curve(net_ret: pd.Series, *, out_path: str, title: str = "Equity curve") -> None:
    """Plot and save an equity curve.

    Parameters
    ----------
    net_ret:
        Daily strategy net returns.
    out_path:
        Destination path (PNG recommended).
    title:
        Figure title.
    """

    if not isinstance(net_ret, pd.Series) or not isinstance(net_ret.index, pd.DatetimeIndex):
        raise TypeError("net_ret must be a pd.Series indexed by DatetimeIndex")

    plt = _require_matplotlib()

    x = net_ret.dropna()
    eq = (1.0 + x).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values, lw=1.5)
    ax.set_title(title)
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown(net_ret: pd.Series, *, out_path: str, title: str = "Drawdown") -> None:
    """Plot and save drawdown series."""

    if not isinstance(net_ret, pd.Series) or not isinstance(net_ret.index, pd.DatetimeIndex):
        raise TypeError("net_ret must be a pd.Series indexed by DatetimeIndex")

    plt = _require_matplotlib()

    x = net_ret.dropna()
    eq = (1.0 + x).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0.0, color="tab:red", alpha=0.3)
    ax.plot(dd.index, dd.values, lw=1.0, color="tab:red")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_growth_curve(L: np.ndarray, g: np.ndarray, *, out_path: str, title: str = "Growth curve") -> None:
    """Plot and save a leverage growth curve g(L)."""

    if len(L) != len(g):
        raise ValueError("L and g must have same length")

    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(L, g, lw=1.5)
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Leverage L")
    ax.set_ylabel("Growth proxy g(L)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = [
    "FigurePaths",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_growth_curve",
]
