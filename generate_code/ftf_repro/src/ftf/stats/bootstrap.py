"""ftf.stats.bootstrap

Bootstrap utilities used throughout the reproduction.

Implements:
- Block bootstrap (fixed block length)
- Stationary bootstrap (Politis & Romano)

Both return re-sampled series with the *same length* as the input and preserve
index values (resamples the *values* while keeping the original index).

The statistical procedures in this repo use these bootstraps for Sharpe CIs and
SPA / Reality Check p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "block_bootstrap_indices",
    "block_bootstrap",
    "stationary_bootstrap_indices",
    "stationary_bootstrap",
    "bootstrap_statistic",
    "SharpeCI",
    "bootstrap_sharpe_ci",
]


def _check_1d(x: pd.Series) -> None:
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series")
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("x must have a DatetimeIndex")


def block_bootstrap_indices(n: int, block_len: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate bootstrap indices using fixed-length blocks.

    Parameters
    ----------
    n:
        Series length.
    block_len:
        Block length (>=1).
    rng:
        Numpy generator.

    Returns
    -------
    np.ndarray
        Array of indices in [0, n-1] of length n.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if block_len <= 0:
        raise ValueError("block_len must be positive")

    out = np.empty(n, dtype=np.int64)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        end = min(pos + block_len, n)
        take = end - pos
        # wrap-around indexing
        idx = (start + np.arange(take, dtype=np.int64)) % n
        out[pos:end] = idx
        pos = end
    return out


def block_bootstrap(x: pd.Series, *, block_len: int = 20, rng: Optional[np.random.Generator] = None) -> pd.Series:
    """Block bootstrap resample of a series (resamples values, keeps index)."""

    _check_1d(x)
    rng = rng or np.random.default_rng()
    idx = block_bootstrap_indices(len(x), block_len, rng=rng)
    vals = x.to_numpy(copy=False)[idx]
    return pd.Series(vals, index=x.index, name=x.name)


def stationary_bootstrap_indices(
    n: int,
    mean_block_len: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate indices for the stationary bootstrap.

    mean_block_len is the expected block length. We use probability p=1/mean_block_len
    of starting a new block at each step.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if mean_block_len <= 0:
        raise ValueError("mean_block_len must be positive")

    p = 1.0 / float(mean_block_len)

    out = np.empty(n, dtype=np.int64)
    # initial start position
    j = int(rng.integers(0, n))
    for t in range(n):
        if t == 0:
            out[t] = j
            continue
        # with probability p, start a new block
        if rng.random() < p:
            j = int(rng.integers(0, n))
        else:
            j = (j + 1) % n
        out[t] = j
    return out


def stationary_bootstrap(
    x: pd.Series,
    *,
    mean_block_len: float = 20,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Stationary bootstrap resample of a series (resamples values, keeps index)."""

    _check_1d(x)
    rng = rng or np.random.default_rng()
    idx = stationary_bootstrap_indices(len(x), mean_block_len, rng=rng)
    vals = x.to_numpy(copy=False)[idx]
    return pd.Series(vals, index=x.index, name=x.name)


def bootstrap_statistic(
    x: pd.Series,
    stat_fn: Callable[[pd.Series], float],
    *,
    B: int,
    method: str = "block",
    block_len: int = 20,
    mean_block_len: float = 20,
    seed: int = 123,
) -> np.ndarray:
    """Generic bootstrap driver.

    Parameters
    ----------
    x:
        Input series.
    stat_fn:
        Function mapping series->scalar.
    B:
        Number of bootstrap replicates.
    method:
        "block" or "stationary".

    Returns
    -------
    np.ndarray
        Bootstrapped statistics of length B.
    """

    _check_1d(x)
    if B <= 0:
        raise ValueError("B must be positive")
    method = method.lower()
    if method not in {"block", "stationary"}:
        raise ValueError("method must be 'block' or 'stationary'")

    rng = np.random.default_rng(int(seed))

    out = np.empty(B, dtype=float)
    for b in range(B):
        if method == "block":
            xb = block_bootstrap(x, block_len=block_len, rng=rng)
        else:
            xb = stationary_bootstrap(x, mean_block_len=mean_block_len, rng=rng)
        out[b] = float(stat_fn(xb))
    return out


@dataclass(frozen=True)
class SharpeCI:
    sharpe: float
    ci_low: float
    ci_high: float
    method: str
    B: int
    block_len: int


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3:
        return float("nan")
    m = float(x.mean())
    s = float(x.std(ddof=0))
    if s <= 0:
        return float("nan")
    return (m / s) * np.sqrt(252.0)


def bootstrap_sharpe_ci(
    net_ret: pd.Series,
    *,
    B: int = 1000,
    block_len: int = 20,
    alpha: float = 0.05,
    seed: int = 123,
) -> SharpeCI:
    """Block bootstrap Sharpe confidence interval (percentile CI)."""

    s0 = _sharpe(net_ret)
    stats = bootstrap_statistic(
        net_ret,
        _sharpe,
        B=B,
        method="block",
        block_len=block_len,
        seed=seed,
    )
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return SharpeCI(sharpe=s0, ci_low=float("nan"), ci_high=float("nan"), method="block", B=B, block_len=block_len)
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return SharpeCI(sharpe=s0, ci_low=float(lo), ci_high=float(hi), method="block", B=B, block_len=block_len)
