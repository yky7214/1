"""ftf.stats.spa

Implements White's Reality Check (RC) and Hansen's Superior Predictive Ability (SPA)
style tests for a family of strategy configurations.

This is not a full academic implementation of all SPA nuances, but it follows the
reproduction plan:

- Build a grid of alternative configurations.
- For each configuration i compute a performance differential series d_i,t
  relative to a baseline 0 (default: daily net return differential).
- Test statistic: max_i mean(d_i) (optionally studentized).
- Use stationary or block bootstrap to estimate p-values.

The code is deterministic and designed to work with the rest of the pipeline.

References (high-level):
- White (2000) "A Reality Check for Data Snooping"
- Hansen (2005) "A Test for Superior Predictive Ability"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.stats.bootstrap import (
    block_bootstrap,
    bootstrap_statistic,
    stationary_bootstrap,
)


BootstrapMethod = Literal["block", "stationary"]
TestKind = Literal["RC", "SPA"]
DiffMetric = Literal["mean", "sharpe"]


@dataclass(frozen=True)
class SPAResult:
    test_kind: TestKind
    metric: DiffMetric
    B: int
    method: BootstrapMethod
    block_len: int
    mean_block_len: int
    t_obs: float
    p_value: float
    t_boot: np.ndarray
    best_name: str
    best_value: float


def _check_panel(panel: Dict[str, pd.Series]) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]:
    if not isinstance(panel, dict) or len(panel) == 0:
        raise TypeError("panel must be a non-empty dict of name -> pd.Series")
    for k, v in panel.items():
        if not isinstance(v, pd.Series):
            raise TypeError(f"panel[{k!r}] must be a pd.Series")
        if not isinstance(v.index, pd.DatetimeIndex):
            raise TypeError(f"panel[{k!r}] must have DatetimeIndex")
    # inner-join across all series
    idx = None
    for v in panel.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    if idx is None or len(idx) == 0:
        raise ValueError("No overlapping dates across panel series")
    aligned = {k: v.reindex(idx) for k, v in panel.items()}
    return idx, aligned


def _annualized_sharpe(x: pd.Series) -> float:
    y = x.dropna().to_numpy(dtype=float)
    if y.size < 3:
        return float("nan")
    mu = float(np.mean(y))
    sd = float(np.std(y, ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return mu / sd * float(np.sqrt(252.0))


def _metric_value(x: pd.Series, metric: DiffMetric) -> float:
    if metric == "mean":
        return float(x.dropna().mean())
    if metric == "sharpe":
        return _annualized_sharpe(x)
    raise ValueError(f"Unknown metric: {metric}")


def compute_differentials(
    panel: Dict[str, pd.Series],
    *,
    baseline_name: str,
) -> Dict[str, pd.Series]:
    """Compute differential return series d_i = r_i - r_0.

    Parameters
    ----------
    panel:
        Mapping from configuration name -> daily return series.
    baseline_name:
        Name of baseline series within panel.

    Returns
    -------
    dict
        Mapping name -> differential series. Baseline is excluded.
    """

    _, aligned = _check_panel(panel)
    if baseline_name not in aligned:
        raise KeyError(f"baseline_name {baseline_name!r} not found in panel")
    base = aligned[baseline_name]
    out: Dict[str, pd.Series] = {}
    for name, s in aligned.items():
        if name == baseline_name:
            continue
        d = (s - base).rename(f"d_{name}_minus_{baseline_name}")
        out[name] = d
    return out


def _test_stat_from_diffs(diffs: Dict[str, pd.Series], metric: DiffMetric) -> Tuple[float, str, float]:
    vals: Dict[str, float] = {k: _metric_value(v, metric) for k, v in diffs.items()}
    # Treat NaN as -inf (cannot be best)
    best_name = max(vals.keys(), key=lambda k: (-np.inf if not np.isfinite(vals[k]) else vals[k]))
    best_value = float(vals[best_name])
    t_obs = float(np.nanmax(list(vals.values())))
    return t_obs, best_name, best_value


def _bootstrap_max_stat(
    diffs: Dict[str, pd.Series],
    metric: DiffMetric,
    *,
    method: BootstrapMethod,
    block_len: int,
    mean_block_len: int,
    seed: int,
) -> np.ndarray:
    # We'll resample each diff series with the same indices by resampling one
    # reference and applying those indices. Since our bootstrap helpers return a
    # resampled Series (values only) but preserve original index, easiest is to
    # bootstrap a matrix by shared indices.
    names = list(diffs.keys())
    idx = None
    for s in diffs.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    if idx is None or len(idx) == 0:
        raise ValueError("No overlapping dates for differentials")

    X = np.column_stack([diffs[n].reindex(idx).to_numpy(dtype=float) for n in names])

    rng = np.random.default_rng(seed)
    n = X.shape[0]

    def one_draw() -> float:
        # resample indices
        if method == "block":
            # re-use block_bootstrap_indices via bootstrap_statistic indirectly by
            # sampling a dummy series and taking its index mapping.
            dummy = pd.Series(np.arange(n, dtype=float), index=idx)
            boot = block_bootstrap(dummy, block_len=block_len, rng=rng)
            sel = boot.to_numpy(dtype=int)
        else:
            dummy = pd.Series(np.arange(n, dtype=float), index=idx)
            boot = stationary_bootstrap(dummy, mean_block_len=mean_block_len, rng=rng)
            sel = boot.to_numpy(dtype=int)

        Xb = X[sel, :]
        # compute metric per config on bootstrap sample
        if metric == "mean":
            vb = np.nanmean(Xb, axis=0)
            return float(np.nanmax(vb))
        # sharpe
        out = []
        for j in range(Xb.shape[1]):
            x = Xb[:, j]
            x = x[np.isfinite(x)]
            if x.size < 3:
                out.append(np.nan)
                continue
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=0))
            out.append(np.nan if sd <= 0 else mu / sd * float(np.sqrt(252.0)))
        return float(np.nanmax(out))

    # vectorize loop
    B = 1  # placeholder; caller uses bootstrap_statistic? We'll implement direct draws below.
    raise RuntimeError("internal")


def spa_reality_check(
    panel: Dict[str, pd.Series],
    *,
    baseline_name: str,
    test_kind: TestKind = "SPA",
    metric: DiffMetric = "mean",
    method: BootstrapMethod = "stationary",
    B: int = 800,
    block_len: int = 20,
    mean_block_len: int = 20,
    seed: int = 123,
    studentize: bool = False,
) -> SPAResult:
    """Compute SPA or Reality Check p-value for a panel of strategies.

    Parameters
    ----------
    panel:
        Mapping name -> daily net return series.
    baseline_name:
        Baseline strategy name.
    test_kind:
        "SPA" or "RC". Currently affects only labeling; both use the same
        max-mean-differential test statistic in this implementation.
    metric:
        Differential metric: "mean" (default) or "sharpe".
    method:
        Bootstrap method: "stationary" or "block".
    studentize:
        If True, approximate studentization by dividing each diff mean by its
        sample std/sqrt(n). This is a rough option; default False.

    Returns
    -------
    SPAResult
        Contains observed statistic, bootstrap distribution, and p-value.
    """

    if method not in ("block", "stationary"):
        raise ValueError("method must be 'block' or 'stationary'")
    if metric not in ("mean", "sharpe"):
        raise ValueError("metric must be 'mean' or 'sharpe'")
    if B <= 50:
        raise ValueError("B too small for SPA/RC")

    diffs = compute_differentials(panel, baseline_name=baseline_name)

    # Optionally studentize the differential series per configuration.
    if studentize and metric == "mean":
        diffs2: Dict[str, pd.Series] = {}
        for k, d in diffs.items():
            x = d.dropna()
            if len(x) < 3:
                diffs2[k] = d * np.nan
                continue
            se = float(np.std(x.to_numpy(dtype=float), ddof=0)) / float(np.sqrt(len(x)))
            if se <= 0 or not np.isfinite(se):
                diffs2[k] = d * np.nan
            else:
                diffs2[k] = (d / se).rename(d.name)
        diffs = diffs2

    t_obs, best_name, best_value = _test_stat_from_diffs(diffs, metric)

    # Build a matrix aligned across diffs
    names = list(diffs.keys())
    idx = None
    for s in diffs.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    if idx is None or len(idx) == 0:
        raise ValueError("No overlapping dates across differentials")
    X = np.column_stack([diffs[n].reindex(idx).to_numpy(dtype=float) for n in names])

    rng = np.random.default_rng(seed)
    n = X.shape[0]

    def resample_indices() -> np.ndarray:
        # Resample integer indices 0..n-1 using chosen method.
        # We use bootstrap helpers on a dummy series of ints.
        dummy = pd.Series(np.arange(n, dtype=float), index=idx)
        if method == "block":
            boot = block_bootstrap(dummy, block_len=block_len, rng=rng)
        else:
            boot = stationary_bootstrap(dummy, mean_block_len=mean_block_len, rng=rng)
        return boot.to_numpy(dtype=int)

    t_boot = np.empty(B, dtype=float)
    for b in range(B):
        sel = resample_indices()
        Xb = X[sel, :]
        if metric == "mean":
            vb = np.nanmean(Xb, axis=0)
            t_boot[b] = float(np.nanmax(vb))
        else:
            vals = []
            for j in range(Xb.shape[1]):
                x = Xb[:, j]
                x = x[np.isfinite(x)]
                if x.size < 3:
                    vals.append(np.nan)
                    continue
                mu = float(np.mean(x))
                sd = float(np.std(x, ddof=0))
                vals.append(np.nan if sd <= 0 else mu / sd * float(np.sqrt(252.0)))
            t_boot[b] = float(np.nanmax(vals))

    # p-value: proportion of bootstrap stats >= observed
    # Add +1 smoothing to avoid 0 p-values due to finite B.
    p = (1.0 + float(np.sum(t_boot >= t_obs))) / float(B + 1)

    return SPAResult(
        test_kind=test_kind,
        metric=metric,
        B=B,
        method=method,
        block_len=int(block_len),
        mean_block_len=int(mean_block_len),
        t_obs=float(t_obs),
        p_value=float(p),
        t_boot=t_boot,
        best_name=str(best_name),
        best_value=float(best_value),
    )


__all__ = [
    "BootstrapMethod",
    "TestKind",
    "DiffMetric",
    "SPAResult",
    "compute_differentials",
    "spa_reality_check",
]
