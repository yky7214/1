"""ftf.stats.regression

Benchmark regression utilities.

Implements the plan's benchmark-neutrality evaluation:

  r_strat,t = alpha_d + beta * r_bench,t + eps_t

with HAC (Newey–West) standard errors.

The implementation prefers statsmodels when available. If statsmodels is not
installed, it falls back to a small internal OLS + Newey–West estimator.

All inputs are daily return Series indexed by timezone-naive DatetimeIndex.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:  # optional
    import statsmodels.api as sm  # type: ignore

    _HAS_SM = True
except Exception:  # pragma: no cover
    sm = None
    _HAS_SM = False


@dataclass(frozen=True)
class HACRegressionResult:
    """Results bundle for benchmark regression with HAC errors."""

    alpha_daily: float
    beta: float
    alpha_se: float
    beta_se: float
    alpha_t: float
    beta_t: float
    nw_lags: int
    n_obs: int
    r2: float
    te_ann: float
    alpha_ann: float
    ir: float


def _check_series(x: pd.Series, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index must be a DatetimeIndex")
    if x.index.tz is not None:
        x = x.copy()
        x.index = x.index.tz_convert(None)  # pragma: no cover
    if not x.index.is_monotonic_increasing:
        raise ValueError(f"{name}.index must be monotonic increasing")
    if x.index.has_duplicates:
        raise ValueError(f"{name}.index must not contain duplicates")
    return x


def align_returns(
    strat_ret: pd.Series, bench_ret: pd.Series, *, dropna: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """Align strategy and benchmark returns on the common date intersection."""

    s = _check_series(strat_ret, "strat_ret")
    b = _check_series(bench_ret, "bench_ret")

    s2, b2 = s.align(b, join="inner")
    if dropna:
        m = np.isfinite(s2.to_numpy(dtype=float)) & np.isfinite(b2.to_numpy(dtype=float))
        s2 = s2.loc[m]
        b2 = b2.loc[m]
    return s2.astype(float), b2.astype(float)


def _ols_nw_fallback(y: np.ndarray, x: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """OLS with Newey–West covariance for (const, x).

    Returns (beta_hat, se_hat, r2).

    This is a minimal, deterministic implementation meant as a fallback.
    """

    n = y.shape[0]
    if n <= 3:
        raise ValueError("Not enough observations for regression")

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x])

    # OLS
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # R^2
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)

    # Newey–West / HAC
    # S = sum_{k=-L..L} w_k * Gamma_k ; with Bartlett weights
    # where Gamma_k = sum_t X_t' e_t e_{t-k} X_{t-k}
    L = int(max(0, lags))
    # Compute meat
    S = np.zeros((2, 2), dtype=float)
    Xe = X * resid[:, None]
    # lag 0
    S += Xe.T @ Xe
    # lags
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        Gamma = Xe[k:].T @ Xe[:-k]
        S += w * (Gamma + Gamma.T)

    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se, float(r2)


def hac_regression(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    *,
    nw_lags: int = 10,
    periods_per_year: int = 252,
) -> HACRegressionResult:
    """Run strategy-vs-benchmark regression with HAC (Newey–West) SE.

    Parameters
    ----------
    strat_ret:
        Strategy daily returns.
    bench_ret:
        Benchmark daily returns.
    nw_lags:
        Newey–West lag length.
    periods_per_year:
        Annualization factor.

    Returns
    -------
    HACRegressionResult
    """

    y_s, x_b = align_returns(strat_ret, bench_ret, dropna=True)
    y = y_s.to_numpy(dtype=float)
    x = x_b.to_numpy(dtype=float)

    if y.shape[0] < max(30, nw_lags + 5):
        # still run, but warn via deterministic behavior: allow small samples
        pass

    if _HAS_SM:
        X = sm.add_constant(x, has_constant="add")
        model = sm.OLS(y, X, missing="drop")
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(nw_lags)})
        params = res.params
        bse = res.bse
        tvals = res.tvalues
        r2 = float(res.rsquared)
        resid = res.resid
        alpha_d = float(params[0])
        beta = float(params[1])
        alpha_se = float(bse[0])
        beta_se = float(bse[1])
        alpha_t = float(tvals[0])
        beta_t = float(tvals[1])
    else:  # pragma: no cover
        beta_hat, se_hat, r2 = _ols_nw_fallback(y, x, int(nw_lags))
        resid = y - (beta_hat[0] + beta_hat[1] * x)
        alpha_d = float(beta_hat[0])
        beta = float(beta_hat[1])
        alpha_se = float(se_hat[0])
        beta_se = float(se_hat[1])
        alpha_t = float(alpha_d / alpha_se) if alpha_se > 0 else np.nan
        beta_t = float(beta / beta_se) if beta_se > 0 else np.nan

    # Tracking error and information ratio
    resid = np.asarray(resid, dtype=float)
    te_ann = float(np.std(resid, ddof=0) * np.sqrt(periods_per_year))
    alpha_ann = float(periods_per_year * alpha_d)
    ir = float(alpha_ann / te_ann) if te_ann > 0 else np.nan

    return HACRegressionResult(
        alpha_daily=alpha_d,
        beta=beta,
        alpha_se=alpha_se,
        beta_se=beta_se,
        alpha_t=alpha_t,
        beta_t=beta_t,
        nw_lags=int(nw_lags),
        n_obs=int(len(y)),
        r2=r2,
        te_ann=te_ann,
        alpha_ann=alpha_ann,
        ir=ir,
    )


def hac_regression_sensitivity(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    *,
    nw_lags_list: Iterable[int] = (5, 10, 20),
    periods_per_year: int = 252,
) -> Dict[int, HACRegressionResult]:
    """Run HAC regression across multiple NW lag choices."""

    out: Dict[int, HACRegressionResult] = {}
    for L in nw_lags_list:
        out[int(L)] = hac_regression(
            strat_ret, bench_ret, nw_lags=int(L), periods_per_year=periods_per_year
        )
    return out


def result_to_dict(res: HACRegressionResult) -> Dict[str, float]:
    """Convert to a JSON/YAML-friendly dict."""

    d = asdict(res)
    # Ensure plain python scalars
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}


__all__ = [
    "HACRegressionResult",
    "align_returns",
    "hac_regression",
    "hac_regression_sensitivity",
    "result_to_dict",
]
