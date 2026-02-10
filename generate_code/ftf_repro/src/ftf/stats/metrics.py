"""ftf.stats.metrics

Performance and attribution metrics.

Implements the metrics specified in the reproduction plan:
- Sharpe (annualized)
- CAGR
- Annual volatility
- Max drawdown and Calmar
- Active-day stats (active if w_exec[t-1] > 1e-3): hit rate, payoff ratio, expectancy (bps)

All functions are deterministic and operate on daily return series that follow
project conventions (see :mod:`ftf.utils.config`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerfStats:
    sharpe: float
    cagr: float
    ann_vol: float
    max_dd: float
    calmar: float
    mean_daily: float
    ann_mean: float
    n_days: int


@dataclass(frozen=True)
class ActiveDayStats:
    active_rate: float
    n_active: int
    hit_rate: float
    payoff_ratio: float
    expectancy_bps: float
    mean_win: float
    mean_loss: float


def _check_series(x: pd.Series, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    if x.index.has_duplicates:
        raise ValueError(f"{name} index has duplicates")
    if not x.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")
    return x


def annualized_sharpe(net_ret: pd.Series, *, periods_per_year: int = 252) -> float:
    """Annualized Sharpe = mean/std*sqrt(252) on daily net returns."""
    net_ret = _check_series(net_ret, "net_ret").dropna()
    if len(net_ret) < 2:
        return float("nan")
    mu = float(net_ret.mean())
    sig = float(net_ret.std(ddof=0))
    if not np.isfinite(sig) or sig <= 0:
        return float("nan")
    return mu / sig * float(np.sqrt(periods_per_year))


def equity_curve(net_ret: pd.Series, *, start_value: float = 1.0) -> pd.Series:
    """Compounded equity curve from daily returns: E_t = E_0 * Π(1+r_t)."""
    net_ret = _check_series(net_ret, "net_ret")
    r = net_ret.fillna(0.0).astype(float)
    eq = start_value * (1.0 + r).cumprod()
    eq.name = "equity"
    return eq


def max_drawdown(net_ret: pd.Series) -> float:
    """Maximum drawdown on compounded equity curve."""
    eq = equity_curve(net_ret)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min())
    # Return as positive magnitude per common reporting? Plan mentions MaxDD; keep magnitude.
    return abs(mdd)


def cagr_from_returns(net_ret: pd.Series, *, periods_per_year: int = 252) -> float:
    net_ret = _check_series(net_ret, "net_ret")
    r = net_ret.dropna().astype(float)
    if len(r) == 0:
        return float("nan")
    eq_end = float((1.0 + r).prod())
    years = len(r) / float(periods_per_year)
    if years <= 0:
        return float("nan")
    return eq_end ** (1.0 / years) - 1.0


def annualized_vol(net_ret: pd.Series, *, periods_per_year: int = 252) -> float:
    net_ret = _check_series(net_ret, "net_ret").dropna()
    if len(net_ret) < 2:
        return float("nan")
    return float(net_ret.std(ddof=0) * np.sqrt(periods_per_year))


def perf_stats(net_ret: pd.Series, *, periods_per_year: int = 252) -> PerfStats:
    net_ret = _check_series(net_ret, "net_ret")
    r = net_ret.dropna().astype(float)
    n = int(len(r))
    mu = float(r.mean()) if n else float("nan")
    ann_mu = mu * periods_per_year if np.isfinite(mu) else float("nan")
    sh = annualized_sharpe(r, periods_per_year=periods_per_year)
    vol = annualized_vol(r, periods_per_year=periods_per_year)
    cagr = cagr_from_returns(r, periods_per_year=periods_per_year)
    mdd = max_drawdown(r)
    calmar = float("nan")
    if np.isfinite(cagr) and np.isfinite(mdd) and mdd > 0:
        calmar = cagr / mdd
    return PerfStats(
        sharpe=float(sh),
        cagr=float(cagr),
        ann_vol=float(vol),
        max_dd=float(mdd),
        calmar=float(calmar),
        mean_daily=float(mu),
        ann_mean=float(ann_mu),
        n_days=n,
    )


def active_day_stats(
    net_ret: pd.Series,
    w_exec: pd.Series,
    *,
    active_threshold: float = 1e-3,
) -> ActiveDayStats:
    """Active-day statistics.

    Active is defined using executed weight *held over* (t-1->t): w_exec[t-1].
    We therefore compute active mask as w_exec.shift(1) > threshold and use
    net_ret[t] conditioned on that.
    """

    net_ret = _check_series(net_ret, "net_ret")
    w_exec = _check_series(w_exec, "w_exec")

    r, w = net_ret.align(w_exec, join="inner")
    held = w.shift(1)
    active = (held.abs() > active_threshold) & r.notna() & held.notna()

    n = int(r.notna().sum())
    n_active = int(active.sum())
    active_rate = float(n_active / n) if n > 0 else float("nan")

    if n_active == 0:
        return ActiveDayStats(
            active_rate=active_rate,
            n_active=0,
            hit_rate=float("nan"),
            payoff_ratio=float("nan"),
            expectancy_bps=float("nan"),
            mean_win=float("nan"),
            mean_loss=float("nan"),
        )

    r_a = r[active].astype(float)
    wins = r_a[r_a > 0]
    losses = r_a[r_a < 0]
    hit_rate = float((r_a > 0).mean())
    mean_win = float(wins.mean()) if len(wins) else 0.0
    mean_loss = float(losses.mean()) if len(losses) else 0.0

    payoff_ratio = float("nan")
    if len(wins) and len(losses) and mean_loss != 0:
        payoff_ratio = float(mean_win / abs(mean_loss))

    expectancy_bps = float(r_a.mean() * 1e4)

    return ActiveDayStats(
        active_rate=active_rate,
        n_active=n_active,
        hit_rate=hit_rate,
        payoff_ratio=payoff_ratio,
        expectancy_bps=expectancy_bps,
        mean_win=mean_win,
        mean_loss=mean_loss,
    )


def summarize(
    net_ret: pd.Series,
    *,
    w_exec: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Convenience flat dict summary for quick tables."""

    ps = perf_stats(net_ret, periods_per_year=periods_per_year)
    out: Dict[str, float] = {
        "sharpe": ps.sharpe,
        "cagr": ps.cagr,
        "ann_vol": ps.ann_vol,
        "max_dd": ps.max_dd,
        "calmar": ps.calmar,
        "mean_daily": ps.mean_daily,
        "ann_mean": ps.ann_mean,
        "n_days": float(ps.n_days),
    }

    if w_exec is not None:
        ads = active_day_stats(net_ret, w_exec)
        out.update(
            {
                "active_rate": ads.active_rate,
                "hit_rate": ads.hit_rate,
                "payoff_ratio": ads.payoff_ratio,
                "expectancy_bps": ads.expectancy_bps,
                "n_active": float(ads.n_active),
            }
        )
    return out


__all__ = [
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
]
