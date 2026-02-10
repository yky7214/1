"""ftf.data.validation

Validation helpers for prepared/continuous futures data.

The reproduction plan requires:
- Verify roll dates match the "2 business days before FND" rule.
- Ensure calendar alignment and no NaNs after warmup.
- Sanity check distributions of daily returns and ATR.

These functions are intentionally deterministic and light-weight; they are
used by scripts and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.data.calendar import CalendarSpec, get_calendar
from ftf.utils.config import DataConfig


@dataclass(frozen=True)
class ContinuousValidationReport:
    ok: bool
    n_days: int
    n_rolls: int
    roll_rule_violations: int
    nan_after_warmup: int
    ret_summary: Dict[str, float]
    atr_summary: Dict[str, float]


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_returns_from_close(close: pd.Series) -> pd.Series:
    """Close-to-close simple returns from (filled) close series."""
    close = close.astype(float)
    r = close.pct_change()
    return r.replace([np.inf, -np.inf], np.nan)


def compute_atr14(df: pd.DataFrame, *, high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.Series:
    """Compute ATR(14) using the plan's definition (simple mean of TR window)."""
    _require_cols(df, [high_col, low_col, close_col])
    h = df[high_col].astype(float)
    l = df[low_col].astype(float)
    c = df[close_col].astype(float)
    c_prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    return atr


def validate_roll_rule(
    active_contract: pd.Series,
    fnd_by_contract: pd.Series,
    *,
    calendar: Optional[CalendarSpec] = None,
    roll_bd_before_fnd: int = 2,
) -> Tuple[int, pd.DataFrame]:
    """Validate that the active contract never violates the roll cutoff.

    Rule: active contract c on date d must satisfy d < shift(FND[c], -roll_bd_before_fnd).

    Returns:
        (n_violations, violations_df)
    """
    if calendar is None:
        calendar = get_calendar("NYSE")

    ac = active_contract.dropna().copy()
    if not isinstance(ac.index, pd.DatetimeIndex):
        raise ValueError("active_contract must be indexed by DatetimeIndex")

    violations = []
    for d, c in ac.items():
        if c not in fnd_by_contract.index:
            continue
        cutoff = calendar.shift(pd.Timestamp(fnd_by_contract.loc[c]), -roll_bd_before_fnd)
        if not (pd.Timestamp(d) < cutoff):
            violations.append((pd.Timestamp(d), str(c), pd.Timestamp(fnd_by_contract.loc[c]), cutoff))

    vdf = pd.DataFrame(violations, columns=["date", "contract", "fnd", "cutoff"])
    return len(violations), vdf


def validate_continuous_df(
    df_cont: pd.DataFrame,
    active_contract: Optional[pd.Series] = None,
    fnd_by_contract: Optional[pd.Series] = None,
    *,
    cfg: Optional[DataConfig] = None,
    warmup_bd: int = 60,
    calendar_name: str = "NYSE",
) -> ContinuousValidationReport:
    """Run a suite of sanity validations for a continuous futures series."""

    cfg = cfg or DataConfig()
    cal = get_calendar(calendar_name)

    if not isinstance(df_cont.index, pd.DatetimeIndex):
        raise ValueError("df_cont must have a DatetimeIndex")
    if df_cont.index.has_duplicates:
        raise ValueError("df_cont has duplicate timestamps")
    if not df_cont.index.is_monotonic_increasing:
        raise ValueError("df_cont index must be monotonic increasing")

    # Calendar sanity: no weekends
    if (df_cont.index.dayofweek >= 5).any():
        raise ValueError("df_cont index contains weekend dates; calendar alignment likely failed")

    _require_cols(df_cont, [cfg.price_col, cfg.high_col, cfg.low_col])

    close = df_cont[cfg.price_col].astype(float)
    r = compute_returns_from_close(close)
    atr = compute_atr14(df_cont, high_col=cfg.high_col, low_col=cfg.low_col, close_col=cfg.price_col)

    # NaN checks after warmup
    if len(df_cont) > warmup_bd:
        sub = df_cont.iloc[warmup_bd:]
        nan_after = int(sub[[cfg.price_col, cfg.high_col, cfg.low_col]].isna().any(axis=1).sum())
    else:
        nan_after = int(df_cont[[cfg.price_col, cfg.high_col, cfg.low_col]].isna().any(axis=1).sum())

    # Roll rule checks
    roll_viol = 0
    if active_contract is not None and fnd_by_contract is not None:
        roll_viol, _ = validate_roll_rule(
            active_contract=active_contract,
            fnd_by_contract=fnd_by_contract,
            calendar=cal,
            roll_bd_before_fnd=cfg.roll_bd_before_fnd,
        )

    def _summary(x: pd.Series) -> Dict[str, float]:
        x = x.dropna()
        if len(x) == 0:
            return {"count": 0.0}
        return {
            "count": float(len(x)),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "p01": float(x.quantile(0.01)),
            "p50": float(x.quantile(0.50)),
            "p99": float(x.quantile(0.99)),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    ok = (nan_after == 0) and (roll_viol == 0)

    return ContinuousValidationReport(
        ok=bool(ok),
        n_days=int(len(df_cont)),
        n_rolls=int(active_contract.ne(active_contract.shift(1)).sum() - 1) if active_contract is not None else 0,
        roll_rule_violations=int(roll_viol),
        nan_after_warmup=int(nan_after),
        ret_summary=_summary(r),
        atr_summary=_summary(atr),
    )


__all__ = [
    "ContinuousValidationReport",
    "compute_returns_from_close",
    "compute_atr14",
    "validate_roll_rule",
    "validate_continuous_df",
]
