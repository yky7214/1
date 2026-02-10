"""ftf.trading.atr

ATR(14) computation used by the exit/state machine.

The reproduction plan specifies a *simple rolling mean* ATR:

- True Range (TR_t) = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)
- ATR_N(t) = mean(TR_{t-N+1..t}) with N=14 by default

This module intentionally does not perform any forward-looking operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ftf.utils.config import ATRExitConfig, DataConfig


@dataclass(frozen=True)
class ATRState:
    window: int = 14
    hard_stop_atr: float = 2.0
    trailing_stop_atr: float = 1.5
    timeout_days: int = 30


def fit_atr_state(*, cfg: Optional[ATRExitConfig] = None) -> ATRState:
    """Create a frozen ATR/exit state container from config.

    Note: In the plan, these are constants (not estimated). We still wrap
    them in a state object for consistent per-anchor serialization.
    """

    if cfg is None:
        cfg = ATRExitConfig()
    if cfg.atr_window <= 1:
        raise ValueError("atr_window must be >= 2")
    return ATRState(
        window=int(cfg.atr_window),
        hard_stop_atr=float(cfg.hard_stop_atr),
        trailing_stop_atr=float(cfg.trailing_stop_atr),
        timeout_days=int(cfg.timeout_days),
    )


def true_range(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Compute daily true range series."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a DatetimeIndex")
    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.name = "tr"
    return tr


def compute_atr(
    df: Optional[pd.DataFrame] = None,
    *,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    ATR using simple rolling mean (NOT Wilder smoothing).

    Accepts either:
      - df with columns [high_col, low_col, close_col], OR
      - explicit Series high/low/close.

    TR_t = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)
    ATR_t = mean(TR_{t-window+1..t})
    """
    if window <= 0:
        raise ValueError("window must be positive")

    if df is not None:
        h = df[high_col].astype(float)
        l = df[low_col].astype(float)
        c = df[close_col].astype(float)
    else:
        if high is None or low is None or close is None:
            raise TypeError("Provide either df=... or (high=..., low=..., close=...)")
        h = high.astype(float)
        l = low.astype(float)
        c = close.astype(float)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    atr.name = f"atr{window}"
    return atr
    # Replace negative zeros / -0.0 artifacts.
    atr = atr.where(~np.isclose(atr.to_numpy(dtype=float), 0.0), 0.0)
    return atr


def compute_atr_from_cfg(
    df: pd.DataFrame,
    *,
    data_cfg: Optional[DataConfig] = None,
    exit_cfg: Optional[ATRExitConfig] = None,
) -> pd.Series:
    """Convenience wrapper using canonical column names from config."""

    if data_cfg is None:
        data_cfg = DataConfig()
    if exit_cfg is None:
        exit_cfg = ATRExitConfig()

    return compute_atr(
        df,
        window=int(exit_cfg.atr_window),
        high_col=data_cfg.high_col,
        low_col=data_cfg.low_col,
        close_col=data_cfg.price_col,
    )


__all__ = [
    "ATRState",
    "fit_atr_state",
    "true_range",
    "compute_atr",
    "compute_atr_from_cfg",
]
