"""ftf.trading.exits

ATR-based long-only position state machine.

This module turns a *raw* desired weight series (sizing output) plus decision-time
signals (p_bull/p_bear, eligibility gate) and ATR into a *target* weight series
that respects entry/exit rules:

- Entry gate: eligible_to_enter and slope>0 (encoded upstream as eligible_to_enter)
- Exits while active:
  * hard stop: close <= entry - hard_stop_atr * ATR
  * trailing stop: close <= peak - trailing_stop_atr * ATR
  * timeout: age_days >= timeout_days
- Regime de-risk:
  * DERISK_HALF: while p_bear > 0.50, halve target weight
  * DERISK_CLOSE: when p_bear > 0.50, exit

Stop fill convention:
- Baseline STOP_FILL_T_PLUS_1: if stop triggers at close t, the engine sets
  w_target[t]=0 (decision at t), and execution latency applies downstream.
- Sensitivity STOP_FILL_SAME_CLOSE: additionally produces a diagnostic series
  w_target_stopfill0 that is zeroed on the *same* day in attribution; the engine
  may choose to use it for T+0 style analysis.

The state machine is fully causal: decisions for date t only use information up
through close/high/low of t (and close_{t-1} for ATR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.trading.atr import ATRState, fit_atr_state
from ftf.trading.logs import TradeEvent, TradingLog
from ftf.utils.config import ATRExitConfig, DeriskPolicy, PeakRef, StopFillPolicy


@dataclass(frozen=True)
class ATRExitState:
    """Frozen exit hyperparameters.

    Kept separate from :class:`~ftf.trading.atr.ATRState` because exit logic also
    depends on peak price convention and de-risk policy.
    """

    atr: ATRState
    price_reference_for_peak: PeakRef = "close"
    derisk_policy: DeriskPolicy = "DERISK_HALF"
    stop_fill_policy: StopFillPolicy = "STOP_FILL_T_PLUS_1"


def fit_atr_exit_state(*, cfg: Optional[ATRExitConfig] = None, stop_fill_policy: StopFillPolicy = "STOP_FILL_T_PLUS_1") -> ATRExitState:
    cfg = cfg or ATRExitConfig()
    atr_state = fit_atr_state(cfg=cfg)
    if cfg.price_reference_for_peak not in ("close", "high"):
        raise ValueError("price_reference_for_peak must be 'close' or 'high'")
    if cfg.derisk_policy not in ("DERISK_HALF", "DERISK_CLOSE"):
        raise ValueError("derisk_policy must be DERISK_HALF or DERISK_CLOSE")
    if stop_fill_policy not in ("STOP_FILL_T_PLUS_1", "STOP_FILL_SAME_CLOSE"):
        raise ValueError("Unsupported stop_fill_policy")
    return ATRExitState(
        atr=atr_state,
        price_reference_for_peak=cfg.price_reference_for_peak,
        derisk_policy=cfg.derisk_policy,
        stop_fill_policy=stop_fill_policy,
    )


@dataclass
class _Pos:
    active: bool = False
    entry_price: float = np.nan
    peak_price: float = np.nan
    age_days: int = 0
    derisk_half_on: bool = False


def _require_series(name: str, s: pd.Series) -> None:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index must be a DatetimeIndex")



def generate_target_weights(
    *,
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,  # 互換のため受け取る（ロジック上は未使用でもOK）
    atr: pd.Series,
    p_bull: Optional[pd.Series] = None,  # 互換のため受け取る（今回は未使用）
    p_bear: pd.Series,
    eligible_to_enter: pd.Series,
    w_raw: pd.Series,
    cfg: Optional[ATRExitConfig] = None,      # engine.py 互換
    time_cfg: Optional[Any] = None,           # engine.py 互換（stop_fill_policyに利用）
    initial_state: Optional[ATRExitState] = None,  # engine.py 互換
    exit_state: Optional[ATRExitState] = None,  # tests互換（alias）
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """Run the ATR exit/state machine and produce target weights.

    Parameters
    ----------
    close, high : pd.Series
        Daily close and (optional) high. If `high` is None, peak tracking uses close.
    atr : pd.Series
        ATR series aligned to the same calendar as close.
    w_raw : pd.Series
        Raw (pre-gating) desired weights (long-only). Typically output from sizing.
    eligible_to_enter : pd.Series
        1/0 eligibility gate; NaNs treated as not eligible.
    p_bear : pd.Series
        Bear probability; used for de-risk policy.

    Returns
    -------
    w_target : pd.Series
        Target weight series (decision-time), same index as the aligned inputs.
    log : TradingLog
        Event log.
    w_target_stopfill0 : Optional[pd.Series]
        If stop_fill_policy==STOP_FILL_SAME_CLOSE, a diagnostic series that exits on
        the same day (T+0 style) for sensitivity analysis.
    """

    _require_series("close", close)
    _require_series("atr", atr)
    _require_series("w_raw", w_raw)
    _require_series("eligible_to_enter", eligible_to_enter)
    _require_series("p_bear", p_bear)
    # prefer explicit initial_state, else accept exit_state (tests alias)
    if initial_state is None and exit_state is not None:
        initial_state = exit_state    


    if high is not None:
        _require_series("high", high)

    # resolve stop_fill_policy from time_cfg if present
    stop_fill_policy = None
    if time_cfg is not None and hasattr(time_cfg, "stop_fill_policy"):
        stop_fill_policy = getattr(time_cfg, "stop_fill_policy")

    # engine passes initial_state (ATRExitState) or None
    exit_state = initial_state
    if exit_state is None:
        exit_state = fit_atr_exit_state(cfg=cfg or ATRExitConfig(), stop_fill_policy=stop_fill_policy or "STOP_FILL_T_PLUS_1")
    elif stop_fill_policy is not None:
        # override stop fill policy deterministically
        exit_state = ATRExitState(
            atr=exit_state.atr,
            price_reference_for_peak=exit_state.price_reference_for_peak,
            derisk_policy=exit_state.derisk_policy,
            stop_fill_policy=stop_fill_policy,
        )
    # Align all inputs on common index.
    series = [close.rename("close"), atr.rename("atr"), w_raw.rename("w_raw"), eligible_to_enter.rename("eligible"), p_bear.rename("p_bear")]
    if high is not None:
        series.append(high.rename("high"))
    df = pd.concat(series, axis=1, join="inner").sort_index()

    peak_ref = exit_state.price_reference_for_peak
    if peak_ref == "high" and "high" not in df.columns:
        # fall back deterministically
        peak_ref = "close"

    pos = _Pos()
    w_target = np.zeros(len(df), dtype=float)
    w_target_stop0 = np.zeros(len(df), dtype=float) if exit_state.stop_fill_policy == "STOP_FILL_SAME_CLOSE" else None

    log = TradingLog(header={"module": "atr_exits"})

    print("timeout_days_used=", exit_state.atr.timeout_days)

    # helper: note that eligibility and w_raw can be NaN during warmup
    def _eligible(val: float) -> bool:
        return bool(np.isfinite(val) and val > 0.5)

    for i, (dt, row) in enumerate(df.iterrows()):
        c = float(row["close"])
        a = float(row["atr"])
        wr = float(row["w_raw"]) if np.isfinite(row["w_raw"]) else np.nan
        elig = _eligible(float(row["eligible"]) if np.isfinite(row["eligible"]) else np.nan)
        pbear = float(row["p_bear"]) if np.isfinite(row["p_bear"]) else np.nan

        peak_price_ref = float(row["high"]) if (peak_ref == "high" and "high" in row and np.isfinite(row["high"])) else c

        # Default desired target while active is current w_raw (can vary daily).
        desired = 0.0
        if np.isfinite(wr):
            desired = max(0.0, wr)

        exit_today = False
        exit_event: Optional[str] = None

        if not pos.active:
            pos.derisk_half_on = False
            if elig and desired > 0.0 and np.isfinite(a) and np.isfinite(c):
                pos.active = True
                pos.entry_price = c
                pos.peak_price = peak_price_ref
                pos.age_days = 0
                log.add(dt, "ENTRY", price=c, entry_price=c)
                w_target[i] = desired
                if w_target_stop0 is not None:
                    w_target_stop0[i] = desired
            else:
                w_target[i] = 0.0
                if w_target_stop0 is not None:
                    w_target_stop0[i] = 0.0
            continue

        # Active position
        pos.age_days += 1
        if np.isfinite(peak_price_ref):
            if not np.isfinite(pos.peak_price):
                pos.peak_price = peak_price_ref
            else:
                pos.peak_price = max(pos.peak_price, peak_price_ref)

        hard_stop = False
        trailing_stop = False
        timeout = False

        if np.isfinite(a) and np.isfinite(c) and np.isfinite(pos.entry_price):
            hard_stop = c <= (pos.entry_price - exit_state.atr.hard_stop_atr * a)
        if np.isfinite(a) and np.isfinite(c) and np.isfinite(pos.peak_price):
            trailing_stop = c <= (pos.peak_price - exit_state.atr.trailing_stop_atr * a)
        timeout = pos.age_days >= exit_state.atr.timeout_days

        # Regime de-risk
        if np.isfinite(pbear) and pbear > 0.50:
            if exit_state.derisk_policy == "DERISK_CLOSE":
                exit_today = True
                exit_event = "EXIT_DERISK_CLOSE"
            else:
                if not pos.derisk_half_on:
                    pos.derisk_half_on = True
                    log.add(dt, "DERISK_HALF_ON", price=c, p_bear=pbear)
        else:
            if pos.derisk_half_on:
                pos.derisk_half_on = False
                log.add(dt, "DERISK_HALF_OFF", price=c, p_bear=pbear)

        if hard_stop:
            exit_today = True
            exit_event = "EXIT_HARD_STOP"
        elif trailing_stop:
            exit_today = True
            exit_event = "EXIT_TRAILING_STOP"
        elif timeout:
            exit_today = True
            exit_event = "EXIT_TIMEOUT"

        if exit_today:
            # Decision at close t: set target 0. Latency applied downstream.
            w_target[i] = 0.0
            if w_target_stop0 is not None:
                # same-close: flatten today in diagnostic series
                w_target_stop0[i] = 0.0
            log.add(dt, exit_event or "FLAT", price=c, age_days=pos.age_days)
            pos = _Pos()  # reset
            continue

        # Still active: apply desired sizing, possibly derisk
        wt = desired
        if pos.derisk_half_on:
            wt = 0.5 * wt
        w_target[i] = wt
        if w_target_stop0 is not None:
            w_target_stop0[i] = wt

    w_target_s = pd.Series(w_target, index=df.index, name="w_target")
    if w_target_stop0 is not None:
        w_target_stop0_s = pd.Series(w_target_stop0, index=df.index, name="w_target_stopfill0")
    else:
        w_target_stop0_s = None

    # Normalize -0.0
    w_target_s = w_target_s.where(~np.isclose(w_target_s.to_numpy(), 0.0), 0.0)
    if w_target_stop0_s is not None:
        w_target_stop0_s = w_target_stop0_s.where(~np.isclose(w_target_stop0_s.to_numpy(), 0.0), 0.0)

    # Convert TradingLog -> list[dict] events (engine expects list of dict-like events)
    events: List[Dict[str, Any]] = []
    if hasattr(log, "events"):
        # TradingLog.events is expected to be a list already
        for e in log.events:
            if isinstance(e, dict):
                events.append(e)
            else:
                # best-effort conversion
                try:
                    events.append(dict(e))
                except Exception:
                    events.append({"event": str(e)})
    else:
        events = []

    return w_target_s, log, w_target_stop0_s

__all__ = ["ATRExitState", "fit_atr_exit_state", "generate_target_weights"]
