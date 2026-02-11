"""ftf.trading.engine

Deterministic daily trading engine implementing the paper's *forecast-to-fill*
conventions.

This module is the core glue between:
- signals/regime probabilities
- risk targeting (EWMA vol)
- sizing (confidence + Kelly) producing raw weights
- ATR-based exit / state machine producing target weights
- execution latency buffer producing executed weights
- costs/impact accounting

Time/P&L convention (critical, baseline):
- Decision time uses info up to close of day t (F_t).
- Targets decided at t are executed after an execution lag d:
    w_exec[t] = w_target[t-d]
  (baseline d=1 == T+1 close).
- P&L attribution:
    gross_ret[t] = w_exec[t-1] * r[t]
  where r[t] = P[t]/P[t-1]-1 computed from (filled) continuous close.
- Costs are charged on turnover at time t:
    turnover[t] = |w_exec[t] - w_exec[t-1]|
    net_ret[t] = gross_ret[t] - cost(turnover[t])

The engine is designed to be walk-forward friendly: all required states/params
are passed in as arguments and are assumed already *frozen* from the training
window.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ftf.execution.costs import compute_costs
from ftf.execution.fills import apply_exec_lag
from ftf.risk.ewma_vol import EWMAVolState, ewma_variance_forecast, vol_target_weight
from ftf.signals.regime import RegimeState, compute_regime_features
from ftf.sizing.policy_weight import PolicyWeightState, compute_w_raw
from ftf.trading.atr import compute_atr
from ftf.trading.exits import ATRExitState, generate_target_weights
from ftf.trading.logs import TradingLog
from ftf.utils.config import FTFConfig


@dataclass(frozen=True)
class EngineResult:
    """Container for engine outputs."""

    daily: pd.DataFrame
    log: TradingLog


def _compute_returns(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series) or not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be a pandas Series with a DatetimeIndex")
    r = close.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan)
    r.name = "r"
    return r

###2/3実装###                                                                                    #####################################
def apply_delta_w_cap_series(w_exec: pd.Series, cap: float, w0: float = 0.0) -> pd.Series:
    """
    逐次で |w_t - w_{t-1}| <= cap を強制する（実際の執行ポジションを滑らかにする）
    """
    if cap is None:
        return w_exec

    cap = float(cap)
    if cap <= 0:
        return w_exec * 0.0

    out = pd.Series(index=w_exec.index, dtype=float)

    prev = float(w0)
    for t, w_t in w_exec.items():
        w_t = float(w_t) if np.isfinite(w_t) else prev

        dw = w_t - prev
        if dw > cap:
            dw = cap
        elif dw < -cap:
            dw = -cap

        prev = prev + dw
        out.loc[t] = prev

    out.name = w_exec.name
    return out

###                                                                  ###############2/4 20:49
def compute_dd_from_net_ret(net_ret_ftf: pd.Series) -> pd.Series:
    equity = (1.0 + net_ret_ftf.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    dd.name = "dd"
    return dd


def run_engine(
    df_cont: pd.DataFrame,
    *,
    cfg: FTFConfig,
    regime_state: RegimeState,
    vol_state: EWMAVolState,
    policy_state: PolicyWeightState,
    f_tilde: float,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    exit_state: Optional[ATRExitState] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EngineResult:
    """Run the full daily engine on a continuous futures DataFrame.

    Parameters
    ----------
    df_cont:
        Continuous futures dataframe with at least columns close/high/low.
    cfg:
        Global configuration containing conventions, costs, ATR settings, etc.
    regime_state:
        Frozen regime signal parameters (EMA slope stats, momentum K, omega, threshold).
    vol_state:
        Frozen EWMA risk state (theta, init variance, target vol, cap).
    policy_state:
        Frozen policy weight settings (caps and baseline floor behavior).
    f_tilde:
        Frozen fractional Kelly scalar for this anchor.
    start/end:
        Optional date clipping.
    exit_state:
        Optional initial exit state. If None, starts flat.
    metadata:
        Optional arbitrary dict copied into the TradingLog header.

    Returns
    -------
    EngineResult
        daily dataframe includes required fields; log contains event stream.
    """

    price_col = cfg.data.price_col
    high_col = cfg.data.high_col
    low_col = cfg.data.low_col

    for c in (price_col, high_col, low_col):
        if c not in df_cont.columns:
            raise ValueError(f"df_cont missing required column: {c}")

    df = df_cont.copy()
    if start is not None:
        df = df.loc[pd.Timestamp(start) :]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]

    if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates:
        raise ValueError("df_cont must have a unique DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    close = df[price_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    r = _compute_returns(close)

    # Signals/regime features at decision time t
    sig = compute_regime_features(close, state=regime_state)

    # ATR at time t uses H,L,C up to t (C_{t-1} allowed)
    atr = compute_atr(
        high=high,
        low=low,
        close=close,
        window=cfg.atr_exit.atr_window,
    )

    # Risk weights based on EWMA variance forecast sigma^2_{t+1}
    ewma_sigma2_next = ewma_variance_forecast(r, state=vol_state)
    w_vol = vol_target_weight(r, state=vol_state)

    # Raw sizing before gating/exits
    w_raw = compute_w_raw(w_vol=w_vol, p_bull=sig["p_bull"], f_tilde=float(f_tilde), state=policy_state)

    lam_w = float(getattr(cfg, "w_smooth_lambda", 0.94))  # 0.90〜0.97で比較
    w_raw_smooth = w_raw.ewm(alpha=(1.0-lam_w), adjust=False).mean()
    w_raw_smooth.name = "w_raw_smooth"


    # Exit / gating to produce w_target at decision time
    w_target, exit_log, w_target_stop0 = generate_target_weights(
        close=close,
        high=high,
        low=low,
        atr=atr,
        p_bull=sig["p_bull"],
        p_bear=sig["p_bear"],
        eligible_to_enter=sig["eligible_to_enter"],
        w_raw=w_raw,
        cfg=cfg.atr_exit,
        time_cfg=cfg.time,
        initial_state=exit_state,
    )

    # Apply execution lag buffer
    w_exec = apply_exec_lag(w_target, exec_lag=int(cfg.time.exec_lag), fill_value=0.0)

    # ------------------------------------------------------------
# DELTA_W CAP (NEW)
#   実際の執行ポジション w_exec の変化量を制限する
# ------------------------------------------------------------     ####                 ##############################2/3
    delta_w_cap = float(getattr(cfg.time, "delta_w_cap", 0.0))
    if delta_w_cap > 0:
        
        w_exec = apply_delta_w_cap_series(w_exec, cap=delta_w_cap, w0=0.0)
#      # 例: 0.5

    # Turnover and costs
    turnover = (w_exec - w_exec.shift(1)).abs()
    costs = compute_costs(w_exec, k_linear=cfg.costs.k_linear, gamma_impact=cfg.costs.gamma_impact)
    cost_lin = costs.cost_linear
    cost_imp = costs.cost_impact
    turnover = costs.turnover

    gross_ret = w_exec.shift(1).fillna(0.0) * r
    gross_ret.name = "gross_ret"
    net_ret = gross_ret - cost_lin - cost_imp
    net_ret.name = "net_ret"
    gross_ret = w_exec.shift(1).fillna(0.0) * r
    gross_ret.name = "gross_ret"

    net_ret = gross_ret - cost_lin - cost_imp
    net_ret.name = "net_ret"



    # =========================================================
    # Forecast-to-Fill (FTF) extension
    # =========================================================

    # execution lag (days)
    d = int(cfg.time.exec_lag)


    # --- forecast execution price using decision-time info ---
    # logP_hat(t+d) = ema_log(t) + d * slope(t)
    logP_hat = sig["ema_log"] + d * sig["slope"]
    P_hat = np.exp(logP_hat)


    # align forecast made at t to execution day t+d
    P_hat_exec = P_hat.shift(d)
    P_hat_exec.name = "P_hat_exec"


    # realized execution price (engine assumes close fill)
    P_exec = close


    # fill error return (positive if forecast > realized fill)
    # NOTE:
    #   Paper convention is positive alpha when we increase position (dw>0)
    #   and get filled below the forecasted execution price.
    #   Therefore sign must be (forecast - realized), not the reverse.
    fill_error_ret = (P_hat_exec - P_exec) / (P_exec + 1e-12)
    fill_error_ret.name = "fill_error_ret"


    # scale for FTF alpha (start with 1.0)
    alpha_scale = float(getattr(cfg, "ftf_alpha_scale", 1.0))

    dw = (w_exec - w_exec.shift(1)).fillna(0.0)
    dw.name = "dw"

    alpha_ret = alpha_scale * (dw * fill_error_ret)
    alpha_ret.name = "alpha_ret"



    # FTF-augmented gross / net returns
    gross_ret_ftf = gross_ret + alpha_ret
    gross_ret_ftf.name = "gross_ret_ftf"


    net_ret_ftf = gross_ret_ftf - cost_lin - cost_imp
    net_ret_ftf.name = "net_ret_ftf"




    out = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "r": r,
            # signals
            "ema_log": sig["ema_log"],
            "slope": sig["slope"],
            "p_trend": sig["p_trend"],
            "momentum": sig["momentum"],
            "p_bull": sig["p_bull"],
            "p_bear": sig["p_bear"],
            "eligible_to_enter": sig["eligible_to_enter"],
            "regime": sig["regime"],
            # risk/sizing
            "atr": atr,
            "ewma_sigma2_next": ewma_sigma2_next,
            "w_vol": w_vol,
            "f_tilde": float(f_tilde),
            "w_raw": w_raw,
            # trading/execution
            "w_target": w_target,
            "w_exec": w_exec,
            "P_hat_exec": P_hat_exec,
            "fill_error_ret": fill_error_ret,
            "gross_ret_ftf": gross_ret_ftf,
            "net_ret_ftf": net_ret_ftf,
            
            
            
            "gross_ret": gross_ret,
            
            "turnover": turnover,
            "cost_lin": cost_lin,
            "cost_imp": cost_imp,
            "net_ret": net_ret,
            "alpha_ret": alpha_ret,
            "dw": dw,

        },
        index=df.index,
    )

    # Build log
    header: Dict[str, Any] = {
        "run_name": cfg.run_name,
        "time": asdict(cfg.time),
        "signal": {
            "ema_lambda": regime_state.ema_state.ema_lambda,
            "slope_mu": regime_state.ema_state.slope_mu,
            "slope_sigma": regime_state.ema_state.slope_sigma,
            "z_clip": regime_state.ema_state.z_clip,
            "momentum_k": regime_state.mom_state.k,
            "blend_omega": regime_state.blend_omega,
            "pbull_threshold": regime_state.pbull_threshold,
        },
        "risk": asdict(vol_state),
        "policy": asdict(policy_state),
        "kelly": {"f_tilde": float(f_tilde)},
        "costs": asdict(cfg.costs),
        "atr_exit": asdict(cfg.atr_exit),
    }
    if metadata:
        header.update(metadata)

    

    log = TradingLog(header=header, events=exit_log.events)
    return EngineResult(daily=out, log=log)    




   
    


__all__ = ["EngineResult", "run_engine"]
