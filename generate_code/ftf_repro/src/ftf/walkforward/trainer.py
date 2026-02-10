"""Walk-forward trainer.

This module is responsible for *train-only* estimation of all frozen parameters
used by the trading engine for a given walk-forward anchor.

Two modes are supported (selected by cfg.walkforward.trainer_mode):

- FIXED: uses reproducible constant hyperparameters.
- GRID: grid-searches hyperparameters on the training window to maximize
  *net* Sharpe of the strategy when run on the training slice, using the same
  entry/exit engine and the same turnover/cost model, but without walk-forward
  stitching.

It also estimates:
- EMA slope distribution parameters (mu/sigma) for z-scoring
- EWMA variance init from training returns
- Unit-notional sleeve returns from training (engine simulation with unit weight)
- Friction-adjusted Kelly fraction f* and fractional Kelly f_tilde

All returned objects are designed to be serializable (dataclasses + plain types)
so the runner can snapshot per-anchor frozen params.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.risk.ewma_vol import EWMAVolState, fit_ewma_vol_state
from ftf.sizing.kelly import (
    KellyInputs,
    estimate_kelly_inputs,
    fractional_kelly,
    solve_friction_adjusted_kelly,
)
from ftf.sizing.policy_weight import PolicyWeightState, fit_policy_weight_state
from ftf.signals.regime import RegimeState, fit_regime_state
from ftf.trading.engine import run_engine
from ftf.trading.exits import ATRExitState, fit_atr_exit_state
from ftf.utils.config import FTFConfig, TrainerMode


@dataclass(frozen=True)
class AnchorFit:
    """Frozen parameters and states for one anchor."""

    regime_state: RegimeState
    vol_state: EWMAVolState
    policy_state: PolicyWeightState
    exit_state: ATRExitState

    # Kelly sizing
    kelly_inputs: KellyInputs
    f_star: float
    f_tilde: float

    # Chosen hyperparameters for traceability
    chosen: Dict[str, Any]


def _annualized_sharpe(net_ret: pd.Series) -> float:
    x = net_ret.dropna().to_numpy(dtype=float)
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return mu / sd * float(np.sqrt(252.0))


def _mean_turnover(daily: pd.DataFrame) -> float:
    if "turnover" not in daily.columns:
        return float("nan")
    x = daily["turnover"].dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _grid_candidates() -> List[Dict[str, Any]]:
    ema_lams = [0.90, 0.92, 0.94, 0.96, 0.97, 0.98]
    thetas = [0.94, 0.96, 0.9659]
    omegas = [0.5, 0.6, 0.7]
    ths = [0.50, 0.52, 0.55, 0.60]
    out: List[Dict[str, Any]] = []
    for lam, th, om, pb in product(ema_lams, thetas, omegas, ths):
        out.append(
            {
                "signal.ema_lambda": lam,
                "risk.ewma_theta": th,
                "signal.blend_omega": om,
                "signal.pbull_threshold": pb,
            }
        )
    return out


def _apply_overrides(cfg: FTFConfig, overrides: Dict[str, Any]) -> FTFConfig:
    """Create a shallow-updated cfg with selected overrides.

    We avoid mutating dataclasses by rebuilding a new FTFConfig.
    """

    # Convert to nested dict and patch via simple paths.
    d = cfg.to_dict()
    for k, v in overrides.items():
        parts = k.split(".")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v

    # Rebuild typed config (replicating scripts' helper to avoid circular imports)
    from ftf.utils.config import (
        ATRExitConfig,
        BootstrapConfig,
        CapacityConfig,
        CostImpactConfig,
        DataConfig,
        KellyConfig,
        RegressionConfig,
        RiskConfig,
        SignalConfig,
        TimeConvention,
        WalkForwardConfig,
        validate_config,
    )

    cfg2 = FTFConfig(
        time=TimeConvention(**d["time"]),
        data=DataConfig(**d["data"]),
        signal=SignalConfig(**d["signal"]),
        risk=RiskConfig(**d["risk"]),
        atr_exit=ATRExitConfig(**d["atr_exit"]),
        costs=CostImpactConfig(**d["costs"]),
        kelly=KellyConfig(**d["kelly"]),
        walkforward=WalkForwardConfig(**d["walkforward"]),
        regression=RegressionConfig(**d["regression"]),
        bootstrap=BootstrapConfig(**d["bootstrap"]),
        capacity=CapacityConfig(**d["capacity"]),
        run_name=d.get("run_name", "base"),
    )
    validate_config(cfg2)
    return cfg2


def _unit_notional_sleeve_returns(
    df_train: pd.DataFrame,
    *,
    cfg: FTFConfig,
    regime_state: RegimeState,
    vol_state: EWMAVolState,
    exit_state: ATRExitState,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Simulate the policy on training with fixed notional weight=1 when active.

    Per plan: recommended sleeve uses same entry/exit engine but fixed weight.

    Implementation detail:
      - Use w_vol=1 and p_bull=1 so that w_raw==1 when signals are defined.
      - Entry is still gated by eligible_to_enter and ΔEMA>0 (encoded in regime).
      - Costs are set to 0 for sleeve return series.

    Returns:
      R_train: pd.Series of sleeve gross returns.
      daily: engine daily DataFrame for diagnostics.
    """

    # Set f_tilde=1 and w_max>=1 via policy_state; set costs to zero.
    cfg_sleeve = _apply_overrides(
        cfg,
        {
            "risk.w_max": max(cfg.risk.w_max, 1.0),
            "costs.k_linear": 0.0,
            "costs.gamma_impact": 0.0,
        },
    )

    policy_state = fit_policy_weight_state(risk_cfg=cfg_sleeve.risk, kelly_cfg=cfg_sleeve.kelly)

    res = run_engine(
        df_train,
        cfg=cfg_sleeve,
        regime_state=regime_state,
        vol_state=vol_state,
        policy_state=policy_state,
        f_tilde=1.0,
        exit_state=exit_state,
        metadata={"purpose": "unit_notional_sleeve"},
    )
    # Sleeve returns should be gross-of-cost by construction
    R = res.daily["gross_ret"].copy()
    R.name = "sleeve_R"
    return R, res.daily


def fit_anchor(
    df_train: pd.DataFrame,
    *,
    cfg: FTFConfig,
) -> AnchorFit:
    """Fit all frozen parameters for one walk-forward anchor.

    Parameters
    ----------
    df_train:
        Continuous futures DataFrame on the training window. Must contain at
        least configured close/high/low columns.
    cfg:
        Full configuration.

    Returns
    -------
    AnchorFit
        Frozen states and scalars.
    """

    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_train.index, pd.DatetimeIndex):
        raise TypeError("df_train must be a pandas DataFrame with DatetimeIndex")

    mode: TrainerMode = cfg.walkforward.trainer_mode

    # Determine hyperparameters either fixed or via grid search.
    if mode == "FIXED":
        chosen_overrides: Dict[str, Any] = {
            "signal.ema_lambda": cfg.signal.ema_lambda,
            "risk.ewma_theta": cfg.risk.ewma_theta,
            "signal.blend_omega": cfg.signal.blend_omega,
            "signal.pbull_threshold": cfg.signal.pbull_threshold,
        }
        cfg_fit = cfg
    elif mode == "GRID":
        best = None
        best_sh = -np.inf
        best_to = np.inf
        best_cfg = None
        for ov in _grid_candidates():
            cfg_try = _apply_overrides(cfg, ov)

            close_train = df_train[cfg_try.data.price_col].astype(float)
            r_train = close_train.pct_change().replace([np.inf, -np.inf], np.nan)

            regime_state = fit_regime_state(
                close_train,
                cfg=cfg_try.signal,
                ema_lambda=cfg_try.signal.ema_lambda,
                blend_omega=cfg_try.signal.blend_omega,
                pbull_threshold=cfg_try.signal.pbull_threshold,
                momentum_k=cfg_try.signal.momentum_k,
            )
            vol_state = fit_ewma_vol_state(r_train, cfg=cfg_try.risk, theta=cfg_try.risk.ewma_theta)
            exit_state = fit_atr_exit_state(cfg=cfg_try.atr_exit, stop_fill_policy=cfg_try.time.stop_fill_policy)
            policy_state = fit_policy_weight_state(risk_cfg=cfg_try.risk, kelly_cfg=cfg_try.kelly)

            # Use a fixed small positive f_tilde during grid search to avoid all-zero
            # weights when Kelly would be ~0; still allows turnover/cost differences.
            f_tilde_grid = 1.0

            res = run_engine(
                df_train,
                cfg=cfg_try,
                regime_state=regime_state,
                vol_state=vol_state,
                policy_state=policy_state,
                f_tilde=f_tilde_grid,
                exit_state=exit_state,
                metadata={"purpose": "grid_search"},
            )
            sh = _annualized_sharpe(res.daily["net_ret"])
            to = _mean_turnover(res.daily)
            if np.isnan(sh):
                continue
            if (sh > best_sh + 1e-12) or (abs(sh - best_sh) <= 1e-12 and np.isfinite(to) and to < best_to):
                best_sh = sh
                best_to = to
                best = ov
                best_cfg = cfg_try

        if best_cfg is None or best is None:
            # Fallback to FIXED
            best_cfg = cfg
            best = {
                "signal.ema_lambda": cfg.signal.ema_lambda,
                "risk.ewma_theta": cfg.risk.ewma_theta,
                "signal.blend_omega": cfg.signal.blend_omega,
                "signal.pbull_threshold": cfg.signal.pbull_threshold,
            }
        chosen_overrides = best
        cfg_fit = best_cfg
    else:
        raise ValueError(f"Unknown trainer_mode: {mode}")

    # Now fit all frozen states using cfg_fit (train-only)
    close_train = df_train[cfg_fit.data.price_col].astype(float)
    r_train = close_train.pct_change().replace([np.inf, -np.inf], np.nan)

    regime_state = fit_regime_state(
        close_train,
        cfg=cfg_fit.signal,
        ema_lambda=cfg_fit.signal.ema_lambda,
        blend_omega=cfg_fit.signal.blend_omega,
        pbull_threshold=cfg_fit.signal.pbull_threshold,
        momentum_k=cfg_fit.signal.momentum_k,
    )
    vol_state = fit_ewma_vol_state(r_train, cfg=cfg_fit.risk, theta=cfg_fit.risk.ewma_theta)
    exit_state = fit_atr_exit_state(cfg=cfg_fit.atr_exit, stop_fill_policy=cfg_fit.time.stop_fill_policy)
    policy_state = fit_policy_weight_state(risk_cfg=cfg_fit.risk, kelly_cfg=cfg_fit.kelly)

    # Unit-notional sleeve (gross-of-costs) to estimate Kelly.
    R_sleeve, _daily_sleeve = _unit_notional_sleeve_returns(
        df_train,
        cfg=cfg_fit,
        regime_state=regime_state,
        vol_state=vol_state,
        exit_state=exit_state,
    )
    kelly_inputs = estimate_kelly_inputs(R_sleeve)
    f_star = solve_friction_adjusted_kelly(inputs=kelly_inputs, costs=cfg_fit.costs)
    f_tilde = fractional_kelly(f_star, kelly_cfg=cfg_fit.kelly)

    chosen = {
        "trainer_mode": mode,
        "grid_overrides": chosen_overrides,
        "ema_lambda": cfg_fit.signal.ema_lambda,
        "ewma_theta": cfg_fit.risk.ewma_theta,
        "blend_omega": cfg_fit.signal.blend_omega,
        "pbull_threshold": cfg_fit.signal.pbull_threshold,
        "f_star": float(f_star),
        "f_tilde": float(f_tilde),
    }

    return AnchorFit(
        regime_state=regime_state,
        vol_state=vol_state,
        policy_state=policy_state,
        exit_state=exit_state,
        kelly_inputs=kelly_inputs,
        f_star=float(f_star),
        f_tilde=float(f_tilde),
        chosen=chosen,
    )


def anchor_fit_to_dict(fit: AnchorFit) -> Dict[str, Any]:
    """Serialize an AnchorFit to a JSON/YAML-friendly dict."""

    d: Dict[str, Any] = {
        "chosen": dict(fit.chosen),
        "f_star": float(fit.f_star),
        "f_tilde": float(fit.f_tilde),
        "kelly_inputs": asdict(fit.kelly_inputs),
        "regime_state": {
            "blend_omega": fit.regime_state.blend_omega,
            "pbull_threshold": fit.regime_state.pbull_threshold,
            "ema_state": asdict(fit.regime_state.ema_state),
            "mom_state": asdict(fit.regime_state.mom_state),
        },
        "vol_state": {
            "theta": fit.vol_state.theta,
            "sigma2_init": fit.vol_state.sigma2_init,
            "vol_target_annual": fit.vol_state.vol_target_annual,
            "w_max": fit.vol_state.w_max,
        },
        "policy_state": asdict(fit.policy_state),
        "exit_state": {
            "atr": asdict(fit.exit_state.atr),
            "price_reference_for_peak": fit.exit_state.price_reference_for_peak,
            "derisk_policy": fit.exit_state.derisk_policy,
            "stop_fill_policy": fit.exit_state.stop_fill_policy,
        },
    }
    return d


__all__ = ["AnchorFit", "fit_anchor", "anchor_fit_to_dict"]
