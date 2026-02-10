"""Configuration utilities.

This module establishes the project-wide time and P&L conventions and provides
strict, explicit configuration dataclasses.

Key conventions (must match reproduction plan):
- Decision time uses information up to close of day t (F_t).
- Baseline execution is T+1 close: a target decided at t becomes executed
  weight at t+1.
- P&L attribution: net_ret[t] uses executed weight held over (t-1 -> t):
    gross_ret[t] = w_exec[t-1] * r[t]
  where r[t] = P[t]/P[t-1] - 1 computed from calendar-aligned continuous close.

Hyperparameters are explicit and can be serialized per walk-forward anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Literal, Optional, Tuple


DeriskPolicy = Literal["DERISK_HALF", "DERISK_CLOSE"]
StopFillPolicy = Literal["STOP_FILL_T_PLUS_1", "STOP_FILL_SAME_CLOSE"]
BaselineFloorMode = Literal["FLOOR_ON_WVOL", "FLOOR_ON_WCONF"]
PeakRef = Literal["close", "high"]
TrainerMode = Literal["FIXED", "GRID"]
StitchRule = Literal["FIRST_STEP_ONLY", "FULL_TEST_DIAGNOSTIC"]


@dataclass(frozen=True)
class TimeConvention:
    """Execution/attribution convention parameters."""

    # Execution lag in business days: 1 is baseline (T+1 close)
    exec_lag: int = 1

    delta_w_cap: float = 0.0

    # Stop fill policy: baseline stop triggers at close t and exits at close t+1
    stop_fill_policy: StopFillPolicy = "STOP_FILL_T_PLUS_1"

    # Stitching rule for overlapping OOS slices
    stitch_rule: StitchRule = "FIRST_STEP_ONLY"

    


@dataclass(frozen=True)
class DataConfig:
    """Data inputs and calendar alignment settings."""

    tz_naive_dates: bool = True
    calendar: Literal["NYSE"] = "NYSE"

    # Continuous roll rule: roll this many business days before first notice date
    roll_bd_before_fnd: int = 2

    # Required columns expectations for contract bars
    price_col: str = "close"
    high_col: str = "high"
    low_col: str = "low"
    open_col: Optional[str] = "open"
    volume_col: Optional[str] = "volume"
    adv_col: Optional[str] = "adv"

    # Futures contract multiplier for capacity mapping
    contract_multiplier: float = 100.0


@dataclass(frozen=True)
class SignalConfig:
    """Signal engine hyperparameters."""

    # EMA smoothing lambda (higher => smoother)
    ema_lambda: float = 0.94

    # Momentum lookback
    momentum_k: int = 50

    # Blend weight for p_trend vs momentum
    blend_omega: float = 0.6

    # Entry threshold on p_bull
    pbull_threshold: float = 0.52

    # z-score clipping range
    z_clip: Tuple[float, float] = (-3.0, 3.0)


@dataclass(frozen=True)
class RiskConfig:
    """Risk model hyperparameters."""

    # EWMA variance decay theta
    ewma_theta: float = 0.94

    # Annual vol target
    vol_target_annual: float = 0.15

    # Max leverage/weight cap
    w_max: float = 2.0


@dataclass(frozen=True)
class ATRExitConfig:
    """ATR computation and exit-state-machine parameters."""

    atr_window: int = 14

    hard_stop_atr: float = 2.0
    trailing_stop_atr: float = 1.5

    timeout_days: int = 30

    price_reference_for_peak: PeakRef = "close"

    derisk_policy: DeriskPolicy = "DERISK_HALF"


@dataclass(frozen=True)
class CostImpactConfig:
    """Execution cost model parameters."""

    # Linear cost per unit turnover (decimal return units)
    k_linear: float = 0.00007  # 0.7 bps

    # Impact coefficient
    gamma_impact: float = 0.02


@dataclass(frozen=True)
class KellyConfig:
    """Friction-adjusted fractional Kelly sizing."""

    lambda_kelly: float = 0.40

    baseline_floor: float = 0.25
    baseline_floor_mode: BaselineFloorMode = "FLOOR_ON_WVOL"
    baseline_floor_eps: float = 1e-6


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward schedule and training mode."""

    # Window lengths in business days
    train_bd: int = 2520
    test_bd: int = 126
    step_bd: int = 21

    # Anchor boundaries (inclusive start, inclusive end) for schedule builder
    anchor_start: str = "2015-01-01"
    anchor_end: str = "2025-10-31"

    trainer_mode: TrainerMode = "FIXED"


@dataclass(frozen=True)
class RegressionConfig:
    nw_lags: int = 10
    nw_lags_sensitivity: Tuple[int, ...] = (5, 10, 20)


@dataclass(frozen=True)
class BootstrapConfig:
    block_bootstrap_B: int = 1000
    block_len: int = 20

    stationary_bootstrap_B: int = 800
    stationary_mean_block: int = 20

    seed: int = 123


@dataclass(frozen=True)
class CapacityConfig:
    participation_cap: float = 0.01


@dataclass(frozen=True)
class FTFConfig:
    """Top-level configuration container."""

    time: TimeConvention = field(default_factory=TimeConvention)
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    atr_exit: ATRExitConfig = field(default_factory=ATRExitConfig)
    costs: CostImpactConfig = field(default_factory=CostImpactConfig)
    kelly: KellyConfig = field(default_factory=KellyConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)

    # Misc
    run_name: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries."""

    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(cfg: FTFConfig) -> None:
    """Basic sanity checks for critical conventions."""

    if cfg.time.exec_lag not in (0, 1, 2):
        raise ValueError("exec_lag must be in {0,1,2}")
    if cfg.signal.z_clip[0] >= cfg.signal.z_clip[1]:
        raise ValueError("z_clip must be (low, high) with low < high")
    if cfg.atr_exit.atr_window < 2:
        raise ValueError("atr_window too small")
    if cfg.walkforward.train_bd <= 0 or cfg.walkforward.test_bd <= 0 or cfg.walkforward.step_bd <= 0:
        raise ValueError("walkforward window lengths must be positive")
    if cfg.walkforward.step_bd > cfg.walkforward.test_bd:
        raise ValueError("step_bd must be <= test_bd")
    if not (0.0 < cfg.risk.vol_target_annual < 1.0):
        raise ValueError("vol_target_annual should be in (0,1)")
    if cfg.risk.w_max <= 0:
        raise ValueError("w_max must be positive")


__all__ = [
    "FTFConfig",
    "TimeConvention",
    "DataConfig",
    "SignalConfig",
    "RiskConfig",
    "ATRExitConfig",
    "CostImpactConfig",
    "KellyConfig",
    "WalkForwardConfig",
    "RegressionConfig",
    "BootstrapConfig",
    "CapacityConfig",
    "deep_update",
    "validate_config",
]
