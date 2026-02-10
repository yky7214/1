"""Utility subpackage.

These helpers are intentionally small and dependency-light to support reproducible
walk-forward experiments.

Public exports:
- Config dataclasses and validation utilities
- YAML/JSON/Parquet IO convenience helpers
- Seed utilities for deterministic bootstraps
"""

from .config import (
    ATRExitConfig,
    BaselineFloorMode,
    BootstrapConfig,
    CapacityConfig,
    CostImpactConfig,
    DataConfig,
    DeriskPolicy,
    FTFConfig,
    KellyConfig,
    PeakRef,
    RegressionConfig,
    RiskConfig,
    SignalConfig,
    StopFillPolicy,
    StitchRule,
    TimeConvention,
    TrainerMode,
    WalkForwardConfig,
    deep_update,
    validate_config,
)
from .io import (
    ensure_dir,
    load_json,
    load_parquet,
    load_yaml,
    save_json,
    save_parquet,
    save_yaml,
)
from .seed import set_global_seed

__all__ = [
    # config
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
    # literals
    "DeriskPolicy",
    "StopFillPolicy",
    "BaselineFloorMode",
    "PeakRef",
    "TrainerMode",
    "StitchRule",
    # utils
    "deep_update",
    "validate_config",
    # io
    "ensure_dir",
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "load_parquet",
    "save_parquet",
    # seed
    "set_global_seed",
]
