"""Run the baseline FAST walk-forward OOS backtest.

Main entry point:

- Loads processed continuous futures dataset (Parquet)
- Loads base configuration YAML (configs/base_fast.yaml)
- Runs strict walk-forward (10y train / 6m test / 1m step)
- Saves stitched OOS daily table and anchor summary to reports/

Example
-------
python scripts/02_run_fast_oos.py `
  --config configs/base_fast.yaml `
  --data data/processed/gc_continuous.parquet `
  --out reports/base_fast
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ftf.utils import (
    FTFConfig,
    deep_update,
    ensure_dir,
    load_parquet,
    load_yaml,
    save_json,
    save_parquet,
    save_yaml,
    set_global_seed,
    validate_config,
)
from ftf.walkforward.runner import run_walkforward


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline FAST walk-forward OOS backtest")

    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "base_fast.yaml"),
        help="Path to base configuration YAML",
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed continuous futures Parquet (df_cont)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "fast_oos"),
        help="Output directory for reports/artifacts",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config.bootstrap.seed)",
    )
    p.add_argument(
        "--trainer-mode",
        type=str,
        default=None,
        choices=["FIXED", "GRID"],
        help="Override walkforward.trainer_mode",
    )
    p.add_argument(
        "--exec-lag",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Override time.exec_lag (latency days)",
    )

    return p.parse_args()


def _dict_to_cfg(d: Dict[str, Any]) -> FTFConfig:
    # Late import to avoid circular import surfaces.
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
    )

    cfg = FTFConfig(
        time=TimeConvention(**d.get("time", {})),
        data=DataConfig(**d.get("data", {})),
        signal=SignalConfig(**d.get("signal", {})),
        risk=RiskConfig(**d.get("risk", {})),
        atr_exit=ATRExitConfig(**d.get("atr_exit", {})),
        costs=CostImpactConfig(**d.get("costs", {})),
        kelly=KellyConfig(**d.get("kelly", {})),
        walkforward=WalkForwardConfig(**d.get("walkforward", {})),
        regression=RegressionConfig(**d.get("regression", {})),
        bootstrap=BootstrapConfig(**d.get("bootstrap", {})),
        capacity=CapacityConfig(**d.get("capacity", {})),
        run_name=str(d.get("run_name", "base_fast")),
    )
    validate_config(cfg)
    return cfg


def _build_anchor_summary(res) -> pd.DataFrame:
    rows = []
    for a in res.anchors:
        rows.append(
            {
                "anchor": str(a.anchor.date()),
                "train_start": str(a.train_start.date()),
                "train_end": str((a.train_end - pd.Timedelta(days=1)).date()),
                "test_start": str(a.test_start.date()),
                "test_end": str((a.test_end - pd.Timedelta(days=1)).date()),
                "kept_start": str(a.kept_start.date()),
                "kept_end": str((a.kept_end - pd.Timedelta(days=1)).date()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    out_dir = ensure_dir(args.out)

    base_cfg_dict = load_yaml(args.config)

    overrides: Dict[str, Any] = {}
    if args.trainer_mode is not None:
        overrides.setdefault("walkforward", {})["trainer_mode"] = args.trainer_mode
    if args.exec_lag is not None:
        overrides.setdefault("time", {})["exec_lag"] = int(args.exec_lag)

    cfg_dict = deep_update(base_cfg_dict, overrides) if overrides else base_cfg_dict
    cfg = _dict_to_cfg(cfg_dict)

    seed = int(args.seed) if args.seed is not None else int(cfg.bootstrap.seed)
    set_global_seed(seed)

    df_cont = load_parquet(args.data)

    res = run_walkforward(df_cont=df_cont, cfg=cfg, out_dir=out_dir)

    # -----------------
    # Persist artifacts
    # -----------------
    save_yaml(cfg.to_dict(), out_dir / "config_snapshot.yaml")
    save_json({"seed": seed}, out_dir / "seed.json")

    # Main stitched OOS daily table (contains gross_ret/net_ret etc.)
    save_parquet(res.oos_daily, out_dir / "oos_daily.parquet")

    # Anchor schedule summary
    anchor_summary = _build_anchor_summary(res)
    save_parquet(anchor_summary, out_dir / "anchor_summary.parquet")

    # Frozen params: YAML is safest (dict of dicts). Parquet is not ideal for nested dict.
    save_yaml(res.frozen_params, out_dir / "frozen_params.yaml")

    # Convenience CSV
    try:
        res.oos_daily.to_csv(out_dir / "oos_daily.csv")
        anchor_summary.to_csv(out_dir / "anchor_summary.csv", index=False)
    except Exception:
        pass

    print("Wrote:", out_dir)
    print("OOS rows:", len(res.oos_daily))
    if len(res.oos_daily) > 0:
        print("OOS date range:", res.oos_daily.index.min().date(), "->", res.oos_daily.index.max().date())


if __name__ == "__main__":
    main()
