"""Run SPA / White Reality Check grid test.

This script expects that you have already built continuous futures data via
`scripts/01_build_data.py` and (optionally) run the baseline walk-forward.

It will:
  1) load processed continuous futures data
  2) build a panel of OOS return series for each configuration in a grid
  3) run SPA and RC tests vs the baseline config

Outputs are written under: reports/<run_name>/spa/

The implementation is intentionally deterministic and may be slow for large grids.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ftf.stats.spa import spa_reality_check
from ftf.utils import (
    FTFConfig,
    deep_update,
    ensure_dir,
    load_parquet,
    load_yaml,
    save_json,
    save_yaml,
    set_global_seed,
    validate_config,
)
from ftf.walkforward.runner import run_walkforward


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ftf_repro/configs/base_fast.yaml")
    ap.add_argument("--grid", default="ftf_repro/configs/grids/spa_grid.yaml")
    ap.add_argument("--processed", default="ftf_repro/data/processed/gc_continuous.parquet")
    ap.add_argument("--out", default="ftf_repro/reports")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--method", choices=["block", "stationary"], default=None)
    return ap.parse_args()


def _dict_to_cfg(d: dict) -> FTFConfig:
    # Import inside to avoid circular import issues.
    from ftf.utils.config import (
        ATRExitConfig,
        BootstrapConfig,
        CapacityConfig,
        CostImpactConfig,
        DataConfig,
        FTFConfig,
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
        run_name=d.get("run_name", "base"),
    )
    validate_config(cfg)
    return cfg


def _run_one(df_cont: pd.DataFrame, base_cfg_dict: dict, overrides: dict) -> pd.Series:
    cfg_dict = deep_update(dict(base_cfg_dict), overrides)
    cfg = _dict_to_cfg(cfg_dict)
    res = run_walkforward(df_cont, cfg=cfg)
    return res.oos_net_ret


def main() -> None:
    args = _parse_args()

    base = load_yaml(args.config)
    grid = load_yaml(args.grid)

    cfg0 = _dict_to_cfg(base)

    B = int(args.B) if args.B is not None else int(cfg0.bootstrap.stationary_bootstrap_B)
    method = args.method if args.method is not None else "stationary"
    seed = int(args.seed) if args.seed is not None else int(cfg0.bootstrap.seed)
    set_global_seed(seed)

    df_cont = load_parquet(args.processed)

    out_root = Path(args.out) / cfg0.run_name / "spa"
    ensure_dir(out_root)
    save_yaml(base, out_root / "base_config_snapshot.yaml")
    save_yaml(grid, out_root / "spa_grid_snapshot.yaml")

    # Grid file format:
    #   baseline_name: "baseline"
    #   baseline_overrides: {...}
    #   configs:
    #     - name: "cfg1"
    #       overrides: {...}
    baseline_name = grid.get("baseline_name", "baseline")
    baseline_overrides = grid.get("baseline_overrides", {})
    configs = grid.get("configs", [])
    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("spa_grid.yaml must contain a non-empty list under 'configs'")

    panel: dict[str, pd.Series] = {}

    # Baseline
    panel[baseline_name] = _run_one(df_cont, base, baseline_overrides)

    # Alternatives
    for item in configs:
        name = item.get("name")
        overrides = item.get("overrides", {})
        if not name:
            raise ValueError("Each grid config must contain a 'name'")
        panel[str(name)] = _run_one(df_cont, base, overrides)

    spa_res = spa_reality_check(
        panel,
        baseline_name=baseline_name,
        test_kind="SPA",
        metric="sharpe",
        method=method,
        B=B,
        block_len=int(cfg0.bootstrap.block_len),
        mean_block_len=int(cfg0.bootstrap.stationary_mean_block),
        seed=seed,
    )
    rc_res = spa_reality_check(
        panel,
        baseline_name=baseline_name,
        test_kind="RC",
        metric="sharpe",
        method=method,
        B=B,
        block_len=int(cfg0.bootstrap.block_len),
        mean_block_len=int(cfg0.bootstrap.stationary_mean_block),
        seed=seed,
    )

    save_json(asdict(spa_res), out_root / "spa_result.json")
    save_json(asdict(rc_res), out_root / "rc_result.json")

    # Small summary table
    summary = pd.DataFrame(
        [
            {
                "test": "SPA",
                "metric": spa_res.metric,
                "method": spa_res.method,
                "B": spa_res.B,
                "t_obs": spa_res.t_obs,
                "p_value": spa_res.p_value,
                "best_name": spa_res.best_name,
                "best_value": spa_res.best_value,
            },
            {
                "test": "RC",
                "metric": rc_res.metric,
                "method": rc_res.method,
                "B": rc_res.B,
                "t_obs": rc_res.t_obs,
                "p_value": rc_res.p_value,
                "best_name": rc_res.best_name,
                "best_value": rc_res.best_value,
            },
        ]
    )
    summary.to_csv(out_root / "spa_rc_summary.csv", index=False)

    # Also write per-config Sharpe for inspection.
    def sharpe(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return float("nan")
        s = float(x.std(ddof=0))
        if s <= 0:
            return float("nan")
        return float(x.mean() / s * np.sqrt(252.0))

    perfs = pd.DataFrame(
        [{"name": k, "sharpe": sharpe(v), "mean": float(v.dropna().mean())} for k, v in panel.items()]
    ).sort_values("sharpe", ascending=False)
    perfs.to_csv(out_root / "panel_performance.csv", index=False)

    print("Wrote:")
    print(f"  {out_root / 'spa_result.json'}")
    print(f"  {out_root / 'rc_result.json'}")
    print(f"  {out_root / 'spa_rc_summary.csv'}")


if __name__ == "__main__":
    main()
