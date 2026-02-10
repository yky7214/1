"""Run latency robustness grid.

This script mirrors the paper's latency sensitivity experiment by re-running the
strict walk-forward backtest under different execution delays d ∈ {0,1,2}.

Usage
-----
python ftf_repro/scripts/03_latency.py \
  --config ftf_repro/configs/base_fast.yaml \
  --grid ftf_repro/configs/grids/latency_grid.yaml \
  --processed ftf_repro/data/processed/gc_continuous.parquet \
  --out ftf_repro/reports/latency

The script writes, for each grid entry:
- per-variant run directory containing config snapshot + stitched oos_daily
- a summary CSV with metrics and baseline-relative diffs

"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ftf.reporting import performance_table
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
    p = argparse.ArgumentParser(description="Latency robustness grid runner")
    p.add_argument("--config", type=str, required=True, help="Base config YAML (e.g. base_fast.yaml)")
    p.add_argument("--grid", type=str, required=True, help="Latency grid YAML (e.g. grids/latency_grid.yaml)")
    p.add_argument("--processed", type=str, required=True, help="Processed continuous futures parquet")
    p.add_argument("--out", type=str, required=True, help="Output directory root")
    p.add_argument("--seed", type=int, default=None, help="Override seed (defaults to base config bootstrap.seed)")
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress inside walk-forward runner (slower due to tqdm overhead)",
    )
    return p.parse_args()


def _dict_to_cfg(d: Dict[str, Any]) -> FTFConfig:
    # Mirror scripts/02_run_fast_oos.py for deterministic typed config reconstruction.
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
        run_name=str(d.get("run_name", "run")),
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
    )
    validate_config(cfg)
    return cfg


def _run_one(df_cont: pd.DataFrame, base_cfg_dict: Dict[str, Any], overrides: Dict[str, Any], out_dir: Path, *, progress: bool) -> Dict[str, Any]:
    cfg_dict = deep_update(dict(base_cfg_dict), overrides)
    cfg = _dict_to_cfg(cfg_dict)

    run_dir = out_dir / cfg.run_name
    ensure_dir(run_dir)

    # Run WF
    res = run_walkforward(df_cont, cfg=cfg, out_dir=run_dir, progress=progress)

    # Persist stitched series and snapshot
    save_yaml(cfg.to_dict(), run_dir / "config_snapshot.yaml")
    save_parquet(res.oos_daily, run_dir / "reports" / "oos_daily.parquet")
    save_json({"summary": res.oos_daily.columns.tolist()}, run_dir / "reports" / "schema.json")

    net_ret = res.oos_daily["net_ret"].copy()
    w_exec = res.oos_daily["w_exec"].copy() if "w_exec" in res.oos_daily.columns else None

    # Return minimal objects for aggregation
    return {
        "cfg": cfg,
        "run_dir": str(run_dir),
        "net_ret": net_ret,
        "w_exec": w_exec,
    }


def main() -> None:
    args = _parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    base_cfg_dict = load_yaml(args.config)
    grid = load_yaml(args.grid)

    df_cont = load_parquet(args.processed)

    base_seed = int(base_cfg_dict.get("bootstrap", {}).get("seed", 123))
    seed = base_seed if args.seed is None else int(args.seed)
    set_global_seed(seed)

    baseline_name = str(grid.get("baseline_name", "baseline"))
    configs = grid.get("configs", [])
    if not configs:
        raise ValueError("Grid YAML must contain non-empty 'configs'")

    panel_rets: Dict[str, pd.Series] = {}
    panel_w: Dict[str, pd.Series] = {}
    meta_rows = []

    for entry in configs:
        name = str(entry["name"])
        overrides = dict(entry.get("overrides", {}))
        # Use a stable per-variant run_name
        overrides = deep_update(overrides, {"run_name": name})

        result = _run_one(df_cont, base_cfg_dict, overrides, out_root, progress=args.progress)
        panel_rets[name] = result["net_ret"]
        if result["w_exec"] is not None:
            panel_w[name] = result["w_exec"]

        cfg: FTFConfig = result["cfg"]
        meta_rows.append(
            {
                "name": name,
                "run_dir": result["run_dir"],
                "exec_lag": cfg.time.exec_lag,
                "k_linear": cfg.costs.k_linear,
                "gamma_impact": cfg.costs.gamma_impact,
            }
        )

    if baseline_name not in panel_rets:
        raise ValueError(f"baseline_name '{baseline_name}' not found in grid configs")

    # Tables
    perf = performance_table(panel_rets, w_exec_panel=panel_w if panel_w else None)
    perf.index.name = "name"

    # Baseline-relative diffs for headline fields
    base = perf.loc[baseline_name]
    for col in ["sharpe", "cagr", "ann_vol", "max_dd", "calmar", "ann_mean"]:
        if col in perf.columns:
            perf[f"{col}_diff_vs_{baseline_name}"] = perf[col] - float(base[col])

    # Simple rank by Sharpe
    if "sharpe" in perf.columns:
        perf["sharpe_rank"] = perf["sharpe"].rank(ascending=False, method="min")

    meta_df = pd.DataFrame(meta_rows).set_index("name").sort_index()
    out_reports = ensure_dir(out_root / "reports")

    perf.to_csv(out_reports / "latency_performance.csv")
    meta_df.to_csv(out_reports / "latency_meta.csv")

    # Persist panel summary JSON
    save_json(
        {
            "seed": seed,
            "baseline_name": baseline_name,
            "variants": meta_rows,
            "performance": perf.reset_index().to_dict(orient="records"),
        },
        out_reports / "latency_summary.json",
    )

    # Also save the aligned panel returns matrix for convenience
    # (inner-join alignment)
    common_idx = None
    for s in panel_rets.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    common_idx = common_idx.sort_values() if common_idx is not None else None

    if common_idx is not None and len(common_idx) > 0:
        panel_mat = pd.DataFrame({k: v.reindex(common_idx) for k, v in panel_rets.items()})
        panel_mat.to_parquet(out_reports / "latency_panel_net_ret.parquet")


if __name__ == "__main__":
    main()
