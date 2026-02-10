"""Run transaction cost + market impact stress grid.

This script mirrors :mod:`scripts/03_latency.py` but varies the deterministic cost
coefficients used by the execution cost model.

Usage
-----
python scripts/04_cost_impact.py \
  --config configs/base_fast.yaml \
  --grid configs/grids/cost_impact_grid.yaml \
  --processed data/processed/gc_continuous.parquet \
  --out reports/cost_impact

Outputs
-------
Writes per-variant run artifacts under ``<out>/<name>/`` and aggregated
comparison tables under ``<out>/summary/``.

Notes
-----
- Uses the same strict walk-forward runner and canonical stitching rule.
- Each variant deep-merges its overrides onto the base config dict, then
  reconstructs a validated :class:`ftf.utils.config.FTFConfig`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

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
    p = argparse.ArgumentParser(description="Run cost/impact robustness grid")
    p.add_argument("--config", type=str, default=str(Path("configs/base_fast.yaml")))
    p.add_argument(
        "--grid", type=str, default=str(Path("configs/grids/cost_impact_grid.yaml"))
    )
    p.add_argument(
        "--processed",
        type=str,
        default=str(Path("data/processed/gc_continuous.parquet")),
        help="Processed continuous futures parquet produced by scripts/01_build_data.py",
    )
    p.add_argument("--out", type=str, default=str(Path("reports/cost_impact")))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    return p.parse_args()


def _dict_to_cfg(d: Dict[str, Any]) -> FTFConfig:
    """Reconstruct a typed config tree from a raw nested dict."""

    from ftf.utils.config import (  # local import to avoid circulars in some contexts
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
        run_name=d.get("run_name", "run"),
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


def _run_one(
    df_cont: pd.DataFrame,
    base_cfg_dict: Dict[str, Any],
    overrides: Dict[str, Any],
    out_dir: Path,
    *,
    progress: bool,
) -> Dict[str, Any]:
    merged = deep_update(dict(base_cfg_dict), overrides)
    cfg = _dict_to_cfg(merged)

    run_dir = ensure_dir(out_dir / cfg.run_name)
    save_yaml(merged, run_dir / "config_snapshot.yaml")

    res = run_walkforward(df_cont, cfg=cfg, out_dir=run_dir, progress=progress)

    # save stitched outputs for aggregation
    save_parquet(res.oos_daily, run_dir / "oos_daily.parquet")

    return {
        "cfg": cfg,
        "run_dir": str(run_dir),
        "net_ret": res.oos_net_ret,
        "w_exec": res.oos_daily.get("w_exec"),
        "summary": {
            "run_name": cfg.run_name,
            "k_linear": cfg.costs.k_linear,
            "gamma_impact": cfg.costs.gamma_impact,
            "n_days": int(res.oos_net_ret.dropna().shape[0]),
        },
    }


def main() -> None:
    args = _parse_args()

    base_cfg_dict = load_yaml(args.config)
    grid = load_yaml(args.grid)
    df_cont = load_parquet(args.processed)

    seed = int(args.seed) if args.seed is not None else int(base_cfg_dict.get("bootstrap", {}).get("seed", 123))
    set_global_seed(seed)

    out_root = ensure_dir(args.out)

    panel: Dict[str, pd.Series] = {}
    w_panel: Dict[str, pd.Series] = {}
    meta_rows = []

    for entry in grid.get("configs", []):
        name = entry["name"]
        overrides = entry.get("overrides", {})

        # Force run_name to variant name for deterministic folder naming.
        overrides = deep_update({"run_name": name}, overrides)

        payload = _run_one(
            df_cont,
            base_cfg_dict,
            overrides,
            out_root,
            progress=bool(args.progress),
        )

        panel[name] = payload["net_ret"]
        if payload["w_exec"] is not None:
            w_panel[name] = payload["w_exec"]
        meta_rows.append(payload["summary"])

    summary_dir = ensure_dir(out_root / "summary")

    # Aggregations
    perf = performance_table(panel, w_exec_panel=w_panel if w_panel else None)
    perf.to_csv(summary_dir / "performance.csv")

    meta = pd.DataFrame(meta_rows).set_index("run_name").sort_index()
    meta.to_csv(summary_dir / "meta.csv")

    # Save aligned panel returns
    df_panel = pd.concat(panel, axis=1)
    save_parquet(df_panel, summary_dir / "panel_net_ret.parquet")

    summary = {
        "seed": seed,
        "config": str(Path(args.config)),
        "grid": str(Path(args.grid)),
        "processed": str(Path(args.processed)),
        "baseline_name": grid.get("baseline_name"),
        "n_variants": int(len(panel)),
    }
    save_json(summary, summary_dir / "summary.json")

    # Also store the performance table as JSON for convenience
    save_json({"performance": perf.reset_index().to_dict(orient="records")}, summary_dir / "performance.json")


if __name__ == "__main__":
    main()
