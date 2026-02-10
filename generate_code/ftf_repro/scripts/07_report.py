"""Generate a lightweight reproduction report.

This script is intentionally pragmatic (not a full paper-quality reporting suite).
It loads a stitched walk-forward output (or re-runs the walk-forward pipeline),
computes headline performance statistics, regression vs LBMA spot (if provided),
bootstrap Sharpe confidence intervals, and writes small tables/figures to a
reports/ directory.

Usage
-----
Baseline (after running scripts/02_run_fast_oos.py):

    python ftf_repro/scripts/07_report.py \
        --run_dir reports/base_fast \
        --lbma_path data/raw/lbma_pm_fix.parquet

Or run end-to-end from processed continuous futures:

    python ftf_repro/scripts/07_report.py \
        --config ftf_repro/configs/base_fast.yaml \
        --processed_path data/processed/gc_continuous.parquet \
        --out_dir reports/base_fast \
        --lbma_path data/raw/lbma_pm_fix.parquet

Outputs
-------
- reports/report_summary.json
- reports/perf_table.csv
- reports/regression_table.csv (if lbma supplied)
- reports/sharpe_ci.json
- reports/equity_curve.png

"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ftf.stats.bootstrap import bootstrap_sharpe_ci
from ftf.stats.metrics import summarize
from ftf.stats.regression import hac_regression_sensitivity, result_to_dict
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
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "Existing run directory created by scripts/02_run_fast_oos.py. "
            "If provided, loads reports/oos_daily.parquet and config_snapshot.yaml from it."
        ),
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML to (re-)run walk-forward if run_dir not provided.",
    )
    p.add_argument(
        "--processed_path",
        type=str,
        default=None,
        help="Processed continuous futures parquet (output of scripts/01_build_data.py).",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for report artifacts. Defaults to <run_dir>/reports or ./reports/report.",
    )

    p.add_argument(
        "--lbma_path",
        type=str,
        default=None,
        help="Optional LBMA spot price file (parquet/csv) for benchmark regression.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for bootstraps (overrides config.bootstrap.seed).",
    )

    return p.parse_args()


def _dict_to_cfg(d: Dict[str, Any]) -> FTFConfig:
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
        run_name=d.get("run_name", "base"),
    )
    validate_config(cfg)
    return cfg


def _read_lbma_returns(path: str, *, calendar_name: str = "NYSE") -> pd.Series:
    from ftf.data.loaders import read_lbma_spot

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    spot = read_lbma_spot(p, calendar_name=calendar_name)
    r = spot.pct_change().replace([np.inf, -np.inf], np.nan)
    r.name = "r_gold"
    return r


def _maybe_plot_equity(net_ret: pd.Series, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        eq = (1.0 + net_ret.fillna(0.0)).cumprod()
        fig, ax = plt.subplots(figsize=(10, 4))
        eq.plot(ax=ax, lw=1.5)
        ax.set_title("Equity curve (net)")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        # matplotlib optional; never fail the reporting pipeline
        return


def _load_or_run(
    *,
    run_dir: Optional[str],
    config_path: Optional[str],
    processed_path: Optional[str],
    out_dir: Optional[str],
) -> tuple[pd.DataFrame, FTFConfig, Path]:
    if run_dir is not None:
        rd = Path(run_dir)
        if not rd.exists():
            raise FileNotFoundError(run_dir)

        cfg_path = rd / "config_snapshot.yaml"
        daily_path = rd / "reports" / "oos_daily.parquet"
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))
        if not daily_path.exists():
            raise FileNotFoundError(str(daily_path))

        cfg = _dict_to_cfg(load_yaml(cfg_path))
        daily = load_parquet(daily_path)
        od = Path(out_dir) if out_dir is not None else (rd / "reports")
        ensure_dir(od)
        return daily, cfg, od

    if config_path is None or processed_path is None:
        raise ValueError("Provide --run_dir OR both --config and --processed_path")

    base = load_yaml(config_path)
    cfg = _dict_to_cfg(base)

    df_cont = load_parquet(processed_path)
    # Run walk-forward and use its stitched daily output
    res = run_walkforward(df_cont, cfg=cfg, out_dir=None, persist_daily=False, persist_per_anchor=False)

    od = Path(out_dir) if out_dir is not None else Path("reports") / "report"
    ensure_dir(od)

    # Persist minimal artifacts to make report self-contained
    save_yaml(cfg.to_dict(), od / "config_snapshot.yaml")
    save_parquet(res.oos_daily, od / "oos_daily.parquet")

    return res.oos_daily, cfg, od


def main() -> None:
    args = _parse_args()

    daily, cfg, out_dir = _load_or_run(
        run_dir=args.run_dir,
        config_path=args.config,
        processed_path=args.processed_path,
        out_dir=args.out_dir,
    )

    seed = int(args.seed) if args.seed is not None else int(cfg.bootstrap.seed)
    set_global_seed(seed)

    if "net_ret" not in daily.columns:
        raise ValueError("daily table missing required column 'net_ret'")

    net_ret = pd.Series(daily["net_ret"].values, index=pd.DatetimeIndex(daily.index), name="net_ret")
    w_exec = None
    if "w_exec" in daily.columns:
        w_exec = pd.Series(daily["w_exec"].values, index=pd.DatetimeIndex(daily.index), name="w_exec")

    perf = summarize(net_ret, w_exec=w_exec)
    perf_df = pd.DataFrame([perf])
    perf_df.to_csv(out_dir / "perf_table.csv", index=False)

    # Bootstrap Sharpe CI
    ci = bootstrap_sharpe_ci(
        net_ret.dropna(),
        B=int(cfg.bootstrap.block_bootstrap_B),
        block_len=int(cfg.bootstrap.block_len),
        seed=seed,
    )
    save_json(asdict(ci), out_dir / "sharpe_ci.json")

    # Regression vs benchmark if provided
    reg_rows = []
    if args.lbma_path is not None:
        bench_ret = _read_lbma_returns(args.lbma_path, calendar_name=cfg.data.calendar)
        sens = hac_regression_sensitivity(
            net_ret,
            bench_ret,
            nw_lags_list=cfg.regression.nw_lags_sensitivity,
        )
        for L, res in sens.items():
            row = result_to_dict(res)
            row["nw_lags"] = L
            reg_rows.append(row)
        reg_df = pd.DataFrame(reg_rows).sort_values("nw_lags")
        reg_df.to_csv(out_dir / "regression_table.csv", index=False)

    # Small summary json
    summary: Dict[str, Any] = {
        "run_name": cfg.run_name,
        "seed": seed,
        "n_days": int(np.isfinite(net_ret.values).sum()),
        "perf": perf,
        "sharpe_ci": asdict(ci),
    }
    if reg_rows:
        summary["regression"] = reg_rows

    save_json(summary, out_dir / "report_summary.json")

    # Plot
    _maybe_plot_equity(net_ret, out_dir / "equity_curve.png")


if __name__ == "__main__":
    main()
