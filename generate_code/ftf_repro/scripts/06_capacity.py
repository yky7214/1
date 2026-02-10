"""Capacity analysis runner.

This script estimates strategy capacity following the reproduction plan:
- Run (or load) a baseline walk-forward OOS backtest.
- Build a unit-notional sleeve return series proxy and estimate (mu_u, sigma_u).
- Compute a leverage-space growth curve g(L) and solve for L_max where g(L)=0.
- Map executed-weight turnover and market ADV to an AUM capacity estimate under a
  participation cap.

Outputs are written under the chosen output directory:
- capacity_summary.json
- growth_curve.csv
- participation_summaries.csv (q summaries for AUMs)
- growth_curve.png (if matplotlib available)

Usage:
  python scripts/06_capacity.py --run_dir reports/base_fast

Or to re-run from config + processed data:
  python scripts/06_capacity.py --config configs/base_fast.yaml --processed_path data/processed/continuous.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.capacity import (
    estimate_aum_capacity,
    estimate_unit_notional_stats,
    growth_curve,
    solve_L_max,
)
from ftf.capacity.aum_mapping import aum_participation_summary, capacity_dict
from ftf.reporting import plot_growth_curve
from ftf.sizing.kelly import estimate_kelly_inputs
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
    p = argparse.ArgumentParser(description="Forecast-to-Fill capacity analysis")
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing run directory containing reports/oos_daily.parquet and config_snapshot.yaml",
    )
    p.add_argument("--config", type=str, default=None, help="Base YAML config to run")
    p.add_argument(
        "--processed_path",
        type=str,
        default=None,
        help="Processed continuous futures parquet (from scripts/01_build_data.py)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for capacity artifacts (default: <run_dir>/reports/capacity or ./reports/capacity)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config.bootstrap.seed)")
    p.add_argument(
        "--aum_list",
        type=str,
        default="100e6,500e6,1e9",
        help="Comma-separated AUMs (USD) to compute participation summaries for",
    )
    p.add_argument(
        "--participation_cap",
        type=float,
        default=None,
        help="Override participation cap (default from cfg.capacity.participation_cap)",
    )
    p.add_argument(
        "--L_max_grid",
        type=float,
        default=5.0,
        help="Max leverage for growth curve grid (default 5.0)",
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
        run_name=d.get("run_name", d.get("name", "base")),
    )
    validate_config(cfg)
    return cfg


def _load_or_run(
    *,
    run_dir: Optional[str],
    config_path: Optional[str],
    processed_path: Optional[str],
    out_dir: Optional[str],
) -> Tuple[pd.DataFrame, FTFConfig, Path]:
    if run_dir is not None:
        rd = Path(run_dir)
        cfg_path = rd / "config_snapshot.yaml"
        daily_path = rd / "reports" / "oos_daily.parquet"
        if not cfg_path.exists() or not daily_path.exists():
            raise FileNotFoundError(
                f"run_dir must contain {cfg_path} and {daily_path}. Got run_dir={rd}"
            )
        cfg = _dict_to_cfg(load_yaml(cfg_path))
        daily = load_parquet(daily_path)
        out = Path(out_dir) if out_dir is not None else (rd / "reports" / "capacity")
        ensure_dir(out)
        return daily, cfg, out

    if config_path is None or processed_path is None:
        raise ValueError("Provide either --run_dir or (--config and --processed_path)")

    cfg_dict = load_yaml(Path(config_path))
    cfg = _dict_to_cfg(cfg_dict)
    df_cont = load_parquet(Path(processed_path))
    out = Path(out_dir) if out_dir is not None else Path("reports") / "capacity"
    ensure_dir(out)

    wf_out = out / "walkforward"
    ensure_dir(wf_out)
    res = run_walkforward(df_cont, cfg=cfg, out_dir=wf_out, progress=False)
    daily = res.oos_daily

    # snapshot config used
    save_yaml(cfg.to_dict(), out / "config_snapshot.yaml")
    return daily, cfg, out


def _unit_notional_proxy_returns(daily: pd.DataFrame) -> pd.Series:
    """Build a pragmatic unit-notional sleeve proxy return series.

    The paper defines this sleeve using the *same engine with fixed weight=1 when active*.
    In this codebase, we may not have stored that sleeve directly for OOS. For capacity
    we use an OOS proxy derived from executed exposure:

      pos_proxy[t-1] = 1{ w_exec[t-1] > 1e-3 }
      R_u[t] = pos_proxy[t-1] * r[t]

    This captures the opportunity set conditional on being in a trade.
    """

    if "r" not in daily.columns or "w_exec" not in daily.columns:
        raise ValueError("daily must contain columns 'r' and 'w_exec'")

    r = daily["r"].astype(float)
    w_exec = daily["w_exec"].astype(float)
    pos_prev = (w_exec.shift(1).fillna(0.0) > 1e-3).astype(float)
    R_u = pos_prev * r
    R_u.name = "unit_notional_proxy"
    return R_u


def main() -> None:
    args = _parse_args()

    daily, cfg, out = _load_or_run(
        run_dir=args.run_dir,
        config_path=args.config,
        processed_path=args.processed_path,
        out_dir=args.out_dir,
    )

    seed = int(args.seed) if args.seed is not None else int(cfg.bootstrap.seed)
    set_global_seed(seed)

    # Parse AUM list
    aums: list[float] = []
    for s in str(args.aum_list).split(","):
        s = s.strip()
        if not s:
            continue
        aums.append(float(eval(s, {"__builtins__": {}}, {})))  # allow 1e9 style
    if not aums:
        aums = [100e6, 500e6, 1e9]

    part_cap = (
        float(args.participation_cap)
        if args.participation_cap is not None
        else float(cfg.capacity.participation_cap)
    )

    # Growth curve from unit-notional proxy
    R_u = _unit_notional_proxy_returns(daily)
    mu_u, sigma_u = estimate_unit_notional_stats(R_u)

    # Also record mu/sigma2 like Kelly inputs (daily)
    ki = estimate_kelly_inputs(R_u, n=1.0)

    gc = growth_curve(mu_u=mu_u, sigma_u=sigma_u, L_max=float(args.L_max_grid), costs=cfg.costs)
    Lmax = solve_L_max(mu_u=mu_u, sigma_u=sigma_u, costs=cfg.costs, bracket=(0.0, max(10.0, float(args.L_max_grid))))

    gc_df = pd.DataFrame({"L": gc.L, "g": gc.g})
    gc_df.to_csv(out / "growth_curve.csv", index=False)

    # Participation / AUM mapping (requires adv + price)
    price_col = cfg.data.price_col
    adv_col = cfg.data.adv_col
    if adv_col is None or adv_col not in daily.columns:
        raise ValueError(
            "Capacity mapping requires ADV column in daily logs (cfg.data.adv_col). "
            f"adv_col={adv_col!r} is missing from daily columns."
        )
    if price_col not in daily.columns:
        raise ValueError(f"daily missing price column {price_col!r}")

    price = daily[price_col].astype(float)
    adv = daily[adv_col].astype(float)
    w_exec = daily["w_exec"].astype(float)

    cap_res = estimate_aum_capacity(
        w_exec,
        price=price,
        adv=adv,
        participation_cap=part_cap,
        contract_multiplier=float(cfg.data.contract_multiplier),
    )

    part_rows = []
    for A in aums:
        q, summ = aum_participation_summary(
            w_exec,
            aum=float(A),
            price=price,
            adv=adv,
            contract_multiplier=float(cfg.data.contract_multiplier),
        )
        part_rows.append({"aum": float(A), **asdict(summ)})
        # Save participation series for each AUM
        q.to_frame("participation").to_parquet(out / f"participation_A{int(A):d}.parquet")

    pd.DataFrame(part_rows).to_csv(out / "participation_summaries.csv", index=False)

    summary = {
        "seed": seed,
        "mu_u_daily": float(mu_u),
        "sigma_u_daily": float(sigma_u),
        "kelly_inputs": asdict(ki),
        "growth_curve": {
            "k_linear": float(cfg.costs.k_linear),
            "gamma_impact": float(cfg.costs.gamma_impact),
            "L_max_grid": float(args.L_max_grid),
            "L_star_zero_cross": float(Lmax),
        },
        "aum_capacity": capacity_dict(cap_res),
        "participation_cap": float(part_cap),
        "aums_checked": [float(x) for x in aums],
    }

    save_json(summary, out / "capacity_summary.json")
    save_yaml({"capacity_summary": summary}, out / "capacity_summary.yaml")

    # Optional plot
    try:
        plot_growth_curve(gc.L, gc.g, out_path=str(out / "growth_curve.png"), title="Growth curve g(L)")
    except Exception:
        pass

    print(f"Wrote capacity artifacts to: {out}")


if __name__ == "__main__":
    main()
