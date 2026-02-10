# scripts/09_sweep_lambda_wmax.py
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ============================================================
# Make src importable: src/ftf/... を import できるようにする
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftf.utils.config import (
    FTFConfig,
    TimeConvention,
    DataConfig,
    SignalConfig,
    RiskConfig,
    ATRExitConfig,
    CostImpactConfig,
    KellyConfig,
    WalkForwardConfig,
    RegressionConfig,
    BootstrapConfig,
    CapacityConfig,
    deep_update,
)
from ftf.walkforward.runner import run_walkforward


# ============================================================
# Helpers
# ============================================================

def sharpe_annual(ret: pd.Series, periods: int = 252) -> float:
    r = ret.astype(float).fillna(0.0)
    return float((r.mean() / (r.std(ddof=0) + 1e-12)) * np.sqrt(periods))


def equity_from_ret(ret: pd.Series) -> pd.Series:
    r = ret.astype(float).fillna(0.0)
    return (1.0 + r).cumprod()


def max_drawdown_from_equity(eq: pd.Series) -> float:
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def cfg_from_dict(d: dict) -> FTFConfig:
    """
    YAML + deep_update 後のネストdict から frozen dataclass を構築。
    """
    return FTFConfig(
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


def load_cfg_from_yaml(path: str | Path) -> FTFConfig:
    base = FTFConfig().to_dict()
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    merged = deep_update(base, updates)
    return cfg_from_dict(merged)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base_fast.yaml など")
    ap.add_argument("--data", required=True, help="data/processed/gc_continuous.parquet など")
    ap.add_argument("--out", required=True, help="reports/sweep_cost0204 など")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- load data ---
    df_cont = pd.read_parquet(args.data).sort_index()

    # --- load base cfg ---
    cfg0 = load_cfg_from_yaml(args.config)

    # --- FIX costs (your choice) ---
    cfg0 = replace(
        cfg0,
        costs=replace(cfg0.costs, k_linear=0.000345, gamma_impact=0.0),
    )

    # --- sweep grid ---
    # 「DD10%を使い切る」方向なら、後でここを攻めたグリッドにしてOK
    lambda_grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
    wmax_grid   = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    rows = []
    best = None

    for lam, wmax in itertools.product(lambda_grid, wmax_grid):
        run_name = f"sweep_lam{lam:.2f}_wmax{wmax:.2f}"
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = replace(
            cfg0,
            run_name=run_name,
            kelly=replace(cfg0.kelly, lambda_kelly=float(lam)),
            risk=replace(cfg0.risk, w_max=float(wmax)),
        )

        res = run_walkforward(
            df_cont=df_cont,
            cfg=cfg,
            out_dir=out_dir,
            persist_daily=True,
            persist_per_anchor=False,  # sweep は軽くする（必要なら True に）
            progress=args.progress,
        )

        df_oos = res.oos_daily
        if "net_ret" not in df_oos.columns:
            raise RuntimeError("oos_daily に net_ret 列が見つかりません")

        r = df_oos["net_ret"].astype(float).fillna(0.0)
        sh = sharpe_annual(r)
        eq = equity_from_ret(r)
        mdd = max_drawdown_from_equity(eq)
        feq = float(eq.iloc[-1])

        row = {
            "lambda_kelly": float(lam),
            "w_max": float(wmax),
            "sharpe": float(sh),
            "maxdd": float(mdd),
            "final_eq": float(feq),
            "out": str(out_dir),
        }
        rows.append(row)

        if best is None or row["sharpe"] > best["sharpe"]:
            best = row

        print(f"[{run_name}] Sharpe={sh:.3f}  MaxDD={mdd:.2%}  FinalEq={feq:.3f}")

    df_res = pd.DataFrame(rows).sort_values(["sharpe", "final_eq"], ascending=[False, False])
    df_res.to_csv(out_root / "sweep_results.csv", index=False)

    with open(out_root / "best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n=== BEST (by Sharpe on net_ret) ===")
    print(best)


if __name__ == "__main__":
    main()
