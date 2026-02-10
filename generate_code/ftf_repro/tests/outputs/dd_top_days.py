import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../ftf_repro
PARQUET = ROOT / "reports/base_fast_eventclose/reports/oos_daily.parquet"

# 上位何日見る？
TOP_N = 30

# 見たい列（存在するやつだけ出す）
CAND_COLS = [
    "net_ret_ftf", "gross_ret",
    "fill_error_ret",
    "cost_lin", "cost_imp",
    "turnover", "dw",
    "w_target", "w_exec",
    "p_bear", "p_bull",
    "eligible_to_enter",
    "atr", "atr14", "TR",
]

def main():
    assert PARQUET.exists(), f"missing: {PARQUET}"
    df = pd.read_parquet(PARQUET).sort_index()

    # 使うリターン列を自動選択（netがあればnet、なければgross）
    ret_col = "net_ret_ftf" if "net_ret_ftf" in df.columns else "gross_ret"
    assert ret_col in df.columns, f"missing return col: {ret_col}"

    # 欠損は0扱い（安全側）
    df[ret_col] = df[ret_col].fillna(0.0)

    # equity & drawdown
    equity = (1.0 + df[ret_col]).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    out = df.copy()
    out["equity"] = equity
    out["dd"] = dd
    out["dd_change"] = out["dd"].diff().fillna(0.0)   # DDが悪化した量（=その日の“DD寄与”）

    # DD悪化が大きい順（=DDに最も効いた日）
    worst = out.sort_values("dd_change").head(TOP_N)

    # 表示する列を存在チェックして並べる
    cols = ["dd_change", "dd", "equity", ret_col]
    cols += [c for c in CAND_COLS if c in out.columns and c not in cols]

    # 見やすく
    worst_view = worst[cols].copy()
    worst_view.index = worst_view.index.astype(str)

    print("\n=== DD TOP DAYS (most negative dd_change) ===")
    print(f"Parquet : {PARQUET}")
    print(f"Return  : {ret_col}")
    print(worst_view.to_string())

    # “犯人パターン”のざっくり統計（上位TOP_N日の特徴）
    print("\n=== SUMMARY on TOP DAYS ===")
    def _mean(s): 
        return float(np.nanmean(s)) if len(s) else np.nan

    summary = {}
    if "p_bear" in out.columns:
        summary["p_bear_mean_top"] = _mean(worst["p_bear"])
        summary["p_bear_mean_all"] = _mean(out["p_bear"])
    if "w_exec" in out.columns:
        summary["abs_w_exec_mean_top"] = _mean(worst["w_exec"].abs())
        summary["abs_w_exec_mean_all"] = _mean(out["w_exec"].abs())
    if "w_target" in out.columns:
        summary["abs_w_target_mean_top"] = _mean(worst["w_target"].abs())
        summary["abs_w_target_mean_all"] = _mean(out["w_target"].abs())
    if "dw" in out.columns:
        summary["dw_mean_top"] = _mean(worst["dw"])
        summary["dw_mean_all"] = _mean(out["dw"])
    if "turnover" in out.columns:
        summary["turnover_mean_top"] = _mean(worst["turnover"])
        summary["turnover_mean_all"] = _mean(out["turnover"])

    for k, v in summary.items():
        print(f"{k:24s}: {v}")

if __name__ == "__main__":
    main()
