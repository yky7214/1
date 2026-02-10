# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Utilities
# ============================================================

def compute_drawdown(ret: pd.Series) -> pd.Series:
    equity = (1.0 + ret).cumprod()
    dd = equity / equity.cummax() - 1.0
    return dd


def extract_dd_window(df: pd.DataFrame, dd: pd.Series, center_date, window: int = 20):
    if center_date not in dd.index:
        raise KeyError(f"center_date not in index: {center_date}")

    idx = dd.index.get_loc(center_date)
    lo = max(0, idx - window)
    hi = min(len(dd) - 1, idx + window)

    return df.iloc[lo:hi + 1], dd.iloc[lo:hi + 1]


def sharpe_ann(x: pd.Series) -> float:
    sd = x.std(ddof=0)
    if sd < 1e-12:
        return 0.0
    return (x.mean() / sd) * np.sqrt(252)


# ============================================================
# Main
# ============================================================

def dd_rebalance_analysis():

    # ------------------------------------------------------------
    # PATH
    #   ここだけ変えれば、baseline / cap どっちでも評価できる
    # ------------------------------------------------------------

    ROOT = Path(__file__).resolve().parents[1]

    # ★ここを cap を回した out_dir に変える（上書き防止で別dir推奨）
    # PARQUET = ROOT / "reports" / "base_fast_nocost_dw_baseline" / "reports" / "oos_daily.parquet"
    PARQUET = ROOT / "reports" / "base_fast_nocost_dw_timeout30" / "reports" / "oos_daily.parquet"

    OUTDIR = ROOT / "tests" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    df = pd.read_parquet(PARQUET).sort_index()

    ret_col = "net_ret_ftf"
    pos_col = "w_exec"

    if ret_col not in df.columns:
        raise KeyError(f"Missing column: {ret_col}  (available={list(df.columns)[:30]}...)")

    if pos_col not in df.columns:
        raise KeyError(f"Missing column: {pos_col}  (available={list(df.columns)[:30]}...)")

    ret = df[ret_col].fillna(0.0)
    w = df[pos_col].fillna(0.0)

    # ------------------------------------------------------------
    # SUMMARY STATS (RAW = このparquetの結果そのもの)
    # ------------------------------------------------------------

    dd = compute_drawdown(ret)

    worst_day_all = dd.idxmin()
    worst_dd_all = dd.min()

    print("\n=== SUMMARY (THIS PARQUET) ===")
    print(f"Parquet : {PARQUET}")
    print(f"Sharpe  : {sharpe_ann(ret):.3f}")
    print(f"MaxDD   : {worst_dd_all:.2%}")
    print(f"WorstDD : {worst_day_all.date()}")

    # ------------------------------------------------------------
    # CHECK ALIGN（確認用：ret[t] が w[t-1] か）
    # ------------------------------------------------------------

    try:
        print("\n=== CHECK ALIGN (2020-11-07 to 2020-11-10) ===")
        print(
            df[[ret_col, pos_col]]
            .assign(w_lag=df[pos_col].shift(1))
            .loc["2020-11-07":"2020-11-10"]
            .to_string()
        )
    except Exception:
        pass

    # ------------------------------------------------------------
    # ALL-TIME WORST DD WINDOW を CSV に保存
    # ------------------------------------------------------------

    base_cols = [ret_col, pos_col]

    sub_df_all, sub_dd_all = extract_dd_window(
        df[base_cols],
        dd,
        worst_day_all,
        window=40
    )

    sub_df_all = sub_df_all.copy()
    sub_df_all["delta_w"] = w.diff().loc[sub_df_all.index]
    sub_df_all["equity"] = (1.0 + ret.loc[sub_df_all.index]).cumprod()
    sub_df_all["drawdown"] = sub_dd_all
    sub_df_all["delta_w_exec"] = df["w_exec"].diff().loc[sub_df_all.index]
    sub_df_all["delta_w_tgt"]  = df["w_target"].diff().loc[sub_df_all.index]

    path_all = OUTDIR / "dd_alltime_rebalance_window.csv"
    sub_df_all.to_csv(path_all)
    print(f"\nSaved: {path_all}")

    # ------------------------------------------------------------
    # 月ごとの WORST DD day を特定 → window 出力 → DD悪化リバランス抽出
    # ------------------------------------------------------------

    target_months = [
        ("2020-03", "dd_2020_03"),
        ("2022-06", "dd_2022_06"),
        ("2022-11", "dd_2022_11"),
        ("2021-01", "dd_2021_01")
    ]

    for ym, tag in target_months:

        mask_m = dd.index.strftime("%Y-%m") == ym

        if not mask_m.any():
            print(f"\n=== {ym} ===")
            print("No data for this month")
            continue

        worst_day = dd[mask_m].idxmin()
        worst_dd = dd.loc[worst_day]

        print(f"\n=== {ym} WORST DD ===")
        print(f"Date : {worst_day.date()}")
        print(f"DD   : {worst_dd:.2%}")

        sub_df, sub_dd = extract_dd_window(
            df[base_cols],
            dd,
            worst_day,
            window=20
        )

        sub_df = sub_df.copy()
        sub_df["delta_w"] = w.diff().loc[sub_df.index]
        sub_df["equity"] = (1.0 + ret.loc[sub_df.index]).cumprod()

        out = sub_df.copy()
        out["drawdown"] = sub_dd

        path = OUTDIR / f"{tag}_rebalance_window.csv"
        out.to_csv(path)
        print(f"Saved: {path}")

        worsening = out[out["drawdown"].diff() < 0].copy()
        print(f"worsening count: {len(worsening)}")

        if len(worsening) == 0:
            print("Top DD-worsening rebalances: (none)")
            continue

        top_moves = worsening.sort_values(
            "delta_w",
            key=lambda x: x.abs(),
            ascending=False
        )

        print("\nTop DD-worsening rebalances:")
        print(
            top_moves[
                ["delta_w", pos_col, ret_col, "drawdown"]
            ].head(5).to_string()
        )

    assert dd.min() < 0
    print("\nDONE")


if __name__ == "__main__":
    dd_rebalance_analysis()
