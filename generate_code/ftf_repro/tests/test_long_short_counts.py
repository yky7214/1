import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/base_fast_nocost_dw_timeout30/reports/oos_daily.parquet"

def _load():
    assert REPORT_PATH.exists(), f"missing parquet: {REPORT_PATH}"
    df = pd.read_parquet(REPORT_PATH).sort_index()

    # 必須列チェック
    for c in ["w_exec", "p_bear"]:
        assert c in df.columns, f"missing column: {c}"

    # NaN対策
    df["w_exec"] = df["w_exec"].fillna(0.0)
    df["p_bear"] = df["p_bear"].fillna(0.5)

    return df


def test_pbear_derisk_relation_prints():
    """
    p_bear が上がると w_exec が落ちる（=de-risk）傾向があるかを確認する。
    目的：DD低下に効くかの sanity check。
    """
    df = _load()

    # 閾値は論文ニュアンスに合わせて 0.50 を基本に
    bear_mask = df["p_bear"] > 0.50
    bull_mask = df["p_bear"] < 0.50

    # 集計（強いほど良い）
    abs_w = df["w_exec"].abs()

    bear_mean_abs = abs_w[bear_mask].mean()
    bull_mean_abs = abs_w[bull_mask].mean()

    bear_flat_rate = (df.loc[bear_mask, "w_exec"] == 0).mean()
    bull_flat_rate = (df.loc[bull_mask, "w_exec"] == 0).mean()

    # p_bear の変化と w_exec の変化（翌日反映っぽい挙動をざっくり確認）
    dp = df["p_bear"].diff()
    dw = df["w_exec"].diff()

    # p_bear が上がった日（dp>0）に、w_exec が減った割合
    # （同日反応 or 翌日反応が混ざるので、まずは同日で雑に見て、弱ければshiftで調整）
    up_mask = dp > 0
    derisk_hit = (dw[up_mask] < 0).mean() if up_mask.any() else np.nan

    # 相関（単純な符号の確認用）
    corr = df["p_bear"].corr(abs_w)

    print("\n=== P_BEAR DE-RISK CHECK ===")
    print(f"Parquet: {REPORT_PATH}")
    print(f"bear days (p_bear>0.50): {int(bear_mask.sum())}")
    print(f"bull days (p_bear<0.50): {int(bull_mask.sum())}")
    print(f"mean |w_exec| in BEAR : {bear_mean_abs:.6f}")
    print(f"mean |w_exec| in BULL : {bull_mean_abs:.6f}")
    print(f"flat rate in BEAR     : {bear_flat_rate:.3%}")
    print(f"flat rate in BULL     : {bull_flat_rate:.3%}")
    print(f"corr(p_bear, |w_exec|): {corr:.4f}")
    print(f"P(bear up => w_exec down) (same-day): {derisk_hit:.3%}")

    # --- “強すぎない” sanity assertions ---
    # BEAR の方がリスク落ちてるはず（同じなら p_bear が効いてない疑い）
    assert bear_mean_abs < bull_mean_abs, "p_bear高いのに |w_exec| が小さくなっていない"

    # BEAR の方がフラット率が高いのが自然
    assert bear_flat_rate >= bull_flat_rate, "p_bear高いのにフラット率が上がっていない"
