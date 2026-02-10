import numpy as np
import pandas as pd
from pathlib import Path


def _build_trades_from_daily(df: pd.DataFrame, ret_col: str, pos_col: str) -> pd.DataFrame:
    """
    日次データから「0→非0をentry」「非0→0をexit」として擬似トレードを作る（round-trip数）。
    ret_col: 戦略日次リターン（例: net_ret_ftf）
    pos_col: ポジション（例: w_exec）
    """
    ret = df[ret_col].fillna(0.0).copy().sort_index()
    pos = df[pos_col].fillna(0.0).reindex(ret.index).copy()

    in_pos = (pos != 0).astype(int)
    entry = (in_pos.diff() == 1)
    exit_ = (in_pos.diff() == -1)

    entries = list(ret.index[entry.fillna(False)])
    exits = list(ret.index[exit_.fillna(False)])

    # 最後が未クローズなら最終日でクローズ扱い
    if len(exits) < len(entries):
        exits.append(ret.index[-1])

    trades = []
    for tid, (en, ex) in enumerate(zip(entries, exits), start=1):
        seg_ret = ret.loc[en:ex]
        seg_pos = pos.loc[en:ex]

        trade_pnl = (1.0 + seg_ret).prod() - 1.0  # 単純複利

        trades.append({
            "trade_id": tid,
            "entry_date": en,
            "exit_date": ex,
            "days_held": int(len(seg_ret)),
            "entry_pos": float(seg_pos.iloc[0]),
            "avg_abs_pos": float(seg_pos.abs().mean()),
            "max_abs_pos": float(seg_pos.abs().max()),
            "trade_pnl": float(trade_pnl),
            "sum_daily_ret": float(seg_ret.sum()),
            "max_daily_ret": float(seg_ret.max()),
            "min_daily_ret": float(seg_ret.min()),
        })

    return pd.DataFrame(trades)


def _build_rebalance_log(df: pd.DataFrame, pos_col: str, eps: float = 1e-6) -> pd.DataFrame:
    """
    日次の w_exec 変化から「リバランス（売買が発生した日）」を抽出する。
    eps: 数値誤差除け（これ以下の変化は無視）
    """
    w = df[pos_col].fillna(0.0).copy().sort_index()
    dw = w.diff()

    mask = dw.abs() > eps

    rebalance = pd.DataFrame({
        "date": w.index[mask],
        "w_prev": w.shift(1)[mask],
        "w_new": w[mask],
        "delta_w_abs": dw[mask].abs(),
        "delta_w_signed": dw[mask],
    })

    return rebalance.reset_index(drop=True)


def test_export_trade_and_rebalance_logs():
    # ===== パス設定 =====
    ROOT = Path(__file__).resolve().parents[1]
    PARQUET = ROOT / "reports" / "base_fast_kelly020_exit15_ema090" / "reports" / "oos_daily.parquet"


    OUTDIR = ROOT / "tests" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ===== 読み込み =====
    df = pd.read_parquet(PARQUET).sort_index()

    # ===== 列名（このparquetに合わせる）=====
    ret_col = "net_ret_ftf"   # Strategy (FTF) の日次リターン
    pos_col = "w_exec"        # 実行ウェイト（ポジション）

    assert ret_col in df.columns, f"missing {ret_col}"
    assert pos_col in df.columns, f"missing {pos_col}"

    # ===== 1) round-trip の擬似取引履歴 =====
    trades_df = _build_trades_from_daily(df, ret_col=ret_col, pos_col=pos_col)
    trades_sorted = trades_df.sort_values("trade_pnl", ascending=False)

    trades_path = OUTDIR / "strategy_ftf_trades.csv"
    sorted_path = OUTDIR / "strategy_ftf_trades_sorted_by_pnl.csv"

    trades_df.to_csv(trades_path, index=False)
    trades_sorted.to_csv(sorted_path, index=False)

    # ===== 2) リバランス履歴（売買が発生した日） =====
    rebalance_df = _build_rebalance_log(df, pos_col=pos_col, eps=1e-6)
    rebalance_path = OUTDIR / "strategy_ftf_rebalances.csv"
    rebalance_df.to_csv(rebalance_path, index=False)

    # ===== 3) サマリー表示（pytest -s で見える）=====
    total_days = len(df)
    rebalance_days = len(rebalance_df)
    rebalance_ratio = rebalance_days / total_days if total_days > 0 else np.nan

    # turnover 列があれば追加で表示
    has_turnover = "turnover" in df.columns
    total_turnover = float(df["turnover"].sum()) if has_turnover else None
    avg_turnover = float(df["turnover"].mean()) if has_turnover else None

    print("\n=== SAVED FILES ===")
    print(f"- {trades_path}")
    print(f"- {sorted_path}")
    print(f"- {rebalance_path}")

    print("\n=== REBALANCE STATS ===")
    print(f"Total days      : {total_days}")
    print(f"Rebalance days  : {rebalance_days}")
    print(f"Rebalance ratio : {rebalance_ratio:.3f}")

    if has_turnover:
        print("\n=== TURNOVER STATS (if provided by parquet) ===")
        print(f"Total turnover      : {total_turnover:.6f}")
        print(f"Avg daily turnover  : {avg_turnover:.6f}")

    print("\nTop 10 largest rebalances:")
    if rebalance_days > 0:
        print(
            rebalance_df.sort_values("delta_w_abs", ascending=False)
            .head(10)
            .to_string(index=False)
        )
    else:
        print("No rebalances detected.")

    print("\n=== ROUND-TRIP (0->non0->0) TRADES ===")
    print(f"Trades (round-trip count): {len(trades_df)}")
    if len(trades_df) > 0:
        print(trades_sorted.head(10).to_string(index=False))

    # ===== 4) 最低限のチェック =====
    assert rebalance_days > 0, "No rebalances detected (pos might not be changing?)"
