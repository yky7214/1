import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Utilities
# ============================================================

def _safe_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Missing column: {col} (available={list(df.columns)[:30]}...)")
    s = df[col].copy()
    if isinstance(default, float):
        s = s.astype(float).fillna(default)
    return s


def _sign(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    # -1 / 0 / +1
    return np.where(x > eps, 1, np.where(x < -eps, -1, 0))


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return lengths of consecutive True runs in a boolean array."""
    if mask.size == 0:
        return np.array([], dtype=int)
    # indices where value changes
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    return (ends - starts).astype(int)


# ============================================================
# Main
# ============================================================

def test_trade_counts():
    ROOT = Path(__file__).resolve().parents[1]

    # ★ここを好きな parquet に変える
    PARQUET = (
        ROOT / "reports" / "base_fast_nocost_dw_timeout30" / "reports" / "oos_daily.parquet"
    )

    df = pd.read_parquet(PARQUET).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("parquet index must be DatetimeIndex")

    # ---- columns ----
    # 基本：w_exec と net_ret_ftf があればOK
    w = _safe_series(df, "w_exec", default=0.0)

    # 追加：w_target でも同じ集計（Timeoutが効いてるか判定する本丸）
    wt = None
    if "w_target" in df.columns:
        wt = _safe_series(df, "w_target", default=0.0)


    # turnover が無い場合は自分で作る
    if "turnover" in df.columns:
        turnover = _safe_series(df, "turnover", default=0.0).abs()
    else:
        turnover = (w - w.shift(1)).abs().fillna(0.0)

    # ---- derived ----
    eps = 1e-12
    w_prev = w.shift(1).fillna(0.0)

    # 1) 執行回数（実際にポジションが変わった日数）
    exec_days = int((turnover > eps).sum())

    # 2) トレード回数（entry/exit を数える）
    #    - entry: w が 0 → 非ゼロ
    #    - exit : w が 非ゼロ → 0
    active = (w.abs() > eps)
    active_prev = (w_prev.abs() > eps)

    # ---- derived (w_target) ----
    if wt is not None:
        active_t = (wt.abs() > eps)
        runs_t = _run_lengths(active_t.to_numpy())
        avg_hold_t = float(runs_t.mean()) if runs_t.size > 0 else 0.0
        med_hold_t = float(np.median(runs_t)) if runs_t.size > 0 else 0.0
        max_hold_t = int(runs_t.max()) if runs_t.size > 0 else 0


    entries = int((~active_prev & active).sum())
    exits   = int((active_prev & ~active).sum())

    # 3) 反転回数（ロング↔ショート）
    s = _sign(w, eps=eps)
    s_prev = pd.Series(_sign(w_prev, eps=eps), index=w.index)
    flips = int(((s_prev != 0) & (s != 0) & (s_prev != s)).sum())

    # 4) 有効日数（ポジション保有してた日数）
    active_days = int(active.sum())

    # 5) 平均保有期間（連続保有ブロックの長さ）
    runs = _run_lengths(active.to_numpy())
    avg_hold = float(runs.mean()) if runs.size > 0 else 0.0
    med_hold = float(np.median(runs)) if runs.size > 0 else 0.0
    max_hold = int(runs.max()) if runs.size > 0 else 0

    # 6) 回転（turnover）
    total_turnover = float(turnover.sum())
    avg_turnover = float(turnover.mean())
    p95_turnover = float(turnover.quantile(0.95))

    # 7) 追加：最大 |w|
    max_abs_w = float(w.abs().max())

    # ---- output ----
    print("\n=== TRADE / EXECUTION COUNTS ===")
    print(f"Parquet          : {PARQUET}")
    print(f"Date range       : {df.index.min().date()} -> {df.index.max().date()}")
    print("")
    print(f"Active days      : {active_days}")
    print(f"Entries          : {entries}")
    print(f"Exits            : {exits}")
    print(f"Flips (L<->S)    : {flips}")
    print(f"Exec days (dw!=0): {exec_days}")
    print("")
    print(f"Hold length avg  : {avg_hold:.2f} days")
    print(f"Hold length med  : {med_hold:.2f} days")
    print(f"Hold length max  : {max_hold} days")
    print("")
    print(f"Turnover total   : {total_turnover:.3f}")
    print(f"Turnover avg/day : {avg_turnover:.4f}")
    print(f"Turnover p95/day : {p95_turnover:.4f}")
    print("")
    print(f"Max |w_exec|     : {max_abs_w:.3f}")
    if wt is not None:
        print("")
        print("=== TARGET-WEIGHT HOLDS (w_target) ===")
        print(f"Active days      : {int((wt.abs()>eps).sum())}")
        print(f"Hold length avg  : {avg_hold_t:.2f} days")
        print(f"Hold length med  : {med_hold_t:.2f} days")
        print(f"Hold length max  : {max_hold_t} days")
        print(f"Max |w_target|   : {float(wt.abs().max()):.3f}")


    # 最低限の sanity
    assert len(df) > 100
    assert max_abs_w >= 0.0


if __name__ == "__main__":
    test_trade_counts()
