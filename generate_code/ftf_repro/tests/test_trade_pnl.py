import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Helpers
# ============================================================

def _require_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Missing column: {col} (available={list(df.columns)[:30]}...)")
    return df[col]


def _build_trade_segments_from_active(active: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    active(bool) から (entry_ts, exit_ts) のリストを作る。
    - entry: 0 -> 1
    - exit : 1 -> 0 の「直前の1の日」
    ここは out-of-bounds やズレが出やすいので、敢えてループで堅牢に書く。
    """
    a = active.astype(bool).to_numpy()
    idx = active.index

    segments = []
    in_trade = False
    entry = None

    for i in range(len(a)):
        if not in_trade and a[i]:
            in_trade = True
            entry = idx[i]

        # トレード中で、次が False になる（または最後）
        if in_trade:
            is_last = (i == len(a) - 1)
            next_inactive = (not is_last) and (not a[i + 1])

            if is_last or next_inactive:
                exit_ = idx[i]  # 最後にactiveだった日
                segments.append((entry, exit_))
                in_trade = False
                entry = None

    return segments


def _trade_metrics(ret: pd.Series) -> dict:
    """
    ret: trade期間の1日リターン（net_ret_ftfなど）
    戻り:
      trade_return: 期間合成リターン
      mae: Max Adverse Excursion（トレード内の最大ドローダウン, マイナス値）
      mfe: Max Favorable Excursion（トレード内の最大含み益, プラス値）
    """
    r = ret.fillna(0.0).astype(float)

    # ret <= -1 があると cumprod が壊れるのでクリップ（安全策）
    r = r.clip(lower=-0.999999)

    eq = (1.0 + r).cumprod()

    trade_return = float(eq.iloc[-1] - 1.0)
    mfe = float(eq.max() - 1.0)

    dd = eq / eq.cummax() - 1.0
    mae = float(dd.min())

    return {"trade_return": trade_return, "mae": mae, "mfe": mfe}


# ============================================================
# Main
# ============================================================

def test_trade_pnl():
    ROOT = Path(__file__).resolve().parents[1]

    # ★ここだけ好きなparquetに変更
    PARQUET = ROOT / "reports" / "base_fast_nocost_dw" / "reports" / "oos_daily.parquet"

    OUTDIR = ROOT / "tests" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("parquet index must be DatetimeIndex")

    ret_col = "net_ret_ftf"
    w_col = "w_exec"

    ret = _require_col(df, ret_col).astype(float).fillna(0.0)
    w = _require_col(df, w_col).astype(float).fillna(0.0)

    eps = 1e-12
    active = (w.abs() > eps)

    # セグメント作成（堅牢版）
    segments = _build_trade_segments_from_active(active)

    rows = []
    skipped = 0

    for i, (entry, exit_) in enumerate(segments, start=1):
        # inclusive slice
        slc = df.loc[entry:exit_]

        # 空ならスキップ（今回のエラー対策）
        if slc.shape[0] == 0:
            skipped += 1
            continue

        ret_t = ret.loc[slc.index]
        w_t = w.loc[slc.index]

        # side は entry日の w の符号で決める（entry日は active のはずだが安全に）
        w0 = float(w_t.iloc[0]) if len(w_t) > 0 else 0.0
        if w0 > eps:
            side = "LONG"
        elif w0 < -eps:
            side = "SHORT"
        else:
            side = "NA"

        hold_days = int(len(slc))
        m = _trade_metrics(ret_t)

        rows.append({
            "trade_id": i,
            "entry_date": entry.date(),
            "exit_date": exit_.date(),
            "hold_days": hold_days,
            "side": side,
            "entry_w": w0,
            "avg_abs_w": float(w_t.abs().mean()),
            "max_abs_w": float(w_t.abs().max()),
            "trade_return": m["trade_return"],
            "mfe": m["mfe"],
            "mae": m["mae"],
        })

    trades = pd.DataFrame(rows)

    if trades.empty:
        print("\nNo trades detected. Check w_exec / eps threshold.")
        print(f"Parquet: {PARQUET}")
        return

    # 保存（リターン降順）
    trades_sorted = trades.sort_values(["trade_return"], ascending=False).reset_index(drop=True)

    out_path = OUTDIR / "trade_pnl_table.csv"
    trades_sorted.to_csv(out_path, index=False)

    # サマリ
    n = len(trades)
    win_rate = float((trades["trade_return"] > 0).mean())
    avg_ret = float(trades["trade_return"].mean())
    med_ret = float(trades["trade_return"].median())
    avg_days = float(trades["hold_days"].mean())
    med_days = float(trades["hold_days"].median())
    max_days = int(trades["hold_days"].max())

    max_row = trades.loc[trades["hold_days"].idxmax()]
    tid_max = int(max_row["trade_id"])
    max_period = f'{max_row["entry_date"]} -> {max_row["exit_date"]}'

    print("\n=== TRADE PNL TABLE ===")
    print(f"Parquet     : {PARQUET}")
    print(f"Trades      : {n}  (skipped empty slices: {skipped})")
    print(f"Win rate    : {win_rate:.2%}")
    print(f"Avg ret     : {avg_ret:.2%}")
    print(f"Med ret     : {med_ret:.2%}")
    print(f"Avg hold    : {avg_days:.2f} days")
    print(f"Med hold    : {med_days:.2f} days")
    print(f"Max hold    : {max_days} days")

    print(f"\nMax-hold trade_id : {tid_max}  period: {max_period}")
    print("Max-hold row:")

    # ============================================================
    # Export: Max-hold trade daily time series
    # ============================================================

    entry_ts = pd.Timestamp(str(max_row["entry_date"]))
    exit_ts  = pd.Timestamp(str(max_row["exit_date"]))

    slc = df.loc[entry_ts:exit_ts].copy()

    # columns (存在するものだけ使う)
    cols = []
    for c in ["close", "r", "w_exec", "net_ret_ftf", "net_ret", "alpha_ret", "turnover"]:
        if c in slc.columns:
            cols.append(c)

    ts = slc[cols].copy()

    # trade-local equity / dd
    ret_for_eq = ts["net_ret_ftf"] if "net_ret_ftf" in ts.columns else ret.loc[ts.index]
    ret_for_eq = ret_for_eq.fillna(0.0).clip(lower=-0.999999)

    ts["trade_equity"] = (1.0 + ret_for_eq).cumprod()
    ts["trade_dd"]     = ts["trade_equity"] / ts["trade_equity"].cummax() - 1.0

    # worst day / best day inside trade
    worst_day = ts["trade_dd"].idxmin()
    best_day  = ts["trade_equity"].idxmax()

    print("\n=== MAX-HOLD TRADE DETAIL ===")
    print(f"trade_id    : {tid_max}")
    print(f"period      : {entry_ts.date()} -> {exit_ts.date()}")
    print(f"worst day   : {worst_day.date()}  trade_dd={ts.loc[worst_day, 'trade_dd']:.2%}")
    print(f"best day    : {best_day.date()}   trade_equity={ts.loc[best_day, 'trade_equity']:.4f}")

    out_ts_path = OUTDIR / f"trade_{tid_max:03d}_timeseries.csv"
    ts.to_csv(out_ts_path)

    print(f"Saved: {out_ts_path}")    


    print(max_row.to_string())

    print(f"\nSaved: {out_path}")

    # sanity
    assert len(df) > 100
    assert (active.sum() >= 0)


if __name__ == "__main__":
    test_trade_pnl()
