import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# Utilities
# ============================================================

def compute_equity(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod()


def compute_drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def annualized_sharpe(ret: pd.Series, periods: int = 252) -> float:
    r = ret.fillna(0.0).astype(float)
    return float((r.mean() / (r.std(ddof=0) + 1e-12)) * np.sqrt(periods))


def pick_buyhold_return(df: pd.DataFrame) -> pd.Series:
    if "r" in df.columns:
        return df["r"].astype(float).fillna(0.0)
    if "ret" in df.columns:
        return df["ret"].astype(float).fillna(0.0)
    if "close" in df.columns:
        return df["close"].astype(float).pct_change().fillna(0.0)
    raise ValueError("Buy&Hold 用の r / ret / close が parquet にありません")


def rebuild_net_ret(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """
    parquet の符号規約に合わせて net_ret を再構築する（自動判定）。

    候補式:
      F1: gross + fill - cost_lin - cost_imp   (コストが正のとき)
      F2: gross + fill + cost_lin + cost_imp   (コストが負のとき)
      F3: gross - cost_lin - cost_imp          (fill が無い/既に含まれてるとき)
      F4: gross + cost_lin + cost_imp
      F5: gross + fill                         (cost が無い/既に含まれてるとき)

    まず列が揃う候補だけ作り、net_ret が存在するなら「最も一致する式」を選ぶ。
    """
    notes = []

    have = set(df.columns)

    def s(col):
        return df[col].astype(float).fillna(0.0)

    candidates = []

    if {"gross_ret", "fill_error_ret", "cost_lin", "cost_imp"} <= have:
        gross = s("gross_ret")
        fill  = s("fill_error_ret")
        c_lin = s("cost_lin")
        c_imp = s("cost_imp")

        # 代表値で「コスト列が負で入ってるか」を推定（多くの実装は cost はマイナスで格納される）
        cost_med = float((c_lin + c_imp).median())
        cost_signed_hint = "negative" if cost_med < 0 else "positive"

        candidates.append(("F1", gross + fill - c_lin - c_imp,
                           f"F1: gross+fill - cost_lin - cost_imp (assumes costs positive)"))
        candidates.append(("F2", gross + fill + c_lin + c_imp,
                           f"F2: gross+fill + cost_lin + cost_imp (assumes costs negative)"))
        candidates.append(("F3", gross - c_lin - c_imp,
                           f"F3: gross - cost_lin - cost_imp (assumes fill already included / not used)"))
        candidates.append(("F4", gross + c_lin + c_imp,
                           f"F4: gross + cost_lin + cost_imp (assumes costs negative, no fill)"))
        candidates.append(("F5", gross + fill,
                           f"F5: gross + fill (assumes costs already included / not used)"))

        notes.append(f"cost_sign_hint_by_median: {cost_signed_hint} (median(cost_lin+cost_imp)={cost_med:.6g})")

    elif {"gross_ret", "cost_lin", "cost_imp"} <= have:
        gross = s("gross_ret")
        c_lin = s("cost_lin")
        c_imp = s("cost_imp")
        candidates.append(("F3", gross - c_lin - c_imp, "F3: gross - cost_lin - cost_imp"))
        candidates.append(("F4", gross + c_lin + c_imp, "F4: gross + cost_lin + cost_imp"))

    elif {"w_exec"} <= have:
        # 最終手段：これは厳密な net ではない
        r_bh = pick_buyhold_return(df)
        gross_like = df["w_exec"].astype(float).shift(1).fillna(0.0) * r_bh
        notes.append("fallback: w_exec.shift(1) * r (NOT true net_ret; costs/fill missing)")
        return gross_like, notes

    else:
        raise ValueError("net_ret 再構築に必要な列が足りません。")

    # net_ret があれば「一番一致する式」を選ぶ
    if "net_ret" in have:
        net_raw = s("net_ret")
        # すべての候補について abs diff mean を比較（極端値に引っ張られにくい）
        scores = []
        for key, series, desc in candidates:
            diff = (net_raw - series).abs()
            scores.append((float(diff.mean()), float(diff.quantile(0.95)), key, series, desc))

        scores.sort(key=lambda x: (x[0], x[1]))
        best_mean, best_q95, best_key, best_series, best_desc = scores[0]
        notes.append(f"picked_best_formula: {best_key} (mean_abs_diff={best_mean:.6g}, q95_abs_diff={best_q95:.6g})")
        notes.append(best_desc)
        return best_series, notes

    # net_ret が無いなら、ヒントに沿って妥当そうなものを返す
    # （cost_med<0 なら F2, >=0 なら F1 を優先）
    if any(k == "F2" for k, *_ in candidates) and "cost_sign_hint_by_median: negative" in " ".join(notes):
        for k, series, desc in candidates:
            if k == "F2":
                notes.append("net_ret missing -> choose F2 by cost sign hint")
                notes.append(desc)
                return series, notes
    for k, series, desc in candidates:
        if k == "F1":
            notes.append("net_ret missing -> choose F1 by default")
            notes.append(desc)
            return series, notes

    # 保険
    k, series, desc = candidates[0]
    notes.append(f"net_ret missing -> choose {k} fallback")
    notes.append(desc)
    return series, notes


def validate_net_ret(df: pd.DataFrame, net_col: str = "net_ret") -> dict:
    out = {"has_net_col": net_col in df.columns}

    if net_col not in df.columns:
        out["status"] = "missing"
        return out

    net_raw = df[net_col].astype(float).fillna(0.0)
    net_rebuilt, notes = rebuild_net_ret(df)

    diff = (net_raw - net_rebuilt)

    # どの程度ヤバいスケールかも出す（%/bps 混入の検知に役立つ）
    out.update({
        "status": "ok",
        "notes": notes,
        "abs_diff_max": float(diff.abs().max()),
        "abs_diff_mean": float(diff.abs().mean()),
        "raw_abs_p95": float(net_raw.abs().quantile(0.95)),
        "rebuilt_abs_p95": float(net_rebuilt.abs().quantile(0.95)),
    })
    return out



# ============================================================
# Plot
# ============================================================

def plot_equity_and_dd(
    equity_strat: pd.Series,
    equity_bh: pd.Series,
    dd_strat: pd.Series,
    dd_bh: pd.Series,
    title: str,
    use_log: bool = False,
    show_dd: bool = True,
):
    if show_dd:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax2 = None

    ax1.plot(equity_strat, label="Strategy (net_ret)", linewidth=2)
    ax1.plot(equity_bh, label="Buy & Hold", linewidth=2, linestyle="--")

    if use_log:
        ax1.set_yscale("log")

    ax1.set_title(title)
    ax1.set_ylabel("Equity")

    if use_log:
        ax1.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
        ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(n=3))

    ax1.tick_params(axis="y", which="major", length=6)
    ax1.tick_params(axis="y", which="minor", length=3)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend()

    if show_dd and ax2 is not None:
        ax2.fill_between(dd_strat.index, dd_strat, 0.0, color="red", alpha=0.3, label="Strategy DD")
        ax2.fill_between(dd_bh.index, dd_bh, 0.0, color="gray", alpha=0.3, label="Buy&Hold DD")

        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(n=2))
        ax2.tick_params(axis="y", which="major", length=6)
        ax2.tick_params(axis="y", which="minor", length=3)
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)
        ax2.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=str, required=True, help="oos_daily.parquet へのパス")
    p.add_argument("--net_col", type=str, default="net_ret", help="strategyのnet return列名")
    p.add_argument("--use_log", action="store_true")
    p.add_argument("--no_dd", action="store_true")
    p.add_argument("--diff_tol", type=float, default=1e-10, help="net_ret検証の許容差（abs max）")
    args = p.parse_args()

    parquet = Path(args.parquet)
    if not parquet.exists():
        raise FileNotFoundError(parquet)

    df = pd.read_parquet(parquet).sort_index()

    # 1) Buy&Hold
    ret_bh = pick_buyhold_return(df)
    equity_bh = compute_equity(ret_bh)
    dd_bh = compute_drawdown(equity_bh)

    # 2) Strategy net_ret（検証して、必要なら再構築）
    v = validate_net_ret(df, net_col=args.net_col)

    if v["status"] == "missing":
        print(f"[WARN] {args.net_col} が無いので再構築します。")
        ret_strat, notes = rebuild_net_ret(df)
        for n in notes:
            print("  -", n)
    else:
        # net_ret はある。差分チェック
        abs_max = v["abs_diff_max"]
        abs_mean = v["abs_diff_mean"]
        print(f"[CHECK] {args.net_col} validation:")
        for n in v.get("notes", []):
            print("  -", n)
        print(f"  abs diff max : {abs_max:.6g}")
        print(f"  abs diff mean: {abs_mean:.6g}")

        if abs_max > args.diff_tol:
            print(f"[WARN] 差分が許容 {args.diff_tol} を超えました。再構築版(net_ret_rebuilt)を採用します。")
            ret_strat, _ = rebuild_net_ret(df)
        else:
            ret_strat = df[args.net_col].astype(float).fillna(0.0)

    equity_strat = compute_equity(ret_strat)
    dd_strat = compute_drawdown(equity_strat)

    # 3) Plot
    plot_equity_and_dd(
        equity_strat=equity_strat,
        equity_bh=equity_bh,
        dd_strat=dd_strat,
        dd_bh=dd_bh,
        title="Equity Curve: Strategy vs Buy & Hold",
        use_log=args.use_log,
        show_dd=(not args.no_dd),
    )

    # 4) Stats
    sharpe_strat = annualized_sharpe(ret_strat)
    sharpe_bh = annualized_sharpe(ret_bh)

    print("\n=== SUMMARY ===")
    print(f"Parquet           : {parquet}")
    print(f"Strategy col used : {args.net_col} (or rebuilt)")
    print(f"Strategy Sharpe   : {sharpe_strat:.3f}")
    print(f"Strategy MaxDD    : {dd_strat.min():.2%}")
    print(f"Strategy FinalEq  : {equity_strat.iloc[-1]:.6f}")
    print("")
    print(f"Buy&Hold Sharpe   : {sharpe_bh:.3f}")
    print(f"Buy&Hold MaxDD    : {dd_bh.min():.2%}")
    print(f"Buy&Hold FinalEq  : {equity_bh.iloc[-1]:.6f}")


if __name__ == "__main__":
    main()
