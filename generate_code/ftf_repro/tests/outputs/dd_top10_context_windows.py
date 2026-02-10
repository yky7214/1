# tests/outputs/dd_top10_context_windows.py
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def compute_dd_from_returns(r: pd.Series) -> pd.DataFrame:
    """
    r: daily returns (net_ret_ftf)
    returns:
      equity, dd, dd_change
    """
    r = r.fillna(0.0).astype(float)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    dd_change = dd.diff()  # negative means DD worsened
    return pd.DataFrame({"equity": equity, "dd": dd, "dd_change": dd_change})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to oos_daily.parquet",
    )
    ap.add_argument(
        "--retcol",
        type=str,
        default="net_ret_ftf",
        help="Return column name",
    )
    ap.add_argument(
        "--topn",
        type=int,
        default=10,
        help="How many worst DD-worsening days to export",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=10,
        help="Days before/after each event day to export (calendar trading days in index)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="tests/outputs/dd_top10_context",
        help="Output directory",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="dd_change",
        choices=["dd_change", "dd_level"],
        help=(
            "dd_change = pick days with most negative dd_change (DD worsened most that day)\n"
            "dd_level  = pick days where dd is lowest (max drawdown points)"
        ),
    )
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    assert parquet_path.exists(), f"Parquet not found: {parquet_path}"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path).sort_index()
    assert args.retcol in df.columns, f"Return column not found: {args.retcol}"

    # Compute dd metrics
    dd_df = compute_dd_from_returns(df[args.retcol])
    df2 = df.join(dd_df)

    # Pick TOP N
    if args.mode == "dd_change":
        # Most negative dd_change (largest worsening that day)
        ranked = df2["dd_change"].sort_values(ascending=True)
        top_days = ranked.dropna().head(args.topn).index
        title = "worst_dd_change_days"
    else:
        # Lowest dd level (deepest drawdown points)
        ranked = df2["dd"].sort_values(ascending=True)
        top_days = ranked.dropna().head(args.topn).index
        title = "worst_dd_level_days"

    # Summary table
    summary_cols = [
        "dd_change", "dd", "equity",
        args.retcol,
        "gross_ret", "fill_error_ret", "cost_lin", "cost_imp",
        "turnover", "dw", "w_target", "w_exec",
        "p_bear", "p_bull", "eligible_to_enter",
        "atr",
    ]
    summary_cols = [c for c in summary_cols if c in df2.columns]

    summary = df2.loc[top_days, summary_cols].copy()
    summary.index.name = "date"
    summary = summary.sort_values("dd_change" if args.mode == "dd_change" else "dd", ascending=True)

    summary_csv = outdir / f"{title}_top{args.topn}_summary.csv"
    summary.to_csv(summary_csv, encoding="utf-8-sig")

    print("\n=== DD TOP DAYS EXPORT ===")
    print("Parquet :", parquet_path.resolve())
    print("Mode    :", args.mode)
    print("TopN    :", args.topn)
    print("Window  : +/-", args.window, "days")
    print("Saved summary:", summary_csv.resolve())
    print("\nTop days:")
    for d in summary.index[: args.topn]:
      row = summary.loc[d]

      dd = float(row["dd"]) if "dd" in row.index else float("nan")
      ddc = float(row["dd_change"]) if "dd_change" in row.index else float("nan")

      d_str = d.date() if hasattr(d, "date") else d

      print(f" - {d_str}   dd={dd:.2%}   dd_change={ddc:.2%}")

    # Export per-day windows
    perdir = outdir / f"{title}_top{args.topn}_windows"
    perdir.mkdir(parents=True, exist_ok=True)

    # Choose columns to include in window CSV (keep readable)
    window_cols = [
        args.retcol, "equity", "dd", "dd_change",
        "gross_ret", "fill_error_ret",
        "turnover", "dw", "w_target", "w_exec",
        "p_bear", "p_bull", "eligible_to_enter",
        "atr",
    ]
    window_cols = [c for c in window_cols if c in df2.columns]

    idx = df2.index
    for d in summary.index[: args.topn]:
        # locate by positional window in the time index
        if d not in idx:
            continue
        loc = idx.get_loc(d)
        start = max(0, loc - args.window)
        end = min(len(idx) - 1, loc + args.window)
        wdf = df2.iloc[start : end + 1][window_cols].copy()
        wdf.index.name = "date"

        # nice filename
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        path = perdir / f"{ds}_window_pm{args.window}.csv"
        wdf.to_csv(path, encoding="utf-8-sig")

    print("Saved per-day windows:", perdir.resolve())
    print("DONE")


if __name__ == "__main__":
    main()
