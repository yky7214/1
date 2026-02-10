# tests/test_rolling_sharpe_252d.py
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rolling_sharpe(
    r: pd.Series,
    window: int = 252,
    ann_factor: int = 252,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Rolling Sharpe (annualized) using simple daily returns.
    Sharpe = sqrt(ann_factor) * mean(r) / std(r)
    """
    r = r.astype(float).fillna(0.0)

    mu = r.rolling(window, min_periods=window).mean()
    sd = r.rolling(window, min_periods=window).std(ddof=0)

    sharpe = np.sqrt(float(ann_factor)) * (mu / (sd + eps))
    sharpe.name = f"sharpe_{window}d"

    return sharpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=str,
        default="reports/base_fast_nocost_wmax3/oos_daily.parquet",
        help="Path to oos_daily.parquet (relative to repo root is OK)",
    )
    ap.add_argument(
        "--retcol",
        type=str,
        default="net_ret_ftf",
        help="Return column to use for Sharpe",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=252,
        help="Rolling window length (trading days)",
    )
    ap.add_argument(
        "--ann",
        type=int,
        default=252,
        help="Annualization factor (252 for daily)",
    )
    ap.add_argument(
        "--paper",
        type=float,
        default=2.88,
        help="Reference Sharpe line (paper) to draw",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="tests/outputs/rolling_sharpe_252d.png",
        help="Output PNG path",
    )
    args = ap.parse_args()

    # Repo root (tests/ の1つ上)
    ROOT = Path(__file__).resolve().parents[1]

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = ROOT / parquet_path

    assert parquet_path.exists(), f"Parquet not found: {parquet_path}"

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path).sort_index()
    assert args.retcol in df.columns, f"Return column not found: {args.retcol}"

    r = df[args.retcol].copy()
    rs = rolling_sharpe(r, window=args.window, ann_factor=args.ann)

    # Print quick stats
    last = rs.dropna().iloc[-1] if rs.notna().any() else float("nan")
    print("\n=== ROLLING SHARPE (252d) ===")
    print("Parquet :", parquet_path.resolve())
    print("Ret col :", args.retcol)
    print("Window  :", args.window)
    print("Last    :", f"{last:.3f}" if np.isfinite(last) else "nan")

    # Plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(rs.index, rs.values)
    ax.set_title(f"Rolling Sharpe Ratio ({args.window}d)")
    ax.set_ylabel("Sharpe")
    ax.grid(True)

    if args.paper is not None:
        ax.axhline(float(args.paper), linestyle="--")
        ax.text(
            rs.index[int(len(rs.index) * 0.02)],
            float(args.paper),
            f"Paper ({args.paper:.2f})",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()

    print("Saved :", out_path.resolve())
    print("DONE")


if __name__ == "__main__":
    main()
