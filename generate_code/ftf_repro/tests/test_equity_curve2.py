import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Config
# ============================================================

USE_LOG = False   # True = log scale, False = linear
SHOW_DD = True    # drawdown を下段に表示


# ============================================================
# Utilities
# ============================================================

def compute_equity(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod()


def compute_drawdown_from_equity(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


# ============================================================
# Main
# ============================================================

def test_equity_curve_v2():

    ROOT = Path(__file__).resolve().parents[1]

    PARQUET = (
        ROOT
        / "reports"
        / "base_fast_nocost_final"
        / "reports"
        / "oos_daily.parquet"
    )

    df = pd.read_parquet(PARQUET).sort_index()

    # ===== columns =====
    ret_col = "net_ret_ftf"
    w_col   = "w_exec"

    assert ret_col in df.columns
    assert w_col in df.columns

    ret = df[ret_col].fillna(0.0)

    equity = compute_equity(ret)
    dd = compute_drawdown_from_equity(equity)

    # ========================================================
    # Plot
    # ========================================================

    if SHOW_DD:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax2 = None

    # ---- equity ----
    ax1.plot(equity, label="Equity (net_ret_ftf)", linewidth=2)

    if USE_LOG:
        ax1.set_yscale("log")

    ax1.set_title("Equity Curve (delta_w_cap applied)")
    ax1.set_ylabel("Equity")
    ax1.grid(True)
    ax1.legend()

    # ---- drawdown ----
    if SHOW_DD:
        ax2.fill_between(dd.index, dd, 0.0, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # Quick stats
    # ========================================================

    sharpe = (
        ret.mean() / (ret.std(ddof=0) + 1e-12)
    ) * np.sqrt(252)

    print("\n=== EQUITY SUMMARY ===")
    print(f"Parquet : {PARQUET}")
    print(f"Sharpe  : {sharpe:.3f}")
    print(f"MaxDD   : {dd.min():.2%}")
    print(f"FinalEq : {equity.iloc[-1]:.3f}")


if __name__ == "__main__":
    test_equity_curve_v2()
