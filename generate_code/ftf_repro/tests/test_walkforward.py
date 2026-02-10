"""Walk-forward pipeline tests.

These tests are intentionally synthetic and light-weight so they can run in CI
without proprietary data.

We validate:
- Schedule sizing and kept-window stitching semantics (FIRST_STEP_ONLY)
- Walk-forward result does not contain overlapping dates in stitched OOS output
- Basic determinism of the runner in FIXED trainer mode

The goal is not to assert a particular Sharpe, but to ensure the mechanics and
no-overlap stitching rule are correct.
"""

from __future__ import annotations

import pandas as pd

from ftf.utils.config import (
    ATRExitConfig,
    BootstrapConfig,
    CapacityConfig,
    CostImpactConfig,
    DataConfig,
    FTFConfig,
    KellyConfig,
    RegressionConfig,
    RiskConfig,
    SignalConfig,
    TimeConvention,
    WalkForwardConfig,
)
from ftf.walkforward.runner import run_walkforward
from ftf.walkforward.schedule import build_walkforward_schedule


def _mk_continuous_df(n: int = 3600) -> pd.DataFrame:
    """Synthetic continuous OHLCV/ADV.

    n should be sufficiently large to support the default train/test windows.
    """

    idx = pd.bdate_range("2005-01-03", periods=n)
    # Mild trend with some cyclical variation; deterministic.
    t = pd.Series(range(n), index=idx, dtype=float)
    close = 1800.0 + 0.03 * t + 2.0 * (t / 50.0).apply(lambda x: __import__("math").sin(x))
    high = close * 1.002
    low = close * 0.998
    adv = pd.Series(20000.0, index=idx)
    vol = pd.Series(15000.0, index=idx)

    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "open": close,
            "adv": adv,
            "volume": vol,
        },
        index=idx,
    )


def _mk_cfg(*, stitch_rule: str = "FIRST_STEP_ONLY") -> FTFConfig:
    return FTFConfig(
        time=TimeConvention(exec_lag=1, stop_fill_policy="STOP_FILL_T_PLUS_1", stitch_rule=stitch_rule),
        data=DataConfig(
            calendar="NYSE",
            roll_bd_before_fnd=2,
            price_col="close",
            high_col="high",
            low_col="low",
            open_col="open",
            volume_col="volume",
            adv_col="adv",
            contract_multiplier=100.0,
        ),
        signal=SignalConfig(
            ema_lambda=0.94,
            momentum_k=50,
            blend_omega=0.6,
            pbull_threshold=0.52,
            z_clip=(-3.0, 3.0),
        ),
        risk=RiskConfig(ewma_theta=0.94, vol_target_annual=0.15, w_max=2.0),
        atr_exit=ATRExitConfig(
            atr_window=14,
            hard_stop_atr=2.0,
            trailing_stop_atr=1.5,
            timeout_days=30,
            price_reference_for_peak="close",
            derisk_policy="DERISK_HALF",
        ),
        costs=CostImpactConfig(k_linear=0.00007, gamma_impact=0.02),
        kelly=KellyConfig(lambda_kelly=0.40, baseline_floor=0.25, baseline_floor_mode="FLOOR_ON_WVOL"),
        walkforward=WalkForwardConfig(
            train_bd=2520,
            test_bd=126,
            step_bd=21,
            anchor_start="2015-01-01",
            anchor_end="2025-10-31",
            trainer_mode="FIXED",
        ),
        regression=RegressionConfig(nw_lags=10, nw_lags_sensitivity=(5, 10, 20)),
        bootstrap=BootstrapConfig(
            block_bootstrap_B=200,
            block_len=20,
            stationary_bootstrap_B=200,
            stationary_mean_block=20,
            seed=123,
        ),
        capacity=CapacityConfig(participation_cap=0.01),
        run_name="test",
    )


def test_schedule_kept_window_first_step_only_has_step_length_or_less() -> None:
    df = _mk_continuous_df(3600)
    cfg = _mk_cfg(stitch_rule="FIRST_STEP_ONLY")
    anchors = build_walkforward_schedule(df.index, cfg=cfg)
    assert len(anchors) > 0

    for a in anchors[:5]:
        # kept window is [kept_start, kept_end) in schedule implementation
        kept_len = len(df.loc[(df.index >= a.kept_start) & (df.index < a.kept_end)])
        assert kept_len <= cfg.walkforward.step_bd
        assert kept_len > 0


def test_walkforward_stitched_oos_has_no_duplicate_dates() -> None:
    df = _mk_continuous_df(3600)
    cfg = _mk_cfg(stitch_rule="FIRST_STEP_ONLY")

    res = run_walkforward(df, cfg=cfg, out_dir=None, persist_daily=False, persist_per_anchor=False, progress=False)

    assert isinstance(res.oos_daily, pd.DataFrame)
    assert res.oos_daily.index.is_monotonic_increasing
    assert res.oos_daily.index.has_duplicates is False

    # also sanity-check kept OOS length equals anchors * ~step_bd (allow last anchor shorter)
    n = len(res.oos_daily)
    assert n > 0
    assert n <= len(res.anchors) * cfg.walkforward.step_bd


def test_walkforward_is_deterministic_for_fixed_mode() -> None:
    df = _mk_continuous_df(3600)
    cfg = _mk_cfg(stitch_rule="FIRST_STEP_ONLY")

    res1 = run_walkforward(df, cfg=cfg, out_dir=None, persist_daily=False, persist_per_anchor=False, progress=False)
    res2 = run_walkforward(df, cfg=cfg, out_dir=None, persist_daily=False, persist_per_anchor=False, progress=False)

    pd.testing.assert_index_equal(res1.oos_daily.index, res2.oos_daily.index)
    pd.testing.assert_series_equal(res1.oos_net_ret, res2.oos_net_ret)
    pd.testing.assert_series_equal(res1.oos_gross_ret, res2.oos_gross_ret)
