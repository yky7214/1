"""Unit tests for ATR exit/state-machine logic.

Covers deterministic triggering of:
- hard stop
- trailing stop
- timeout
and validates peak tracking convention (close vs high).

These tests use synthetic OHLC paths so they do not require any market data.
"""

from __future__ import annotations

import pandas as pd

from ftf.trading.atr import compute_atr
from ftf.trading.exits import fit_atr_exit_state, generate_target_weights
from ftf.utils.config import ATRExitConfig


def _mk_path_for_hard_stop() -> pd.DataFrame:
    # Create a stable path then a sharp drop after entry.
    idx = pd.bdate_range("2020-01-01", periods=80)
    close = pd.Series(100.0, index=idx)
    # keep constant until after entry, then drop hard
    close.iloc[30:] = 100.0
    close.iloc[40] = 95.0
    close.iloc[41:] = 95.0

    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _mk_path_for_trailing_stop(*, peak_ref: str = "close") -> pd.DataFrame:
    # Ramp up to create a peak, then decline enough to trigger trailing.
    idx = pd.bdate_range("2020-01-01", periods=120)
    close = pd.Series(100.0, index=idx)
    # Uptrend from day 30 to 60
    for i in range(30, 61):
        close.iloc[i] = close.iloc[i - 1] + 0.5
    # Flat
    close.iloc[61:70] = close.iloc[60]
    # Then a drop
    close.iloc[70] = close.iloc[69] - 4.0
    close.iloc[71:] = close.iloc[70]

    if peak_ref == "high":
        high = close.copy()
        # Make highs a bit higher during the uptrend so peak differs from close.
        high.iloc[30:61] = close.iloc[30:61] + 1.0
    else:
        high = close + 0.5

    low = close - 0.5
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _mk_path_for_timeout() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=80)
    close = pd.Series(100.0, index=idx)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _run_exit_engine(
    df: pd.DataFrame,
    *,
    entry_day: int,
    exit_cfg: ATRExitConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    atr = compute_atr(df, window=exit_cfg.atr_window)

    idx = df.index
    # w_raw wants to be long whenever active.
    w_raw = pd.Series(1.0, index=idx)

    eligible = pd.Series(False, index=idx)
    eligible.iloc[entry_day:] = True

    # Default: no derisk (p_bear <= 0.5)
    p_bear = pd.Series(0.0, index=idx)

    exit_state = fit_atr_exit_state(cfg=exit_cfg, stop_fill_policy="STOP_FILL_T_PLUS_1")
    w_target, log, _ = generate_target_weights(
        close=df["close"],
        high=df.get("high"),
        atr=atr,
        w_raw=w_raw,
        eligible_to_enter=eligible.astype(float),
        p_bear=p_bear,
        exit_state=exit_state,
    )
    events = log.to_frame()
    return w_target, events


def test_hard_stop_triggers_exit_event_and_flattens_target():
    df = _mk_path_for_hard_stop()
    # Use a small ATR window to get ATR available early and predictable.
    cfg = ATRExitConfig(
        atr_window=5,
        hard_stop_atr=2.0,
        trailing_stop_atr=1.5,
        timeout_days=30,
        price_reference_for_peak="close",
        derisk_policy="DERISK_HALF",
    )

    # Enter at day 35 (ATR should be available with window=5)
    w_target, events = _run_exit_engine(df, entry_day=35, exit_cfg=cfg)

    # Expect an entry and then a hard stop exit after the drop.
    assert (events["event"] == "ENTRY").any()
    assert (events["event"] == "EXIT_HARD_STOP").any()

    # After the stop triggers at close t, target weight at t should be 0.
    # Find the date the hard stop triggered.
    hard_date = pd.to_datetime(events.loc[events["event"] == "EXIT_HARD_STOP", "date"].iloc[0])
    assert w_target.loc[hard_date] == 0.0


def test_trailing_stop_uses_peak_reference_close_vs_high():
    # With peak_ref='high', peak is higher, so trailing threshold is higher,
    # making a trailing stop more likely/earlier for the same close path.

    base_cfg = dict(
        atr_window=5,
        hard_stop_atr=999.0,  # disable hard stop so we isolate trailing
        trailing_stop_atr=1.0,
        timeout_days=999,
        derisk_policy="DERISK_HALF",
    )

    df_close_peak = _mk_path_for_trailing_stop(peak_ref="close")
    cfg_close = ATRExitConfig(price_reference_for_peak="close", **base_cfg)
    w_close, ev_close = _run_exit_engine(df_close_peak, entry_day=35, exit_cfg=cfg_close)

    df_high_peak = _mk_path_for_trailing_stop(peak_ref="high")
    cfg_high = ATRExitConfig(price_reference_for_peak="high", **base_cfg)
    w_high, ev_high = _run_exit_engine(df_high_peak, entry_day=35, exit_cfg=cfg_high)

    # Both should have a trailing stop eventually.
    assert (ev_close["event"] == "EXIT_TRAILING_STOP").any()
    assert (ev_high["event"] == "EXIT_TRAILING_STOP").any()

    dt_close = pd.to_datetime(ev_close.loc[ev_close["event"] == "EXIT_TRAILING_STOP", "date"].iloc[0])
    dt_high = pd.to_datetime(ev_high.loc[ev_high["event"] == "EXIT_TRAILING_STOP", "date"].iloc[0])

    # With higher peak (using highs), trailing threshold is tighter, so it should
    # exit no later than the close-based peak.
    assert dt_high <= dt_close

    # At trailing stop date, target should be flat.
    assert w_close.loc[dt_close] == 0.0
    assert w_high.loc[dt_high] == 0.0


def test_timeout_exit_after_timeout_days():
    df = _mk_path_for_timeout()
    cfg = ATRExitConfig(
        atr_window=5,
        hard_stop_atr=999.0,
        trailing_stop_atr=999.0,
        timeout_days=10,
        price_reference_for_peak="close",
        derisk_policy="DERISK_HALF",
    )

    w_target, events = _run_exit_engine(df, entry_day=20, exit_cfg=cfg)

    assert (events["event"] == "ENTRY").any()
    assert (events["event"] == "EXIT_TIMEOUT").any()

    timeout_date = pd.to_datetime(events.loc[events["event"] == "EXIT_TIMEOUT", "date"].iloc[0])
    assert w_target.loc[timeout_date] == 0.0
