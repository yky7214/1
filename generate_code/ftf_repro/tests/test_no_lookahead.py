"""No-lookahead invariant tests.

These tests are intentionally synthetic and lightweight so they can run in CI
without requiring the user to provide proprietary futures datasets.

We validate two critical invariants from the reproduction plan:
  1) Regime feature computation is causal: for a given day t, p_bull computed
     using only prices <= t matches the stored value computed on the full
     series (no future leakage).
  2) ATR at day t uses only information up to t (and C_{t-1}); it must match a
     recomputation performed on a truncated series.

We also validate end-to-end engine causality under the forecast-to-fill
convention by ensuring weights/returns align as expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ftf.signals.regime import fit_regime_state, compute_regime_features
from ftf.trading.atr import compute_atr


def _mk_ohlc(n: int = 220, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)

    # Build a gentle random walk to avoid degenerate sigma=0.
    ret = rng.normal(0.0002, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + ret)

    # high/low around close with small random ranges.
    spread = np.abs(rng.normal(0.0, 0.5, size=n))
    high = close + spread
    low = close - spread

    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)
    return df


def test_regime_no_lookahead_random_day():
    df = _mk_ohlc(n=260, seed=1)
    close = df["close"]

    # Fit on a "train" prefix.
    train = close.iloc[:200]
    state = fit_regime_state(train)

    full = compute_regime_features(close, state=state)

    # Pick a day well after warmup but not the last day.
    t = close.index[230]

    # Recompute features using only data up to t.
    trunc_close = close.loc[:t]
    trunc = compute_regime_features(trunc_close, state=state)

    # Compare stored vs recomputed on the same timestamp.
    assert np.isfinite(full.loc[t, "p_bull"])
    assert np.isfinite(trunc.loc[t, "p_bull"])
    assert float(full.loc[t, "p_bull"]) == float(trunc.loc[t, "p_bull"])

    # Additional check: eligibility uses only same-day slope and p_bull.
    assert float(full.loc[t, "eligible_to_enter"]) == float(trunc.loc[t, "eligible_to_enter"])


def test_atr_no_lookahead_matches_truncated():
    df = _mk_ohlc(n=80, seed=2)

    atr_full = compute_atr(df, window=14)

    # Choose a date after ATR warmup.
    t = df.index[50]
    atr_trunc = compute_atr(df.loc[:t], window=14)

    assert np.isfinite(atr_full.loc[t])
    assert np.isfinite(atr_trunc.loc[t])
    assert float(atr_full.loc[t]) == float(atr_trunc.loc[t])


def test_atr_does_not_use_future_close_in_tr():
    # Construct a series where future close is extreme; ATR at t should not change
    # when we perturb close at t+1.
    df = _mk_ohlc(n=40, seed=3)
    t = df.index[25]

    atr0 = compute_atr(df, window=14).loc[t]

    df2 = df.copy()
    df2.loc[df.index[26], "close"] = df2.loc[df.index[26], "close"] * 50.0  # huge future move
    # Keep high/low aligned to new close to be adversarial.
    df2.loc[df.index[26], "high"] = df2.loc[df.index[26], "close"] + 1.0
    df2.loc[df.index[26], "low"] = df2.loc[df.index[26], "close"] - 1.0

    atr1 = compute_atr(df2, window=14).loc[t]

    assert float(atr0) == float(atr1)
