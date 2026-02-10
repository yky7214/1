"""ftf.walkforward.schedule

Builds the canonical walk-forward schedule used throughout the reproduction.

Conventions
-----------
- Business-day based windows:
  * train_bd ~ 10y = 2520
  * test_bd  ~ 6m  = 126
  * step_bd  ~ 1m  = 21

- Anchors are *decision dates* (t0) such that:
  train = [t0 - train_bd, t0)
  test  = [t0, t0 + test_bd)

The schedule uses the index of the provided continuous futures dataframe to
respect the actual available business-day calendar (NYSE-like).

This module is intentionally deterministic and free of side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from ftf.utils.config import FTFConfig


@dataclass(frozen=True)
class WalkForwardAnchor:
    """A single walk-forward anchor definition (all bounds inclusive/exclusive).

    All timestamps are timezone-naive and normalized to midnight.

    Attributes
    ----------
    anchor:
        The anchor date t0.
    train_start:
        First date in the training slice (inclusive).
    train_end:
        End of training slice (exclusive), equals anchor.
    test_start:
        Start of the test slice (inclusive), equals anchor.
    test_end:
        End of the test slice (exclusive).
    kept_start:
        Start of the kept OOS segment for canonical stitching.
    kept_end:
        End of the kept OOS segment for canonical stitching.
        For FIRST_STEP_ONLY, kept_end = anchor shifted by step_bd.
        For FULL_TEST_DIAGNOSTIC, kept_end = test_end.
    """

    anchor: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    kept_start: pd.Timestamp
    kept_end: pd.Timestamp


def _normalize_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _require_bday_in_index(idx: pd.DatetimeIndex, date: pd.Timestamp, *, name: str) -> pd.Timestamp:
    """Return the first index date >= date, else raise."""

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("idx must be a DatetimeIndex")
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    date = _normalize_ts(date)

    pos = idx.searchsorted(date, side="left")
    if pos >= len(idx):
        raise ValueError(f"{name}={date.date()} is after last available date {idx[-1].date()}")
    return idx[pos]


def build_walkforward_schedule(
    index: pd.DatetimeIndex | Iterable[pd.Timestamp],
    *,
    cfg: Optional[FTFConfig] = None,
    train_bd: Optional[int] = None,
    test_bd: Optional[int] = None,
    step_bd: Optional[int] = None,
    anchor_start: Optional[str | pd.Timestamp] = None,
    anchor_end: Optional[str | pd.Timestamp] = None,
) -> List[WalkForwardAnchor]:
    """Construct the walk-forward schedule as a list of anchors.

    Parameters
    ----------
    index:
        Business-day date index of the underlying continuous series.
        The schedule is built *on this index* to avoid calendar mismatches.
    cfg:
        Optional top-level config; provides defaults.
    train_bd, test_bd, step_bd:
        Window lengths in business days.
    anchor_start, anchor_end:
        Inclusive anchor bounds. Anchors are snapped to the first available
        date in `index` on/after the provided start.

    Returns
    -------
    list[WalkForwardAnchor]

    Notes
    -----
    Only anchors that have a full training and full test window available
    within the provided `index` are returned.
    """

    if isinstance(index, pd.DatetimeIndex):
        idx = index
    else:
        idx = pd.DatetimeIndex(list(index))

    if idx.tz is not None:
        idx = idx.tz_convert(None)
    idx = idx.normalize()
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")
    if idx.has_duplicates:
        raise ValueError("index must not contain duplicates")

    if cfg is None:
        # late import defaults but without importing heavy modules
        from ftf.utils.config import FTFConfig as _FTFConfig

        cfg = _FTFConfig()  # type: ignore[call-arg]

    train_bd = int(train_bd if train_bd is not None else cfg.walkforward.train_bd)
    test_bd = int(test_bd if test_bd is not None else cfg.walkforward.test_bd)
    step_bd = int(step_bd if step_bd is not None else cfg.walkforward.step_bd)

    if train_bd <= 0 or test_bd <= 0 or step_bd <= 0:
        raise ValueError("train_bd/test_bd/step_bd must be positive")

    a_start = anchor_start if anchor_start is not None else cfg.walkforward.anchor_start
    a_end = anchor_end if anchor_end is not None else cfg.walkforward.anchor_end
    a_start_ts = _normalize_ts(a_start)
    a_end_ts = _normalize_ts(a_end)
    if a_end_ts < a_start_ts:
        raise ValueError("anchor_end must be >= anchor_start")

    # Snap start/end to available business days.
    start_anchor = _require_bday_in_index(idx, a_start_ts, name="anchor_start")
    
    if a_end_ts > idx[-1]:
        end_anchor = idx[-1]

    else:
        end_anchor = _require_bday_in_index(idx, a_end_ts, name="anchor_end")

    anchors: List[WalkForwardAnchor] = []

    # Iterate anchors using index positions with step_bd.
    pos = idx.get_indexer([start_anchor], method=None)[0]
    if pos < 0:
        pos = int(idx.searchsorted(start_anchor, side="left"))

    last_possible_anchor_pos = len(idx) - test_bd
    # Ensure we also have full training.
    first_possible_anchor_pos = train_bd

    # Move forward until in feasible range.
    if pos < first_possible_anchor_pos:
        pos = first_possible_anchor_pos

    while pos <= last_possible_anchor_pos and idx[pos] <= end_anchor:
        t0 = idx[pos]
        train_start = idx[pos - train_bd]
        train_end = t0
        test_start = t0
        test_end = idx[pos + test_bd] if (pos + test_bd) < len(idx) else idx[-1] + pd.Timedelta(days=1)

        kept_start = test_start
        if cfg.time.stitch_rule == "FIRST_STEP_ONLY":
            kept_end = idx[pos + step_bd] if (pos + step_bd) < len(idx) else idx[-1] + pd.Timedelta(days=1)
            # kept_end should not exceed test_end.
            if kept_end > test_end:
                kept_end = test_end
        else:
            kept_end = test_end

        anchors.append(
            WalkForwardAnchor(
                anchor=t0,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                kept_start=kept_start,
                kept_end=kept_end,
            )
        )

        pos += step_bd

    return anchors


__all__ = ["WalkForwardAnchor", "build_walkforward_schedule"]
