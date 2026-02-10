"""ftf.data.futures_roll

Continuous futures construction for GC (or similar) from per-contract daily bars.

Implements the reproduction-plan rule:
- Active contract on date d is the nearest-to-expiry contract such that
  d < (FND(contract) - roll_bd_before_fnd business days).
- Roll occurs exactly `roll_bd_before_fnd` business days before FND.

The builder *splices* prices (no back-adjustment). Returns computed from the
spliced continuous close therefore include roll P&L implicitly.

Expected inputs
---------------
Per-contract OHLC tables in a dict: {contract: df}, where each df is aligned to
business-day calendar already (recommended) and has columns at least:
  close, high, low (optionally open, volume, adv)
Index must be timezone-naive daily DatetimeIndex.

Contract metadata must include first notice date (FND) per contract.

Outputs
-------
- df_cont: DataFrame indexed by daily business dates with continuous columns.
- active_contract: Series of active contract per date
- roll_table: DataFrame with roll_date, from_contract, to_contract

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.data.calendar import CalendarSpec, get_calendar
from ftf.data.loaders import validate_daily_index
from ftf.utils.config import DataConfig

__all__ = [
    "ContinuousFuturesResult",
    "build_continuous_front_month",
    "determine_active_contract",
]


@dataclass(frozen=True)
class ContinuousFuturesResult:
    df_cont: pd.DataFrame
    active_contract: pd.Series
    roll_table: pd.DataFrame


def _ensure_timestamp(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def determine_active_contract(
    date: pd.Timestamp,
    contracts_sorted: Iterable[str],
    fnd_by_contract: pd.Series,
    *,
    calendar: CalendarSpec,
    roll_bd_before_fnd: int,
) -> Optional[str]:
    """Return active contract for a given date.

    Parameters
    ----------
    date:
        Business date.
    contracts_sorted:
        Contracts sorted by expiry/FND ascending.
    fnd_by_contract:
        Series indexed by contract, value is FND timestamp.
    calendar:
        CalendarSpec.
    roll_bd_before_fnd:
        How many business days before FND we roll away.

    Returns
    -------
    Optional[str]
        Active contract symbol or None if none eligible.
    """

    d = _ensure_timestamp(date)
    for c in contracts_sorted:
        fnd = _ensure_timestamp(fnd_by_contract.loc[c])
        roll_cutoff = calendar.shift(fnd, -roll_bd_before_fnd)
        # Eligible strictly before cutoff
        if d < roll_cutoff:
            return c
    return None


def build_continuous_front_month(
    contract_bars: Dict[str, pd.DataFrame],
    contract_meta: pd.DataFrame,
    *,
    cfg: Optional[DataConfig] = None,
    calendar_name: str = "NYSE",
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ContinuousFuturesResult:
    """Build a spliced continuous front-month series.

    Notes
    -----
    - Assumes `contract_bars` are already calendar-aligned (business days).
      If not, align upstream using ftf.data.loaders.
    - The continuous series uses OHLC columns from the active contract on each
      day. No back-adjustment.
    """

    cfg = cfg or DataConfig()
    cal = get_calendar(calendar_name)

    if "contract" in contract_meta.columns:
        meta = contract_meta.set_index("contract")
    else:
        meta = contract_meta.copy()
    if "fnd" not in meta.columns and "FND" in meta.columns:
        meta = meta.rename(columns={"FND": "fnd"})
    if "fnd" not in meta.columns:
        raise ValueError("contract_meta must include 'fnd' column (first notice date)")

    # Normalize meta FNDs
    fnd_by_contract = meta["fnd"].apply(_ensure_timestamp)

    # Determine union calendar range
    all_idx = []
    for c, df in contract_bars.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"contract_bars[{c}] must be a DataFrame")
        validate_daily_index(df, name=f"contract_bars[{c}]")
        all_idx.append(df.index)

    if not all_idx:
        raise ValueError("contract_bars is empty")

    idx_union = all_idx[0]
    for idx in all_idx[1:]:
        idx_union = idx_union.union(idx)

    if start is None:
        start_ts = idx_union.min()
    else:
        start_ts = _ensure_timestamp(start)
    if end is None:
        end_ts = idx_union.max()
    else:
        end_ts = _ensure_timestamp(end)

    bdays = cal.bdays(start_ts, end_ts)

    # Sort contracts by FND ascending (nearest first)
    contracts_sorted = [c for c in fnd_by_contract.sort_values().index if c in contract_bars]
    if not contracts_sorted:
        raise ValueError("No overlap between contract_meta and contract_bars keys")

    active = pd.Series(index=bdays, dtype=object, name="active_contract")
    for d in bdays:
        c = determine_active_contract(
            d,
            contracts_sorted,
            fnd_by_contract,
            calendar=cal,
            roll_bd_before_fnd=cfg.roll_bd_before_fnd,
        )
        active.loc[d] = c

    # Build roll table
    # roll occurs when active changes from t-1 to t
    prev = active.shift(1)
    roll_mask = (active != prev) & active.notna() & prev.notna()
    roll_dates = active.index[roll_mask.fillna(False).to_numpy()]
    roll_table = pd.DataFrame(
        {
            "roll_date": roll_dates,
            "from_contract": prev.loc[roll_dates].to_numpy(),
            "to_contract": active.loc[roll_dates].to_numpy(),
        }
    )

    # Prepare output frame
    cols = [
        c
        for c in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.price_col, cfg.volume_col, cfg.adv_col]
        if c is not None
    ]
    cols = list(dict.fromkeys(cols))

    out = pd.DataFrame(index=bdays, columns=cols, dtype=float)

    for c in contracts_sorted:
        dfc = contract_bars[c]
        # reindex to bdays (if contract missing some dates, keep NaNs)
        dfc2 = dfc.reindex(bdays)
        mask = active == c
        if mask.any():
            out.loc[mask, cols] = dfc2.loc[mask, cols].to_numpy()

    # Enforce dtypes
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Basic forward-fill of OHLC within active contract gaps is intentionally
    # *not* performed here; upstream loader alignment should have handled price
    # ffill. For safety we can ffill continuous prices only if entire day missing.
    price_cols = [x for x in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.price_col] if x is not None]
    out[price_cols] = out[price_cols].ffill()

    return ContinuousFuturesResult(df_cont=out, active_contract=active, roll_table=roll_table)
