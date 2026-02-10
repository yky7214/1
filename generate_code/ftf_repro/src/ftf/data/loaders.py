"""ftf.data.loaders

Data ingestion and calendar alignment.

This project intentionally does not hardcode a particular vendor format. Instead,
this module provides:

- Canonical loaders for common file types (CSV/Parquet) and expected columns.
- Business-day calendar alignment (NYSE-like) with strict validation.
- Forward-fill rules:
    * prices (close/settle, open, high, low) may be forward-filled if an entire
      business day is missing.
    * returns are *never* forward-filled; they are computed from the filled close.

All dates are treated as timezone-naive daily timestamps normalized to midnight.

The most important helper is :func:`align_ohlc_to_calendar` which reindexes a
DataFrame to a given business-day index and enforces monotonic unique dates.

"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from ftf.data.calendar import CalendarSpec, get_calendar, to_date_index
from ftf.utils.config import DataConfig

__all__ = [
    "read_contract_ohlc",
    "read_contract_metadata",
    "read_lbma_spot",
    "align_ohlc_to_calendar",
    "validate_daily_index",
    "infer_date_col",
]


def validate_daily_index(df: pd.DataFrame, *, name: str = "data") -> None:
    """Validate daily time index invariants.

    Requirements:
      - monotonic increasing
      - no duplicates
      - timezone-naive

    Raises:
        ValueError on violation.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: expected DatetimeIndex, got {type(df.index)}")

    if df.index.tz is not None:
        raise ValueError(f"{name}: expected timezone-naive dates")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index not monotonic increasing")

    if df.index.has_duplicates:
        dups = df.index[df.index.duplicated()].unique()
        raise ValueError(f"{name}: duplicate timestamps found (e.g. {dups[:5].tolist()})")


def infer_date_col(columns: Iterable[str]) -> Optional[str]:
    cols = [c.lower() for c in columns]
    for cand in ["date", "timestamp", "time", "dt"]:
        if cand in cols:
            return list(columns)[cols.index(cand)]
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _set_datetime_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    if date_col is None:
        date_col = infer_date_col(df.columns)
    if date_col is None:
        raise ValueError("Could not infer date column; please provide date_col")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.tz_localize(None).dt.normalize()
    out = out.set_index(date_col).sort_index()
    validate_daily_index(out, name="raw")
    return out


def align_ohlc_to_calendar(
    df: pd.DataFrame,
    *,
    calendar: CalendarSpec,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    ffill_price_cols: Tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """Reindex a daily OHLC dataframe to business days.

    Forward-fills only the specified price columns, to avoid fabricating returns.

    Args:
        df: indexed by daily timestamps.
        calendar: business-day calendar.
        start/end: optional boundaries; if omitted, inferred from df.
        ffill_price_cols: columns to forward-fill when a full day is missing.

    Returns:
        aligned DataFrame with business-day index.
    """

    validate_daily_index(df, name="df")

    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()

    bidx = calendar.bdays(start, end)
    out = df.reindex(bidx)

    cols = [c for c in ffill_price_cols if c in out.columns]
    if cols:
        out[cols] = out[cols].ffill()

    validate_daily_index(out, name="aligned")
    return out


def read_contract_ohlc(
    path: str | Path,
    *,
    cfg: Optional[DataConfig] = None,
    date_col: Optional[str] = None,
    calendar_name: str = "NYSE",
) -> pd.DataFrame:
    """Read a single-contract daily OHLC(+volume/ADV) file.

    Expected columns are controlled by DataConfig. Missing optional columns are
    allowed.
    """

    cfg = cfg or DataConfig()
    df = _read_table(path)
    df = _set_datetime_index(df, date_col=date_col)

    # Normalize column names: we only ensure required price col exists.
    if cfg.price_col not in df.columns:
        # common alternatives
        for alt in ["settle", "Settle", "Close", "SETTLE", "CLOSE"]:
            if alt in df.columns:
                df = df.rename(columns={alt: cfg.price_col})
                break
    if cfg.price_col not in df.columns:
        raise ValueError(f"Missing price column '{cfg.price_col}' in {path}")

    cal = get_calendar(calendar_name)
    ffill_cols = tuple(c for c in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.price_col] if c)
    df = align_ohlc_to_calendar(df, calendar=cal, ffill_price_cols=ffill_cols)
    return df


def read_contract_metadata(path: str | Path, *, date_col: str = "fnd") -> pd.DataFrame:
    """Read contract metadata.

    Required columns:
      - contract: contract identifier (string)
      - fnd: first notice date (date)

    Returns:
        DataFrame indexed by contract with column 'fnd' as Timestamp.
    """

    df = _read_table(path)
    # common column names
    cols = {c.lower(): c for c in df.columns}
    if "contract" not in cols:
        raise ValueError("metadata must contain 'contract' column")
    contract_col = cols["contract"]
    if date_col not in df.columns:
        # try infer
        for cand in ["fnd", "first_notice", "first_notice_date", "firstnoticedate"]:
            if cand in cols:
                date_col = cols[cand]
                break
    if date_col not in df.columns:
        raise ValueError("metadata must contain a first notice date column (e.g., 'fnd')")

    out = df[[contract_col, date_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.tz_localize(None).dt.normalize()
    out[contract_col] = out[contract_col].astype(str)
    out = out.set_index(contract_col).sort_index()
    return out


def read_lbma_spot(
    path: str | Path,
    *,
    date_col: Optional[str] = None,
    price_col: str = "price",
    calendar_name: str = "NYSE",
) -> pd.Series:
    """Read LBMA PM fix spot series for benchmark regression.

    The file should contain columns [date, price] (names flexible). The returned
    Series is aligned to the NYSE business-day index and forward-filled on price.
    """

    df = _read_table(path)
    df = _set_datetime_index(df, date_col=date_col)

    if price_col not in df.columns:
        # guess first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("LBMA spot file must contain a numeric price column")
        df = df.rename(columns={num_cols[0]: price_col})

    cal = get_calendar(calendar_name)
    aligned = align_ohlc_to_calendar(df[[price_col]], calendar=cal, ffill_price_cols=(price_col,))
    return aligned[price_col]
