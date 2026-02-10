"""Trading calendar helpers.

The reproduction plan requires a canonical NYSE business-day index for aligning
all daily series. We implement a lightweight calendar using pandas
``CustomBusinessDay`` with US Federal holidays as a practical proxy.

Note: The paper specifies "NYSE business days". A perfect NYSE calendar
requires an exchange holiday calendar; for reproducibility without extra
dependencies we use `USFederalHolidayCalendar`, which is close for the 2015–2025
range. If users have `pandas_market_calendars` they can swap the implementation.

All dates are timezone-naive at daily frequency (normalized to midnight).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


_US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def to_date_index(idx: Iterable) -> pd.DatetimeIndex:
    """Convert an iterable of timestamps to a normalized, tz-naive DatetimeIndex."""
    di = pd.DatetimeIndex(idx)
    # normalize and drop tz info
    if di.tz is not None:
        di = di.tz_convert(None)
    di = di.normalize()
    return di


def nyse_business_days(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Generate a NYSE-like business day index between start and end (inclusive)."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must be >= start")
    return pd.date_range(start_ts, end_ts, freq=_US_BDAY)


def shift_bdays(date: pd.Timestamp, n: int) -> pd.Timestamp:
    """Shift a date by n business days in the NYSE-like calendar."""
    return (pd.Timestamp(date).normalize() + n * _US_BDAY).normalize()


@dataclass(frozen=True)
class CalendarSpec:
    """A minimal calendar object to pass around in roll logic."""

    name: str = "NYSE"

    def bdays(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        return nyse_business_days(start, end)

    def shift(self, date: pd.Timestamp, n: int) -> pd.Timestamp:
        return shift_bdays(date, n)

    def offset(self) -> CustomBusinessDay:
        return _US_BDAY


def get_calendar(name: str = "NYSE") -> CalendarSpec:
    name_u = name.upper()
    if name_u != "NYSE":
        raise ValueError(f"Only 'NYSE' supported, got {name}")
    return CalendarSpec(name="NYSE")


def infer_calendar_from_index(index: pd.DatetimeIndex) -> Optional[str]:
    """Best-effort inference (used only for diagnostics)."""
    if len(index) < 3:
        return None
    # crude: if weekends missing, assume business day
    weekdays = pd.Series(index.weekday)
    if (weekdays >= 5).any():
        return None
    return "NYSE"


__all__ = [
    "CalendarSpec",
    "get_calendar",
    "infer_calendar_from_index",
    "nyse_business_days",
    "shift_bdays",
    "to_date_index",
]
