"""Data subpackage.

This package contains utilities for:
- calendar creation/alignment
- raw data loaders (CSV/Parquet)
- continuous futures construction (roll logic)
- prepared-data validation checks

The submodules are intentionally lightweight and deterministic to support
walk-forward reproducibility and no-lookahead testing.
"""

from .calendar import CalendarSpec, get_calendar, infer_calendar_from_index, nyse_business_days, shift_bdays, to_date_index
from .futures_roll import ContinuousFuturesResult, build_continuous_front_month, determine_active_contract
from .loaders import (
    align_ohlc_to_calendar,
    infer_date_col,
    read_contract_metadata,
    read_contract_ohlc,
    read_lbma_spot,
    validate_daily_index,
)
from .validation import (
    ContinuousValidationReport,
    compute_atr14,
    compute_returns_from_close,
    validate_continuous_df,
    validate_roll_rule,
)

__all__ = [
    # calendar
    "CalendarSpec",
    "get_calendar",
    "infer_calendar_from_index",
    "nyse_business_days",
    "shift_bdays",
    "to_date_index",
    # loaders
    "validate_daily_index",
    "infer_date_col",
    "align_ohlc_to_calendar",
    "read_contract_ohlc",
    "read_contract_metadata",
    "read_lbma_spot",
    # roll
    "ContinuousFuturesResult",
    "determine_active_contract",
    "build_continuous_front_month",
    # validation
    "ContinuousValidationReport",
    "compute_returns_from_close",
    "compute_atr14",
    "validate_roll_rule",
    "validate_continuous_df",
]
