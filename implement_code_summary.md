# Code Implementation Progress Summary
*Accumulated implementation progress for all files*


================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/utils/config.py; ROUND 0 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:52:13
**File Implemented**: ftf_repro/src/ftf/utils/config.py

**Core Purpose**
- Defines the project’s canonical configuration schema and critical time/P&L conventions via frozen dataclasses, plus utilities to merge and sanity-check configs before running the walk-forward trading pipeline.

**Public Interface**
- Class `TimeConvention`: Execution/attribution conventions (exec lag, stop fill, OOS stitching) | Key methods: *(none)* | Constructor params: `exec_lag: int = 1`, `stop_fill_policy: StopFillPolicy = "STOP_FILL_T_PLUS_1"`, `stitch_rule: StitchRule = "FIRST_STEP_ONLY"`
- Class `DataConfig`: Data/calendar and continuous futures roll expectations | Key methods: *(none)* | Constructor params: `tz_naive_dates: bool = True`, `calendar: Literal["NYSE"] = "NYSE"`, `roll_bd_before_fnd: int = 2`, `price_col: str = "close"`, `high_col: str = "high"`, `low_col: str = "low"`, `open_col: Optional[str] = "open"`, `volume_col: Optional[str] = "volume"`, `adv_col: Optional[str] = "adv"`, `contract_multiplier: float = 100.0`
- Class `SignalConfig`: Signal hyperparameters (EMA/momentum/blend/thresholds) | Key methods: *(none)* | Constructor params: `ema_lambda: float = 0.94`, `momentum_k: int = 50`, `blend_omega: float = 0.6`, `pbull_threshold: float = 0.52`, `z_clip: Tuple[float, float] = (-3.0, 3.0)`
- Class `RiskConfig`: EWMA vol targeting and leverage cap | Key methods: *(none)* | Constructor params: `ewma_theta: float = 0.94`, `vol_target_annual: float = 0.15`, `w_max: float = 2.0`
- Class `ATRExitConfig`: ATR/exit state machine knobs | Key methods: *(none)* | Constructor params: `atr_window: int = 14`, `hard_stop_atr: float = 2.0`, `trailing_stop_atr: float = 1.5`, `timeout_days: int = 30`, `price_reference_for_peak: PeakRef = "close"`, `derisk_policy: DeriskPolicy = "DERISK_HALF"`
- Class `CostImpactConfig`: Execution cost model parameters | Key methods: *(none)* | Constructor params: `k_linear: float = 0.00007`, `gamma_impact: float = 0.02`
- Class `KellyConfig`: Fractional Kelly and floor behavior | Key methods: *(none)* | Constructor params: `lambda_kelly: float = 0.40`, `baseline_floor: float = 0.25`, `baseline_floor_mode: BaselineFloorMode = "FLOOR_ON_WVOL"`, `baseline_floor_eps: float = 1e-6`
- Class `WalkForwardConfig`: Walk-forward window sizes and trainer mode | Key methods: *(none)* | Constructor params: `train_bd: int = 2520`, `test_bd: int = 126`, `step_bd: int = 21`, `anchor_start: str = "2015-01-01"`, `anchor_end: str = "2025-10-31"`, `trainer_mode: TrainerMode = "FIXED"`
- Class `RegressionConfig`: Newey–West lag settings | Key methods: *(none)* | Constructor params: `nw_lags: int = 10`, `nw_lags_sensitivity: Tuple[int, ...] = (5, 10, 20)`
- Class `BootstrapConfig`: Bootstrap parameters and seed | Key methods: *(none)* | Constructor params: `block_bootstrap_B: int = 1000`, `block_len: int = 20`, `stationary_bootstrap_B: int = 800`, `stationary_mean_block: int = 20`, `seed: int = 123`
- Class `CapacityConfig`: Capacity/participation settings | Key methods: *(none)* | Constructor params: `participation_cap: float = 0.01`
- Class `FTFConfig`: Top-level config container bundling all sub-configs | Key methods: `to_dict()` | Constructor params:  
  `time: TimeConvention`, `data: DataConfig`, `signal: SignalConfig`, `risk: RiskConfig`, `atr_exit: ATRExitConfig`, `costs: CostImpactConfig`, `kelly: KellyConfig`, `walkforward: WalkForwardConfig`, `regression: RegressionConfig`, `bootstrap: BootstrapConfig`, `capacity: CapacityConfig`, `run_name: str = "base"`
  - Method `to_dict(self) -> Dict[str, Any]`: Serialize full config to a plain nested dict (via `dataclasses.asdict`).
- Function `deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]`: Recursively merges `updates` into `base` (nested dict merge) -> `Dict[str, Any]`: merged result.
- Function `validate_config(cfg: FTFConfig) -> None`: Validates critical invariants (allowed exec lag, z-clip ordering, walk-forward windows, vol target range, etc.) -> `None`.
- Constants/Types:
  - `DeriskPolicy`: `Literal["DERISK_HALF", "DERISK_CLOSE"]`
  - `StopFillPolicy`: `Literal["STOP_FILL_T_PLUS_1", "STOP_FILL_SAME_CLOSE"]`
  - `BaselineFloorMode`: `Literal["FLOOR_ON_WVOL", "FLOOR_ON_WCONF"]`
  - `PeakRef`: `Literal["close", "high"]`
  - `TrainerMode`: `Literal["FIXED", "GRID"]`
  - `StitchRule`: `Literal["FIRST_STEP_ONLY", "FULL_TEST_DIAGNOSTIC"]`
  - `__all__`: Explicit re-export list for the module’s public API.

**Internal Dependencies**
- From Python stdlib:
  - `dataclasses`: `dataclass`, `asdict`, `field` (define immutable config objects + serialization)
  - `typing`: `Any`, `Dict`, `Literal`, `Optional`, `Tuple` (type safety for config schema)

**External Dependencies**
- Expected to be imported by: walk-forward runner/trainer, signal/risk/sizing/trading/execution components, scripts that load YAML and need a typed config object (e.g., `scripts/02_run_fast_oos.py`), and reporting/stats modules that rely on NW/bootstrapping parameters.
- Key exports used elsewhere: `FTFConfig`, all sub-config dataclasses, `validate_config`, `deep_update`, and the `Literal` policy types (to constrain config options).

**Implementation Notes**
- Architecture decisions:
  - Uses `@dataclass(frozen=True)` everywhere to enforce immutability and reproducibility (critical for walk-forward “frozen params per anchor”).
  - Centralizes time/P&L conventions (`TimeConvention`) so downstream code can consistently implement T+1 execution and return attribution.
  - Provides minimal but essential validation in `validate_config` to fail fast on convention-breaking settings.
- Cross-File Relationships:
  - Downstream modules should read from `cfg.time.exec_lag`, `cfg.time.stop_fill_policy`, and `cfg.time.stitch_rule` to implement execution delay, stop timing, and OOS stitching rules consistently.
  - Column name fields in `DataConfig` let data loaders/validators standardize required inputs without hardcoding.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/__init__.py; ROUND 1 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:52:27
**File Implemented**: ftf_repro/src/ftf/__init__.py

**Core Purpose**
- Defines the top-level `ftf` package entrypoint and provides a high-level description of the project’s module layout.
- Re-exports the central configuration object so downstream code can import it directly from `ftf`.

**Public Interface**
- Class `FTFConfig`: Central configuration model for the pipeline (imported and re-exported here) | Key methods: *(not defined in this file)* | Constructor params: *(defined in `ftf.utils.config`)*  
- Constants/Types:
  - `__all__ = ["FTFConfig"]`: Controls the public API surface of `ftf` for `from ftf import *`.

**Internal Dependencies**
- From `ftf.utils.config`: `FTFConfig`
- External packages: None

**External Dependencies**
- Expected to be imported by: scripts and library consumers needing configuration access (e.g., `ftf_repro/scripts/*`, and any module doing `from ftf import FTFConfig`)
- Key exports used elsewhere: `FTFConfig` (top-level re-export)

**Implementation Notes**
- Architecture decisions: Keeps the package API minimal—only configuration is exposed at the package root to encourage consistent config handling across scripts/modules.
- Cross-File Relationships: Serves as the stable import path (`ftf.FTFConfig`) while the actual implementation resides in `ftf/utils/config.py`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/utils/__init__.py; ROUND 2 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:52:50
**File Implemented**: ftf_repro/src/ftf/utils/__init__.py

**Core Purpose**
- Provides the public entry point for the `ftf.utils` subpackage by re-exporting configuration dataclasses/enums, IO helpers, and seeding utilities behind a stable, dependency-light interface for reproducible experiments.

**Public Interface**
- Classes (re-exported):
  - Class `FTFConfig`: top-level experiment configuration container | Key methods: *(not defined here; re-export)* | Constructor params: *(defined in `utils/config.py`)*
  - Class `DataConfig`: data ingestion/alignment config | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `SignalConfig`: signal generation config | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `RiskConfig`: risk/vol targeting config | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `ATRExitConfig`: ATR/exit-state-machine config | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `CostImpactConfig`: transaction cost + impact settings | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `KellyConfig`: Kelly sizing settings | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `WalkForwardConfig`: walk-forward schedule/freezing settings | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `RegressionConfig`: benchmark regression/HAC settings | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `BootstrapConfig`: bootstrap settings | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
  - Class `CapacityConfig`: capacity/AUM mapping settings | Key methods: *(re-export)* | Constructor params: *(in `utils/config.py`)*
- Functions (re-exported):
  - Function `deep_update`: deep merge helper for nested config dicts -> *(return type defined in `utils/config.py`)*
  - Function `validate_config`: validates/normalizes configuration -> *(return type defined in `utils/config.py`)*
  - Function `ensure_dir`: create/check output directory -> *(return type defined in `utils/io.py`)*
  - Function `load_yaml`: load YAML config/data -> *(return type defined in `utils/io.py`)*
  - Function `save_yaml`: save YAML config/data -> *(return type defined in `utils/io.py`)*
  - Function `load_json`: load JSON -> *(return type defined in `utils/io.py`)*
  - Function `save_json`: save JSON -> *(return type defined in `utils/io.py`)*
  - Function `load_parquet`: load Parquet -> *(return type defined in `utils/io.py`)*
  - Function `save_parquet`: save Parquet -> *(return type defined in `utils/io.py`)*
  - Function `set_global_seed`: set deterministic seeds across libs -> *(return type defined in `utils/seed.py`)*
- Constants/Types (re-exported enums/typing-literals):
  - `TimeConvention`, `DeriskPolicy`, `StopFillPolicy`, `BaselineFloorMode`, `PeakRef`, `TrainerMode`, `StitchRule`: shared config enums used across the project.
- Module export control:
  - `__all__`: explicitly lists the supported public API for `from ftf.utils import *`.

**Internal Dependencies**
- From `ftf.utils.config`: `ATRExitConfig, BaselineFloorMode, BootstrapConfig, CapacityConfig, CostImpactConfig, DataConfig, DeriskPolicy, FTFConfig, KellyConfig, PeakRef, RegressionConfig, RiskConfig, SignalConfig, StopFillPolicy, StitchRule, TimeConvention, TrainerMode, WalkForwardConfig, deep_update, validate_config`
- From `ftf.utils.io`: `ensure_dir, load_json, load_parquet, load_yaml, save_json, save_parquet, save_yaml`
- From `ftf.utils.seed`: `set_global_seed`
- External packages: none directly (only via the re-exported modules).

**External Dependencies**
- Expected to be imported by: most pipeline modules (data loading, walk-forward runner, scripts) via `from ftf.utils import ...` to access config objects, IO, and seed control.
- Key exports used elsewhere: `FTFConfig`, `validate_config`, `load_yaml/save_yaml`, `load_parquet/save_parquet`, `set_global_seed`, and the various enum types (e.g., `TrainerMode`, `StitchRule`, `StopFillPolicy`).

**Implementation Notes**
- Architecture decisions: centralizes `utils` public API to keep downstream imports stable and minimal (single import surface), and enforces explicit export curation via `__all__`.
- Cross-File Relationships: acts as a façade over `utils/config.py`, `utils/io.py`, and `utils/seed.py`, re-exporting their primary interfaces so other packages don’t need to import those modules directly.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/data/calendar.py; ROUND 3 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:53:16
**File Implemented**: ftf_repro/src/ftf/data/calendar.py

**Core Purpose**
- Provides lightweight trading-calendar utilities to create and manipulate a canonical “NYSE-like” business-day index for aligning daily time series, using US Federal holidays as a proxy.

**Public Interface**
- Class `CalendarSpec`: Minimal calendar object to pass around (e.g., for roll logic) | Key methods: `bdays`, `shift`, `offset` | Constructor params: `name: str = "NYSE"`
  - `bdays(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex`: Generate business days (inclusive) using the module’s NYSE-like calendar.
  - `shift(self, date: pd.Timestamp, n: int) -> pd.Timestamp`: Shift a date by `n` business days.
  - `offset(self) -> CustomBusinessDay`: Return the underlying pandas business-day offset object.
- Function `to_date_index(idx: Iterable) -> pd.DatetimeIndex`: Converts an iterable of timestamps into a normalized (midnight), timezone-naive `DatetimeIndex`.
- Function `nyse_business_days(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex`: Generates a NYSE-like business-day index between `start` and `end` (inclusive); validates `end >= start`.
- Function `shift_bdays(date: pd.Timestamp, n: int) -> pd.Timestamp`: Shifts `date` by `n` NYSE-like business days using the module offset.
- Function `get_calendar(name: str = "NYSE") -> CalendarSpec`: Factory returning a `CalendarSpec`; currently only supports `"NYSE"` (case-insensitive) and errors otherwise.
- Function `infer_calendar_from_index(index: pd.DatetimeIndex) -> Optional[str]`: Best-effort diagnostic inference; returns `"NYSE"` if no weekend days appear and index length is sufficient, else `None`.
- Constants/Types:
  - `_US_BDAY: CustomBusinessDay`: `CustomBusinessDay(calendar=USFederalHolidayCalendar())` used as the core business-day frequency/offset.
  - `__all__`: Exports `CalendarSpec`, `get_calendar`, `infer_calendar_from_index`, `nyse_business_days`, `shift_bdays`, `to_date_index`.

**Internal Dependencies**
- From `dataclasses`: `dataclass`
- From `typing`: `Iterable`, `Optional`
- From `pandas.tseries.holiday`: `USFederalHolidayCalendar`
- From `pandas.tseries.offsets`: `CustomBusinessDay`
- External packages:
  - `pandas` - builds/normalizes `DatetimeIndex`, date ranges, and business-day offsets.

**External Dependencies**
- Expected to be imported by: `ftf_repro/src/ftf/data/futures_roll.py`, `ftf_repro/src/ftf/data/loaders.py`, `ftf_repro/src/ftf/data/validation.py` (and any walk-forward scheduling code needing business-day shifting).
- Key exports used elsewhere: `CalendarSpec`, `get_calendar`, `nyse_business_days`, `shift_bdays`, `to_date_index`.

**Implementation Notes**
- Architecture decisions: Uses `USFederalHolidayCalendar` as a pragmatic approximation of NYSE holidays to avoid extra dependencies (explicitly noted in docstring); all dates are normalized to midnight and made timezone-naive for consistent daily indexing.
- Cross-File Relationships: `CalendarSpec` is designed to be passed into roll/calendar-sensitive logic (e.g., “roll 2 business days before FND”), ensuring a consistent definition of “business day” across the pipeline.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/data/loaders.py; ROUND 4 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:53:56
**File Implemented**: ftf_repro/src/ftf/data/loaders.py

**Core Purpose**:
- Provides vendor-agnostic utilities to load daily futures/spot data from CSV/Parquet and align it to a NYSE-like business-day calendar with strict index validation and controlled forward-filling (prices only).

**Public Interface**:
- Function `validate_daily_index(df: pd.DataFrame, *, name: str = "data")`: Validates that `df.index` is a timezone-naive, monotonic increasing `DatetimeIndex` with no duplicates -> `None` (raises `ValueError` on violations).
- Function `infer_date_col(columns: Iterable[str])`: Attempts to infer a date column name from common candidates (`date`, `timestamp`, `time`, `dt`) -> `Optional[str]`.
- Function `align_ohlc_to_calendar(df: pd.DataFrame, *, calendar: CalendarSpec, start: Optional[str | pd.Timestamp] = None, end: Optional[str | pd.Timestamp] = None, ffill_price_cols: Tuple[str, ...] = ("open","high","low","close"))`: Reindexes to business days and forward-fills only specified price columns -> `pd.DataFrame`.
- Function `read_contract_ohlc(path: str | Path, *, cfg: Optional[DataConfig] = None, date_col: Optional[str] = None, calendar_name: str = "NYSE")`: Loads a single-contract OHLC(+optional) table, normalizes date index and price column, aligns to business days -> `pd.DataFrame`.
- Function `read_contract_metadata(path: str | Path, *, date_col: str = "fnd")`: Loads contract metadata with required `contract` and first-notice-date column; returns indexed by contract -> `pd.DataFrame`.
- Function `read_lbma_spot(path: str | Path, *, date_col: Optional[str] = None, price_col: str = "price", calendar_name: str = "NYSE")`: Loads spot price series, aligns to business days, forward-fills price -> `pd.Series`.
- Constants/Types:
  - `__all__`: Exposes `read_contract_ohlc`, `read_contract_metadata`, `read_lbma_spot`, `align_ohlc_to_calendar`, `validate_daily_index`, `infer_date_col`.

**Internal Dependencies**:
- From `ftf.data.calendar`: `CalendarSpec`, `get_calendar`, `to_date_index` (note: `to_date_index` is imported but not used in this file).
- From `ftf.utils.config`: `DataConfig` (used to specify canonical column names like `price_col`, `open_col`, etc.).
- External packages:
  - `pandas` - core IO (`read_csv`, `read_parquet`), datetime parsing/normalization, reindexing to business-day calendars, and forward-filling.

**External Dependencies**:
- Expected to be imported by: data preparation/build scripts (e.g., `scripts/01_build_data.py`), continuous futures builder (`src/ftf/data/futures_roll.py`), and validation modules (`src/ftf/data/validation.py`) to ensure consistent ingestion and calendar alignment.
- Key exports used elsewhere: `read_contract_ohlc`, `read_contract_metadata`, `read_lbma_spot`, `align_ohlc_to_calendar`, `validate_daily_index`.

**Implementation Notes**:
- Architecture decisions:
  - Vendor-format neutrality: supports CSV/Parquet via `_read_table` and infers date column where possible.
  - Strict daily index invariants enforced centrally (`validate_daily_index`) before/after calendar alignment to prevent subtle downstream lookahead/duplication issues.
  - Forward-fill policy is deliberately limited to price columns only; returns are expected to be computed later from filled close/settle.
- Cross-File Relationships:
  - Uses `get_calendar("NYSE")` from `ftf.data.calendar` to generate a business-day index (`calendar.bdays(start, end)`), then reindexes/ffills.
  - `DataConfig` supplies canonical column names and drives normalization/compatibility with multiple vendor schemas (e.g., mapping `settle`/`Close` to configured `price_col`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/data/futures_roll.py; ROUND 5 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:54:40
**File Implemented**: ftf_repro/src/ftf/data/futures_roll.py

**Core Purpose**
- Constructs a spliced (non-back-adjusted) continuous front-month futures time series (OHLC/volume/ADV) from per-contract daily bars, rolling a fixed number of business days before each contract’s first notice date (FND).
- Produces the continuous DataFrame plus diagnostics: the active contract per date and an explicit roll table.

**Public Interface**
- Class `ContinuousFuturesResult`: Immutable container for continuous futures build outputs | Key methods: *(dataclass; no methods)* | Constructor params: `df_cont: pd.DataFrame`, `active_contract: pd.Series`, `roll_table: pd.DataFrame`
- Function `determine_active_contract(date: pd.Timestamp, contracts_sorted: Iterable[str], fnd_by_contract: pd.Series, *, calendar: CalendarSpec, roll_bd_before_fnd: int)`: Selects the active (front) contract on a given business date using rule `date < shift(FND, -roll_bd_before_fnd)` -> `Optional[str]`: active contract symbol or `None` if none eligible
- Function `build_continuous_front_month(contract_bars: Dict[str, pd.DataFrame], contract_meta: pd.DataFrame, *, cfg: Optional[DataConfig] = None, calendar_name: str = "NYSE", start: Optional[str | pd.Timestamp] = None, end: Optional[str | pd.Timestamp] = None)`: Builds spliced continuous series and roll diagnostics -> `ContinuousFuturesResult`: contains `df_cont`, `active_contract`, `roll_table`
- Constants/Types:
  - `__all__ = ["ContinuousFuturesResult", "build_continuous_front_month", "determine_active_contract"]`

**Internal Dependencies**
- From `ftf.data.calendar`: `CalendarSpec`, `get_calendar` (business-day generation and business-day shifting for roll cutoff)
- From `ftf.data.loaders`: `validate_daily_index` (asserts input contract bars have valid daily DatetimeIndex)
- From `ftf.utils.config`: `DataConfig` (column name conventions and `roll_bd_before_fnd`)
- External packages:
  - `pandas` - DataFrame/Series manipulation, indexing, reindexing, forward-fill, roll table construction
  - `numpy` - minor array conversion (`to_numpy`) for masks/assignment

**External Dependencies**
- Expected to be imported by: `ftf_repro/scripts/01_build_data.py`, `ftf_repro/src/ftf/data/validation.py`, and downstream pipeline components that require continuous prices (signals/risk/trading engine).
- Key exports used elsewhere: `build_continuous_front_month`, `determine_active_contract`, `ContinuousFuturesResult`

**Implementation Notes**
- Architecture decisions:
  - Uses *splicing* (no back-adjustment): continuous price series is assembled by switching the source contract on roll dates; roll P&L is implicitly captured by close-to-close returns on the spliced series.
  - Roll rule is strict: contract is eligible only when `date < (FND - roll_bd_before_fnd business days)`; selection is “nearest” by iterating contracts sorted by ascending FND.
  - Builds a union business-day calendar (`NYSE` default) across all contract indices, optionally clipped by `start`/`end`.
- Cross-File Relationships:
  - Relies on `get_calendar(...).bdays(...)` and `calendar.shift(...)` to ensure roll timing is based on business days (not calendar days).
  - Assumes upstream loader alignment already handled contract-level missing data; only performs a safety `ffill()` on continuous OHLC price columns after splicing (volume/ADV not forward-filled).
  - Column selection is driven by `DataConfig` (`open_col`, `high_col`, `low_col`, `price_col`, `volume_col`, `adv_col`), enabling consistent downstream consumption.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/data/validation.py; ROUND 6 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:55:17
**File Implemented**: ftf_repro/src/ftf/data/validation.py

**Core Purpose**:
- Provides deterministic validation utilities for prepared/continuous futures datasets, focusing on roll-rule compliance, calendar/index sanity, NaN checks after warmup, and basic distribution summaries for returns and ATR(14).

**Public Interface**:
- Class `ContinuousValidationReport`: Immutable container for validation outcomes and summary stats | Key methods: (dataclass; no methods) | Constructor params: `ok, n_days, n_rolls, roll_rule_violations, nan_after_warmup, ret_summary, atr_summary`
- Function `_require_cols(df: pd.DataFrame, cols: Iterable[str])`: (internal helper) ensure required columns exist; raises `ValueError` if missing -> `None`
- Function `compute_returns_from_close(close: pd.Series)`: compute close-to-close simple returns via `pct_change()` with inf-to-NaN cleanup -> `pd.Series`: daily returns
- Function `compute_atr14(df: pd.DataFrame, *, high_col: str = "high", low_col: str = "low", close_col: str = "close")`: compute ATR(14) as a simple rolling mean of True Range (TR) with `min_periods=14` -> `pd.Series`: ATR values (NaN until day 14)
- Function `validate_roll_rule(active_contract: pd.Series, fnd_by_contract: pd.Series, *, calendar: Optional[CalendarSpec] = None, roll_bd_before_fnd: int = 2)`: verify active contract never persists on/after the cutoff date defined as `FND shifted by -roll_bd_before_fnd` business days -> `Tuple[int, pd.DataFrame]`: (violation_count, violations_df with columns `date, contract, fnd, cutoff`)
- Function `validate_continuous_df(df_cont: pd.DataFrame, active_contract: Optional[pd.Series] = None, fnd_by_contract: Optional[pd.Series] = None, *, cfg: Optional[DataConfig] = None, warmup_bd: int = 60, calendar_name: str = "NYSE")`: run full validation suite (index checks, weekend check, required columns, returns/ATR summaries, NaNs-after-warmup, optional roll-rule check) -> `ContinuousValidationReport`

**Internal Dependencies**:
- From `ftf.data.calendar`: `CalendarSpec`, `get_calendar` (business-day shifting for roll cutoff; calendar selection by name)
- From `ftf.utils.config`: `DataConfig` (column names and `roll_bd_before_fnd` configuration)
- External packages:
  - `pandas` - time series indexing, rolling computations, DataFrame/Series ops, quantiles
  - `numpy` - inf constants for returns cleanup
  - `dataclasses` - `@dataclass(frozen=True)` report object
  - `typing` - type annotations (`Optional`, `Tuple`, etc.)

**External Dependencies**:
- Expected to be imported by: data build scripts (e.g., `scripts/01_build_data.py`), continuous futures construction workflows (`ftf.data.futures_roll`), and tests that assert roll/no-lookahead/data sanity (e.g., `tests/test_roll.py`, broader pipeline tests).
- Key exports used elsewhere: `validate_continuous_df`, `validate_roll_rule`, `compute_atr14`, `compute_returns_from_close`, `ContinuousValidationReport`

**Implementation Notes**:
- Architecture decisions:
  - Keeps validations “light-weight” and deterministic (no randomness; simple summaries).
  - Enforces strict time-series hygiene early (DatetimeIndex, monotonic, no duplicates, and no weekend dates).
  - ATR(14) matches the plan: TR computed from (H-L, |H-C_prev|, |L-C_prev|) and averaged (not Wilder smoothing).
- Cross-File Relationships:
  - Uses `DataConfig` to avoid hardcoding column names and roll rule parameters, keeping validation consistent with loader/roll builder outputs.
  - Uses `CalendarSpec.shift()` to implement the “2 business days before FND” roll cutoff check, aligning with the continuous futures roll logic.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/signals/ema_trend.py; ROUND 7 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:55:49
**File Implemented**: ftf_repro/src/ftf/signals/ema_trend.py

**Core Purpose**:
- Computes the paper’s EMA-trend feature at decision time: EMA of log-price → EMA slope → z-score standardized using training-window stats → clipped z mapped into a probability-like signal `p_trend ∈ [0,1]`.
- Provides both (a) a train-window fitting routine that freezes parameters for walk-forward anchors and (b) an inference routine to compute per-day features.

**Public Interface**:
- Class **EMATrendState**: Frozen EMA-trend parameters for walk-forward anchors | Key methods: *(dataclass; no methods)* | Constructor params: `ema_lambda: float, slope_mu: float, slope_sigma: float, z_clip: Tuple[float, float]=(-3.0, 3.0)`
- Function **ema_log_price**(`close: pd.Series, *, ema_lambda: float`): Compute EMA of log-price via explicit recursion -> `pd.Series`: returns `ema_log` series aligned to `close.index`
- Function **ema_slope_from_ema**(`ema_log: pd.Series`): Compute slope proxy `Δỹ_t = ỹ_t - ỹ_{t-1}` -> `pd.Series`: returns `ema_slope` (diff) series
- Function **fit_ema_trend_state**(`close_train: pd.Series, *, cfg: Optional[SignalConfig]=None, ema_lambda: Optional[float]=None`): Fit training distribution parameters (μ, σ) for EMA slope and produce a frozen state -> `EMATrendState`
- Function **compute_p_trend**(`close: pd.Series, *, state: EMATrendState`): Compute full EMA-trend feature set -> `pd.DataFrame`: columns `ema_log`, `slope`, `z`, `z_clipped`, `p_trend`
- Constants/Types:
  - `__all__`: exports `EMATrendState`, `ema_log_price`, `ema_slope_from_ema`, `fit_ema_trend_state`, `compute_p_trend`

**Internal Dependencies**:
- From `ftf.utils.config`: `SignalConfig` (provides defaults like `ema_lambda` and `z_clip`)
- External packages:
  - `numpy` - log transform, array recursion, mean/std estimation
  - `pandas` - Series/DataFrame handling, index validation, diff/clip/concat operations
  - `dataclasses` - immutable frozen parameter container (`EMATrendState`)
  - `typing` - `Optional`, `Tuple` for type annotations

**External Dependencies**:
- Expected to be imported by: walk-forward training/runner code and signal composition code, e.g. `ftf/walkforward/trainer.py`, `ftf/walkforward/runner.py`, and higher-level signal modules like `ftf/signals/regime.py` / `ftf/signals/__init__.py`.
- Key exports used elsewhere: `fit_ema_trend_state` (train-only parameter freezing) and `compute_p_trend` (OOS feature generation), plus `EMATrendState` for serialization/transport across anchors.

**Implementation Notes**:
- Architecture decisions:
  - Uses manual EMA recursion instead of `pandas.Series.ewm` to eliminate initialization ambiguity and improve determinism for unit tests.
  - Enforces `DatetimeIndex` on inputs to prevent silent misalignment issues.
  - Training fit computes μ and σ from `slope.dropna()`; uses `ddof=0` for population std.
  - Protects against degenerate/flat training data by flooring `slope_sigma` to `1e-12`, which collapses `p_trend` toward ~0.5 rather than exploding.
  - Applies two-stage clipping: first to `z_clip` (default `[-3, 3]`), then ensures `p_trend` is within `[0, 1]`.
- Cross-File Relationships:
  - This module is explicitly “decision-time” and does not apply latency; execution lag is expected to be applied downstream in execution/trading components.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/signals/momentum.py; ROUND 8 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:56:09
**File Implemented**: ftf_repro/src/ftf/signals/momentum.py

**Core Purpose**:
- Provides a simple “momentum confirmation” signal for the pipeline: a binary indicator of whether today’s close is above the close `K` business days ago, with a frozen state object to lock the lookback used across walk-forward anchors.

**Public Interface**:
- Class `MomentumState`: Frozen container for momentum parameters (lookback) | Key methods: (dataclass; no methods) | Constructor params: `k: int = 50`
- Function `compute_momentum_indicator(close: pd.Series, *, k: int)`: Computes `m_t = 1{P_t / P_{t-K} > 1}` with NaN for first `k` observations -> `pd.Series`: `{0.0, 1.0}` plus NaNs
- Function `fit_momentum_state(*, cfg: Optional[SignalConfig] = None, k: Optional[int] = None)`: Creates a frozen `MomentumState` from config or override -> `MomentumState`
- Function `compute_momentum(close: pd.Series, *, state: MomentumState)`: Convenience wrapper using frozen state -> `pd.Series`
- Constants/Types:
  - `__all__`: exports `MomentumState`, `fit_momentum_state`, `compute_momentum_indicator`, `compute_momentum`

**Internal Dependencies**:
- From `ftf.utils.config`: `SignalConfig` (reads `momentum_k` default)
- External packages:
  - `pandas` (`pd.Series`, `DatetimeIndex`, shifting / alignment)
  - `numpy` (sets NaNs via `np.nan`)
  - `dataclasses` (`@dataclass(frozen=True)` for immutable state)
  - `typing` (`Optional`)

**External Dependencies**:
- Expected to be imported by: `ftf/signals/regime.py` (blending with EMA-trend), `ftf/trading/engine.py` or walk-forward trainer/runner modules for signal generation.
- Key exports used elsewhere: `fit_momentum_state` (to freeze per-anchor params), `compute_momentum` / `compute_momentum_indicator` (to generate the momentum series).

**Implementation Notes**:
- Architecture decisions:
  - Separates “fit” (parameter freezing) from “compute” (pure transformation), matching walk-forward freeze protocols.
  - Enforces time/index correctness: requires `close` be `DatetimeIndex`-indexed; validates `k > 0`.
  - Produces NaN for insufficient history (`t < k`) rather than forcing 0/0.5, leaving filling policy to callers.
- Cross-File Relationships:
  - Intended to complement `ftf.signals.ema_trend` by supplying the `m_t` term in the blended bull probability; `SignalConfig` provides default `momentum_k`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/signals/regime.py; ROUND 9 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:56:38
**File Implemented**: ftf_repro/src/ftf/signals/regime.py

**Core Purpose**:
- Provides a causal, decision-time “regime” signal composition layer that combines EMA-slope trend probability and K-day momentum into a blended bull probability, an entry eligibility flag, and optional regime labels for attribution.

**Public Interface**:
- Class `RegimeState`: Frozen per-anchor regime parameters bundling fitted EMA-trend state, momentum state, and blending/threshold hyperparameters | Key methods: *(dataclass; no methods)* | Constructor params: `ema_state: EMATrendState, mom_state: MomentumState, blend_omega: float, pbull_threshold: float`
- Function `fit_regime_state(close_train: pd.Series, *, cfg: Optional[SignalConfig] = None, ema_lambda: Optional[float] = None, blend_omega: Optional[float] = None, pbull_threshold: Optional[float] = None, momentum_k: Optional[int] = None)`: Fit/freeze regime parameters on training data (EMA slope distribution via `fit_ema_trend_state`, momentum K via `fit_momentum_state`, validate omega/threshold) -> `RegimeState`: frozen parameters for walk-forward anchor
- Function `label_regime(p_bull: pd.Series)`: Map bull probability to categorical labels (`"bull"`, `"bear"`, `"chop"`) using fixed cutoffs (0.55/0.45) -> `pd.Series`: object dtype labels indexed by the same `DatetimeIndex`
- Function `compute_regime_features(close: pd.Series, *, state: RegimeState)`: Compute trend features, momentum, blended probabilities, entry gate, and regime labels -> `pd.DataFrame`: columns `ema_log, slope, z, z_clipped, p_trend, momentum, p_bull, p_bear, eligible_to_enter, regime`
- Constants/Types:
  - `__all__ = ["RegimeState", "fit_regime_state", "compute_regime_features", "label_regime"]`

**Internal Dependencies**:
- From `ftf.signals.ema_trend`: `EMATrendState`, `compute_p_trend`, `fit_ema_trend_state`
- From `ftf.signals.momentum`: `MomentumState`, `compute_momentum`, `fit_momentum_state`
- From `ftf.utils.config`: `SignalConfig`
- External packages:
  - `pandas` - Series/DataFrame inputs/outputs, index validation, concatenation
  - `numpy` - `np.nan` assignment for eligibility during warmup/NaN regions
  - `dataclasses` - frozen state container (`@dataclass(frozen=True)`)

**External Dependencies**:
- Expected to be imported by: `ftf/trading/engine.py`, `ftf/walkforward/trainer.py`, `ftf/walkforward/runner.py` (for per-anchor fitting and OOS feature generation), and reporting modules for regime attribution.
- Key exports used elsewhere: `RegimeState`, `fit_regime_state`, `compute_regime_features`

**Implementation Notes**:
- Architecture decisions:
  - Separates *fitting/freezing* (`fit_regime_state`) from *feature generation* (`compute_regime_features`) to enforce walk-forward train-only estimation.
  - Ensures causality by operating purely on the `close` series at time *t* (no forward references); execution latency is explicitly deferred to execution modules.
  - Eligibility is returned as float with NaNs preserved during warmup (momentum K window and first EMA-slope day), avoiding false binary signals early.
- Cross-File Relationships:
  - Delegates EMA slope standardization and probability mapping to `ema_trend.compute_p_trend` (using frozen `EMATrendState`).
  - Delegates momentum computation to `momentum.compute_momentum` (using frozen `MomentumState`).
  - Uses `SignalConfig` defaults when overrides are not provided; validates `blend_omega` and `pbull_threshold` in [0, 1].
  - Adds regime labeling for attribution as a lightweight post-processing step (fixed thresholds independent of config/state).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/risk/ewma_vol.py; ROUND 10 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:57:10
**File Implemented**: ftf_repro/src/ftf/risk/ewma_vol.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Implements an EWMA variance forecaster and converts that forecast into a deterministic volatility-targeting position weight, following a strict “no lookahead” convention (uses \(r_t\) to forecast \(\sigma^2_{t+1}\) and size at time \(t\)).

**Public Interface** (what other files can use, if any):
- Class `EWMAVolState`: Frozen per-anchor risk state containing EWMA decay and initialization variance, plus vol-targeting parameters | Key methods: `sigma_target_daily` (property) | Constructor params: `theta: float`, `sigma2_init: float`, `vol_target_annual: float = 0.15`, `w_max: float = 2.0`
- Function `fit_ewma_vol_state(r_train: pd.Series, *, cfg: Optional[RiskConfig] = None, theta: Optional[float] = None)`: Fit/freeze EWMA state from training returns (train-only) -> `EWMAVolState`: computes `sigma2_init = Var_train(r)` and pulls defaults from `RiskConfig`
- Function `ewma_variance_forecast(r: pd.Series, *, state: EWMAVolState)`: Compute EWMA next-day variance forecast series \(\sigma^2_{t+1}\) indexed like `r` -> `pd.Series`: returns `"ewma_sigma2_next"`
- Function `vol_target_weight(r: pd.Series, *, state: EWMAVolState)`: Compute volatility targeting weights \(w_{vol}(t)\) using \(\sigma^2_{t+1}\) -> `pd.Series`: returns `"w_vol"`
- Constants/Types: `__all__` exports `["EWMAVolState", "fit_ewma_vol_state", "ewma_variance_forecast", "vol_target_weight"]`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.utils.config`: `RiskConfig` (supplies defaults: `ewma_theta`, `vol_target_annual`, `w_max`)
- External packages:
  - `numpy` - variance computation, sqrt, clipping floors, array loop for EWMA recursion
  - `pandas` - Series I/O, DatetimeIndex validation, returning forecast/weight series

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: walk-forward training/runner components and sizing/policy assembly (e.g., `ftf/walkforward/trainer.py`, `ftf/sizing/policy_weight.py`, `ftf/trading/engine.py`)
- Key exports used elsewhere: `fit_ewma_vol_state` (freeze per-anchor risk params), `vol_target_weight` (daily risk budget weight), `EWMAVolState` (state serialization/transport between train/test)

**Implementation Notes**: (if any)
- Architecture decisions:
  - Enforces strict no-lookahead by outputting \(\sigma^2_{t+1}\) aligned to index date \(t\) (forecast computed using only \(r_t\) and prior state).
  - Deterministic, minimal state: only `theta` and `sigma2_init` are required; targeting uses annual vol converted to daily via `sqrt(252)`.
  - Robustness features: validates `pd.Series` with `DatetimeIndex`, floors variance at `1e-18` to avoid division-by-zero, and treats NaN returns as “missing” by carrying forward the prior forecast.
- Cross-File Relationships:
  - `RiskConfig` centralizes risk hyperparameters; `fit_ewma_vol_state` allows per-anchor freezing consistent with walk-forward protocols.
  - Output `w_vol` is intended to be multiplied by confidence/Kelly sizing layers downstream.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/risk/confidence.py; ROUND 11 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:57:35
**File Implemented**: ftf_repro/src/ftf/risk/confidence.py

**Core Purpose**
- Implements the deterministic “confidence shaping” layer that converts a blended bull probability series (`p_bull`) into a [0,1] confidence share and then scales a volatility-target weight (`w_vol`) into a confidence-shaped weight (`w_conf`).

**Public Interface**
- Class `ConfidenceState`: Frozen parameters controlling the linear mapping from `p_bull` to confidence share (pivot/scale). | Key methods: *(dataclass; no methods)* | Constructor params: `pivot: float = 0.5`, `scale: float = 0.5`
- Function `fit_confidence_state()`: Returns the canonical (non-fitted) confidence state -> `ConfidenceState`: Provides a standard state object for symmetry with other fitted components.
- Function `confidence_share(p_bull: pd.Series, *, state: Optional[ConfidenceState] = None)`: Maps `p_bull` in [0,1] to `conf_share` in [0,1] with NaNs preserved -> `pd.Series`: Linear transform `(p_bull - pivot)/scale`, clipped to [0,1].
- Function `confidence_weight(w_vol: pd.Series, p_bull: pd.Series, *, state: Optional[ConfidenceState] = None)`: Computes `w_conf = w_vol * conf_share` -> `pd.Series`: Aligns inputs on the inner date intersection, propagates NaNs, clips negative outputs to 0, and normalizes “-0.0” to 0.0.
- Constants/Types:
  - `__all__`: Exposes `ConfidenceState`, `fit_confidence_state`, `confidence_share`, `confidence_weight`.

**Internal Dependencies**
- From standard library: `dataclasses.dataclass` (define immutable state), `typing.Optional` (optional state injection).
- External packages:
  - `pandas` - Series/DatetimeIndex validation, alignment (`align`), clipping, naming outputs.
  - `numpy` - numeric cleanup via `np.isclose` to avoid `-0.0`.

**External Dependencies**
- Expected to be imported by: sizing and/or trading pipeline components that combine `p_bull` with volatility targeting, e.g. `ftf_repro/src/ftf/sizing/policy_weight.py`, and potentially the walk-forward runner/engine once they assemble daily features and weights.
- Key exports used elsewhere: `confidence_share`, `confidence_weight`, `ConfidenceState` (and `fit_confidence_state` for consistent construction).

**Implementation Notes**
- Architecture decisions:
  - Uses a small immutable `ConfidenceState` even though parameters are fixed, enabling future extensibility (e.g., nonlinear shaping) while keeping a consistent “fit → state → apply” pattern.
  - Strict input validation: requires `pd.Series` with `pd.DatetimeIndex`; enforces `state.scale > 0`.
  - Deterministic alignment behavior: `confidence_weight` inner-joins indices to avoid accidental forward-filling or misalignment across pipeline stages.
- Cross-File Relationships:
  - Intended to sit between signal generation (`p_bull`) and sizing/weight construction (`w_vol` → `w_conf`), before later components apply Kelly scaling, caps/floors, and execution lag/costs.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/sizing/kelly.py; ROUND 12 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:58:14
**File Implemented**: ftf_repro/src/ftf/sizing/kelly.py

**Core Purpose**:
- Provides friction- and impact-adjusted Kelly sizing utilities, including estimating Kelly inputs from a training return series and computing a closed-form optimal Kelly fraction under linear costs and concave market impact.

**Public Interface**:
- Class `KellyInputs`: Frozen container for Kelly estimation inputs (`mu`, `sigma2`, `n`) | Key methods: *(dataclass; none)* | Constructor params: `mu: float, sigma2: float, n: float = 1.0`
- Function `estimate_kelly_inputs(R_train: pd.Series, *, n: float = 1.0)`: Estimate mean/variance of unit-notional sleeve returns on training data -> `KellyInputs`: returns frozen inputs for sizing
- Function `growth_proxy(f: np.ndarray | float, *, inputs: KellyInputs, k_linear: float, gamma_impact: float)`: Compute reduced-form growth proxy \(g(f)\) including linear costs and impact -> `np.ndarray | float`: growth values for scalar/array `f`
- Function `solve_friction_adjusted_kelly(*, inputs: KellyInputs, costs: Optional[CostImpactConfig] = None, k_linear: Optional[float] = None, gamma_impact: Optional[float] = None)`: Closed-form optimizer for friction-adjusted Kelly \(f^*\) -> `float`: optimal nonnegative Kelly fraction
- Function `fractional_kelly(f_star: float, *, kelly_cfg: Optional[KellyConfig] = None, lambda_kelly: Optional[float] = None)`: Apply fractional Kelly scaling \( \tilde f = \lambda f^* \) -> `float`: scaled nonnegative Kelly fraction
- Constants/Types:
  - `__all__`: exports `KellyInputs`, `estimate_kelly_inputs`, `growth_proxy`, `solve_friction_adjusted_kelly`, `fractional_kelly`

**Internal Dependencies**:
- From `ftf.utils.config`: `CostImpactConfig`, `KellyConfig`
- External packages:
  - `numpy` - numeric computation, quadratic root, array handling, safeguards
  - `pandas` - input type for training return series (`pd.Series`)
  - `dataclasses` - immutable `KellyInputs` container
  - `typing` - `Optional`

**External Dependencies**:
- Expected to be imported by: walk-forward training/orchestration and sizing composition modules, e.g. `ftf/walkforward/trainer.py` (to estimate/freeze Kelly inputs) and `ftf/sizing/policy_weight.py` (to convert Kelly fraction into final target weights).
- Key exports used elsewhere: `estimate_kelly_inputs`, `solve_friction_adjusted_kelly`, `fractional_kelly`, and `KellyInputs`

**Implementation Notes**:
- Architecture decisions:
  - Kept independent of the trading engine; operates purely on summary statistics (`mu`, `sigma2`) of a precomputed unit-notional sleeve return series.
  - Implements the plan’s closed-form solution using substitution \(x=\sqrt{f}\) and taking the positive root; clamps to nonnegative and guards numerical issues.
- Cross-File Relationships:
  - Training pipeline should generate the unit-notional sleeve return series and call `estimate_kelly_inputs(...)`, then call `solve_friction_adjusted_kelly(...)` with `CostImpactConfig` (or overrides) and finally scale via `fractional_kelly(...)` using `KellyConfig.lambda_kelly`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/sizing/policy_weight.py; ROUND 13 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:58:54
**File Implemented**: ftf_repro/src/ftf/sizing/policy_weight.py

**Core Purpose**:
- Constructs a *raw* (pre-gating, pre-latency, pre-cost) long-only target weight series by combining volatility targeting (`w_vol`), confidence shaping from `p_bull`, and a (frozen) fractional Kelly scalar `f_tilde`, with optional “baseline floor” behavior when `f_tilde` is effectively zero.

**Public Interface**:
- Class `PolicyWeightState`: Frozen parameters controlling caps and baseline-floor behavior | Key methods: *(none; dataclass container)* | Constructor params: `w_max: float = 2.0`, `baseline_floor: float = 0.25`, `baseline_floor_mode: BaselineFloorMode = "FLOOR_ON_WVOL"`, `baseline_floor_eps: float = 1e-6`
- Function `fit_policy_weight_state(*, risk_cfg: Optional[RiskConfig] = None, kelly_cfg: Optional[KellyConfig] = None, w_max: Optional[float] = None)`: Builds/validates a `PolicyWeightState` from config objects and optional overrides -> `PolicyWeightState`: returns a validated frozen state object for later reuse in walk-forward “fit/freeze/apply”.
- Function `compute_w_raw(*, w_vol: pd.Series, p_bull: pd.Series, f_tilde: float, state: Optional[PolicyWeightState] = None)`: Computes `w_raw(t)` series aligned to inputs -> `pd.Series`: nonnegative weight series capped at `state.w_max`, preserving NaNs from upstream signals.
- Constants/Types:
  - `__all__ = ["PolicyWeightState", "fit_policy_weight_state", "compute_w_raw"]`: explicit export surface.
  - `BaselineFloorMode` (imported type): string-like mode selecting floor behavior (`"FLOOR_ON_WVOL"` or `"FLOOR_ON_WCONF"`).

**Internal Dependencies**:
- From `ftf.risk.confidence`: `confidence_weight` (computes `w_conf = w_vol * conf_share(p_bull)` with index alignment handled there)
- From `ftf.utils.config`: `BaselineFloorMode`, `KellyConfig`, `RiskConfig` (config dataclasses/types providing defaults for `w_max`, baseline floor parameters)
- External packages:
  - `numpy` - numeric validation (`isfinite`), zero-cleanup (`isclose`)
  - `pandas` - Series/DatetimeIndex validation, alignment (`reindex`), masking, clipping, naming

**External Dependencies**:
- Expected to be imported by: walk-forward training/runner and trading pipeline components that need the raw sizing series before applying entry/exit gating and execution (likely `ftf/trading/engine.py`, `ftf/walkforward/trainer.py`, `ftf/walkforward/runner.py`).
- Key exports used elsewhere: `compute_w_raw` for daily raw weight construction; `fit_policy_weight_state` for per-anchor parameter freezing; `PolicyWeightState` for serialization/storage of sizing policy knobs.

**Implementation Notes**:
- Architecture decisions:
  - Separates sizing (`w_raw`) from trading logic (entry/exit gating), latency, and costs to keep modules composable and avoid look-ahead coupling.
  - Enforces long-only behavior by zeroing negative `f_tilde` (treated as `0.0`) and clipping final weights to `[0, w_max]`.
  - Implements configurable “baseline floor” when `f_tilde < baseline_floor_eps`:
    - `"FLOOR_ON_WVOL"` uses `baseline_floor * w_vol` (reindexed to output index)
    - `"FLOOR_ON_WCONF"` uses `baseline_floor * w_conf`
  - Preserves NaNs: final `w_raw` is masked wherever either `w_vol` or `p_bull` is NaN after alignment.
  - Normalizes `-0.0` to `0.0` for cleaner downstream logs/tests.
- Cross-File Relationships:
  - Relies on `confidence_weight()` to implement the confidence shaping mapping from `p_bull` into a scaling factor applied to `w_vol`.
  - Intended to multiply with a *frozen* `f_tilde` produced by the Kelly sizing module, and then be gated/lagged/costed in trading/execution modules.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/trading/engine.py; ROUND 14 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 20:59:50
**File Implemented**: ftf_repro/src/ftf/trading/engine.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Implements the deterministic, end-to-end daily trading engine that converts continuous futures OHLC data into executed weights and daily gross/net returns under the paper’s forecast-to-fill (T+lag) convention.
- Orchestrates signal computation, ATR exits, volatility targeting, Kelly/confidence sizing, execution lagging, and cost/impact accounting, emitting a full daily diagnostics DataFrame and structured event log.

**Public Interface** (what other files can use, if any):
- Class `EngineResult`: Container for engine outputs (`daily` DataFrame + `log`) | Key methods: *(dataclass, no methods)* | Constructor params: `daily: pd.DataFrame`, `log: TradingLog`
- Function `_compute_returns(close: pd.Series)`: Compute close-to-close simple returns with inf→NaN handling -> `pd.Series`
- Function `run_engine(df_cont: pd.DataFrame, *, cfg: FTFConfig, regime_state: RegimeState, vol_state: EWMAVolState, policy_state: PolicyWeightState, f_tilde: float, start: Optional[pd.Timestamp]=None, end: Optional[pd.Timestamp]=None, exit_state: Optional[ATRExitState]=None, metadata: Optional[Dict[str, Any]]=None)`: Runs full pipeline, producing daily fields + trade event log -> `EngineResult`
- Constants/Types:
  - `__all__ = ["EngineResult", "run_engine"]`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.execution.costs`: `compute_costs`
- From `ftf.execution.fills`: `apply_exec_lag`
- From `ftf.risk.ewma_vol`: `EWMAVolState`, `ewma_variance_forecast`, `vol_target_weight`
- From `ftf.signals.regime`: `RegimeState`, `compute_regime_features`
- From `ftf.sizing.policy_weight`: `PolicyWeightState`, `compute_w_raw`
- From `ftf.trading.atr`: `compute_atr`
- From `ftf.trading.exits`: `ATRExitState`, `generate_target_weights`
- From `ftf.trading.logs`: `TradingLog`
- From `ftf.utils.config`: `FTFConfig`
- External packages:
  - `pandas` - time-indexed series ops, slicing, output DataFrame assembly
  - `numpy` - inf constants and NaN sanitation
  - `dataclasses` - `@dataclass` and `asdict` for log headers

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: walk-forward orchestration and experiment scripts (e.g., `ftf.walkforward.runner`, `scripts/02_run_fast_oos.py`, robustness scripts for latency/costs).
- Key exports used elsewhere: `run_engine`, `EngineResult`

**Implementation Notes**: (if any)
- Architecture decisions:
  - Enforces paper’s timing conventions: decision at day *t*, executed weight via lag buffer `w_exec[t] = w_target[t-d]`, and P&L attribution `gross_ret[t] = w_exec[t-1] * r[t]`.
  - Costs charged on turnover at *t* using executed weights (`|w_exec[t]-w_exec[t-1]|`), then subtracted from daily gross return.
  - Produces a single “wide” daily diagnostics table with signals, risk, sizing, execution, and P&L fields to support debugging, reporting, and tests.
- Cross-File Relationships:
  - `compute_regime_features()` provides `p_bull/p_bear/eligible_to_enter` and related signal diagnostics used both for sizing (`compute_w_raw`) and entry gating (`generate_target_weights`).
  - `compute_atr()` feeds `generate_target_weights()` which encapsulates the position state machine and produces `w_target` plus an event stream.
  - `apply_exec_lag()` converts `w_target` to `w_exec` for forecast-to-fill simulation; `compute_costs()` converts turnover into linear + impact costs.
  - `TradingLog` header captures frozen parameters (`regime_state`, `vol_state`, `policy_state`, `f_tilde`) plus config snapshots for reproducibility.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/trading/logs.py; ROUND 15 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:00:13
**File Implemented**: ftf_repro/src/ftf/trading/logs.py

**Core Purpose**
- Provides a lightweight, deterministic, append-only structured trade/event logging facility for the trading state machine (entries/exits/derisk transitions), designed to be attached to engine outputs and serialized for reproducible analysis.

**Public Interface**
- Class `TradeEvent`: Represents a single discrete trade/state-machine event. | Key fields: `date`, `event`, `price`, `info` | Constructor params: `date: str`, `event: EventType`, `price: Optional[float]=None`, `info: Dict[str, Any]=...`
- Class `TradingLog`: Append-only event log with metadata header. | Key methods: `add`, `extend`, `to_dict`, `to_frame` | Constructor params: `header: Dict[str, Any]=...`, `events: List[TradeEvent]=...`
  - `add(date: pd.Timestamp, event: EventType, *, price: Optional[float]=None, **info: Any) -> None`: Appends one event; normalizes `date` to `pd.Timestamp` and stores as `YYYY-MM-DD`.
  - `extend(events: Iterable[TradeEvent]) -> None`: Appends multiple `TradeEvent` objects; validates types (raises `TypeError` if any element is not `TradeEvent`).
  - `to_dict() -> Dict[str, Any]`: Serializes to `{"header": ..., "events": [asdict(TradeEvent), ...]}` for JSON-friendly output.
  - `to_frame() -> pd.DataFrame`: Converts events into a sorted DataFrame; expands `info` dict keys into columns; returns empty frame with `["date","event","price"]` if no events.
- Constants/Types:
  - `EventType`: `typing.Literal[...]` enumerating allowed event strings:
    - `"ENTRY"`, `"EXIT_HARD_STOP"`, `"EXIT_TRAILING_STOP"`, `"EXIT_TIMEOUT"`, `"EXIT_DERISK_CLOSE"`, `"DERISK_HALF_ON"`, `"DERISK_HALF_OFF"`, `"FLAT"`
  - `__all__ = ["EventType", "TradeEvent", "TradingLog"]`

**Internal Dependencies**
- From standard library: `dataclasses` (`dataclass`, `field`, `asdict`), `typing` (`Any`, `Dict`, `Iterable`, `List`, `Literal`, `Optional`)
- External packages:
  - `pandas` - used for timestamp normalization/parsing and for building/sorting a tabular event log (`DataFrame`).

**External Dependencies**
- Expected to be imported by: `ftf_repro/src/ftf/trading/engine.py` and `ftf_repro/src/ftf/trading/exits.py` (or other state-machine components) to record discrete transitions alongside per-day engine outputs.
- Key exports used elsewhere: `TradingLog` (primary), `TradeEvent`, `EventType`.

**Implementation Notes**
- Architecture decisions:
  - Uses `dataclass` structures for stable schema and deterministic serialization (`asdict`).
  - Enforces append-only semantics (no mutation helpers beyond append/extend).
  - Stores event dates as ISO strings in events for JSON stability, while `to_frame()` converts back to datetime for analysis.
- Cross-File Relationships:
  - Complements the engine’s per-day DataFrame logging by capturing discrete trade lifecycle transitions (entries/exits/derisk toggles) in a compact event stream that can be serialized alongside engine results.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/trading/atr.py; ROUND 16 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:00:42
**File Implemented**: ftf_repro/src/ftf/trading/atr.py

**Core Purpose**:
- Provides ATR(14)-style volatility measurement utilities (True Range and rolling-mean ATR) plus a small immutable configuration/state container used by the trading exit/state machine. Emphasizes no forward-looking calculations.

**Public Interface**:
- Class `ATRState`: Frozen container for ATR/exit parameters | Key methods: (dataclass; no methods) | Constructor params: `window: int = 14`, `hard_stop_atr: float = 2.0`, `trailing_stop_atr: float = 1.5`, `timeout_days: int = 30`
- Function `fit_atr_state(*, cfg: Optional[ATRExitConfig] = None)`: Builds an `ATRState` from an `ATRExitConfig` (or defaults) with validation (`atr_window >= 2`) -> `ATRState`: normalized, typed, per-anchor-serializable state.
- Function `true_range(df: pd.DataFrame, *, high_col: str = "high", low_col: str = "low", close_col: str = "close")`: Computes daily True Range as `max(H-L, |H-prevC|, |L-prevC|)` -> `pd.Series`: TR series named `"tr"`.
- Function `compute_atr(df: pd.DataFrame, *, window: int = 14, high_col: str = "high", low_col: str = "low", close_col: str = "close")`: Rolling mean of TR with `min_periods=window` -> `pd.Series`: ATR series named `"atr"` (NaN for first `window` rows); coerces near-zero floats to `0.0` to avoid `-0.0` artifacts.
- Function `compute_atr_from_cfg(df: pd.DataFrame, *, data_cfg: Optional[DataConfig] = None, exit_cfg: Optional[ATRExitConfig] = None)`: Wrapper that uses canonical column names from configs -> `pd.Series`: ATR series.
- Constants/Types:
  - `__all__`: `["ATRState", "fit_atr_state", "true_range", "compute_atr", "compute_atr_from_cfg"]`

**Internal Dependencies**:
- From `ftf.utils.config`: `ATRExitConfig`, `DataConfig`
- External packages:
  - `pandas` - DataFrame/Series manipulation, `shift`, `rolling`, index validation
  - `numpy` - `isclose` used to normalize tiny/negative-zero ATR values
  - `dataclasses` - defines immutable `ATRState`
  - `typing` - `Optional` annotations

**External Dependencies**:
- Expected to be imported by: `ftf_repro/src/ftf/trading/exits.py` (stop/timeout logic), potentially `ftf_repro/src/ftf/trading/engine.py` (feature precompute/logging).
- Key exports used elsewhere: `compute_atr_from_cfg`, `compute_atr`, `ATRState`, `fit_atr_state`.

**Implementation Notes**:
- Architecture decisions:
  - Uses a simple rolling mean ATR (not Wilder’s smoothing), matching the reproduction plan.
  - Enforces no-lookahead: only uses `close.shift(1)` for previous close; ATR is computed from current/previous day fields only.
  - Input validation is strict: requires `DatetimeIndex` and presence of required columns.
- Cross-File Relationships:
  - `ATRExitConfig`/`DataConfig` provide centralized parameter/column naming, enabling consistent usage across the trading engine and any per-anchor frozen parameter snapshots.
  - `ATRState` mirrors exit parameters as an immutable “state” object to support walk-forward freezing/serialization even though values are constant.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/trading/exits.py; ROUND 17 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:01:43
**File Implemented**: ftf_repro/src/ftf/trading/exits.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Implements a causal, ATR-based long-only position state machine that converts a raw desired weight series into a gated/managed *target* weight series with entry rules, stop/timeout exits, and regime de-risking.  
- Produces both target weights and a structured trading event log, with an optional diagnostic series for “same-close” stop-fill sensitivity.

**Public Interface** (what other files can use, if any):
- Class `ATRExitState`: Frozen container for exit hyperparameters (ATR settings + peak reference + de-risk + stop-fill convention) | Key methods: (dataclass, no methods) | Constructor params: `atr: ATRState`, `price_reference_for_peak: PeakRef="close"`, `derisk_policy: DeriskPolicy="DERISK_HALF"`, `stop_fill_policy: StopFillPolicy="STOP_FILL_T_PLUS_1"`
- Function `fit_atr_exit_state(*, cfg: Optional[ATRExitConfig] = None, stop_fill_policy: StopFillPolicy = "STOP_FILL_T_PLUS_1")`: Builds/validates an `ATRExitState` from config and stop-fill policy -> `ATRExitState`: validated frozen exit settings
- Function `generate_target_weights(*, close: pd.Series, high: Optional[pd.Series], atr: pd.Series, w_raw: pd.Series, eligible_to_enter: pd.Series, p_bear: pd.Series, exit_state: Optional[ATRExitState] = None, exit_cfg: Optional[ATRExitConfig] = None, stop_fill_policy: Optional[StopFillPolicy] = None, initial_position: Optional[Dict[str, Any]] = None)`: Runs the state machine day-by-day to produce decision-time weights and an event log -> `Tuple[pd.Series, TradingLog, Optional[pd.Series]]`: `(w_target, log, w_target_stopfill0)`
- Constants/Types:
  - `__all__ = ["ATRExitState", "fit_atr_exit_state", "generate_target_weights"]`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.trading.atr`: `ATRState`, `fit_atr_state`
- From `ftf.trading.logs`: `TradeEvent`, `TradingLog` (note: `TradeEvent` imported but not used directly; events are logged via `TradingLog.add`)
- From `ftf.utils.config`: `ATRExitConfig`, `DeriskPolicy`, `PeakRef`, `StopFillPolicy`
- External packages:
  - `numpy` - numeric checks (`isfinite`, `isclose`), array backing for weights, NaN handling
  - `pandas` - series validation/alignment, iteration over aligned dataframe, output series construction

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf_repro/src/ftf/trading/engine.py` (to transform sizing output into tradable targets), and potentially walk-forward training/runner components that need consistent entry/exit simulation.
- Key exports used elsewhere: `generate_target_weights`, `fit_atr_exit_state`, `ATRExitState`

**Implementation Notes**: (if any)
- Architecture decisions:
  - Fully causal daily loop: decisions at date *t* use only same-day OHLC-derived values and provided decision-time signals; no forward-looking fields.
  - Strict input alignment: concatenates all required inputs with `join="inner"` to enforce a common index and avoid accidental lookahead/misalignment.
  - State machine modeled via private mutable `_Pos` (active flag, entry/peak, age, derisk flag), reset on exit.
  - Stop-fill convention support:
    - Baseline: stop triggers set `w_target[t]=0` (execution lag handled downstream).
    - Optional diagnostic: `STOP_FILL_SAME_CLOSE` creates `w_target_stopfill0` for T+0 attribution sensitivity.
  - Deterministic peak tracking: uses `high` or `close` depending on config; if configured for `high` but missing, falls back to `close`.
- Cross-File Relationships:
  - Uses ATR hyperparameters from `ftf.trading.atr.ATRState` (hard stop multiple, trailing stop multiple, timeout days).
  - Logs discrete trade lifecycle and risk-regime events via `ftf.trading.logs.TradingLog` (e.g., `ENTRY`, `EXIT_*`, `DERISK_HALF_ON/OFF`).
  - Relies on upstream components to provide:
    - `w_raw` (sizing output),
    - `eligible_to_enter` gate (slope/threshold logic encoded upstream),
    - `p_bear` regime probability (from signal/regime modules).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/execution/latency.py; ROUND 18 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:02:07
**File Implemented**: ftf_repro/src/ftf/execution/latency.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Implements the deterministic “forecast-to-fill” execution latency transform by converting decision-time target weights into executed weights via an integer business-day lag.  
- Enforces basic input validation and produces a clean, float executed-weight series suitable for downstream turnover/cost and P&L attribution.

**Public Interface** (what other files can use, if any):
- Function `apply_exec_lag(w_target: pd.Series, *, exec_lag: int, fill_value: float = 0.0)`: Apply a shift-based execution lag to a target weight series -> `pd.Series`: returns `w_exec` aligned to the same index as `w_target`.
- Constants/Types:
  - `__all__ = ["apply_exec_lag"]`: defines the module’s public export surface.

**Internal Dependencies** (what this file imports/requires, if any):
- External packages:
  - `pandas` (`pd`) - uses `Series.shift`, index typing (`DatetimeIndex`), NA handling (`fillna`), naming.
  - `numpy` (`np`) - uses `np.isclose` to normalize “negative zero”/tiny floating artifacts to exact `0.0`.
- Standard library:
  - `typing.Optional` is imported but not used.

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf_repro/src/ftf/trading/engine.py` (to convert `w_target` → `w_exec` prior to turnover/cost computation and return attribution).
- Key exports used elsewhere: `apply_exec_lag`.

**Implementation Notes**: (if any)
- Architecture decisions:
  - Uses `pd.Series.shift(exec_lag)` to implement the lag, preserving index alignment and business-day indexing semantics.
  - Fills the first `exec_lag` entries with a configurable `fill_value` (default `0.0`) to represent “no position before first executable signal”.
  - Performs strict type checks (must be `pd.Series` with `pd.DatetimeIndex`) and bounds checks (`0 <= exec_lag <= 10`) to catch configuration/data issues early.
  - Does not forward-fill weights; NaNs propagate only through the shift/fill policy, consistent with warmup/insufficient-history periods.
  - Normalizes near-zero values to exact `0.0` to avoid downstream noise in turnover/cost calculations.
- Cross-File Relationships:
  - Meant to be applied after signal/sizing produce `w_target` and before execution/costs logic consumes executed weights; complements upcoming `execution/fills.py` and `execution/costs.py` modules.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/execution/costs.py; ROUND 19 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:02:50
**File Implemented**: ftf_repro/src/ftf/execution/costs.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Implements a deterministic, turnover-based transaction cost and impact model driven by executed weights (`w_exec`), producing per-day linear/impact/total costs intended to be subtracted from strategy gross returns under the forecast-to-fill convention.

**Public Interface** (what other files can use, if any):
- Class `CostSeries`: Container for computed series | Key methods: *(dataclass; no methods)* | Constructor params: `turnover: pd.Series`, `cost_linear: pd.Series`, `cost_impact: pd.Series`, `cost_total: pd.Series`
- Function `turnover_from_exec(w_exec: pd.Series, *, fill_first: bool = True)`: Computes daily turnover from executed weights (`|w_t - w_{t-1}|`) -> `pd.Series`: turnover series named `"turnover"`, with optional first-day handling
- Function `compute_costs(w_exec: pd.Series, *, costs_cfg: Optional[CostImpactConfig] = None, k_linear: Optional[float] = None, gamma_impact: Optional[float] = None, turnover: Optional[pd.Series] = None)`: Computes turnover + linear and impact costs (and total) -> `CostSeries`: includes `"cost_linear"`, `"cost_impact"`, `"cost_total"`
- Function `apply_costs_to_returns(gross_ret: pd.Series, *, cost_total: pd.Series)`: Helper to compute `net_ret = gross_ret - cost_total` with index alignment -> `pd.Series`
- Constants/Types:
  - `__all__ = ["CostSeries", "turnover_from_exec", "compute_costs", "apply_costs_to_returns"]`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.utils.config`: `CostImpactConfig` (default coefficients `k_linear`, `gamma_impact`)
- External packages:
  - `numpy` - absolute value, `isclose` near-zero cleanup, `power(turnover, 1.5)`
  - `pandas` - Series operations (`shift`, `abs`, `align`, `clip`, masking/reindexing)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf/trading/engine.py` (net return computation), and potentially scenario scripts for cost/impact stress tests.
- Key exports used elsewhere: `compute_costs`, `turnover_from_exec`, and `CostSeries` for structured logging/diagnostics.

**Implementation Notes**: (if any)
- Architecture decisions:
  - Costs are computed strictly from *executed* weights (not targets), matching forecast-to-fill accounting and avoiding look-ahead.
  - Provides both config-driven defaults (`CostImpactConfig`) and explicit overrides (`k_linear`, `gamma_impact`) to support robustness grids.
  - Near-zero values are snapped to `0.0` using a tight tolerance (`1e-15`) to reduce floating noise in logs/tests.
- Cross-File Relationships:
  - Coefficients come from `ftf.utils.config.CostImpactConfig`; the engine can pass executed weights (potentially latency-shifted) to `compute_costs` to generate daily cost series that are subtracted from `gross_ret`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/execution/fills.py; ROUND 20 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:03:16
**File Implemented**: ftf_repro/src/ftf/execution/fills.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Provides deterministic “forecast-to-fill” execution helpers: applies an integer-day execution lag to target weights and computes gross (pre-cost) return attribution consistent with the project convention.

**Public Interface** (what other files can use, if any):
- Class `FillResult`: Lightweight container for execution outputs | Key methods: *(dataclass; no methods)* | Constructor params: `w_exec: pd.Series`, `gross_ret: pd.Series`
- Function `compute_gross_return(r: pd.Series, w_exec: pd.Series) -> pd.Series`: Computes gross strategy returns using `gross_ret[t] = w_exec[t-1] * r[t]` with inner index alignment and basic numeric cleanup -> `pd.Series`: daily gross returns
- Function `fill_from_targets(w_target: pd.Series, r: pd.Series, *, exec_lag: int, fill_value: float = 0.0) -> FillResult`: Convenience wrapper that lags targets via `apply_exec_lag` then computes gross returns -> `FillResult`: executed weights + gross returns
- Constants/Types:
  - Re-export `apply_exec_lag` via `__all__` for convenient access from this module

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.execution.latency`: `apply_exec_lag`
- External packages:
  - `pandas` - Series/DatetimeIndex handling, alignment, shifting
  - `numpy` - `isclose` used to clean near-zero artifacts (e.g., `-0.0`)
  - `dataclasses` - immutable `FillResult` container (`frozen=True`)
  - `typing.Optional` - imported but not used in current implementation

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: backtest orchestration and tests/scripts needing consistent execution attribution (likely `ftf/trading/engine.py`, and future walk-forward/stats scripts)
- Key exports used elsewhere: `compute_gross_return`, `fill_from_targets`, `FillResult`, and the re-exported `apply_exec_lag`

**Implementation Notes**: (if any)
- Architecture decisions: keeps fills deterministic (daily close) and lightweight; separates lagging (`apply_exec_lag`) from attribution (`compute_gross_return`) while providing a small façade for convenience.
- Cross-File Relationships: relies on `ftf.execution.latency.apply_exec_lag` for the canonical latency model; complements `ftf.execution.costs` by producing *pre-cost* gross returns that other components can net down with cost/impact.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/utils/io.py; ROUND 21 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:03:41
**File Implemented**: ftf_repro/src/ftf/utils/io.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Centralizes minimal, deterministic filesystem IO utilities for the project, standardizing config/artifact formats (YAML/JSON) and large table storage (Parquet) to support reproducible pipelines.

**Public Interface** (what other files can use, if any):
- Function `ensure_dir(path: PathLike)`: Ensure a directory exists (or the parent directory if `path` looks like a file) -> `Path`: returns the created/existing directory path as a `Path`.
- Function `load_yaml(path: PathLike)`: Load YAML from disk -> `Dict[str, Any]`: returns `{}` if YAML is empty/`None`.
- Function `save_yaml(obj: Any, path: PathLike)`: Save object as YAML -> `None`.
- Function `load_json(path: PathLike)`: Load JSON from disk -> `Any`.
- Function `save_json(obj: Any, path: PathLike, *, indent: int = 2)`: Save object as JSON -> `None` (no key sorting; indented by default).
- Function `load_parquet(path: PathLike, *, columns: Optional[list[str]] = None)`: Read Parquet -> `pd.DataFrame` (optional column projection).
- Function `save_parquet(df: pd.DataFrame, path: PathLike, *, index: bool = True)`: Write Parquet -> `None` (index written by default).
- Constants/Types:
  - `PathLike = Union[str, Path]`: accepted path input type alias.
- Module export control:
  - `__all__ = ["PathLike", "ensure_dir", "load_yaml", "save_yaml", "load_json", "save_json", "load_parquet", "save_parquet"]`

**Internal Dependencies** (what this file imports/requires, if any):
- From standard library:
  - `json` (JSON serialization/deserialization)
  - `pathlib.Path` (path handling)
  - `typing` (`Any`, `Dict`, `Optional`, `Union`) for type hints
- External packages:
  - `yaml` (PyYAML) — `safe_load`, `safe_dump` for config/parameter snapshots
  - `pandas as pd` — `read_parquet`, `DataFrame.to_parquet` for timeseries tables

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: experiment scripts (`scripts/*.py`) and orchestration/utility modules that need standardized reading/writing of configs, artifacts, and timeseries outputs (e.g., walk-forward runner/trainer, reporting, stats pipelines).
- Key exports used elsewhere: `load_yaml`/`save_yaml` (configs + frozen params), `save_parquet`/`load_parquet` (prepared data + daily logs), `save_json`/`load_json` (small artifacts/headers), `ensure_dir` (safe writes).

**Implementation Notes**: (if any)
- Architecture decisions:
  - Determinism/reproducibility emphasis: uses `yaml.safe_*` and avoids auto key sorting in dumps (`sort_keys=False`) to preserve human-authored ordering where possible.
  - Write helpers always create parent directories via `ensure_dir`, avoiding scattered directory-creation logic across scripts.
  - `ensure_dir` treats a path with a suffix as a file path and creates its parent directory; suffixless paths are treated as directories.
- Cross-File Relationships:
  - Intended as the single IO convention layer for configs, per-anchor frozen parameter snapshots, logs, and processed datasets produced by scripts and consumed by walk-forward/stat/report components.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/utils/seed.py; ROUND 22 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:04:03
**File Implemented**: ftf_repro/src/ftf/utils/seed.py

**Core Purpose**:
- Centralizes reproducibility/seeding utilities to make randomized procedures (bootstraps, SPA/RC, etc.) deterministic across scripts by setting common RNG seeds and (optionally) deterministic hashing behavior.

**Public Interface**:
- Function `set_global_seed(seed: int, *, deterministic_hash: bool = True)`: Sets seeds for Python’s `random` and `numpy` RNGs and optionally sets `PYTHONHASHSEED` for deterministic hashing -> `None`.
- Constants/Types:
  - `__all__ = ["set_global_seed"]`: Declares `set_global_seed` as the intended public export.

**Internal Dependencies**:
- From standard library: `os` (env var handling), `random` (Python RNG), `typing.Optional` (type hints)
- External packages:
  - `numpy` (`np.random.seed`, `np.integer` type acceptance)
  - Optional: `numba` (best-effort import only; no actual seeding API invoked)

**External Dependencies**:
- Expected to be imported by: experiment orchestration and statistical modules needing deterministic randomness, e.g.:
  - `scripts/02_run_fast_oos.py`, `scripts/03_latency.py`, `scripts/04_cost_impact.py`, `scripts/05_spa.py`, `scripts/06_capacity.py`, `scripts/07_report.py`
  - `src/ftf/stats/bootstrap.py`, `src/ftf/stats/spa.py`, and potentially `src/ftf/walkforward/runner.py`
- Key exports used elsewhere: `set_global_seed`

**Implementation Notes**:
- Architecture decisions:
  - Validates `seed` is an `int`/`np.integer` and normalizes to a Python `int`.
  - Uses `os.environ.setdefault("PYTHONHASHSEED", ...)` so it won’t override an already-specified hash seed; notes that full hash determinism requires setting before interpreter start but this still helps subprocesses.
  - Seeds both `random` and `numpy` for broad determinism coverage.
  - Attempts to import `numba` as a conservative hook without assuming version-specific global seeding APIs.
- Cross-File Relationships:
  - Designed as a single-call utility to be invoked early in scripts/runners before any bootstrap/grid-search sampling occurs, ensuring consistent results across runs and environments.
  - Includes an internal helper `_get_env_seed(name: str = "FTF_SEED") -> Optional[int]` to parse seeds from environment variables, but it is not exported via `__all__` and is intended for internal/convenience use within future scripts/utilities.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/stats/bootstrap.py; ROUND 23 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:04:47
**File Implemented**: ftf_repro/src/ftf/stats/bootstrap.py

**Core Purpose**:
- Provides time-series bootstrap utilities (fixed block bootstrap and Politis–Romano stationary bootstrap) plus a reusable bootstrap driver and a block-bootstrap Sharpe-ratio percentile confidence interval helper.

**Public Interface** (what other files can use, if any):
- Class `SharpeCI`: Container for Sharpe estimate and CI metadata | Key methods: (dataclass; no methods) | Constructor params: `sharpe: float, ci_low: float, ci_high: float, method: str, B: int, block_len: int`
- Function `block_bootstrap_indices(n: int, block_len: int, *, rng: np.random.Generator)`: Generate resample indices using fixed-length blocks (with wrap-around) -> `np.ndarray`: integer indices of length `n`
- Function `block_bootstrap(x: pd.Series, *, block_len: int = 20, rng: Optional[np.random.Generator] = None)`: Resample a Series’ values using block bootstrap while preserving the original index -> `pd.Series`
- Function `stationary_bootstrap_indices(n: int, mean_block_len: float, *, rng: np.random.Generator)`: Generate indices for stationary bootstrap (random block restarts with prob `p=1/mean_block_len`) -> `np.ndarray`
- Function `stationary_bootstrap(x: pd.Series, *, mean_block_len: float = 20, rng: Optional[np.random.Generator] = None)`: Resample Series’ values via stationary bootstrap while preserving index -> `pd.Series`
- Function `bootstrap_statistic(x: pd.Series, stat_fn: Callable[[pd.Series], float], *, B: int, method: str = "block", block_len: int = 20, mean_block_len: float = 20, seed: int = 123)`: Generic bootstrap runner that returns `B` bootstrap replicates of `stat_fn` -> `np.ndarray`
- Function `bootstrap_sharpe_ci(net_ret: pd.Series, *, B: int = 1000, block_len: int = 20, alpha: float = 0.05, seed: int = 123)`: Computes an annualized Sharpe and a percentile CI via block bootstrap -> `SharpeCI`
- Constants/Types:
  - `__all__`: exports `block_bootstrap_indices, block_bootstrap, stationary_bootstrap_indices, stationary_bootstrap, bootstrap_statistic, SharpeCI, bootstrap_sharpe_ci`

**Internal Dependencies** (what this file imports/requires, if any):
- From standard library:  
  - `dataclasses.dataclass` (for `SharpeCI`)  
  - `typing` (`Callable, Optional`) (for type hints)
- External packages:
  - `numpy` (`np`) - random number generation, array indexing, quantiles, sqrt
  - `pandas` (`pd`) - Series container, DatetimeIndex validation, output preservation of index/name

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by:  
  - `ftf_repro/src/ftf/stats/spa.py` (stationary bootstrap / generic bootstrap stats for SPA/Reality Check)  
  - `ftf_repro/src/ftf/stats/metrics.py` or reporting scripts (Sharpe CI computation for summaries/tables)
- Key exports used elsewhere:  
  - `bootstrap_statistic`, `stationary_bootstrap(_indices)`, `block_bootstrap(_indices)`, `bootstrap_sharpe_ci`, `SharpeCI`

**Implementation Notes**:
- Architecture decisions:
  - Enforces input as `pd.Series` with `pd.DatetimeIndex` via `_check_1d`, ensuring time-series semantics.
  - Resampling preserves the original index (only values are resampled), making outputs easy to align/plot alongside original series.
  - Fixed-block bootstrap uses wrap-around indexing to avoid edge issues and always returns exactly length `n`.
  - Stationary bootstrap implements geometric block lengths implicitly via restart probability `p = 1/mean_block_len`.
  - `bootstrap_statistic` uses a single seeded `np.random.default_rng(seed)` and passes it through all replicates for reproducibility.
  - Sharpe computation `_sharpe` drops NaNs, uses `ddof=0`, annualizes with `sqrt(252)`, and returns `nan` when insufficient data or zero volatility.
- Cross-File Relationships:
  - Designed to feed downstream statistical validation (Sharpe confidence intervals and SPA/RC p-values) without requiring callers to manage index handling or bootstrap index generation directly.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/stats/spa.py; ROUND 24 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:06:00
**File Implemented**: ftf_repro/src/ftf/stats/spa.py

**Core Purpose**:
- Provides a deterministic implementation of White’s Reality Check and Hansen’s SPA-style data-snooping tests over a panel of strategy return series by bootstrapping the maximum performance differential vs a baseline configuration.

**Public Interface**:
- Class `SPAResult`: Immutable container for SPA/RC outputs (observed stat, bootstrap distribution, p-value, best alternative). | Key methods: *(dataclass; no methods)* | Constructor params: `test_kind, metric, B, method, block_len, mean_block_len, t_obs, p_value, t_boot, best_name, best_value`
- Function `compute_differentials(panel: Dict[str, pd.Series], *, baseline_name: str) -> Dict[str, pd.Series]`: Computes per-strategy differential series `d_i = r_i - r_baseline`, aligned on common dates, excluding the baseline -> `dict[name -> pd.Series]`.
- Function `spa_reality_check(panel: Dict[str, pd.Series], *, baseline_name: str, test_kind: TestKind = "SPA", metric: DiffMetric = "mean", method: BootstrapMethod = "stationary", B: int = 800, block_len: int = 20, mean_block_len: int = 20, seed: int = 123, studentize: bool = False) -> SPAResult`: Runs SPA/RC-style test using block or stationary bootstrap on the max-across-configs statistic (mean or Sharpe differential) -> `SPAResult`.
- Constants/Types:
  - `BootstrapMethod = Literal["block", "stationary"]`
  - `TestKind = Literal["RC", "SPA"]`
  - `DiffMetric = Literal["mean", "sharpe"]`
  - `__all__` exports: `BootstrapMethod, TestKind, DiffMetric, SPAResult, compute_differentials, spa_reality_check`

**Internal Dependencies**:
- From `ftf.stats.bootstrap`: `block_bootstrap`, `bootstrap_statistic`, `stationary_bootstrap` (only `block_bootstrap` and `stationary_bootstrap` are actually used; `bootstrap_statistic` is imported but unused).
- External packages:
  - `numpy`: array ops, RNG (`default_rng`), mean/std computations, bootstrap loop, p-value computation.
  - `pandas`: Series alignment by `DatetimeIndex`, reindexing, dummy Series used to generate bootstrap-resampled integer indices.

**External Dependencies**:
- Expected to be imported by: SPA/robustness scripts (e.g., `scripts/05_spa.py`) and potentially reporting utilities to summarize SPA/RC p-values.
- Key exports used elsewhere: `spa_reality_check`, `compute_differentials`, and `SPAResult`.

**Implementation Notes**:
- Architecture decisions:
  - Uses a *panel* of named `pd.Series` return streams, inner-joining on overlapping dates via `_check_panel`.
  - Test statistic is `max_i metric(d_i)` where `d_i = r_i - r_0`; metric is either mean differential or annualized Sharpe of differential.
  - Bootstraps a *shared index selection* across all configurations by resampling a dummy integer series, then applying those integer selections to a pre-built `X` matrix of aligned differentials; this preserves cross-config dependence structure.
  - Provides optional “studentization” only for `metric="mean"` by dividing each differential series by its estimated standard error (approximation, not full SPA nuance).
  - P-value uses a +1 smoothing: `(1 + count(t_boot >= t_obs)) / (B + 1)` to avoid exact zero.
- Cross-File Relationships:
  - Relies on `ftf.stats.bootstrap` for block/stationary bootstrap mechanics; this file focuses on SPA/RC-specific panel construction, statistic definition, and p-value estimation.
- Notable quirk:
  - `_bootstrap_max_stat(...)` is a leftover stub that raises `RuntimeError("internal")` and is unused; main logic is implemented directly in `spa_reality_check`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/05_spa.py; ROUND 25 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:06:47
**File Implemented**: ftf_repro/scripts/05_spa.py

**Core Purpose**:
- Runs a grid-based SPA (Superior Predictive Ability) and White Reality Check (RC) evaluation by generating out-of-sample (OOS) net return series for a baseline configuration and multiple alternative configurations, then bootstrapping Sharpe-based test statistics and saving results to disk.

**Public Interface**:
- Function `_parse_args()`: Parses CLI arguments for config paths, processed data path, output dir, seed, bootstrap repetitions (B), and bootstrap method -> `argparse.Namespace`
- Function `_dict_to_cfg(d: dict)`: Converts a nested dict (from YAML) into a validated `FTFConfig` dataclass -> `FTFConfig`
- Function `_run_one(df_cont: pd.DataFrame, base_cfg_dict: dict, overrides: dict)`: Applies overrides to base config, runs walk-forward backtest, returns OOS net returns -> `pd.Series`
- Function `main()`: Orchestrates loading configs/data, running the panel, executing SPA/RC, and writing artifacts -> `None`
- Constants/Types: None (script-style module; entry point is `main()`)

**Internal Dependencies**:
- From `ftf.stats.spa`: `spa_reality_check` (core SPA/RC bootstrap testing routine)
- From `ftf.walkforward.runner`: `run_walkforward` (produces `res.oos_net_ret` per configuration)
- From `ftf.utils`:  
  - `FTFConfig` (type reference)  
  - `deep_update` (merge base config dict with per-grid overrides)  
  - `ensure_dir` (create output directory)  
  - `load_parquet`, `load_yaml` (I/O)  
  - `save_json`, `save_yaml` (artifact persistence)  
  - `set_global_seed` (determinism)  
  - `validate_config` (config validation)
- From `ftf.utils.config` (imported inside `_dict_to_cfg` to avoid circular imports):  
  `ATRExitConfig, BootstrapConfig, CapacityConfig, CostImpactConfig, DataConfig, FTFConfig, KellyConfig, RegressionConfig, RiskConfig, SignalConfig, TimeConvention, WalkForwardConfig`
- External packages:
  - `argparse` - CLI interface
  - `dataclasses.asdict` - serialize SPA/RC results dataclasses to JSON
  - `pathlib.Path` - filesystem paths
  - `numpy` - Sharpe computation (`sqrt(252)`)
  - `pandas` - return series handling, output tables (`DataFrame`, CSV writing)

**External Dependencies**:
- Expected to be invoked by: CLI users (e.g., `python ftf_repro/scripts/05_spa.py ...`); potentially called by higher-level automation scripts (not required).
- Key exports used elsewhere: None (script module; functions are not designed as library API).

**Implementation Notes**:
- Architecture decisions:
  - Deterministic execution: chooses seed from CLI or `cfg0.bootstrap.seed`, calls `set_global_seed(seed)`.
  - Config handling: loads `base_fast.yaml` + `spa_grid.yaml`, merges overrides via `deep_update`, then reconstructs a strongly-typed `FTFConfig` via `_dict_to_cfg()` for validation and consistent defaults.
  - Bootstrap settings: defaults to stationary bootstrap (method `"stationary"`) unless overridden; `B` defaults to `cfg0.bootstrap.stationary_bootstrap_B`.
- Cross-File Relationships:
  - Uses `run_walkforward()` to produce comparable OOS net returns per config, then passes a `{name: Series}` panel into `spa_reality_check()` twice—once for `"SPA"` and once for `"RC"`—with identical panel and baseline name.
  - Writes reproducibility artifacts: snapshots of base config and grid config YAML; JSON results for SPA and RC; CSV summary (`spa_rc_summary.csv`); per-config performance table (`panel_performance.csv`) computed locally via an internal Sharpe helper.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/configs/grids/spa_grid.yaml; ROUND 26 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:07:25
**File Implemented**: ftf_repro/configs/grids/spa_grid.yaml

**Core Purpose**:
- Defines a fully enumerated 64-configuration grid of hyperparameter overrides for SPA/White Reality Check experiments.
- Provides a human-auditable list of named config variants that a runner can deep-merge into a base configuration to generate comparable strategy runs.

**Public Interface**:
- Constants/Types:
  - `baseline_name`: `"baseline"` — identifies which entry in `configs` should be treated as the reference model for SPA/RC comparisons.
  - `configs`: `list[dict]` — each element has:
    - `name`: `str` unique identifier for the configuration.
    - `overrides`: `dict` deep-merge override applied onto `base_fast.yaml`, containing:
      - `signal.ema_lambda`: float (one of 0.90, 0.92, 0.94, 0.96; baseline uses 0.94)
      - `signal.blend_omega`: float (one of 0.50, 0.60, 0.70, 0.80; baseline uses 0.60)
      - `signal.pbull_threshold`: float (grid uses 0.50 or 0.55; baseline uses 0.52)
      - `risk.ewma_theta`: float (0.94 or 0.96; baseline uses 0.94)

**Internal Dependencies**:
- From other modules/files: none (YAML configuration only).
- External packages: none directly (parsed by the project’s YAML/config loader elsewhere).

**External Dependencies**:
- Expected to be imported by: `ftf_repro/scripts/05_spa.py` (explicitly referenced in comments as the SPA/RC runner).
- Key exports used elsewhere:
  - `baseline_name` to select the baseline config for differential performance statistics.
  - `configs[*].overrides` to deep-merge into `base_fast.yaml` via the project’s `deep_update` mechanism (in utils/config).

**Implementation Notes**:
- Architecture decisions:
  - Grid is intentionally expanded explicitly (no programmatic generation) to keep it easy to audit and to ensure stable config ordering/names across runs.
  - Uses a structured naming convention: `l{lambda}_o{omega}_th{threshold}_t{theta}` for traceability in logs/artifacts.
- Cross-File Relationships:
  - Works with `base_fast.yaml` as the base experiment configuration and relies on the runner to deep-merge each `overrides` dict before executing backtests and SPA/RC bootstraps.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/01_build_data.py; ROUND 27 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:08:00
**File Implemented**: ftf_repro/scripts/01_build_data.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Provides a vendor-agnostic CLI script to ingest per-contract GC futures OHLC(+volume/ADV) data plus contract metadata (FND), construct a continuous front-month series using the project’s roll logic, run validation checks, and persist processed datasets + a validation report.

**Public Interface** (what other files can use, if any):
- Function `_parse_args()`: Builds and returns the `argparse.Namespace` for CLI options -> `argparse.Namespace`
- Function `_load_datacfg_from_yaml(path: Path)`: Loads a YAML file and constructs a `DataConfig` from its `data` section -> `DataConfig`
- Function `_find_contract_files(contracts_dir: Path)`: Discovers `*.parquet`/`*.csv` files and maps contract symbol (filename stem) to file path -> `Dict[str, Path]`
- Function `main()`: Orchestrates end-to-end build/validate/persist pipeline -> `None`
- Constants/Types: None (script-style entrypoint; primary execution via `__main__`)

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.data.calendar`: `get_calendar`
- From `ftf.data.futures_roll`: `build_continuous_front_month`
- From `ftf.data.loaders`: `read_contract_metadata`, `read_contract_ohlc`
- From `ftf.data.validation`: `validate_continuous_df`
- From `ftf.utils`: `ensure_dir`, `load_yaml`, `save_json`, `save_parquet`
- From `ftf.utils.config`: `DataConfig`
- External packages:
  - `argparse` — CLI parsing
  - `pathlib.Path` — filesystem paths
  - `pandas` — in-memory DataFrame handling for loaded contract bars and outputs

**External Dependencies** (what depends on this file, if any):
- Expected to be invoked by: end users via CLI (e.g., as `python scripts/01_build_data.py ...`) or as a subprocess in higher-level experiment scripts.
- Key exports used elsewhere: none (not designed as a library module; primary interface is `main()` and CLI execution)

**Implementation Notes**: (if any)
- Architecture decisions:
  - Vendor-agnostic input contract discovery: scans `<data_dir>/contracts` for `.parquet` or `.csv` and infers contract symbol from filename stem.
  - Metadata-gated ingestion: ignores contract files not present in metadata index (allows extra files/artifacts in the directory).
  - Config-driven column conventions: optionally loads `DataConfig` from a YAML config under the `data:` key; otherwise defaults to `DataConfig()`.
  - Outputs multiple artifacts for downstream reproducibility and diagnostics: continuous dataset, active contract mapping, roll diagnostics table, and a JSON validation report.
- Cross-File Relationships:
  - Delegates all market-calendar alignment, rolling logic, and validation to already-implemented modules (`calendar`, `futures_roll`, `validation`).
  - Uses shared IO utilities (`save_parquet`, `save_json`, `ensure_dir`) to standardize artifact persistence across the project.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/02_run_fast_oos.py; ROUND 28 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:08:38
**File Implemented**: ftf_repro/scripts/02_run_fast_oos.py

**Core Purpose**
- Provides the CLI entry point to run the baseline “FAST” strict walk-forward out-of-sample backtest. It loads processed continuous futures data and a YAML config, applies optional CLI overrides, runs the walk-forward pipeline, and writes all artifacts (logs, frozen params, summaries) to an output directory.

**Public Interface**
- Function `_parse_args()`: Builds/executes an `argparse` CLI parser for config/data/out/seed/trainer-mode/exec-lag overrides -> `argparse.Namespace`
- Function `_dict_to_cfg(d: Dict[str, Any])`: Converts a raw nested dict (from YAML + overrides) into a validated `FTFConfig` dataclass instance -> `FTFConfig`
- Function `main()`: Orchestrates config loading/overrides, seeding, data loading, walk-forward execution, and artifact persistence -> `None`
- Script entrypoint: `if __name__ == "__main__": main()`

**Internal Dependencies**
- From `ftf.utils`:
  - `FTFConfig` (configuration root type)
  - `deep_update` (merge YAML dict with CLI overrides)
  - `ensure_dir` (create output directory)
  - `load_parquet`, `load_yaml` (I/O)
  - `save_json`, `save_parquet`, `save_yaml` (artifact persistence)
  - `set_global_seed` (reproducibility)
  - `validate_config` (config validation)
- From `ftf.walkforward.runner`:
  - `run_walkforward` (executes the full walk-forward backtest and returns results bundle)
- Late import inside `_dict_to_cfg` from `ftf.utils.config`:
  - `ATRExitConfig, BootstrapConfig, CapacityConfig, CostImpactConfig, DataConfig, KellyConfig, RegressionConfig, RiskConfig, SignalConfig, TimeConvention, WalkForwardConfig` (typed sub-config dataclasses)
- External packages:
  - `argparse` (CLI)
  - `dataclasses.asdict` (serialize summary dataclass)
  - `pathlib.Path` (portable paths and defaults)
  - `typing` (`Any, Dict`)
  - `pandas as pd` (imported but not directly used in this script; dataframes flow through I/O + runner)

**External Dependencies**
- Expected to be executed by: end users / experiment automation (shell, Makefile, CI) as the main reproduction command for baseline OOS.
- Key exports used elsewhere:
  - None (script-oriented module). The callable `main()` could be imported by tooling, but primary usage is CLI execution.

**Implementation Notes**
- Architecture decisions:
  - Uses a typed config object (`FTFConfig` + nested dataclasses) constructed from YAML dicts and validated via `validate_config`.
  - CLI overrides are intentionally minimal and targeted: `walkforward.trainer_mode` and `time.exec_lag`, plus seed override.
  - Ensures reproducibility by applying a single global seed (CLI `--seed` overrides config `bootstrap.seed`).
- Cross-File Relationships:
  - Delegates all core finance logic (training/freezing, stitching, trading loop, logging) to `ftf.walkforward.runner.run_walkforward`.
  - Relies on `scripts/01_build_data.py` output format (`df_cont` as a processed Parquet) and persists artifacts for downstream scripts (SPA, latency/cost grids, reporting).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/walkforward/runner.py; ROUND 29 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:09:26
**File Implemented**: ftf_repro/src/ftf/walkforward/runner.py

**Core Purpose**:
- Orchestrates the end-to-end strict walk-forward backtest: builds anchors, fits train-only frozen parameters per anchor, runs the trading engine on each test slice, and stitches overlapping OOS slices into a single canonical OOS series (default: first-step-only). Optionally persists per-anchor artifacts and stitched outputs for reproducibility.

**Public Interface**:
- Class `WalkForwardResult`: Container for full walk-forward outputs (stitched OOS + per-anchor results) | Key methods: *(dataclass; no methods)* | Constructor params: `oos_daily, oos_net_ret, oos_gross_ret, anchors, per_anchor, frozen_params`
- Function `run_walkforward(df_cont: pd.DataFrame, *, cfg: FTFConfig, out_dir: Optional[str | Path] = None, persist_daily: bool = True, persist_per_anchor: bool = True, progress: bool = False) -> WalkForwardResult`: Runs the walk-forward schedule, trains per anchor, backtests each test slice, stitches kept OOS windows, and optionally persists artifacts -> `WalkForwardResult`: stitched daily table + returns series + per-anchor engine outputs + frozen params.
- Constants/Types:
  - `__all__ = ["WalkForwardResult", "run_walkforward"]`

**Internal Dependencies**:
- From `ftf.trading.engine`: `EngineResult`, `run_engine`
- From `ftf.utils.config`: `FTFConfig`, `StitchRule`
- From `ftf.utils.io`: `ensure_dir`, `save_json`, `save_parquet`, `save_yaml`
- From `ftf.walkforward.schedule`: `WalkForwardAnchor`, `build_walkforward_schedule`
- From `ftf.walkforward.trainer`: `AnchorFit`, `fit_anchor`
- External packages:
  - `pandas` - time indexing, slicing train/test windows, concatenation/stitching, series extraction.
  - `numpy` - imported but not directly used in this module.
  - `dataclasses` - defines immutable result container (`WalkForwardResult`).
  - `pathlib.Path` / `typing` - filesystem paths and type annotations.

**External Dependencies**:
- Expected to be imported by: experiment scripts and orchestration code (notably `scripts/02_run_fast_oos.py`; also likely robustness scripts that reuse the same pipeline).
- Key exports used elsewhere: `run_walkforward`, `WalkForwardResult`

**Implementation Notes**:
- Architecture decisions:
  - Deterministic walk-forward loop: for each anchor, slices `df_cont` into `[train_start, train_end)` and `[test_start, test_end)`, fits on train only (`fit_anchor`), then runs `run_engine` on the full test slice for diagnostics.
  - Canonical OOS stitching: `_stitch_first_step_only` uses `_kept_oos_window` driven by `cfg.time.stitch_rule`, and asserts *no duplicate dates* to prevent double-counting.
  - Persistence layout (if `out_dir` provided): `artifacts/anchors/{anchor_date}/` for frozen params (`frozen_params.yaml`), daily logs (`daily.parquet`), and event log (`events.json`); stitched outputs under `reports/`.
- Cross-File Relationships:
  - `build_walkforward_schedule(...)` defines anchor windows; `fit_anchor(...)` produces frozen states/params; `run_engine(...)` consumes those frozen states to generate per-day logs and returns; runner stitches outputs and writes snapshots for reproducibility.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/walkforward/schedule.py; ROUND 30 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:10:09
**File Implemented**: ftf_repro/src/ftf/walkforward/schedule.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Builds a deterministic, business-day-aligned walk-forward schedule (train/test/step windows) over a provided continuous-futures date index.
- Encodes canonical OOS “stitching” bounds (kept segment) per anchor based on the configured stitch rule.

**Public Interface** (what other files can use, if any):
- Class `WalkForwardAnchor`: Immutable container describing one walk-forward anchor’s train/test and kept-OOS bounds | Key methods: *(dataclass; no methods)* | Constructor params: `anchor, train_start, train_end, test_start, test_end, kept_start, kept_end`
- Function `build_walkforward_schedule(index, *, cfg=None, train_bd=None, test_bd=None, step_bd=None, anchor_start=None, anchor_end=None)`: Construct feasible anchors on the given business-day index with full train/test coverage -> `List[WalkForwardAnchor]`: list of anchor definitions ready for training/running/stitching
- Constants/Types: `__all__ = ["WalkForwardAnchor", "build_walkforward_schedule"]`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.utils.config`: `FTFConfig` (used for defaults: `walkforward.*` window sizes, anchor bounds; `time.stitch_rule`)
- External packages:
  - `pandas` - `Timestamp`, `DatetimeIndex`, normalization, timezone stripping, index search/snap (`searchsorted`, `get_indexer`)
  - `dataclasses` - `@dataclass(frozen=True)` for immutable anchor records
  - `typing` - type hints (`Iterable`, `List`, `Optional`)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf_repro/src/ftf/walkforward/runner.py`, `ftf_repro/src/ftf/walkforward/trainer.py`, and scripts that orchestrate walk-forward runs (e.g., fast OOS pipelines).
- Key exports used elsewhere: `build_walkforward_schedule` for generating anchor slices; `WalkForwardAnchor` for consistent train/test/kept boundaries.

**Implementation Notes**: (if any)
- Architecture decisions:
  - Schedule is built strictly on the provided `index` to avoid calendar mismatches; `anchor_start`/`anchor_end` are “snapped” to the first available business day on/after the requested date.
  - Enforces determinism and safety: checks monotonic increasing index, no duplicates, timezone-naive normalized timestamps, and only returns anchors with full train and test windows available.
  - Implements canonical stitching support: if `cfg.time.stitch_rule == "FIRST_STEP_ONLY"`, `kept_end` is `anchor + step_bd` (capped at `test_end`); otherwise keeps the full test window (`kept_end = test_end`).
- Cross-File Relationships:
  - Complements `walkforward/runner.py` (stitching/aggregation of kept OOS segments) and `walkforward/trainer.py` (train-only fitting per anchor). Uses `FTFConfig` conventions shared across the project.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/walkforward/trainer.py; ROUND 31 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:11:18
**File Implemented**: ftf_repro/src/ftf/walkforward/trainer.py

**Core Purpose**
- Provides *train-only* fitting of all frozen, per-anchor parameters needed for walk-forward trading, including signal/regime state, EWMA vol init, ATR exit settings, and friction-adjusted fractional Kelly sizing.
- Supports two trainer modes: deterministic **FIXED** hyperparameters and **GRID** search maximizing *net* Sharpe on the training slice (with turnover/costs), then freezes the chosen parameters.

**Public Interface**
- Class `AnchorFit`: Frozen parameters/states for one anchor (serializable container) | Key methods: *(dataclass; no methods)* | Constructor params:  
  - `regime_state: RegimeState`, `vol_state: EWMAVolState`, `policy_state: PolicyWeightState`, `exit_state: ATRExitState`,  
  - `kelly_inputs: KellyInputs`, `f_star: float`, `f_tilde: float`, `chosen: Dict[str, Any]`
- Function `fit_anchor(df_train: pd.DataFrame, *, cfg: FTFConfig) -> AnchorFit`: Fits all frozen states/scalars for a single anchor using only `df_train` (optionally runs grid search) -> `AnchorFit`: per-anchor frozen artifacts.
- Function `anchor_fit_to_dict(fit: AnchorFit) -> Dict[str, Any]`: Converts `AnchorFit` into JSON/YAML-friendly nested dict -> `dict`: includes chosen hyperparams, Kelly inputs, and nested state dicts.
- Constants/Types:
  - `__all__ = ["AnchorFit", "fit_anchor", "anchor_fit_to_dict"]`

**Internal Dependencies**
- From `ftf.risk.ewma_vol`: `EWMAVolState`, `fit_ewma_vol_state`
- From `ftf.sizing.kelly`: `KellyInputs`, `estimate_kelly_inputs`, `fractional_kelly`, `solve_friction_adjusted_kelly`
- From `ftf.sizing.policy_weight`: `PolicyWeightState`, `fit_policy_weight_state`
- From `ftf.signals.regime`: `RegimeState`, `fit_regime_state`
- From `ftf.trading.engine`: `run_engine`
- From `ftf.trading.exits`: `ATRExitState`, `fit_atr_exit_state`
- From `ftf.utils.config`: `FTFConfig`, `TrainerMode`
- External packages:
  - `numpy`: Sharpe computation, numerical guards, grid-search comparisons
  - `pandas`: series/df handling, return computation (`pct_change`), index/type validation
  - `dataclasses`: serializing via `asdict`
  - `itertools.product`: generating grid combinations
  - `typing`: annotations for config overrides and return types

**External Dependencies**
- Expected to be imported by: `ftf_repro/src/ftf/walkforward/runner.py` (to fit per-anchor params) and/or walk-forward orchestration code; scripts that run walk-forward OOS likely call into runner → trainer.
- Key exports used elsewhere: `fit_anchor`, `AnchorFit`, `anchor_fit_to_dict`

**Implementation Notes**
- Architecture decisions:
  - Implements a strict *train-only* protocol: computes train returns, fits regime/vol/exit/policy states on training data only, then estimates Kelly using a unit-notional sleeve simulation.
  - Grid search is self-contained and deterministic: iterates over `_grid_candidates()` and chooses by highest annualized Sharpe of `net_ret`, tie-breaking by lower mean `turnover`.
  - Avoids mutating config dataclasses by rebuilding a new `FTFConfig` in `_apply_overrides()` from `cfg.to_dict()` plus dot-path overrides, then re-validates via `validate_config`.
- Cross-File Relationships:
  - Uses `run_engine()` as the “ground truth” evaluator during grid search and for unit-notional sleeve simulation, ensuring selection criteria aligns with actual execution, costs, and state machine logic.
  - Unit-notional sleeve uses the same entry/exit logic but forces sizing to behave like a fixed-notional strategy by setting `f_tilde=1.0`, ensuring Kelly estimation reflects the policy’s realized opportunity set.
  - Serializes nested states (EMA/momentum, EWMA vol init, ATR config) so the walk-forward runner can snapshot per-anchor artifacts reproducibly.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/stats/metrics.py; ROUND 32 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:12:02
**File Implemented**: ftf_repro/src/ftf/stats/metrics.py

**Core Purpose**
- Provides deterministic performance and attribution metrics for daily strategy return series, including headline stats (Sharpe/CAGR/vol/MaxDD/Calmar) and “active-day” diagnostics conditioned on executed exposure.

**Public Interface**
- Class `PerfStats`: Immutable container for headline performance metrics | Key methods: *(dataclass; no methods)* | Constructor params: `sharpe, cagr, ann_vol, max_dd, calmar, mean_daily, ann_mean, n_days`
- Class `ActiveDayStats`: Immutable container for active-day diagnostics | Key methods: *(dataclass; no methods)* | Constructor params: `active_rate, n_active, hit_rate, payoff_ratio, expectancy_bps, mean_win, mean_loss`
- Function `_check_series(x: pd.Series, name: str)`: Validates a return/weight series (type, DatetimeIndex, monotonic, no duplicates) -> `pd.Series`
- Function `annualized_sharpe(net_ret: pd.Series, *, periods_per_year: int = 252)`: Annualized Sharpe on daily net returns -> `float`
- Function `equity_curve(net_ret: pd.Series, *, start_value: float = 1.0)`: Compounded equity curve from daily returns -> `pd.Series`
- Function `max_drawdown(net_ret: pd.Series)`: Max drawdown magnitude computed from compounded equity curve -> `float`
- Function `cagr_from_returns(net_ret: pd.Series, *, periods_per_year: int = 252)`: CAGR from daily returns via terminal equity and implied years -> `float`
- Function `annualized_vol(net_ret: pd.Series, *, periods_per_year: int = 252)`: Annualized volatility (std * sqrt) -> `float`
- Function `perf_stats(net_ret: pd.Series, *, periods_per_year: int = 252)`: Computes full `PerfStats` bundle -> `PerfStats`
- Function `active_day_stats(net_ret: pd.Series, w_exec: pd.Series, *, active_threshold: float = 1e-3)`: Active-day stats using held exposure `w_exec.shift(1)` to condition `net_ret` -> `ActiveDayStats`
- Function `summarize(net_ret: pd.Series, *, w_exec: Optional[pd.Series] = None, periods_per_year: int = 252)`: Convenience flat dict of metrics (optionally includes active-day fields) -> `Dict[str, float]`
- Constants/Types:
  - `__all__`: Exports `PerfStats, ActiveDayStats` and all public metric functions listed above.

**Internal Dependencies**
- External packages:
  - `numpy` (`np`) - numeric ops, sqrt, finiteness checks
  - `pandas` (`pd`) - Series handling, alignment, time index validation, compounding, shifting
- Standard library:
  - `dataclasses.dataclass` - immutable metric containers
  - `typing` (`Dict, Optional, Tuple`) - type hints (note: `Tuple` imported but unused)

**External Dependencies**
- Expected to be imported by: walk-forward runners/scripts/reporting components that need KPI summaries, e.g. likely `scripts/02_run_fast_oos.py`, `scripts/07_report.py`, and future `ftf_repro/src/ftf/reporting/tables.py`.
- Key exports used elsewhere: `summarize()`, `perf_stats()`, `annualized_sharpe()`, `active_day_stats()`.

**Implementation Notes**
- Architecture decisions:
  - Enforces strict input invariants via `_check_series` (DatetimeIndex, monotonic, no duplicates), which helps ensure correct time-series semantics in backtests.
  - Uses population std (`ddof=0`) for Sharpe/vol for determinism and consistency across environments.
  - Max drawdown is returned as a positive magnitude (`abs(min_drawdown)`), matching common reporting.
  - Active-day logic intentionally conditions on exposure held over the return interval: `active := w_exec.shift(1) > threshold`, aligned with the project’s P&L convention.
- Cross-File Relationships:
  - Assumes `net_ret` follows the engine convention: return at date *t* corresponds to position held from *(t-1→t)*; pairs directly with `trading/engine.py` outputs (`net_ret`, `w_exec`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/stats/regression.py; ROUND 33 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:12:58
**File Implemented**: ftf_repro/src/ftf/stats/regression.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Provides utilities to run a benchmark-neutrality regression of strategy daily returns on benchmark daily returns, reporting alpha/beta with HAC (Newey–West) standard errors plus annualized alpha, tracking error, and information ratio.

**Public Interface** (what other files can use, if any):
- Class `HACRegressionResult`: Immutable dataclass bundling regression outputs (alpha/beta, SEs, t-stats, R², annualized diagnostics) | Key methods: N/A | Constructor params: `alpha_daily, beta, alpha_se, beta_se, alpha_t, beta_t, nw_lags, n_obs, r2, te_ann, alpha_ann, ir`
- Function `align_returns(strat_ret: pd.Series, bench_ret: pd.Series, *, dropna: bool = True) -> Tuple[pd.Series, pd.Series]`: Aligns strategy and benchmark return series on common dates, optionally dropping non-finite values.
- Function `hac_regression(strat_ret: pd.Series, bench_ret: pd.Series, *, nw_lags: int = 10, periods_per_year: int = 252) -> HACRegressionResult`: Runs OLS with HAC SE (statsmodels if available; otherwise internal fallback) and computes TE/alpha_ann/IR.
- Function `hac_regression_sensitivity(strat_ret: pd.Series, bench_ret: pd.Series, *, nw_lags_list: Iterable[int] = (5, 10, 20), periods_per_year: int = 252) -> Dict[int, HACRegressionResult]`: Convenience wrapper to run `hac_regression` across multiple lag choices.
- Function `result_to_dict(res: HACRegressionResult) -> Dict[str, float]`: Converts result dataclass into JSON/YAML-friendly scalars.
- Constants/Types:
  - `__all__`: exports `HACRegressionResult`, `align_returns`, `hac_regression`, `hac_regression_sensitivity`, `result_to_dict`

**Internal Dependencies** (what this file imports/requires, if any):
- External packages:
  - `numpy` - array conversion, finite filtering, OLS linear algebra in fallback, annualization calculations.
  - `pandas` - Series alignment and DatetimeIndex validation.
  - `statsmodels.api` (optional) - primary regression engine with HAC covariance (`cov_type="HAC"`, `maxlags=nw_lags`).
- Standard library:
  - `dataclasses` - `@dataclass`, `asdict` for result bundling/serialization.
  - `typing` - type annotations (`Dict`, `Iterable`, `Tuple`, etc.).

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf_repro/scripts/07_report.py` (report generation), potentially `ftf_repro/src/ftf/reporting/tables.py` (regression table), and any evaluation pipeline consuming benchmark regression.
- Key exports used elsewhere: `hac_regression`, `hac_regression_sensitivity`, `HACRegressionResult`, `result_to_dict`.

**Implementation Notes**: (if any)
- Architecture decisions:
  - Prefers `statsmodels` for correctness and robustness; includes a deterministic internal OLS + Newey–West fallback when `statsmodels` is unavailable.
  - Strict input validation via `_check_series`: enforces `pd.Series` with monotonic, duplicate-free `DatetimeIndex`; timezone-aware indices are converted to naive.
  - Alignment is explicit (`join="inner"`) with optional non-finite filtering for stable regression inputs.
  - Newey–West fallback uses Bartlett weights and the standard sandwich form `cov = (X'X)^-1 S (X'X)^-1` with `pinv` for numerical stability.
- Cross-File Relationships:
  - Intended to support the project’s benchmark neutrality evaluation: downstream reporting can consume `HACRegressionResult` for alpha/beta significance and IR metrics consistent with the pipeline’s daily return conventions.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/07_report.py; ROUND 34 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:13:51
**File Implemented**: ftf_repro/scripts/07_report.py

**Core Purpose**
- Provides a pragmatic end-to-end reporting script that loads an existing stitched walk-forward run (or re-runs walk-forward), computes headline performance metrics, bootstrap Sharpe confidence intervals, optional HAC regression vs LBMA spot, and writes small report artifacts (CSV/JSON/PNG) to a reports directory.

**Public Interface**
- Function `_parse_args() -> argparse.Namespace`: Defines CLI arguments for loading an existing run (`--run_dir`) or re-running from config + processed data (`--config`, `--processed_path`), plus output location, optional LBMA path, and bootstrap seed override.
- Function `_dict_to_cfg(d: Dict[str, Any]) -> FTFConfig`: Reconstructs an `FTFConfig` dataclass (including nested config sections) from a raw dict (YAML-loaded), validates it, and returns the structured config.
- Function `_read_lbma_returns(path: str, *, calendar_name: str = "NYSE") -> pd.Series`: Loads LBMA spot prices and converts them into daily percentage returns (`r_gold`).
- Function `_maybe_plot_equity(net_ret: pd.Series, out_path: Path) -> None`: Attempts to generate and save an equity curve PNG using matplotlib; silently no-ops if plotting fails or matplotlib is unavailable.
- Function `_load_or_run(*, run_dir: Optional[str], config_path: Optional[str], processed_path: Optional[str], out_dir: Optional[str]) -> tuple[pd.DataFrame, FTFConfig, Path]`: Either (a) loads `config_snapshot.yaml` + `reports/oos_daily.parquet` from a prior run directory, or (b) loads config + processed data and runs `run_walkforward`, returning the daily table, config, and output directory path.
- Function `main() -> None`: Orchestrates the workflow: load/run, seed RNG, compute metrics, bootstrap Sharpe CI, optional regression sensitivity table, write artifacts, and plot equity curve.

**Internal Dependencies**
- From `ftf.stats.bootstrap`: `bootstrap_sharpe_ci`
- From `ftf.stats.metrics`: `summarize`
- From `ftf.stats.regression`: `hac_regression_sensitivity`, `result_to_dict`
- From `ftf.utils`: `FTFConfig`, `deep_update`, `ensure_dir`, `load_parquet`, `load_yaml`, `save_json`, `save_yaml`, `set_global_seed`, `validate_config`
  - Note: `deep_update` is imported but not used in this script.
- From `ftf.walkforward.runner`: `run_walkforward`
- Local import inside `_dict_to_cfg` from `ftf.utils.config`: `ATRExitConfig`, `BootstrapConfig`, `CapacityConfig`, `CostImpactConfig`, `DataConfig`, `KellyConfig`, `RegressionConfig`, `RiskConfig`, `SignalConfig`, `TimeConvention`, `WalkForwardConfig`
- Local import inside `_read_lbma_returns` from `ftf.data.loaders`: `read_lbma_spot`
- External packages:
  - `argparse` – CLI interface
  - `dataclasses.asdict` – serializing config/CI dataclasses to JSON
  - `pathlib.Path` – path handling
  - `numpy`, `pandas` – time series manipulation, stats prep, CSV outputs
  - `matplotlib` (optional) – equity curve plotting (guarded import)

**External Dependencies**
- Expected to be imported by: none (intended as a CLI script entry point).
- Key exports used elsewhere: none (functions are script-local; primary usage is `python scripts/07_report.py ...`).

**Implementation Notes**
- Architecture decisions:
  - Supports two modes: “load existing run artifacts” (preferred) vs “rerun walk-forward in-place” for convenience.
  - Keeps reporting robust by making plotting optional and non-fatal (broad exception catch in `_maybe_plot_equity`).
  - Writes a “lightweight but self-contained” report bundle: performance table, Sharpe CI, optional regression table, and a summary JSON aggregating results.
- Cross-File Relationships:
  - Relies on the walk-forward pipeline (`run_walkforward`) to produce the stitched `oos_daily` table.
  - Uses `summarize` for headline metrics and `bootstrap_sharpe_ci` for uncertainty.
  - Uses `hac_regression_sensitivity` + `result_to_dict` to generate a Newey–West lag sensitivity table vs LBMA spot returns loaded via `read_lbma_spot`.
  - Enforces reproducibility for bootstrap outputs via `set_global_seed`, with CLI override taking precedence over config.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/reporting/tables.py; ROUND 35 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:14:16
**File Implemented**: ftf_repro/src/ftf/reporting/tables.py

**Core Purpose**:
- Provides lightweight, pandas-based helper functions to convert common backtest outputs (performance summaries and regression outputs) into consistent tabular `DataFrame` objects for CSV/JSON reporting.

**Public Interface**:
- Function `dict_table(rows: Iterable[Mapping[str, Any]], *, index_col: Optional[str] = None) -> pd.DataFrame`: Converts an iterable of dict-like rows into a `DataFrame`, optionally setting an index column.
- Function `performance_table(panel: Mapping[str, pd.Series], *, w_exec_panel: Optional[Mapping[str, pd.Series]] = None, periods_per_year: int = 252) -> pd.DataFrame`: Builds a one-row-per-strategy performance summary table by calling `ftf.stats.metrics.summarize` for each return series (and optional executed weights for active-day stats).
- Function `regression_table(results: Mapping[int, HACRegressionResult]) -> pd.DataFrame`: Converts a mapping of `nw_lags -> HACRegressionResult` into a `DataFrame` by serializing each result via `result_to_dict`.
- Constants/Types:
  - `__all__ = ["performance_table", "regression_table", "dict_table"]`: Explicit public export list.

**Internal Dependencies**:
- From `ftf.stats.metrics`: `summarize`
- From `ftf.stats.regression`: `HACRegressionResult`, `result_to_dict`
- External packages:
  - `pandas` (`pd`) - constructing and shaping output tables (`DataFrame`, `set_index`, `sort_index`).
- Standard library:
  - `typing` (`Any`, `Dict`, `Iterable`, `Mapping`, `Optional`) for type annotations.
  - `__future__.annotations` for postponed evaluation of annotations.
  - `dataclasses.asdict` is imported but not used (minor cleanup opportunity).

**External Dependencies**:
- Expected to be imported by: reporting and experiment scripts that need consistent CSV tables, e.g. `ftf_repro/scripts/07_report.py` (and likely future `03_latency.py`, `04_cost_impact.py`, `06_capacity.py`).
- Key exports used elsewhere: `performance_table`, `regression_table`, `dict_table`.

**Implementation Notes**:
- Architecture decisions: intentionally “thin” wrappers around existing metric/regression utilities; avoids extra formatting logic and keeps dependencies limited to pandas.
- Cross-File Relationships:
  - `performance_table` delegates all metric computation to `ftf.stats.metrics.summarize`, optionally passing executed weights to include active-day stats.
  - `regression_table` assumes regression objects are produced elsewhere (via `ftf.stats.regression`) and serializes them consistently using `result_to_dict`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/reporting/figures.py; ROUND 36 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:14:47
**File Implemented**: ftf_repro/src/ftf/reporting/figures.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Provides minimal, optional plotting utilities (equity curve, drawdown, and capacity growth curve) used by reporting/robustness scripts. Designed to keep plotting as a non-core dependency by requiring matplotlib only at call time.

**Public Interface** (what other files can use, if any):
- Class `FigurePaths`: Small container for standard output paths for common figures | Key methods: none (dataclass) | Constructor params: `equity_curve: Optional[str] = None`, `drawdown: Optional[str] = None`
- Function `plot_equity_curve(net_ret: pd.Series, *, out_path: str, title: str = "Equity curve")`: Plots `(1+net_ret).cumprod()` and saves to file -> `None`: validates `net_ret` is a `pd.Series` with `DatetimeIndex`, drops NaNs, saves PNG (or any supported format) via `fig.savefig`.
- Function `plot_drawdown(net_ret: pd.Series, *, out_path: str, title: str = "Drawdown")`: Plots drawdown computed from equity curve with fill-under shading -> `None`: validates inputs, computes `dd = eq/eq.cummax() - 1`, saves figure.
- Function `plot_growth_curve(L: np.ndarray, g: np.ndarray, *, out_path: str, title: str = "Growth curve")`: Plots growth proxy `g(L)` with a zero baseline -> `None`: checks equal lengths, saves figure.
- Constants/Types:
  - `__all__`: `["FigurePaths", "plot_equity_curve", "plot_drawdown", "plot_growth_curve"]` (explicit export surface)

**Internal Dependencies** (what this file imports/requires, if any):
- External packages:
  - `pandas` (`pd`) - input typing/validation (`pd.Series`, `pd.DatetimeIndex`), time index handling, cumulative computations.
  - `numpy` (`np`) - array typing for growth curve inputs.
  - `matplotlib` - imported lazily inside `_require_matplotlib()`; used for figure creation and saving.
- Standard library:
  - `dataclasses.dataclass` - defines `FigurePaths`.
  - `typing.Optional` - optional path fields.
- Internal helper:
  - `_require_matplotlib()` (module-private): returns `matplotlib.pyplot` or raises a clear `ImportError` if unavailable.

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf_repro/scripts/07_report.py` (and likely upcoming robustness/capacity scripts such as `03_latency.py`, `04_cost_impact.py`, `06_capacity.py`) to generate and persist figures.
- Key exports used elsewhere: `plot_equity_curve`, `plot_drawdown`, `plot_growth_curve`, and possibly `FigurePaths` to standardize output destinations.

**Implementation Notes**: (if any)
- Architecture decisions: matplotlib is treated as an optional dependency via lazy import; failures raise an actionable ImportError message rather than breaking the package import.
- Input validation is intentionally strict for return series (must be `pd.Series` with `DatetimeIndex`) to avoid silent plotting of misaligned data.
- Plotting is intentionally “lightweight”: fixed figure sizes, basic grids, no styling dependencies; figures are always closed (`plt.close(fig)`) to prevent memory growth in batch runs.
- Cross-File Relationships: complements `ftf.reporting.tables` (tabular outputs) and is expected to visualize return series produced by the trading engine / walk-forward runner and capacity outputs from `ftf.capacity.*`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/reporting/__init__.py; ROUND 37 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:15:05
**File Implemented**: ftf_repro/src/ftf/reporting/__init__.py

**Core Purpose**:
- Defines the public API for the `ftf.reporting` subpackage by re-exporting table/figure helper utilities. Keeps reporting imports lightweight and centralized for downstream consumers.

**Public Interface**:
- Function `dict_table`(…): re-exported table helper (CSV-friendly formatting) -> (type defined in `tables.py`)
- Function `performance_table`(…): re-exported performance summary table builder -> (type defined in `tables.py`)
- Function `regression_table`(…): re-exported regression results table builder -> (type defined in `tables.py`)
- Class `FigurePaths`: re-exported helper for managing figure output paths | Key methods: (defined in `figures.py`) | Constructor params: (defined in `figures.py`)
- Function `plot_drawdown`(…): re-exported plotting helper -> (type defined in `figures.py`)
- Function `plot_equity_curve`(…): re-exported plotting helper -> (type defined in `figures.py`)
- Function `plot_growth_curve`(…): re-exported plotting helper -> (type defined in `figures.py`)
- Constants/Types:
  - `__all__`: explicit export list for `from ftf.reporting import *`

**Internal Dependencies**:
- From `ftf_repro/src/ftf/reporting/tables.py`: `dict_table`, `performance_table`, `regression_table`
- From `ftf_repro/src/ftf/reporting/figures.py`: `FigurePaths`, `plot_drawdown`, `plot_equity_curve`, `plot_growth_curve`
- External packages: none (this file itself uses only relative imports)

**External Dependencies**:
- Expected to be imported by: scripts and reporting entrypoints that want a stable reporting API (e.g., `ftf_repro/scripts/07_report.py`)
- Key exports used elsewhere: the re-exported table builders and plotting functions/classes via `ftf.reporting`

**Implementation Notes**:
- Architecture decisions: acts as a thin “barrel” module to centralize imports and present a clean namespace (`ftf.reporting.*`) without duplicating logic.
- Cross-File Relationships: delegates all real functionality to `reporting/tables.py` and `reporting/figures.py`; this module just re-exports and defines `__all__` for controlled exposure.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/trading/__init__.py; ROUND 38 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:15:26
**File Implemented**: ftf_repro/src/ftf/trading/__init__.py

**Core Purpose**
- Defines the `ftf.trading` subpackage boundary by re-exporting the main trading/engine, ATR, exit-state-machine, and logging interfaces as a single convenient import surface.

**Public Interface**
- Class `ATRState`: ATR/TR computation state container (re-export) | Key methods: N/A (not defined here) | Constructor params: N/A (not defined here)
- Class `ATRExitState`: long-only exit state container (re-export) | Key methods: N/A | Constructor params: N/A
- Class `EngineResult`: engine output/result container (re-export) | Key methods: N/A | Constructor params: N/A
- Class `TradeEvent`: structured trade event record (re-export) | Key methods: N/A | Constructor params: N/A
- Class `TradingLog`: container for structured event logs (re-export) | Key methods: N/A | Constructor params: N/A
- Function `fit_atr_state(...)`: fits/initializes ATR state (re-export) -> (type defined in `atr.py`)
- Function `true_range(...)`: computes true range (re-export) -> (numeric series/array type defined in `atr.py`)
- Function `compute_atr(...)`: computes ATR series (re-export) -> (numeric series/array type defined in `atr.py`)
- Function `compute_atr_from_cfg(...)`: computes ATR using config (re-export) -> (numeric series/array type defined in `atr.py`)
- Function `fit_atr_exit_state(...)`: fits/initializes ATR exit state (re-export) -> `ATRExitState`
- Function `generate_target_weights(...)`: produces target weights from signals/sizing plus exit logic (re-export) -> (series/array of weights)
- Function `run_engine(...)`: top-level orchestrator for the daily loop (re-export) -> `EngineResult`
- Constants/Types:
  - `EventType`: enum/type of event categories used in logs (re-export)
  - `__all__`: explicit export list controlling `from ftf.trading import *`

**Internal Dependencies**
- From `ftf.trading.atr`: `ATRState`, `compute_atr`, `compute_atr_from_cfg`, `fit_atr_state`, `true_range`
- From `ftf.trading.engine`: `EngineResult`, `run_engine`
- From `ftf.trading.exits`: `ATRExitState`, `fit_atr_exit_state`, `generate_target_weights`
- From `ftf.trading.logs`: `EventType`, `TradeEvent`, `TradingLog`
- External packages: None (pure package wiring / re-exports)

**External Dependencies**
- Expected to be imported by: higher-level orchestration and scripts (e.g., walk-forward runner/trainer and experiment scripts) that want `ftf.trading.run_engine` and related types without deep module paths.
- Key exports used elsewhere: `run_engine`, `EngineResult`, `generate_target_weights`, `TradingLog` (likely primary touchpoints).

**Implementation Notes**
- Architecture decisions: uses a “flat” public API via re-exports to keep the subpackage dependency-light and minimize import-path churn for callers.
- Cross-File Relationships: `engine.py` orchestrates the pipeline, while `atr.py` and `exits.py` provide the indicator/state machine and `logs.py` provides structured events; this `__init__.py` stitches them into a cohesive subpackage interface.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/execution/__init__.py; ROUND 39 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:15:49
**File Implemented**: ftf_repro/src/ftf/execution/__init__.py

**Core Purpose**
- Defines the `ftf.execution` package’s public API by re-exporting the key latency, fills/attribution, and transaction cost/impact utilities used by the Forecast-to-Fill pipeline.

**Public Interface**
- Function `apply_exec_lag(...)`: Re-exported helper to apply a deterministic execution delay (target weights → executed weights) -> (type defined in `latency.py`).
- Class `FillResult`: Re-exported result container for fill/attribution outputs | Key methods: (defined in `fills.py`) | Constructor params: (defined in `fills.py`).
- Function `compute_gross_return(...)`: Re-exported gross return attribution utility -> (type defined in `fills.py`).
- Function `fill_from_targets(...)`: Re-exported end-to-end helper to convert targets into executed weights/returns (per convention) -> (type defined in `fills.py`).
- Class `CostSeries`: Re-exported container for per-day cost components | Key methods: (defined in `costs.py`) | Constructor params: (defined in `costs.py`).
- Function `turnover_from_exec(...)`: Re-exported turnover computation from executed weights -> (type defined in `costs.py`).
- Function `compute_costs(...)`: Re-exported linear + concave impact cost computation -> (type defined in `costs.py`).
- Function `apply_costs_to_returns(...)`: Re-exported helper to subtract costs from returns -> (type defined in `costs.py`).
- Constants/Types:
  - `__all__`: Explicit export list containing the names above.

**Internal Dependencies**
- From `.latency`: `apply_exec_lag`
- From `.fills`: `FillResult`, `compute_gross_return`, `fill_from_targets`
- From `.costs`: `CostSeries`, `apply_costs_to_returns`, `compute_costs`, `turnover_from_exec`
- External packages: None (this file only re-exports internal symbols)

**External Dependencies**
- Expected to be imported by: trading/backtest orchestration and experiment scripts that want a stable execution API (e.g., `ftf/trading/engine.py`, robustness scripts for latency/cost stress).
- Key exports used elsewhere: `apply_exec_lag`, `fill_from_targets`, `compute_costs`, `apply_costs_to_returns`, plus the `FillResult`/`CostSeries` datatypes for structured outputs.

**Implementation Notes**
- Architecture decisions: Keeps `ftf.execution` as a lightweight, deterministic subpackage and provides a clean, centralized import surface to avoid deep-module imports throughout the codebase.
- Cross-File Relationships: Acts as the package “facade” over `latency.py` (lag model), `fills.py` (forecast-to-fill attribution), and `costs.py` (turnover + cost/impact model), ensuring consistent use of these components across trading and analysis layers.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/data/__init__.py; ROUND 40 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:16:15
**File Implemented**: ftf_repro/src/ftf/data/__init__.py

**Core Purpose**:
- Defines the `ftf.data` subpackage boundary by re-exporting core calendar, loading, continuous-roll, and validation utilities through a single, stable import surface (`ftf.data.*`).

**Public Interface**:
- Constants/Types:
  - `__all__`: Explicit export list for the data subpackage, controlling what `from ftf.data import *` exposes.

- Re-exported Classes (defined in submodules):
  - Class `CalendarSpec`: Calendar configuration/type used to generate and infer business-day calendars.
  - Class `ContinuousFuturesResult`: Container for outputs from continuous front-month construction.
  - Class `ContinuousValidationReport`: Container/report for validation results on continuous futures outputs.

- Re-exported Functions (defined in submodules):
  - `get_calendar(...)`: Build a calendar index from a `CalendarSpec`.
  - `infer_calendar_from_index(...)`: Infer calendar characteristics from an existing index.
  - `nyse_business_days(...)`: Generate NYSE business-day date index.
  - `shift_bdays(...)`: Shift dates by N business days according to the calendar.
  - `to_date_index(...)`: Normalize/convert an index to a date-based index.

  - `align_ohlc_to_calendar(...)`: Reindex/align OHLC data to a target calendar with deterministic filling rules.
  - `infer_date_col(...)`: Detect the date column name in raw tabular inputs.
  - `read_contract_metadata(...)`: Load contract metadata (e.g., expiries/FNDs).
  - `read_contract_ohlc(...)`: Load per-contract OHLC time series.
  - `read_lbma_spot(...)`: Load LBMA spot series (benchmark/regression input).
  - `validate_daily_index(...)`: Validate monotonicity/uniqueness/frequency expectations of daily indices.

  - `determine_active_contract(...)`: Select the active contract per date under the roll rule.
  - `build_continuous_front_month(...)`: Construct a continuous front-month series using the roll logic.

  - `compute_returns_from_close(...)`: Compute returns from a close series.
  - `compute_atr14(...)`: Compute ATR(14) for validation/sanity checks.
  - `validate_roll_rule(...)`: Check that roll dates match the expected rule.
  - `validate_continuous_df(...)`: Validate the final continuous futures dataframe (NaNs, distributions, etc.).

**Internal Dependencies**:
- From `ftf/data/calendar.py`: `CalendarSpec, get_calendar, infer_calendar_from_index, nyse_business_days, shift_bdays, to_date_index`
- From `ftf/data/futures_roll.py`: `ContinuousFuturesResult, build_continuous_front_month, determine_active_contract`
- From `ftf/data/loaders.py`: `align_ohlc_to_calendar, infer_date_col, read_contract_metadata, read_contract_ohlc, read_lbma_spot, validate_daily_index`
- From `ftf/data/validation.py`: `ContinuousValidationReport, compute_atr14, compute_returns_from_close, validate_continuous_df, validate_roll_rule`
- External packages: None directly (this file only aggregates symbols).

**External Dependencies**:
- Expected to be imported by: pipeline scripts and orchestration modules that want a single import path, e.g. `scripts/01_build_data.py`, walk-forward components, and reporting scripts.
- Key exports used elsewhere: calendar generation helpers, continuous futures builder (`build_continuous_front_month`), and validation utilities (`validate_continuous_df`, `validate_roll_rule`).

**Implementation Notes**:
- Architecture decisions: Centralizes the data-layer API via re-exports and an explicit `__all__`, reducing deep-module imports and making the package interface stable.
- Cross-File Relationships: Acts as a thin façade over `calendar.py` (date index + business day logic), `loaders.py` (I/O + alignment), `futures_roll.py` (roll + continuous series), and `validation.py` (sanity/consistency checks).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/risk/__init__.py; ROUND 41 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:16:39
**File Implemented**: ftf_repro/src/ftf/risk/__init__.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Defines the `ftf.risk` package boundary and re-exports the primary risk-budgeting APIs (EWMA volatility targeting and confidence shaping) as a small, stable public surface for the rest of the pipeline.

**Public Interface** (what other files can use, if any):
- Class `ConfidenceState`: Confidence-shaping fitted state container (defined in `ftf.risk.confidence`) | Key methods: *(n/a in this file; re-export only)* | Constructor params: *(defined in `confidence.py`)*
- Function `fit_confidence_state(...)`: Fit confidence parameters/state (train-only) -> `ConfidenceState`
- Function `confidence_share(...)`: Map probability/score to a [0,1] confidence share -> numeric/series
- Function `confidence_weight(...)`: Apply confidence share to a base risk budget (e.g., vol weight) -> numeric/series
- Class `EWMAVolState`: EWMA variance/volatility fitted state container (defined in `ftf.risk.ewma_vol`) | Key methods: *(n/a in this file; re-export only)* | Constructor params: *(defined in `ewma_vol.py`)*
- Function `fit_ewma_vol_state(...)`: Fit/initialize EWMA variance state (train-only initialization) -> `EWMAVolState`
- Function `ewma_variance_forecast(...)`: Deterministic EWMA variance recursion/forecast -> numeric/series
- Function `vol_target_weight(...)`: Convert EWMA variance forecast into a capped volatility-targeting weight -> numeric/series
- Constants/Types:
  - `__all__`: Explicit export list for `from ftf.risk import *`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.risk.confidence`: `ConfidenceState`, `confidence_share`, `confidence_weight`, `fit_confidence_state`
- From `ftf.risk.ewma_vol`: `EWMAVolState`, `ewma_variance_forecast`, `fit_ewma_vol_state`, `vol_target_weight`
- External packages: None (pure package wiring/re-exports)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: trading/pipeline orchestrators and walk-forward components (e.g., `ftf/trading/engine.py`, `ftf/walkforward/trainer.py`, `ftf/walkforward/runner.py`) to access risk functions via a single namespace.
- Key exports used elsewhere: `fit_ewma_vol_state`, `vol_target_weight`, `fit_confidence_state`, `confidence_weight`

**Implementation Notes**: (if any)
- Architecture decisions: Uses a “facade” pattern via re-exports to keep call sites stable (`from ftf.risk import ...`) and to avoid leaking internal module structure.
- Cross-File Relationships: `ewma_vol.py` provides volatility targeting; `confidence.py` shapes the risk budget based on signal confidence. This `__init__.py` stitches them into one coherent risk API.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/signals/__init__.py; ROUND 42 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:17:07
**File Implemented**: ftf_repro/src/ftf/signals/__init__.py

**Core Purpose**:
- Defines the `ftf.signals` package boundary by documenting the signal stack and re-exporting the key “fit” (train-only) and “compute” (apply state) APIs for EMA-trend, momentum, and regime composition.

**Public Interface** (what other files can use, if any):
- Constants/Types:
  - `__all__`: Explicit export list for the signals subpackage, ensuring stable import surface.
- Class `EMATrendState`: Immutable fitted state for EMA-trend standardization (imported/re-exported) | Key methods: N/A (state object) | Constructor params: defined in `ema_trend.py`
- Function `ema_log_price(...)`: Computes EMA of log prices (re-export) -> return type defined in `ema_trend.py`
- Function `ema_slope_from_ema(...)`: Computes EMA slope proxy from EMA series (re-export) -> return type defined in `ema_trend.py`
- Function `fit_ema_trend_state(...)`: Fits EMA-trend normalization state on training data (re-export) -> `EMATrendState`
- Function `compute_p_trend(...)`: Produces trend probability from standardized slope (re-export) -> typically `pd.Series`/array-like
- Class `MomentumState`: Immutable fitted state for momentum settings (imported/re-exported) | Key methods: N/A | Constructor params: defined in `momentum.py`
- Function `fit_momentum_state(...)`: Fits/locks momentum configuration on training data (re-export) -> `MomentumState`
- Function `compute_momentum_indicator(...)`: Computes K-day momentum indicator (re-export) -> series/array-like
- Function `compute_momentum(...)`: Computes momentum signal/features using `MomentumState` (re-export) -> series/array-like
- Class `RegimeState`: Immutable fitted state for blending/thresholding into regimes (imported/re-exported) | Key methods: N/A | Constructor params: defined in `regime.py`
- Function `fit_regime_state(...)`: Fits regime composition parameters on training data (re-export) -> `RegimeState`
- Function `compute_regime_features(...)`: Produces `p_bull/p_bear`, entry gate, and related regime features (re-export) -> tabular output (likely `pd.DataFrame`)
- Function `label_regime(...)`: Labels bull/bear/chop for attribution (re-export) -> series/array-like

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf/signals/ema_trend.py`: `EMATrendState`, `compute_p_trend`, `ema_log_price`, `ema_slope_from_ema`, `fit_ema_trend_state`
- From `ftf/signals/momentum.py`: `MomentumState`, `compute_momentum`, `compute_momentum_indicator`, `fit_momentum_state`
- From `ftf/signals/regime.py`: `RegimeState`, `compute_regime_features`, `fit_regime_state`, `label_regime`
- External packages: None (pure re-export/namespace management)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: higher-level orchestration/backtest code (e.g., walk-forward trainer/runner and trading engine) to access a stable, centralized signals API.
- Key exports used elsewhere: `fit_*_state` functions (train-only freezing), `compute_*` functions (OOS application), and `RegimeState`/`EMATrendState`/`MomentumState` types.

**Implementation Notes**:
- Architecture decisions: Uses `__all__` to enforce a curated public surface, preventing downstream code from importing internal helpers accidentally and supporting reproducible, stable interfaces.
- Cross-File Relationships: Acts as the aggregator layer over the three signal components:
  - `ema_trend.py` provides normalized trend probability inputs,
  - `momentum.py` provides confirmation inputs,
  - `regime.py` blends them and produces decision-time regime features used by sizing/exits/trading.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/sizing/__init__.py; ROUND 43 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:17:32
**File Implemented**: ftf_repro/src/ftf/sizing/__init__.py

**Core Purpose**:
- Defines the `ftf.sizing` subpackage API by re-exporting the key Kelly sizing and policy-weight construction components used by the Forecast-to-Fill pipeline.
- Provides a small, stable import surface so other modules can depend on `ftf.sizing` without importing internal module paths.

**Public Interface** (what other files can use, if any):
- Class `KellyInputs`: container/type describing inputs needed for Kelly estimation/optimization | Key methods: *(not shown in this file)* | Constructor params: *(defined in `kelly.py`)*
- Class `PolicyWeightState`: container/type for fitted policy-weight parameters/state | Key methods: *(not shown in this file)* | Constructor params: *(defined in `policy_weight.py`)*
- Function `estimate_kelly_inputs(...)`: estimates train-only Kelly inputs -> return type: `KellyInputs`
- Function `growth_proxy(...)`: computes the penalized growth proxy used for sizing -> return type: *(defined in `kelly.py`)*
- Function `solve_friction_adjusted_kelly(...)`: closed-form friction/impact-adjusted Kelly optimizer -> return type: *(defined in `kelly.py`)*
- Function `fractional_kelly(...)`: applies fractional Kelly scaling -> return type: *(defined in `kelly.py`)*
- Function `fit_policy_weight_state(...)`: fits/derives policy-weight state needed to compute sizing -> return type: `PolicyWeightState`
- Function `compute_w_raw(...)`: computes raw target weight from vol/conf/Kelly/baseline-floor logic -> return type: *(defined in `policy_weight.py`)*
- Constants/Types:
  - `__all__`: explicit export list for the package namespace

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf/sizing/kelly.py`: `KellyInputs`, `estimate_kelly_inputs`, `fractional_kelly`, `growth_proxy`, `solve_friction_adjusted_kelly`
- From `ftf/sizing/policy_weight.py`: `PolicyWeightState`, `compute_w_raw`, `fit_policy_weight_state`
- External packages: none (only intra-package imports)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: higher-level orchestration/backtest code that wants a stable sizing API, e.g. `ftf/trading/engine.py`, `ftf/walkforward/trainer.py`, `ftf/walkforward/runner.py`
- Key exports used elsewhere: `compute_w_raw`, `fit_policy_weight_state`, and the Kelly toolchain (`estimate_kelly_inputs`, `solve_friction_adjusted_kelly`, `fractional_kelly`)

**Implementation Notes**:
- Architecture decisions: uses re-exports + `__all__` to define a clean package boundary and prevent consumers from relying on deep module paths.
- Cross-File Relationships: `kelly.py` supplies train-only sizing coefficients (edge/variance/friction-adjusted leverage), while `policy_weight.py` combines those coefficients with risk/confidence inputs to produce the tradable raw weight; this `__init__.py` ties them together as the canonical import point (`from ftf.sizing import ...`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/stats/__init__.py; ROUND 44 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:18:06
**File Implemented**: ftf_repro/src/ftf/stats/__init__.py

**Core Purpose**
- Defines the `ftf.stats` package’s public API by re-exporting key statistical utilities (metrics, HAC regression, bootstraps, and SPA/Reality Check) behind a stable import surface for scripts and reporting.

**Public Interface**
- Classes/Dataclasses (re-exported):
  - Class `PerfStats`: performance summary container | Key methods: N/A (container) | Constructor params: (defined in `metrics.py`)
  - Class `ActiveDayStats`: active-day performance container | Key methods: N/A (container) | Constructor params: (defined in `metrics.py`)
  - Class `HACRegressionResult`: HAC regression results container | Key methods: N/A (container) | Constructor params: (defined in `regression.py`)
  - Class `SharpeCI`: bootstrap Sharpe confidence interval container | Key methods: N/A (container) | Constructor params: (defined in `bootstrap.py`)
  - Class `SPAResult`: SPA/Reality Check output container | Key methods: N/A (container) | Constructor params: (defined in `spa.py`)
- Enums/Types (re-exported):
  - `TestKind`: identifies SPA/RC test variant (as defined in `spa.py`)
  - `DiffMetric`: specifies differential metric used for SPA/RC (as defined in `spa.py`)
- Functions (re-exported; signatures as exposed by this module name-wise):
  - `annualized_sharpe(...)`: compute annualized Sharpe
  - `annualized_vol(...)`: compute annualized volatility
  - `cagr_from_returns(...)`: compute CAGR from a return series
  - `equity_curve(...)`: convert returns to cumulative equity curve
  - `max_drawdown(...)`: compute max drawdown from an equity curve/returns
  - `perf_stats(...)`: compute aggregated performance statistics
  - `active_day_stats(...)`: compute active-day hit-rate/payoff/expectancy style stats
  - `summarize(...)`: higher-level summary convenience wrapper
  - `align_returns(...)`: align strategy/benchmark returns for regression
  - `hac_regression(...)`: run HAC/Newey–West regression vs benchmark
  - `hac_regression_sensitivity(...)`: run HAC regression across lag choices
  - `result_to_dict(...)`: serialize regression result to plain dict
  - `block_bootstrap_indices(...)`: generate block bootstrap resample indices
  - `block_bootstrap(...)`: apply block bootstrap to a series/statistic
  - `stationary_bootstrap_indices(...)`: generate stationary bootstrap indices
  - `stationary_bootstrap(...)`: apply stationary bootstrap
  - `bootstrap_statistic(...)`: generic bootstrap statistic helper
  - `bootstrap_sharpe_ci(...)`: compute bootstrap CI for Sharpe
  - `compute_differentials(...)`: compute per-configuration differentials vs baseline for SPA/RC
  - `spa_reality_check(...)`: run SPA / White Reality Check procedure
- Constants/Types:
  - `__all__`: explicit export list controlling `from ftf.stats import *` and documenting the supported surface.

**Internal Dependencies**
- From `ftf/stats/bootstrap.py`: `SharpeCI`, `block_bootstrap`, `block_bootstrap_indices`, `bootstrap_sharpe_ci`, `bootstrap_statistic`, `stationary_bootstrap`, `stationary_bootstrap_indices`
- From `ftf/stats/metrics.py`: `ActiveDayStats`, `PerfStats`, `active_day_stats`, `annualized_sharpe`, `annualized_vol`, `cagr_from_returns`, `equity_curve`, `max_drawdown`, `perf_stats`, `summarize`
- From `ftf/stats/regression.py`: `HACRegressionResult`, `align_returns`, `hac_regression`, `hac_regression_sensitivity`, `result_to_dict`
- From `ftf/stats/spa.py`: `DiffMetric`, `SPAResult`, `TestKind`, `compute_differentials`, `spa_reality_check`
- External packages: None directly imported in this file (dependencies are encapsulated in the submodules it re-exports).

**External Dependencies**
- Expected to be imported by: experiment scripts (e.g., reporting/SPA runners), reporting modules, and any orchestration code needing a single entry-point for stats.
- Key exports used elsewhere: `perf_stats`, `summarize`, `hac_regression*`, `bootstrap_sharpe_ci`, `spa_reality_check`, plus result container types (`PerfStats`, `HACRegressionResult`, `SPAResult`).

**Implementation Notes**
- Architecture decisions: centralizes the stats subpackage API via re-exports and a strict `__all__`, minimizing churn and keeping downstream imports stable.
- Cross-File Relationships: acts as the “barrel” module tying together `metrics.py`, `regression.py`, `bootstrap.py`, and `spa.py` so scripts/reporting can depend on `ftf.stats` rather than individual modules.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/walkforward/__init__.py; ROUND 45 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:18:24
**File Implemented**: ftf_repro/src/ftf/walkforward/__init__.py

**Core Purpose**
- Defines the `ftf.walkforward` subpackage entry point and re-exports the walk-forward orchestration API (schedule building, per-anchor training, and OOS running/stitching) to provide stable import paths.

**Public Interface**
- Constants/Types:
  - `__all__`: Explicit export list containing:
    - `"WalkForwardAnchor"`, `"build_walkforward_schedule"`, `"AnchorFit"`, `"fit_anchor"`, `"anchor_fit_to_dict"`, `"WalkForwardResult"`, `"run_walkforward"`
- Re-exported classes/functions (defined in other modules, exposed here):
  - Class `WalkForwardAnchor`: Walk-forward anchor descriptor (train/test/step slices).
  - Function `build_walkforward_schedule(...)`: Builds deterministic walk-forward anchors.
  - Class `AnchorFit`: Container for frozen, train-only parameters per anchor.
  - Function `fit_anchor(...)`: Fits/chooses frozen parameters on the train slice for a given anchor.
  - Function `anchor_fit_to_dict(...)`: Serializes an `AnchorFit` to a dict (for artifact writing).
  - Class `WalkForwardResult`: Container for stitched OOS series and per-anchor outputs.
  - Function `run_walkforward(...)`: Runs per-anchor backtests and stitches overlapping OOS segments.

**Internal Dependencies**
- From `ftf.walkforward.schedule`: `WalkForwardAnchor`, `build_walkforward_schedule`
- From `ftf.walkforward.trainer`: `AnchorFit`, `anchor_fit_to_dict`, `fit_anchor`
- From `ftf.walkforward.runner`: `WalkForwardResult`, `run_walkforward`
- External packages: None

**External Dependencies**
- Expected to be imported by: scripts and higher-level orchestration/reporting code (e.g., run pipelines that need schedule/train/run via a single import path).
- Key exports used elsewhere: `build_walkforward_schedule`, `fit_anchor`, `run_walkforward` (plus their result/container types).

**Implementation Notes**
- Architecture decisions: Uses a thin `__init__.py` to centralize and stabilize the public API via re-exports, preventing callers from depending on deeper module paths.
- Cross-File Relationships: This module is the glue layer over:
  - `schedule.py` (anchor generation) → `trainer.py` (train-only fitting per anchor) → `runner.py` (execute engine per anchor + stitch OOS).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/configs/base_fast.yaml; ROUND 46 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:18:51
**File Implemented**: ftf_repro/configs/base_fast.yaml

**Core Purpose**:
- Defines the baseline “FAST” experiment configuration used across the pipeline (data handling, signals, risk, sizing, execution, walk-forward, stats, and capacity), ensuring a fully explicit, reproducible parameter set.

**Public Interface**:
- Constants/Types:
  - `run_name`: `base_fast` — identifier used for naming outputs/artifacts.
  - YAML sections (configuration keys consumed by the codebase):
    - `time`: `{exec_lag, stop_fill_policy, stitch_rule}`
    - `data`: `{tz_naive_dates, calendar, roll_bd_before_fnd, price_col, high_col, low_col, open_col, volume_col, adv_col, contract_multiplier}`
    - `signal`: `{ema_lambda, momentum_k, blend_omega, pbull_threshold, z_clip}`
    - `risk`: `{ewma_theta, vol_target_annual, w_max}`
    - `atr_exit`: `{atr_window, hard_stop_atr, trailing_stop_atr, timeout_days, price_reference_for_peak, derisk_policy}`
    - `costs`: `{k_linear, gamma_impact}`
    - `kelly`: `{lambda_kelly, baseline_floor, baseline_floor_mode, baseline_floor_eps}`
    - `walkforward`: `{train_bd, test_bd, step_bd, anchor_start, anchor_end, trainer_mode}`
    - `regression`: `{nw_lags, nw_lags_sensitivity}`
    - `bootstrap`: `{block_bootstrap_B, block_len, stationary_bootstrap_B, stationary_mean_block, seed}`
    - `capacity`: `{participation_cap}`

**Internal Dependencies**:
- From other modules/files: None (pure YAML configuration).
- External packages: None directly (loaded by the project’s config loader, e.g., via `pyyaml` in `ftf/utils/config.py`).

**External Dependencies**:
- Expected to be imported by: configuration loading and experiment orchestration layers, primarily:
  - `ftf_repro/src/ftf/utils/config.py` (parsing/validation of YAML)
  - `ftf_repro/scripts/01_build_data.py`, `02_run_fast_oos.py`, `05_spa.py`, `07_report.py` (run orchestration)
  - `ftf_repro/src/ftf/walkforward/{schedule.py,trainer.py,runner.py}` (WF settings, trainer mode, stitching)
  - `ftf_repro/src/ftf/trading/engine.py`, `execution/{latency.py,costs.py,fills.py}` (exec lag, stop fill policy, cost params)
  - `ftf_repro/src/ftf/signals/*`, `risk/*`, `sizing/*`, `trading/{atr.py,exits.py}` (model hyperparameters)
  - `ftf_repro/src/ftf/stats/{bootstrap.py,regression.py}` (bootstrap + HAC regression lags)
  - Future capacity scripts/modules (participation cap)
- Key exports used elsewhere: the named keys/values above (treated as a schema contract).

**Implementation Notes**:
- Architecture decisions:
  - “Baseline FAST” is explicitly parameterized (no implicit defaults), to make walk-forward runs deterministic and artifact-reproducible.
  - Uses “FIRST_STEP_ONLY” stitching, enforcing non-overlapping OOS aggregation as the canonical series.
  - Encodes baseline execution conventions: `exec_lag: 1` and `STOP_FILL_T_PLUS_1` to align decisions at *t* with fills at *t+1*.
- Cross-File Relationships:
  - Signal → risk → sizing → exits → execution → stats chain is fully configured here (e.g., `ema_lambda`, `ewma_theta`, `lambda_kelly`, ATR stop multipliers, cost parameters).
  - Regression and bootstrap settings ensure reporting scripts use consistent NW lags and resampling parameters.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/configs/grids/latency_grid.yaml; ROUND 47 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:19:10
**File Implemented**: ftf_repro/configs/grids/latency_grid.yaml

**Core Purpose**
- Defines a small robustness experiment grid for execution latency, allowing the walk-forward pipeline to be rerun under different forecast-to-fill delays (T+0, T+1, T+2) for sensitivity analysis.

**Public Interface**
- Constants/Config Schema (YAML):
  - `run_name`: `latency_grid` — identifier for the grid run/artifacts.
  - `baseline_name`: `lag1` — names the baseline configuration used for comparisons in reports.
  - `configs`: list of experiment variants:
    - `name`: `lag0` | `overrides.time.exec_lag: 0`
    - `name`: `lag1` | `overrides.time.exec_lag: 1`
    - `name`: `lag2` | `overrides.time.exec_lag: 2`

**Internal Dependencies**
- From project config loader (implicit): this YAML is expected to be read/parsed by the project’s configuration utilities (e.g., YAML loader in `ftf.utils.config`) and merged as overrides onto a base config.
- External packages: none directly (data file only).

**External Dependencies**
- Expected to be imported by: `ftf_repro/scripts/03_latency.py` (explicitly referenced in file header comment).
- Key exports used elsewhere:
  - The `configs[*].overrides.time.exec_lag` value, which drives execution lag behavior in the trading/execution pipeline (`w_exec[t] = w_target[t - exec_lag]` per the comment).

**Implementation Notes**
- Architecture decisions:
  - Uses a simple “grid of overrides” pattern: each entry supplies only the delta from baseline (here, `time.exec_lag`).
  - Declares a canonical baseline (`lag1`) to standardize tables/plots and comparisons.
- Cross-File Relationships:
  - `scripts/03_latency.py` likely iterates `configs`, applies each `overrides` onto a base config, runs the walk-forward runner, then aggregates metrics with baseline alignment using `baseline_name`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/03_latency.py; ROUND 48 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:20:05
**File Implemented**: ftf_repro/scripts/03_latency.py

**Core Purpose**:
- Runs a latency robustness experiment by re-executing the walk-forward backtest across a YAML-defined grid of execution delays (and any other overrides), saving per-variant artifacts and aggregated performance comparisons vs a designated baseline.

**Public Interface** (what other files can use, if any):
- Function `_parse_args()`: CLI parser for config/grid/data/out/seed/progress options -> `argparse.Namespace`
- Function `_dict_to_cfg(d: Dict[str, Any])`: reconstructs a validated typed `FTFConfig` from a raw dict (YAML-loaded) -> `FTFConfig`
- Function `_run_one(df_cont: pd.DataFrame, base_cfg_dict: Dict[str, Any], overrides: Dict[str, Any], out_dir: Path, *, progress: bool)`: runs one grid variant, persists outputs, returns minimal aggregation payload -> `Dict[str, Any]` containing `{cfg, run_dir, net_ret, w_exec}`
- Function `main()`: orchestrates grid execution, aggregation, and report writing -> `None`
- Constants/Types: none (script-style module)

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.reporting`: `performance_table` (computes metrics table from return panels; optionally uses w_exec)
- From `ftf.utils`:  
  - `FTFConfig` (type annotation / returned config)  
  - `deep_update` (merge base config with per-variant overrides)  
  - `ensure_dir` (create output folders)  
  - `load_parquet`, `load_yaml` (inputs)  
  - `save_json`, `save_parquet`, `save_yaml` (artifacts)  
  - `set_global_seed` (reproducibility)  
  - `validate_config` (config validation gate)
- From `ftf.walkforward.runner`: `run_walkforward` (core WF engine invoked per variant)
- External packages:
  - `argparse` (CLI)
  - `dataclasses.asdict` (imported but unused in this script)
  - `pathlib.Path` (filesystem paths)
  - `typing` (`Any`, `Dict`)
  - `numpy` (imported but unused in this script)
  - `pandas` (I/O, series/panel assembly, CSVs)

**External Dependencies** (what depends on this file, if any):
- Expected to be invoked by: end users / experiment runners via CLI (`python scripts/03_latency.py ...`)
- Key exports used elsewhere: none (script entrypoint; not designed as a library module)

**Implementation Notes**:
- Architecture decisions:
  - Uses a base YAML config plus grid entry overrides merged via `deep_update`, and forces `run_name` to be stable per grid entry (`name`) for deterministic output folder naming.
  - Rebuilds a *typed* config (`FTFConfig` and nested dataclasses) from dict via `_dict_to_cfg` to ensure validation and consistent defaults (mirrors the approach in `02_run_fast_oos.py`).
  - Produces both per-variant artifacts (config snapshot + stitched `oos_daily.parquet` + schema) and aggregated artifacts (performance CSV, meta CSV, summary JSON, aligned panel net returns parquet).
- Cross-File Relationships:
  - Delegates all trading logic and stitching to `ftf.walkforward.runner.run_walkforward`, and all metric computation to `ftf.reporting.performance_table`.
  - Expects the processed continuous futures dataset schema produced by earlier pipeline steps (loaded via `load_parquet`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/configs/grids/cost_impact_grid.yaml; ROUND 49 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:20:24
**File Implemented**: ftf_repro/configs/grids/cost_impact_grid.yaml

**Core Purpose**  
- Defines a small experiment grid for stress-testing the strategy under higher/lower transaction cost and market impact assumptions by overriding baseline `costs.k_linear` and `costs.gamma_impact` values.

**Public Interface**  
- Constants/Types (YAML keys as the “interface” consumed by grid runners):
  - `run_name`: `cost_impact_grid` — identifier for the grid run/output namespace.
  - `baseline_name`: `x1.0` — marks which grid entry should be treated as the baseline scenario.
  - `configs`: list of grid entries, each with:
    - `name`: scenario label (`x0.5`, `x1.0`, `x1.5`, `x2.0`)
    - `overrides.costs.k_linear`: float — per-unit turnover linear cost rate.
    - `overrides.costs.gamma_impact`: float — turnover^(3/2) impact coefficient.

**Internal Dependencies**  
- From {module/file}: None (pure YAML configuration data).
- External packages: None directly (parsed/loaded by the project’s config utilities elsewhere).

**External Dependencies**  
- Expected to be imported by: `ftf_repro/scripts/04_cost_impact.py` (grid runner) and any config-loading utilities (e.g., deep-merge logic).
- Key exports used elsewhere: the YAML structure (`run_name`, `baseline_name`, `configs[*].overrides`) to parameterize repeated backtest runs.

**Implementation Notes**  
- Architecture decisions: Uses “deep-merge overrides onto a base config” pattern to keep stress tests minimal and composable; only cost parameters are changed per scenario.
- Cross-File Relationships: Intended to be applied on top of a base strategy config (e.g., `configs/base_fast.yaml`) by a grid-running script that iterates `configs`, merges `overrides`, and executes the pipeline with each modified cost model.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/04_cost_impact.py; ROUND 50 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:21:12
**File Implemented**: ftf_repro/scripts/04_cost_impact.py

**Core Purpose**:
- Runs a robustness/stress grid over transaction cost (`k_linear`) and market impact (`gamma_impact`) parameters by repeatedly executing the strict walk-forward backtest, saving per-variant artifacts and producing aggregated comparison tables/panels.

**Public Interface**:
- Function `_parse_args()`: CLI argument parsing for base config, grid spec, processed data path, output directory, seed, and progress flag -> `argparse.Namespace`.
- Function `_dict_to_cfg(d: Dict[str, Any])`: Reconstructs a fully-typed `FTFConfig` tree from a nested dict (typically YAML) and validates it -> `FTFConfig`.
- Function `_run_one(df_cont: pd.DataFrame, base_cfg_dict: Dict[str, Any], overrides: Dict[str, Any], out_dir: Path, *, progress: bool)`: Executes one grid variant (deep-merged config) via walk-forward runner, persists artifacts, and returns payload (cfg, run_dir, net returns, optional executed weights, and a small metadata summary) -> `Dict[str, Any]`.
- Function `main()`: Orchestrates loading inputs, iterating grid variants, aggregating results, and writing summary outputs -> `None`.

**Internal Dependencies**:
- From `ftf.reporting`: `performance_table`
- From `ftf.utils`:  
  `FTFConfig`, `deep_update`, `ensure_dir`, `load_parquet`, `load_yaml`, `save_json`, `save_parquet`, `save_yaml`, `set_global_seed`, `validate_config`
- From `ftf.walkforward.runner`: `run_walkforward`
- Local (inside `_dict_to_cfg`) from `ftf.utils.config`:  
  `ATRExitConfig`, `BootstrapConfig`, `CapacityConfig`, `CostImpactConfig`, `DataConfig`, `KellyConfig`, `RegressionConfig`, `RiskConfig`, `SignalConfig`, `TimeConvention`, `WalkForwardConfig`
- External packages:
  - `argparse` - CLI interface
  - `pathlib.Path` - path handling/output layout
  - `pandas` - series/panel assembly and CSV output
  - `dataclasses.asdict` - imported but not used in this file

**External Dependencies**:
- Expected to be invoked by: command line users / experiment runners (not designed as a library module).
- Key exports used elsewhere: none (script-style; primary entrypoint is `main()` under `__name__ == "__main__"`).

**Implementation Notes**:
- Architecture decisions:
  - Grid iteration uses `deep_update()` to overlay per-variant overrides on the base YAML dict, then rehydrates a validated typed `FTFConfig` via `_dict_to_cfg()` (ensures config schema correctness after merging).
  - Forces `run_name` to the grid variant `name` to make output folders deterministic (`<out>/<run_name>/`).
  - Stores per-variant `config_snapshot.yaml` and `oos_daily.parquet` for reproducibility and later analysis.
- Cross-File Relationships:
  - Delegates actual strategy simulation and canonical OOS stitching to `ftf.walkforward.runner.run_walkforward`.
  - Uses `ftf.reporting.performance_table` to compute aggregated metrics from a dict of OOS return series (and optionally executed weight panels).
  - Reads continuous futures data produced by `scripts/01_build_data.py` and varies only cost model coefficients via grid YAML (typically `configs/grids/cost_impact_grid.yaml`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/capacity/__init__.py; ROUND 51 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:21:33
**File Implemented**: ftf_repro/src/ftf/capacity/__init__.py

**Core Purpose**:
- Defines the `ftf.capacity` subpackage boundary by re-exporting the main capacity-analysis primitives (growth curve, participation diagnostics, and AUM capacity mapping) as a clean public API.

**Public Interface** (what other files can use, if any):
- Type `GrowthCurveResult`: result container/type for growth-curve outputs (defined in `growth_curve.py`; re-exported here)
- Function `estimate_unit_notional_stats(...)`: estimates μ/σ stats from unit-notional sleeve returns (re-export)
- Function `growth_curve(...)`: computes leverage-space growth curve under costs/impact (re-export)
- Function `solve_L_max(...)`: solves for maximum leverage where growth crosses zero (re-export)
- Function `contracts_delta(...)`: converts weight/AUM changes to contract count changes (re-export)
- Function `participation_rate(...)`: computes participation (contracts vs ADV) (re-export)
- Function `summarize_participation(...)`: aggregates participation distribution/summary stats (re-export)
- Function `estimate_aum_capacity(...)`: maps participation constraints to an AUM capacity estimate (re-export)
- Constants/Types:
  - `__all__`: explicitly lists the intended stable exports of the `ftf.capacity` namespace.

**Internal Dependencies** (what this file imports/requires, if any):
- From `.growth_curve`: `GrowthCurveResult`, `estimate_unit_notional_stats`, `growth_curve`, `solve_L_max`
- From `.participation`: `contracts_delta`, `participation_rate`, `summarize_participation`
- From `.aum_mapping`: `estimate_aum_capacity`
- External packages: none (pure package wiring)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: scripts and reporting/workflow code that wants a single import point for capacity analysis (e.g., `scripts/06_capacity.py`, potentially reporting utilities).
- Key exports used elsewhere: the re-exported functions/types listed in `__all__`, enabling `from ftf.capacity import ...` imports.

**Implementation Notes**:
- Architecture decisions: keeps the subpackage deterministic and “dependency-light” by only aggregating interfaces; no runtime logic in `__init__.py`.
- Cross-File Relationships: acts as a façade over three capacity modules—growth modeling (`growth_curve.py`), liquidity/participation calculations (`participation.py`), and translating participation limits into AUM (`aum_mapping.py`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/capacity/growth_curve.py; ROUND 52 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:22:27
**File Implemented**: ftf_repro/src/ftf/capacity/growth_curve.py

**Core Purpose**:
- Implements a lightweight, deterministic capacity “growth curve” in leverage space for the strategy, using a reduced-form objective \( g(L) \) derived from unit-notional sleeve return stats and the project’s linear + impact cost coefficients. Also provides a simple (non-scipy) solver for the maximum leverage \(L_{\max}\) where growth turns non-positive.

**Public Interface**:
- Class `GrowthCurveResult`: Container for computed growth curve outputs | Key methods: *(dataclass; no methods)* | Constructor params: `L, g, mu_u, sigma_u, n, k_linear, gamma_impact`
- Function `estimate_unit_notional_stats(R: pd.Series)`: Cleans a unit-notional sleeve return series and estimates daily mean/std -> `Tuple[float, float]`: `(mu_u, sigma_u)`
- Function `growth_proxy_L(L: np.ndarray | float, *, mu_u: float, sigma_u: float, n: float = 1.0, costs: Optional[CostImpactConfig] = None, k_linear: Optional[float] = None, gamma_impact: Optional[float] = None)`: Computes reduced-form growth proxy \(g(L)\) with linear+impact penalties -> `np.ndarray | float`
- Function `growth_curve(*, mu_u: float, sigma_u: float, L_grid: Optional[Iterable[float]] = None, L_max: float = 5.0, n: float = 1.0, costs: Optional[CostImpactConfig] = None, k_linear: Optional[float] = None, gamma_impact: Optional[float] = None)`: Evaluates \(g(L)\) on a leverage grid and packages results -> `GrowthCurveResult`
- Function `solve_L_max(*, mu_u: float, sigma_u: float, n: float = 1.0, costs: Optional[CostImpactConfig] = None, k_linear: Optional[float] = None, gamma_impact: Optional[float] = None, bracket: Tuple[float, float] = (0.0, 10.0), grid_points: int = 2001)`: Finds first positive-root crossing (downward) of \(g(L)\) within bracket via dense grid + linear interpolation -> `float` (returns `0.0` if no crossing found)
- Constants/Types:
  - `__all__`: `["GrowthCurveResult", "estimate_unit_notional_stats", "growth_proxy_L", "growth_curve", "solve_L_max"]`

**Internal Dependencies**:
- From `ftf.utils.config`: `CostImpactConfig` (default source of `k_linear` and `gamma_impact` coefficients)
- External packages:
  - `numpy` - vectorized computation, grid creation, interpolation, sign-change detection
  - `pandas` - input validation/cleaning for return series stats (`estimate_unit_notional_stats`)
  - `dataclasses` - immutable result container (`@dataclass(frozen=True)`)
  - `typing` - type hints (`Iterable, Optional, Tuple`)

**External Dependencies**:
- Expected to be imported by: `ftf_repro/scripts/06_capacity.py`, and likely future modules `ftf.capacity.participation` / `ftf.capacity.aum_mapping` to obtain `mu_u/sigma_u`, \(g(L)\), and `L_max` for capacity reporting.
- Key exports used elsewhere: `growth_curve`, `solve_L_max`, `estimate_unit_notional_stats`, `GrowthCurveResult`

**Implementation Notes**:
- Architecture decisions:
  - Avoids `scipy` by implementing `solve_L_max` as a deterministic dense grid search with first sign-change detection and linear interpolation.
  - Cost parameters are sourced from `CostImpactConfig` by default, but can be overridden explicitly (`k_linear`, `gamma_impact`) for sensitivity runs.
  - Supports negative leverage inputs in `growth_proxy_L` for diagnostics; impact term uses `abs(L)` while linear term keeps sign (`n*k*L`).
- Cross-File Relationships:
  - Aligns capacity math with sizing/execution assumptions by reusing the same linear cost (`k_linear`) and impact coefficient (`gamma_impact`) configuration object used elsewhere in the backtest.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/capacity/participation.py; ROUND 53 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:23:25
**File Implemented**: ftf_repro/src/ftf/capacity/participation.py

**Core Purpose**:
- Provides deterministic, vectorized liquidity/participation utilities for capacity analysis by converting executed weight changes and AUM into estimated futures contracts traded and participation rates vs ADV, plus summary statistics for mapping to capacity constraints.

**Public Interface**:
- Class `ParticipationSummary`: Summary container for participation distribution stats | Key methods: (dataclass; no methods) | Constructor params: `n, median, p90, p95, p99, mean, max`
- Function `contracts_delta(delta_w: pd.Series, *, aum: float, price: pd.Series, adv: Optional[pd.Series] = None, contract_multiplier: float = 100.0) -> pd.Series`: Converts executed weight changes into estimated absolute contracts traded -> `pd.Series` named `"delta_contracts"`
- Function `participation_rate(delta_w: pd.Series, *, aum: float, price: pd.Series, adv: pd.Series, contract_multiplier: float = 100.0) -> pd.Series`: Computes participation rate `q_t = |Δcontracts_t| / ADV_t` -> `pd.Series` named `"participation"`
- Function `summarize_participation(q: pd.Series) -> ParticipationSummary`: Produces count + distribution stats (median, p90/p95/p99, mean, max) for a participation series -> `ParticipationSummary`
- Function `representative_participation_inputs(w_exec: pd.Series, price: pd.Series, adv: pd.Series, *, active_threshold: float = 1e-3) -> Dict[str, float]`: Computes representative medians used for AUM mapping (median |Δw| on active days, median ADV, median price, active-day count) -> `dict`
- Constants/Types:
  - `__all__`: Exposes `ParticipationSummary`, `contracts_delta`, `participation_rate`, `summarize_participation`, `representative_participation_inputs`

**Internal Dependencies**:
- From `ftf.utils.config`: `DataConfig` (imported but not used in this file)
- External packages:
  - `pandas` - Series alignment (`align`, `concat`), indexing validation, quantiles/median
  - `numpy` - NaN/inf handling, numeric constants

**External Dependencies**:
- Expected to be imported by: `ftf_repro/scripts/06_capacity.py`, and/or `ftf_repro/src/ftf/capacity/aum_mapping.py` (to map participation caps to max AUM)
- Key exports used elsewhere: `participation_rate`, `contracts_delta`, `summarize_participation`, `representative_participation_inputs`, `ParticipationSummary`

**Implementation Notes**:
- Architecture decisions:
  - Strict input validation via `_check_series()` (DatetimeIndex, monotonic, no duplicates) to prevent silent misalignment and ensure deterministic time-series math.
  - Vectorized calculations with explicit inner-join alignment to common dates; robust handling of zeros (converted to NaN) and infinities (replaced with NaN).
  - Contracts conversion uses absolute weight changes (`abs()`), consistent with participation focusing on traded quantity magnitude.
- Cross-File Relationships:
  - Intended to consume executed weights (`w_exec`) from the trading engine/logs, along with futures `price` and `adv` from processed data; outputs participation series and summary stats to support capacity reporting and AUM limit estimation.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/src/ftf/capacity/aum_mapping.py; ROUND 54 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:24:10
**File Implemented**: ftf_repro/src/ftf/capacity/aum_mapping.py

**Core Purpose**:
- Provides deterministic utilities to translate executed strategy turnover (Δ executed weights) and market liquidity (ADV) into participation rates and an estimated AUM capacity limit under a participation cap.

**Public Interface** (what other files can use, if any):
- Class `AUMCapacityResult`: Immutable container for AUM capacity estimation outputs | Key methods: *(dataclass; no methods)* | Constructor params: `aum_max, q_cap, method, rep_delta_w, rep_price, rep_adv, contract_multiplier`
- Function `participation_for_aum(w_exec: pd.Series, *, aum: float, price: pd.Series, adv: pd.Series, contract_multiplier: float = 100.0)`: Computes daily participation series `q_t` implied by executed weight changes for a given AUM -> `pd.Series`: participation rate time series
- Function `estimate_aum_capacity(w_exec: pd.Series, *, price: pd.Series, adv: pd.Series, participation_cap: float = 0.01, contract_multiplier: float = 100.0, active_threshold: float = 1e-3)`: Estimates maximum AUM consistent with a participation cap using representative medians -> `AUMCapacityResult`: capacity estimate + representative inputs used
- Function `aum_participation_summary(w_exec: pd.Series, *, aum: float, price: pd.Series, adv: pd.Series, contract_multiplier: float = 100.0)`: Convenience wrapper computing both `q_t` and summary stats -> `Tuple[pd.Series, ParticipationSummary]`
- Function `capacity_dict(res: AUMCapacityResult)`: Serializes capacity result to JSON-friendly scalars -> `Dict[str, float]`
- Constants/Types:
  - `__all__`: exports `AUMCapacityResult`, `participation_for_aum`, `estimate_aum_capacity`, `aum_participation_summary`, `capacity_dict`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.capacity.participation`:
  - `ParticipationSummary`: return type for summarized participation
  - `participation_rate`: core computation turning `Δw`, AUM, price, ADV into `q_t`
  - `representative_participation_inputs`: computes representative medians (median |Δw| on active days, median ADV, median price)
  - `summarize_participation`: produces summary statistics for a participation series
- External packages:
  - `numpy`: finiteness checks and `inf` handling
  - `pandas`: Series/DatetimeIndex validation, alignment via index behavior, diff/fillna operations
  - `dataclasses`: frozen result struct (`@dataclass(frozen=True)`)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: `ftf_repro/scripts/06_capacity.py` (capacity report script), potentially reporting modules that display capacity estimates.
- Key exports used elsewhere: `estimate_aum_capacity`, `participation_for_aum`, `aum_participation_summary`, `AUMCapacityResult`

**Implementation Notes**: (if any)
- Architecture decisions:
  - Uses executed weights (`w_exec`) rather than targets, and bases trading volume on daily weight changes `delta_w = w_exec.diff()`.
  - Deterministic, index-safe design: `_check_series` enforces `DatetimeIndex`, monotonic increasing order, no duplicates, and strips timezone if present.
  - Capacity estimation uses a “median mapping” heuristic consistent with the plan:  
    `A_max ≈ q_cap * median_price * contract_multiplier * median_ADV / median(|Δw| active)`.
  - Edge case handling: if representative active-day turnover is nonpositive or non-finite, returns `aum_max = inf` with method tag `median_abs_delta_w_active<=0`.
- Cross-File Relationships:
  - Delegates the core microstructure math (participation computation and summarization) to `ftf.capacity.participation`, while this module focuses on AUM-level mapping and packaging results.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/scripts/06_capacity.py; ROUND 55 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:25:12
**File Implemented**: ftf_repro/scripts/06_capacity.py

**Core Purpose** (provide a general overview of the file's main responsibility):
- Provides a CLI runner to perform end-to-end capacity analysis: loads or runs a baseline walk-forward backtest, estimates unit-notional return stats, computes the leverage growth curve and break-even leverage, and maps turnover/ADV into an AUM capacity estimate under a participation cap.

**Public Interface** (what other files can use, if any):
- Function `_parse_args()`: CLI argument parsing for selecting run directory vs config+data, output paths, seed, AUM list, participation cap, and leverage grid -> `argparse.Namespace`
- Function `_dict_to_cfg(d: Dict[str, Any])`: Converts a raw YAML-loaded dict into a validated `FTFConfig` dataclass -> `FTFConfig`
- Function `_load_or_run(*, run_dir: Optional[str], config_path: Optional[str], processed_path: Optional[str], out_dir: Optional[str])`: Loads `oos_daily` + config from an existing run directory, or executes walk-forward from config and processed data -> `(pd.DataFrame, FTFConfig, Path)`
- Function `_unit_notional_proxy_returns(daily: pd.DataFrame)`: Builds an OOS proxy “unit-notional sleeve” return series using executed exposure indicator and raw returns -> `pd.Series`
- Function `main()`: Orchestrates the full capacity pipeline and writes artifacts -> `None`

**Internal Dependencies** (what this file imports/requires, if any):
- From `ftf.capacity`: `estimate_aum_capacity`, `estimate_unit_notional_stats`, `growth_curve`, `solve_L_max`
- From `ftf.capacity.aum_mapping`: `aum_participation_summary`, `capacity_dict`
- From `ftf.reporting`: `plot_growth_curve`
- From `ftf.sizing.kelly`: `estimate_kelly_inputs`
- From `ftf.utils`: `FTFConfig`, `deep_update` (imported but unused), `ensure_dir`, `load_parquet`, `load_yaml`, `save_json`, `save_yaml`, `set_global_seed`, `validate_config`
- From `ftf.walkforward.runner`: `run_walkforward`
- External packages:
  - `argparse` - CLI interface
  - `dataclasses.asdict` - serializing dataclass summaries to dict
  - `pathlib.Path` - filesystem path handling
  - `numpy`, `pandas` - numeric ops and tabular I/O (CSV/Parquet)

**External Dependencies** (what depends on this file, if any):
- Expected to be imported by: none (script-style entrypoint; run via CLI)
- Key exports used elsewhere: none (primary interface is command-line execution)

**Implementation Notes**: (if any)
- Architecture decisions:
  - Supports two modes:
    - **Load mode** via `--run_dir`: expects `config_snapshot.yaml` and `reports/oos_daily.parquet`.
    - **Run mode** via `--config` + `--processed_path`: runs `run_walkforward()` into a subdirectory, then uses its `oos_daily`.
  - Uses a pragmatic **unit-notional proxy** derived from `w_exec` activity (`1{w_exec[t-1] > 1e-3}`) times raw return `r[t]`, rather than requiring a separately-logged unit-notional sleeve.
  - Parses `--aum_list` using restricted `eval(..., {"__builtins__": {}}, {})` to allow scientific notation strings like `1e9`.
  - Writes a consistent artifact bundle: `capacity_summary.{json,yaml}`, `growth_curve.csv`, `participation_summaries.csv`, per-AUM participation parquet series, and an optional `growth_curve.png`.
- Cross-File Relationships:
  - Relies on walk-forward output schema (`oos_daily` containing at least `r`, `w_exec`, plus `cfg.data.price_col` and `cfg.data.adv_col`) produced by `ftf.walkforward.runner`/`ftf.trading.engine`.
  - Capacity math is delegated to `ftf.capacity.*` modules, while participation/AUM summaries are delegated to `ftf.capacity.aum_mapping`, and plotting to `ftf.reporting`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_roll.py; ROUND 56 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:25:44
**File Implemented**: ftf_repro/tests/test_roll.py

**Core Purpose**
- Provides pytest unit tests validating the continuous futures front-month roll logic: (1) the roll occurs exactly `roll_bd_before_fnd` business days before first notice date (FND), and (2) the continuous series price splice switches to the next contract on the roll day.

**Public Interface**
- Function `_mk_contract_df(idx: pd.DatetimeIndex, base: float)`: Builds a deterministic synthetic OHLCV/ADV DataFrame for a single contract over `idx` -> `pd.DataFrame`.
- Function `test_roll_occurs_two_business_days_before_fnd()`: Asserts active-contract selection respects the “date < FND - 2 business days” eligibility rule and cross-validates via `validate_roll_rule()` -> `None` (pytest test).
- Function `test_continuous_splice_switches_prices_on_roll_day()`: Asserts the continuous close equals the new contract’s close on the roll day and equals the prior contract’s close just before -> `None` (pytest test).
- Constants/Types: None (all test-local variables).

**Internal Dependencies**
- From `ftf.data.calendar`: `get_calendar` (build NYSE business-day index; shift business days).
- From `ftf.data.futures_roll`: `build_continuous_front_month` (constructs continuous series and active contract labels).
- From `ftf.data.validation`: `validate_roll_rule` (checks active contract does not violate roll cutoff rule).
- From `ftf.utils.config`: `DataConfig` (provides `roll_bd_before_fnd` config used by roll logic).
- External packages:
  - `pandas` - constructing indices, DataFrames, timestamps, and series used in synthetic fixtures and assertions.
  - `pytest` - `approx` for float comparisons and test discovery/execution.

**External Dependencies**
- Expected to be imported by: none (pytest discovers and runs tests by filename/pattern).
- Key exports used elsewhere: none (tests only).

**Implementation Notes**
- Architecture decisions:
  - Uses deterministic synthetic contract price paths (`base + 0.1 * t`) with distinct bases per contract to make roll/splice visibly testable.
  - Constrains FNDs to business days to avoid ambiguity in calendar shifting.
- Cross-File Relationships:
  - Validates that `build_continuous_front_month()` (data construction) and `validate_roll_rule()` (rule checker) agree, ensuring the roll rule is implemented consistently across builder and validator.
  - Relies on `get_calendar("NYSE")` and its `bdays()` / `shift()` helpers to mirror production calendar logic in tests.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_no_lookahead.py; ROUND 57 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:26:22
**File Implemented**: ftf_repro/tests/test_no_lookahead.py

**Core Purpose**
- Provides lightweight, synthetic unit tests that enforce “no look-ahead” (causality) invariants for regime feature computation and ATR calculation, ensuring values at time *t* do not depend on data from *t+1* onward.

**Public Interface**
- Function `_mk_ohlc(n: int = 220, seed: int = 0)`: Creates a synthetic OHLC pandas DataFrame on a business-day index for deterministic testing -> `pd.DataFrame`.
- Function `test_regime_no_lookahead_random_day()`: Pytest test ensuring `p_bull` and `eligible_to_enter` at a chosen timestamp match when recomputed from truncated history -> `None` (assertions).
- Function `test_atr_no_lookahead_matches_truncated()`: Pytest test ensuring ATR at time *t* matches recomputation from truncated history -> `None` (assertions).
- Function `test_atr_does_not_use_future_close_in_tr()`: Pytest test ensuring ATR at time *t* is invariant to extreme perturbations in `close` at *t+1* -> `None` (assertions).
- Constants/Types: None exported for production use (test module only).

**Internal Dependencies**
- From `ftf.signals.regime`: `fit_regime_state`, `compute_regime_features`
- From `ftf.trading.atr`: `compute_atr`
- External packages:
  - `numpy` - RNG generation, finite checks, numeric comparisons
  - `pandas` - business-day index creation and Series/DataFrame slicing by date

**External Dependencies**
- Expected to be imported by: `pytest` test discovery (not intended as a library module)
- Key exports used elsewhere: None (tests only)

**Implementation Notes**
- Architecture decisions:
  - Uses synthetic OHLC generation (`_mk_ohlc`) to avoid reliance on proprietary datasets and keep CI fast/deterministic.
  - Causality is tested by recomputing indicators on a truncated time series up to a chosen date `t` and requiring exact equality at `t` with the full-series computation.
- Cross-File Relationships:
  - Validates that `ftf.signals.regime.compute_regime_features()` is purely causal given a fixed fitted `state` from `fit_regime_state(train)`.
  - Validates that `ftf.trading.atr.compute_atr()` uses only permissible inputs (up to `t`, with `C_{t-1}` allowed via true range definition), specifically guarding against inadvertent use of `close[t+1]`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_atr_exits.py; ROUND 58 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:27:11
**File Implemented**: ftf_repro/tests/test_atr_exits.py

**Core Purpose**:
- Provides pytest unit tests for the ATR-based exit/state-machine logic, validating deterministic triggering of hard stop, trailing stop, and timeout exits, plus verifying the “peak tracking” reference choice (close vs high) affects trailing-stop behavior as intended.

**Public Interface**:
- Function `_mk_path_for_hard_stop()`: Builds a synthetic OHLC path that produces a sharp drop after entry to trigger a hard stop -> `pd.DataFrame`.
- Function `_mk_path_for_trailing_stop(*, peak_ref: str = "close")`: Builds a synthetic OHLC path with an uptrend then drop, with configurable peak reference behavior via highs vs closes -> `pd.DataFrame`.
- Function `_mk_path_for_timeout()`: Builds a flat synthetic OHLC path to force exit via timeout -> `pd.DataFrame`.
- Function `_run_exit_engine(df: pd.DataFrame, *, entry_day: int, exit_cfg: ATRExitConfig)`: Helper that runs the ATR exit engine end-to-end on synthetic data -> `tuple[pd.Series, pd.DataFrame]` (`w_target`, `events`).
- Test `test_hard_stop_triggers_exit_event_and_flattens_target()`: Asserts ENTRY and EXIT_HARD_STOP occur and that `w_target` is flattened on the hard-stop date.
- Test `test_trailing_stop_uses_peak_reference_close_vs_high()`: Asserts EXIT_TRAILING_STOP occurs for both peak modes, and that using highs produces an exit no later than using closes; also checks flattening on exit date.
- Test `test_timeout_exit_after_timeout_days()`: Asserts ENTRY and EXIT_TIMEOUT occur and that `w_target` is flattened on the timeout date.
- Constants/Types: none exported (this is a test module; helpers are internal by convention).

**Internal Dependencies**:
- From `ftf.trading.atr`: `compute_atr`
- From `ftf.trading.exits`: `fit_atr_exit_state`, `generate_target_weights`
- From `ftf.utils.config`: `ATRExitConfig`
- External packages:
  - `pandas` (`pd`) - constructing business-day indices, Series/DataFrames, and event date extraction.

**External Dependencies**:
- Expected to be imported by: none (pytest discovers and runs tests by filename/pattern).
- Key exports used elsewhere: none (tests only).

**Implementation Notes**:
- Architecture decisions:
  - Uses fully synthetic OHLC series so tests are deterministic and data-independent.
  - Forces “always long when eligible” by setting `w_raw = 1.0` and `eligible_to_enter` true from `entry_day` onward, isolating exit mechanics.
  - Disables unrelated exits in certain tests (e.g., `hard_stop_atr=999.0`) to focus on the targeted rule.
  - Uses `stop_fill_policy="STOP_FILL_T_PLUS_1"` when fitting the exit state to match the project’s default convention, while still asserting that the *target* is flattened on the signal/trigger date.
- Cross-File Relationships:
  - `compute_atr` supplies ATR inputs to `generate_target_weights`, which applies the configured ATR exit state returned by `fit_atr_exit_state`.
  - `ATRExitConfig` parameterizes all exit thresholds and conventions (ATR window, stop multipliers, timeout, and peak reference).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_costs.py; ROUND 59 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:27:32
**File Implemented**: ftf_repro/tests/test_costs.py

**Core Purpose**:
- Provides pytest unit tests validating that execution turnover and trading cost calculations (linear + impact) match the project’s deterministic formulas and behave sensibly under simple scenarios.

**Public Interface**:
- None (test module only; not intended as a reusable library interface).
- Test functions:
  - `test_turnover_and_costs_match_definition()`: Verifies `turnover_from_exec` and `compute_costs` outputs exactly match hand-computed turnover and cost formulas.
  - `test_costs_zero_when_weights_constant()`: Ensures constant executed weights produce zero turnover and zero total costs.
  - `test_costs_nonnegative_and_increase_with_turnover()`: Checks costs are nonnegative and that a higher-turnover path yields strictly higher cumulative total cost.

**Internal Dependencies**:
- From `ftf.execution.costs`: `compute_costs`, `turnover_from_exec`
- External packages:
  - `numpy`: builds constant weight arrays for tests.
  - `pandas`: creates business-day indices and `Series`; uses `pd.testing.assert_series_equal` for exact series comparisons.
  - `pytest`: uses `pytest.approx` for floating-point sum comparisons.

**External Dependencies**:
- Expected to be imported by: `pytest` test discovery (executed as part of the test suite).
- Key exports used elsewhere: None (only assertions/tests).

**Implementation Notes**:
- Architecture decisions:
  - Uses deterministic, small synthetic weight paths to make expected turnover/cost values analytically obvious and stable.
  - Validates both exact per-day series equality (including names like `"turnover"`, `"cost_linear"`, etc.) and aggregate properties (zero-sum, monotonicity with turnover).
- Cross-File Relationships:
  - Exercises the core execution friction model implemented in `ftf.execution.costs` by asserting its outputs match the canonical definitions used throughout the trading engine/backtests.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_hac.py; ROUND 60 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:27:56
**File Implemented**: ftf_repro/tests/test_hac.py

**Core Purpose**:
- Provides pytest-based validation for the HAC/Newey–West regression helper (`hac_regression`) using synthetic return series, ensuring beta recovery, alpha annualization, and correct index alignment behavior.

**Public Interface**:
- Function `_mk_returns(n: int = 800, seed: int = 0, beta: float = 0.0, alpha: float = 0.0002)`: Generates synthetic strategy and benchmark daily return `pd.Series` with controllable alpha/beta -> `tuple[pd.Series, pd.Series]`.
- Test `test_hac_regression_beta_near_zero_when_constructed()`: Asserts beta is finite and near zero when constructed with `beta=0.0`.
- Test `test_hac_regression_recovers_positive_beta()`: Asserts beta is finite and within a reasonable band around the constructed positive beta (`0.6`).
- Test `test_hac_regression_alpha_annualization_sanity()`: Validates `alpha_ann == 252 * alpha_daily` and positive annualized alpha under positive daily alpha.
- Test `test_hac_regression_alignment_inner_join()`: Ensures regression uses intersection of indices (inner join), reflected in `res.n_obs`.

**Internal Dependencies**:
- From `ftf.stats.regression`: `hac_regression`
- External packages:
  - `numpy` (`np`) - RNG and numeric assertions (`isfinite`)
  - `pandas` (`pd`) - business-day index generation and `Series` construction/alignment
  - `pytest` - approximate equality check (`pytest.approx`) and test framework

**External Dependencies**:
- Expected to be imported by: `pytest` test discovery (no runtime imports expected from application code).
- Key exports used elsewhere: None (this is a test module; it consumes `hac_regression`).

**Implementation Notes**:
- Architecture decisions:
  - Uses i.i.d. normal benchmark and noise to create a simple linear data-generating process sufficient to test sign/magnitude and stability of regression outputs without requiring complex time-series structure.
  - Uses `pd.bdate_range` to match project convention of business-day indexed returns.
- Cross-File Relationships:
  - Directly validates `ftf.stats.regression.hac_regression` outputs/fields: `beta`, `alpha_daily`, `alpha_ann`, `n_obs`, and implicitly its alignment logic (intersection of dates) and annualization convention (`252`).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_kelly.py; ROUND 61 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:28:32
**File Implemented**: ftf_repro/tests/test_kelly.py

**Core Purpose**:
- Provides deterministic, synthetic unit tests validating the friction-adjusted Kelly sizing implementation: (1) the closed-form optimizer matches a numerical grid maximization of the growth proxy and (2) the solution behaves correctly under edge/cost/impact changes.

**Public Interface**:
- Function `_grid_argmax_f(*, inputs: KellyInputs, k_linear: float, gamma_impact: float, f_max: float = 5.0, n_grid: int = 20001) -> float`: Dense-grid numerical maximizer for the Kelly growth proxy used only for test validation.
- Function `test_closed_form_matches_grid_argmax_reasonably() -> None`: Randomized scenario test comparing `solve_friction_adjusted_kelly` vs grid argmax and verifying near-optimal growth value.
- Function `test_f_star_zero_when_mu_leq_nk() -> None`: Asserts `f* == 0` when expected edge is non-positive after linear costs (`mu <= n*k`).
- Function `test_f_star_decreases_with_higher_cost_or_impact() -> None`: Monotonicity checks: higher linear cost `k` or impact `gamma` should not increase optimal leverage `f*`.
- Function `test_f_star_is_zero_when_sigma2_is_degenerate() -> None`: Degenerate variance handling: solver should return finite, nonnegative `f*` even when `sigma2 == 0`.

**Internal Dependencies**:
- From `ftf.sizing.kelly`: `KellyInputs`, `growth_proxy`, `solve_friction_adjusted_kelly`
- External packages:
  - `numpy` - random scenario generation, grid creation, numeric argmax, finite checks
  - `pytest` - approximate comparisons (`pytest.approx`) and test discovery/assertions

**External Dependencies**:
- Expected to be imported by: none (pytest discovers and runs it as a test module)
- Key exports used elsewhere: none (all functions are test cases or helpers)

**Implementation Notes**:
- Architecture decisions:
  - Uses a very dense grid (`n_grid=20001` over `[0, 5]`) to create a robust numerical reference for the closed-form solution.
  - Adds tolerance in comparisons (`abs=2.5e-3`, `rel=1e-2`) to account for grid discretization and flat optima.
  - Verifies not only parameter proximity (`f*`) but also objective proximity (`g(f*)` near `max(g)`), reducing false passes.
- Cross-File Relationships:
  - Directly exercises the sizing math in `ftf/sizing/kelly.py` by evaluating both `growth_proxy` and `solve_friction_adjusted_kelly`, ensuring the analytical solution matches the implemented objective.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/tests/test_walkforward.py; ROUND 62 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:29:05
**File Implemented**: ftf_repro/tests/test_walkforward.py

**Core Purpose**:
- Provides lightweight, fully synthetic CI-friendly tests for the walk-forward pipeline, focusing on schedule kept-window semantics, stitched OOS non-overlap, and runner determinism in FIXED trainer mode.

**Public Interface**:
- Function `_mk_continuous_df(n: int = 3600) -> pd.DataFrame`: Builds a deterministic synthetic continuous OHLCV/ADV dataset on a business-day index sized to support default train/test windows.
- Function `_mk_cfg(*, stitch_rule: str = "FIRST_STEP_ONLY") -> FTFConfig`: Constructs a complete `FTFConfig` with FIXED walk-forward parameters and tunable stitching rule.
- Function `test_schedule_kept_window_first_step_only_has_step_length_or_less() -> None`: Asserts that each anchor’s kept OOS segment length is `<= step_bd` under `FIRST_STEP_ONLY`.
- Function `test_walkforward_stitched_oos_has_no_duplicate_dates() -> None`: Runs the walk-forward pipeline and asserts stitched OOS output is monotonic, has no duplicate dates, and has plausible length.
- Function `test_walkforward_is_deterministic_for_fixed_mode() -> None`: Runs walk-forward twice and asserts identical OOS indices and return series.

**Internal Dependencies**:
- From `ftf.utils.config`:
  - `ATRExitConfig, BootstrapConfig, CapacityConfig, CostImpactConfig, DataConfig, FTFConfig, KellyConfig, RegressionConfig, RiskConfig, SignalConfig, TimeConvention, WalkForwardConfig` (used to build a full configuration object for the runner).
- From `ftf.walkforward.runner`:
  - `run_walkforward` (executes end-to-end walk-forward and returns stitched OOS outputs + anchors).
- From `ftf.walkforward.schedule`:
  - `build_walkforward_schedule` (builds anchor schedule; used to validate kept-window semantics).
- External packages:
  - `pandas` (dataframe creation, business-day index, and testing utilities like `pd.testing.*`).

**External Dependencies**:
- Expected to be imported by: `pytest` test discovery/runner (invoked as part of the test suite).
- Key exports used elsewhere: None (test module only; no reusable library exports intended).

**Implementation Notes**:
- Architecture decisions:
  - Uses a deterministic synthetic price path (mild trend + sinusoid) to avoid flaky tests and reliance on proprietary data.
  - Enforces canonical non-overlap behavior by testing the `FIRST_STEP_ONLY` stitching rule and asserting no duplicate dates in stitched OOS.
  - Validates determinism specifically for `trainer_mode="FIXED"` by comparing full OOS return series across repeated runs.
- Cross-File Relationships:
  - Couples `build_walkforward_schedule()` (schedule semantics) with `run_walkforward()` (actual stitched output) to ensure the runner respects schedule-kept windows.
  - Relies on `FTFConfig` structure to mirror real pipeline configuration while remaining test-light (no persistence, no progress output).

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/README.md; ROUND 63 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:29:47
**File Implemented**: ftf_repro/README.md

**Core Purpose** (provide a general overview of the file's main responsibility):
- Documents how to set up the environment, provide required input data, and run the end-to-end Forecast-to-Fill reproduction pipeline (data build → walk-forward backtest → robustness grids → SPA → capacity → reporting).
- Defines the project’s key timing conventions and canonical OOS stitching rule so results are reproducible and comparable.

- No programmatic public interface (no classes/functions/constants). This is user-facing documentation only.
- CLI entrypoints referenced (as commands, not Python-callable interfaces):
  - `scripts/01_build_data.py`: build continuous futures dataset + diagnostics
  - `scripts/02_run_fast_oos.py`: run baseline walk-forward OOS
  - `scripts/03_latency.py`: latency sensitivity grid
  - `scripts/04_cost_impact.py`: cost/impact stress grid
  - `scripts/05_spa.py`: SPA / White Reality Check
  - `scripts/06_capacity.py`: capacity analysis (growth curve + AUM mapping)
  - `scripts/07_report.py`: reporting bundle (metrics/CI/regression)

**Internal Dependencies** (what this file imports/requires, if any):
- None (Markdown documentation; no imports).
- External tooling implied by instructions:
  - `pip` / `venv` for environment setup
  - `pytest` for running tests

**External Dependencies** (what depends on this file, if any):
- Expected to be referenced by: repository users and maintainers; CI/docs workflows (if any).
- Key exports used elsewhere: none (not imported by code).

**Implementation Notes**: (if any)
- Architecture decisions:
  - Emphasizes strict time convention: decisions at close *t*, executed at close *t+lag* (default T+1), and P&L attribution based on `w_exec[t-1] * r[t]` with costs charged on executed turnover.
  - Declares canonical OOS stitching rule: **FIRST_STEP_ONLY** to avoid overlap/double-counting in walk-forward evaluation.
  - Explicitly states data is not shipped and specifies minimal required columns for per-contract bars and contract metadata (FND).
- Cross-File Relationships:
  - Connects the full workflow across scripts: processed continuous futures produced by `01_build_data.py` becomes the `--processed_path` input to backtests/grids/SPA; outputs of the baseline run become inputs to `06_capacity.py` and `07_report.py`.

---
*Auto-generated by Memory Agent*



================================================================================
## IMPLEMENTATION File ftf_repro/requirements.txt; ROUND 64 
================================================================================

# Code Implementation Summary
**Generated**: 2026-01-27 21:30:06
**File Implemented**: ftf_repro/requirements.txt

**Core Purpose** (provide a general overview of the file's main responsibility):
- Defines the Python dependency set (with minimum versions) required to run the `ftf_repro` project, covering data processing, statistics, reporting, testing, and optional acceleration.

**Public Interface** (what other files can use, if any):
- Constants/Types: N/A (this is a dependency specification file; no runtime symbols are exported)

**Internal Dependencies** (what this file imports/requires, if any):
- From {module/file}: N/A
- External packages:
  - `numpy>=1.26` - core numerical computing used throughout signals/risk/stats.
  - `pandas>=2.1` - timeseries/dataframe manipulation across data loaders, backtests, metrics.
  - `scipy>=1.11` - optimization/stats utilities (e.g., roots, distributions) in sizing/stats/capacity.
  - `statsmodels>=0.14` - HAC/Newey–West regression utilities for alpha/beta analysis.
  - `pyyaml>=6.0` - config parsing/serialization for experiment configs and frozen params.
  - `pyarrow>=14` - Parquet IO backend for processed datasets and artifacts.
  - `matplotlib>=3.8` - plotting figures in reporting.
  - `seaborn>=0.13` - higher-level statistical plots (optional but included for reporting).
  - `tqdm>=4.66` - progress bars for bootstraps/grids/walk-forward runs.
  - `pytest>=7.4` - unit/integration testing.
  - `numba>=0.59` - optional performance acceleration for tight loops.

**External Dependencies** (what depends on this file, if any):
- Expected to be used by: project installation workflows (`pip install -r requirements.txt`), CI/test runners, and any user environment setup for running `scripts/*`.
- Key exports used elsewhere: N/A (package list only)

**Implementation Notes**: (if any)
- Architecture decisions: Pins minimum versions aligned with Python 3.11-era ecosystems (pandas 2.x, statsmodels 0.14, pyarrow 14+) to ensure consistent behavior/performance across the walk-forward, regression, and Parquet pipelines.
- Cross-File Relationships: These dependencies enable the entire pipeline: ingestion (`pandas`, `pyarrow`), modeling (`numpy`, `scipy`), inference (`statsmodels`), experiment configuration (`pyyaml`), evaluation/reporting (`matplotlib`, `seaborn`), and reproducibility/testing (`pytest`, `tqdm`, optional `numba`).

---
*Auto-generated by Memory Agent*


