# Forecast-to-Fill (ftf_repro)

Reproduction codebase for the **Forecast-to-Fill** walk-forward gold-futures strategy described in the reproduction plan:

- Strict walk-forward (10y train / 6m test / 1m step)
- Decision at **close of day _t_**, execution with **T+lag close** (default **T+1**) 
- EMA-slope + momentum regime → confidence shaping → vol targeting → (friction+impact) fractional Kelly sizing
- ATR(14) hard/trailing stops + timeout exits
- Deterministic turnover-based cost + impact model
- OOS stitching rule: **FIRST_STEP_ONLY** (canonical non-overlapping OOS)
- Evaluation: performance metrics, Newey–West/HAC regression vs LBMA spot, bootstrap Sharpe CI, SPA/Reality Check
- Capacity: growth curve in leverage space and AUM mapping via participation vs ADV

> This repository **does not ship data**. You must provide your own daily per-contract GC futures bars, contract FND metadata, and (optionally) LBMA spot.

## Data (important) Parquet files are NOT included in this repository. All analysis assumes the following local file: reports/base_fast_nocost_final/reports/oos_daily.parquet To reproduce it, run: python scripts/02_run_fast_oos.py \ --config configs/base_fast_nocost_final.yaml

---

## 1) Environment

Python **3.11** (3.10 acceptable).

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ftf_repro/requirements.txt
```

Run tests:

```bash
pytest -q
```

---

## 2) Data format (inputs)

### 2.1 Contract bars (per contract)

One file per contract (CSV or Parquet). Required columns:

- `close` (or whatever you configure as `data.price_col`)
- `high` (`data.high_col`)
- `low` (`data.low_col`)

Optional columns:

- `open` (`data.open_col`)
- `volume` (`data.volume_col`)
- `adv` (`data.adv_col`)  (contracts/day)

Index / date column:

- Must be daily dates. The loader will infer a date column (e.g. `date`) or use an existing index.
- Dates are normalized to midnight, timezone-naive.

### 2.2 Contract metadata

A single CSV/Parquet with (at least) columns:

- `contract` (string, must match filenames’ stems)
- `fnd` (first notice date)

### 2.3 LBMA spot (optional; regression benchmark)

CSV/Parquet with date + a price column (default `price`).

---

## 3) Pipeline: build continuous futures

Create a continuous front-month GC series, rolling **2 business days before FND**:

```bash
python ftf_repro/scripts/01_build_data.py \
  --contracts_dir /path/to/contracts/ \
  --metadata_path /path/to/contract_metadata.parquet \
  --out_dir ftf_repro/data/processed
```

Outputs (under `--out_dir`):

- `gc_continuous.parquet` (continuous OHLC(+vol/adv))
- `active_contract.parquet` (active contract per day)
- `roll_table.parquet` (roll diagnostics)
- `validation_report.json`

---

## 4) Run baseline FAST walk-forward OOS

Using the baseline config:

```bash
python ftf_repro/scripts/02_run_fast_oos.py \
  --config ftf_repro/configs/base_fast.yaml \
  --processed_path ftf_repro/data/processed/gc_continuous.parquet \
  --out_dir ftf_repro/reports/base_fast
```

Key outputs:

- `config_snapshot.yaml`
- `reports/oos_daily.parquet` (stitched daily log table)
- `reports/oos_summary.json`
- `artifacts/anchors/<anchor_date>/...` (per-anchor frozen params + per-anchor daily logs + events)

Timing convention (critical):

- `w_target[t]` is the decision-time target at close of day **t**.
- `w_exec[t] = w_target[t - exec_lag]` (default `exec_lag=1`).
- P&L attribution:
  - `gross_ret[t] = w_exec[t-1] * r[t]`
  - costs are charged on executed turnover: `|w_exec[t]-w_exec[t-1]|`
  - `net_ret[t] = gross_ret[t] - costs[t]`

OOS stitching (critical): default is **FIRST_STEP_ONLY** to avoid overlapping OOS slices.

---

## 5) Robustness scripts

### 5.1 Latency sensitivity

```bash
python ftf_repro/scripts/03_latency.py \
  --config ftf_repro/configs/base_fast.yaml \
  --grid ftf_repro/configs/grids/latency_grid.yaml \
  --processed_path ftf_repro/data/processed/gc_continuous.parquet \
  --out_dir ftf_repro/reports/latency_grid
```

### 5.2 Cost/impact stress grid

```bash
python ftf_repro/scripts/04_cost_impact.py \
  --config ftf_repro/configs/base_fast.yaml \
  --grid ftf_repro/configs/grids/cost_impact_grid.yaml \
  --processed_path ftf_repro/data/processed/gc_continuous.parquet \
  --out_dir ftf_repro/reports/cost_impact_grid
```

### 5.3 SPA / White Reality Check

```bash
python ftf_repro/scripts/05_spa.py \
  --config ftf_repro/configs/base_fast.yaml \
  --grid ftf_repro/configs/grids/spa_grid.yaml \
  --processed_path ftf_repro/data/processed/gc_continuous.parquet \
  --out_dir ftf_repro/reports/spa
```

---

## 6) Capacity analysis

Capacity analysis requires `adv` in the processed continuous dataset.

```bash
python ftf_repro/scripts/06_capacity.py \
  --run_dir ftf_repro/reports/base_fast \
  --out_dir ftf_repro/reports/capacity
```

This produces a leverage-space growth curve and an AUM mapping under a participation cap (default 1%).

---

## 7) Reporting

Generate a light report bundle (metrics, bootstrap Sharpe CI, optional HAC regression vs LBMA):

```bash
python ftf_repro/scripts/07_report.py \
  --run_dir ftf_repro/reports/base_fast \
  --out_dir ftf_repro/reports/report
```

With LBMA spot regression:

```bash
python ftf_repro/scripts/07_report.py \
  --run_dir ftf_repro/reports/base_fast \
  --lbma_path /path/to/lbma_spot.parquet \
  --out_dir ftf_repro/reports/report
```

---

## 8) Notes / design choices

- **Calendar**: uses a NYSE-like business day calendar based on US Federal holidays (pragmatic proxy).
- **ATR**: simple rolling-mean ATR(14) over True Range (not Wilder smoothing), matching the plan.
- **Stops**: stop triggers occur at close _t_; baseline exits via T+1 lag (execution handled by `exec_lag`). A sensitivity option exists via `time.stop_fill_policy`.
- **Trainer**:
  - `walkforward.trainer_mode: FIXED` uses canonical constants.
  - `walkforward.trainer_mode: GRID` grid-searches hyperparameters on **training only**, selecting by best **train net Sharpe** (tie-breaker: lower turnover).

---

## License / disclaimer

This is research/reproduction code. It is **not** investment advice and is provided without any warranty. You are responsible for validating data quality, trading assumptions, and regulatory constraints.
