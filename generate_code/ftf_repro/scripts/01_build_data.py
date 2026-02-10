"""Build processed continuous futures dataset.

This script is intentionally vendor-agnostic: it expects an input directory
containing per-contract OHLC(+volume/ADV) tables plus a contract metadata table
with first notice dates (FND).

Outputs
-------
- data/processed/gc_continuous.parquet : continuous front-month OHLC(+vol/ADV)
- data/processed/gc_active_contract.parquet : active contract per day
- data/processed/gc_roll_table.parquet : roll diagnostics
- reports/data_validation.json : validation report

Expected input layout (default)
------------------------------
<data_dir>/contracts/*.csv|*.parquet
  each file contains daily bars for one contract. Contract symbol is inferred
  from filename stem.
<data_dir>/metadata.csv|metadata.parquet
  columns: contract, fnd

You can override paths via CLI flags.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from ftf.data.calendar import get_calendar
from ftf.data.futures_roll import build_continuous_front_month
from ftf.data.loaders import read_contract_metadata, read_contract_ohlc
from ftf.data.validation import validate_continuous_df
from ftf.utils import ensure_dir, load_yaml, save_json, save_parquet
from ftf.utils.config import DataConfig


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build continuous GC futures dataset")
    ap.add_argument("--data-dir", type=str, default="data/raw", help="Raw data directory")
    ap.add_argument(
        "--contracts-dir",
        type=str,
        default=None,
        help="Directory of per-contract OHLC files (default: <data-dir>/contracts)",
    )
    ap.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Contract metadata file path (default: <data-dir>/metadata.csv)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed datasets",
    )
    ap.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Output directory for reports/artifacts",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config to get DataConfig column conventions",
    )
    ap.add_argument(
        "--calendar",
        type=str,
        default="NYSE",
        help="Calendar name (default NYSE)",
    )
    ap.add_argument("--start", type=str, default=None, help="Optional start date")
    ap.add_argument("--end", type=str, default=None, help="Optional end date")
    return ap.parse_args()


def _load_datacfg_from_yaml(path: Path) -> DataConfig:
    d = load_yaml(path)
    data_d = d.get("data", {}) if isinstance(d, dict) else {}
    return DataConfig(**data_d)


def _find_contract_files(contracts_dir: Path) -> Dict[str, Path]:
    pats = ["*.parquet", "*.csv"]
    files = []
    for pat in pats:
        files.extend(sorted(contracts_dir.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No contract files found in {contracts_dir}")
    out: Dict[str, Path] = {}
    for p in files:
        sym = p.stem
        out[sym] = p
    return out


def main() -> None:
    args = _parse_args()

    data_dir = Path(args.data_dir)
    contracts_dir = Path(args.contracts_dir) if args.contracts_dir else data_dir / "contracts"
    metadata_path = Path(args.metadata_path) if args.metadata_path else data_dir / "metadata.csv"

    out_dir = ensure_dir(Path(args.output_dir))
    rep_dir = ensure_dir(Path(args.reports_dir))

    data_cfg = _load_datacfg_from_yaml(Path(args.config)) if args.config else DataConfig()

    # Load contract metadata (FND)
    meta = read_contract_metadata(metadata_path, date_col="fnd")

    # Load per-contract bars
    contract_files = _find_contract_files(contracts_dir)
    contract_bars: Dict[str, pd.DataFrame] = {}
    for contract, path in contract_files.items():
        if contract not in meta.index:
            # Ignore unknown files; allow extra artifacts in the directory.
            continue
        contract_bars[contract] = read_contract_ohlc(path, cfg=data_cfg, calendar_name=args.calendar)

    if not contract_bars:
        raise ValueError("No contract bars loaded. Check metadata and contracts directory.")

    # Build continuous
    res = build_continuous_front_month(
        contract_bars,
        meta.reset_index(),
        cfg=data_cfg,
        calendar_name=args.calendar,
        start=args.start,
        end=args.end,
    )

    # Validate
    fnd_by_contract = meta["fnd"]
    report = validate_continuous_df(
        res.df_cont,
        active_contract=res.active_contract,
        fnd_by_contract=fnd_by_contract,
        cfg=data_cfg,
        calendar_name=args.calendar,
    )

    # Persist
    save_parquet(res.df_cont, out_dir / "gc_continuous.parquet")
    save_parquet(res.active_contract.to_frame("active_contract"), out_dir / "gc_active_contract.parquet")
    save_parquet(res.roll_table, out_dir / "gc_roll_table.parquet")
    save_json(report.__dict__, rep_dir / "data_validation.json")

    # Small stdout summary
    cal = get_calendar(args.calendar)
    print("Built continuous front-month futures")
    print(f"Calendar: {cal.name}")
    print(f"Days: {report.n_days}, rolls: {report.n_rolls}, roll violations: {report.roll_rule_violations}")
    print(f"NaNs after warmup: {report.nan_after_warmup}")
    if report.ret_summary:
        print("Return summary:", report.ret_summary)


if __name__ == "__main__":
    main()
