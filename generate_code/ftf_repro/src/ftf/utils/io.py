"""ftf.utils.io

Minimal, deterministic IO helpers used across scripts and modules.

The project intentionally keeps IO conventions centralized to improve
reproducibility:
- YAML for configs and frozen-parameter snapshots
- JSON for small artifacts/log headers
- Parquet for large time-series tables

All paths are treated as filesystem paths (local). Functions accept either
str or pathlib.Path and create parent directories when writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Ensure a directory exists and return it as a Path.

    If ``path`` points to a file (has a suffix), its parent directory is
    created.
    """

    p = Path(path)
    dir_path = p if p.suffix == "" else p.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_yaml(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return {} if obj is None else obj


def save_yaml(obj: Any, path: PathLike) -> None:
    p = Path(path)
    ensure_dir(p)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_json(path: PathLike) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: PathLike, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=False)


def load_parquet(path: PathLike, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p, columns=columns)


def save_parquet(df: pd.DataFrame, path: PathLike, *, index: bool = True) -> None:
    p = Path(path)
    ensure_dir(p)
    df.to_parquet(p, index=index)


__all__ = [
    "PathLike",
    "ensure_dir",
    "load_yaml",
    "save_yaml",
    "load_json",
    "save_json",
    "load_parquet",
    "save_parquet",
]
