"""ftf.utils.seed

Reproducibility helpers.

The reproduction plan requires deterministic bootstraps and any randomized
procedures (SPA/RC, stationary bootstrap, etc.). This module centralizes seeding
logic so scripts can call a single function before running experiments.

Notes
-----
- We avoid importing heavy optional libs unless present.
- The project is CPU-only; no torch/jax dependencies are assumed.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_global_seed"]


def set_global_seed(seed: int, *, deterministic_hash: bool = True) -> None:
    """Set global RNG seeds for common libraries.

    Parameters
    ----------
    seed:
        Integer seed.
    deterministic_hash:
        If True, sets ``PYTHONHASHSEED`` for deterministic hashing behavior.
        Note: this must be set before Python starts to fully affect hash
        randomization, but setting it here is still useful for subprocesses.

    Returns
    -------
    None
    """

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an int")

    seed_int = int(seed)

    if deterministic_hash:
        os.environ.setdefault("PYTHONHASHSEED", str(seed_int))

    random.seed(seed_int)
    np.random.seed(seed_int)

    # Optional: try to seed numba if available (numba uses numpy RNG most of the
    # time; still keep as a best-effort hook).
    try:
        import numba  # type: ignore

        # numba doesn't provide a single global seed setter across all versions.
        # This block is intentionally conservative.
        _ = numba
    except Exception:
        pass


def _get_env_seed(name: str = "FTF_SEED") -> Optional[int]:
    """Internal helper to parse an environment seed, if present."""

    s = os.environ.get(name)
    if s is None or str(s).strip() == "":
        return None
    try:
        return int(s)
    except Exception:
        return None
