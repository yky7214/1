"""Walk-forward orchestration subpackage.

This package wires together:

- :mod:`ftf.walkforward.schedule` to build deterministic train/test/step anchors
- :mod:`ftf.walkforward.trainer` to fit *train-only* frozen parameters per anchor
- :mod:`ftf.walkforward.runner` to run the trading engine on each anchor and
  stitch overlapping out-of-sample slices into one canonical OOS series.

The project’s critical time conventions are implemented in the trading engine
and controlled by :class:`ftf.utils.config.TimeConvention`.

Public API is re-exported here to keep import paths stable.
"""

from .schedule import WalkForwardAnchor, build_walkforward_schedule
from .trainer import AnchorFit, anchor_fit_to_dict, fit_anchor
from .runner import WalkForwardResult, run_walkforward

__all__ = [
    "WalkForwardAnchor",
    "build_walkforward_schedule",
    "AnchorFit",
    "fit_anchor",
    "anchor_fit_to_dict",
    "WalkForwardResult",
    "run_walkforward",
]
