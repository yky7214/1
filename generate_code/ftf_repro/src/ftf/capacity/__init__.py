"""Capacity analysis subpackage.

Implements simplified capacity / scalability diagnostics used in the reproduction plan:

- Growth curve in leverage space L (proxy for long-only strategy growth after
  linear costs and concave impact).
- Participation-based AUM mapping using futures contract ADV.

The modules are intentionally deterministic and dependency-light.
"""

from .growth_curve import GrowthCurveResult, estimate_unit_notional_stats, growth_curve, solve_L_max
from .participation import contracts_delta, participation_rate, summarize_participation
from .aum_mapping import estimate_aum_capacity

__all__ = [
    "GrowthCurveResult",
    "estimate_unit_notional_stats",
    "growth_curve",
    "solve_L_max",
    "contracts_delta",
    "participation_rate",
    "summarize_participation",
    "estimate_aum_capacity",
]
