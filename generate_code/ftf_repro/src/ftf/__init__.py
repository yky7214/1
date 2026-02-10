"""Forecast-to-Fill (ftf) reproduction package.

This repository implements a walk-forward, latency-aware, cost/impact-adjusted
trend+momentum gold-futures strategy pipeline.

The package is organized into submodules:
- ftf.data: ingestion, calendar alignment, continuous futures roll
- ftf.signals: EMA slope + momentum regime signal
- ftf.risk: EWMA volatility targeting + confidence shaping
- ftf.sizing: friction-adjusted fractional Kelly sizing
- ftf.trading: ATR exits + deterministic daily backtest engine + logging
- ftf.execution: latency, fills, and cost/impact accounting
- ftf.walkforward: schedule, training/freeze, and stitching runner
- ftf.stats: metrics, regression (HAC), bootstrap, SPA/Reality Check
- ftf.capacity: growth curve and AUM/participation mapping
- ftf.reporting: tables and figures

Most users will interact via scripts in ftf_repro/scripts.
"""

from .utils.config import FTFConfig

__all__ = ["FTFConfig"]
