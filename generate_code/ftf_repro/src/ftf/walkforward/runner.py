"""Walk-forward runner.

This module orchestrates the strict walk-forward backtest described in the
reproduction plan. It:

- builds a walk-forward schedule (anchors)
- trains/fits *only on the training slice* and freezes parameters per anchor
- runs the trading engine on each anchor's test slice
- stitches overlapping OOS slices into one canonical OOS series
  (baseline: FIRST_STEP_ONLY)
- persists per-anchor artifacts when an output directory is provided

The runner is intentionally deterministic and designed for reproducible
pipelines and unit testing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass,replace

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ftf.trading.engine import EngineResult, run_engine
from ftf.utils.config import FTFConfig, StitchRule
from ftf.utils.io import ensure_dir, save_json, save_parquet, save_yaml
from ftf.walkforward.schedule import WalkForwardAnchor, build_walkforward_schedule
from ftf.walkforward.trainer import AnchorFit, fit_anchor


@dataclass(frozen=True)
class WalkForwardResult:
    """Outputs of a full walk-forward run."""

    oos_daily: pd.DataFrame
    oos_net_ret: pd.Series
    oos_gross_ret: pd.Series
    anchors: List[WalkForwardAnchor]
    per_anchor: Dict[str, EngineResult]
    frozen_params: Dict[str, Dict[str, Any]]


def _anchor_name(t0: pd.Timestamp) -> str:
    return pd.Timestamp(t0).strftime("%Y-%m-%d")


def _slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by DatetimeIndex")
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def _kept_oos_window(anchor: WalkForwardAnchor, *, stitch_rule: StitchRule) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if stitch_rule == "FIRST_STEP_ONLY":
        return anchor.test_start, anchor.kept_end
    if stitch_rule == "FULL_TEST_DIAGNOSTIC":
        return anchor.test_start, anchor.test_end
    raise ValueError(f"Unknown stitch_rule: {stitch_rule}")


def _stitch_first_step_only(engines: Dict[str, EngineResult], anchors: List[WalkForwardAnchor], *, cfg: FTFConfig) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    # Track the end of the already-kept window to avoid overlaps (no double-counting)
    last_end: Optional[pd.Timestamp] = None

    for a in anchors:
        name = _anchor_name(a.test_start)
        daily = engines[name].daily

        keep_start, keep_end = _kept_oos_window(a, stitch_rule=cfg.time.stitch_rule)

        # Prevent overlap with previously kept window
        if last_end is not None and keep_start < last_end:
            keep_start = last_end

        part = daily.loc[(daily.index >= keep_start) & (daily.index < keep_end)].copy()

        # If this anchor contributes nothing after trimming, skip it
        if part.empty:
            last_end = max(last_end, keep_end) if last_end is not None else keep_end
            continue

        part["anchor"] = name
        parts.append(part)

        # Advance last_end
        last_end = max(last_end, keep_end) if last_end is not None else keep_end

    if not parts:
        raise ValueError("No anchors produced stitched output")

    out = pd.concat(parts, axis=0).sort_index()

    # Safety check (should not happen now)
    if out.index.has_duplicates:
        dup = out.index[out.index.duplicated()].unique()
        raise AssertionError(f"Stitched OOS has duplicate dates (double-counting): {list(map(str, dup[:10]))}")

    return out



def run_walkforward(
    df_cont: pd.DataFrame,
    *,
    cfg: FTFConfig,
    out_dir: Optional[str | Path] = None,
    persist_daily: bool = True,
    persist_per_anchor: bool = True,
    progress: bool = False,
) -> WalkForwardResult:
    """Run the strict walk-forward pipeline.

    Parameters
    ----------
    df_cont:
        Continuous OHLC(+volume/adv) dataframe indexed by business-day dates.
    cfg:
        Full experiment configuration.
    out_dir:
        If provided, per-anchor frozen parameters and daily logs are persisted
        under this directory.
    persist_daily:
        If True, writes stitched OOS daily table and returns series.
    persist_per_anchor:
        If True, writes per-anchor daily logs and parameter snapshots.
    progress:
        If True, prints a lightweight progress line per anchor.

    Returns
    -------
    WalkForwardResult
        Includes stitched OOS daily table and net return series.
    """

    if not isinstance(df_cont, pd.DataFrame):
        raise TypeError("df_cont must be a pandas DataFrame")
    if not isinstance(df_cont.index, pd.DatetimeIndex):
        raise TypeError("df_cont must be indexed by a DatetimeIndex")
    if not df_cont.index.is_monotonic_increasing:
        df_cont = df_cont.sort_index()

    anchors = build_walkforward_schedule(df_cont.index, cfg=cfg)
    # --- deduplicate anchors by test_start (avoid duplicate OOS dates) ---
    seen = set()
    dedup: List[WalkForwardAnchor] = []
    for a in anchors:
        key = pd.Timestamp(a.test_start)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(a)
    anchors = dedup



    engines: Dict[str, EngineResult] = {}
    frozen_params: Dict[str, Dict[str, Any]] = {}

    out_path: Optional[Path] = None
    if out_dir is not None:
        out_path = ensure_dir(Path(out_dir))
        ensure_dir(out_path / "artifacts")
        ensure_dir(out_path / "artifacts" / "anchors")

    for i, a in enumerate(anchors):
        name = _anchor_name(a.test_start)
        if progress:
            print(f"[{i+1:03d}/{len(anchors):03d}] anchor {name} train={a.train_start.date()}..{(a.train_end-pd.Timedelta(days=1)).date()} test={a.test_start.date()}..{(a.test_end-pd.Timedelta(days=1)).date()}")

        df_train = _slice_df(df_cont, a.train_start, a.train_end)
        df_test  = _slice_df(df_cont, a.test_start, a.test_end)

        fit: AnchorFit = fit_anchor(df_train, cfg=cfg)

        # =========================================================
        # NEW: add trailing context from train so EMA/momentum/ATR
        #      are continuous at test start (no reset / NaN burst)
        # =========================================================
        # momentum lookback k (from cfg, or from fit if you prefer)
        k = int(getattr(cfg.signal, "momentum_k", 0) or 0)
        atr_w = int(getattr(cfg.atr_exit, "atr_window", 0) or 0)
        exec_lag = int(getattr(cfg.time, "exec_lag", 0) or 0)

        # conservative minimum context:
        # - need at least k bars for momentum shift(k)
        # - need atr_window bars for ATR rolling
        # - need exec_lag bars for execution alignment
        # - plus small buffer to avoid boundary instability
        ema_lam = float(getattr(fit.regime_state.ema_state, "ema_lambda", 0.95))
        ema_burn = int(np.ceil(10.0 / max(1e-6, (1.0 - ema_lam))))
        ctx_len = max(k, atr_w, exec_lag, ema_burn, 50)


        df_ctx = df_train.tail(ctx_len)
        df_run = pd.concat([df_ctx, df_test], axis=0)

        # If any overlap in dates, keep the later (test) row
        df_run = df_run[~df_run.index.duplicated(keep="last")].sort_index()

        # Run engine on ctx+test for correct recursive features; trim later.
        eng_full = run_engine(
            df_run,
            cfg=cfg,
            regime_state=fit.regime_state,
            vol_state=fit.vol_state,
            policy_state=fit.policy_state,
            f_tilde=fit.f_tilde,
            exit_state=fit.exit_state,
            metadata={
                "anchor": name,
                "train_start": str(a.train_start.date()),
                "train_end": str(a.train_end.date()),
                "test_start": str(a.test_start.date()),
                "test_end": str(a.test_end.date()),
                "trainer_mode": cfg.walkforward.trainer_mode,
                "ctx_len": int(ctx_len),
            },
        )

        # =========================================================
        # NEW: trim engine outputs back to pure test window
        #      (so stitch & per-anchor daily.parquet keep same meaning)
        # =========================================================
        daily_test = eng_full.daily.loc[
            (eng_full.daily.index >= a.test_start) &
            (eng_full.daily.index <  a.test_end)
        ].copy()

        # Keep the event log as-is (it may include ctx period events; useful for debugging)
        eng = type(eng_full)(daily=daily_test, log=eng_full.log)

        engines[name] = eng
        frozen_params[name] = asdict(fit)

        if out_path is not None and persist_per_anchor:
            adir = out_path / "artifacts" / "anchors" / name
            ensure_dir(adir)
            # Save frozen params
            save_yaml(frozen_params[name], adir / "frozen_params.yaml")
            # Save daily logs (TEST ONLY, consistent with before)
            save_parquet(eng.daily, adir / "daily.parquet")
            # (optional) Save full ctx+test daily for diagnostics
            # save_parquet(eng_full.daily, adir / "daily_ctx_plus_test.parquet")
            # Save event log (json)
            save_json(eng.log.to_dict(), adir / "events.json")
     # ===== after finishing all anchors: stitch OOS =====
    stitched = _stitch_first_step_only(engines, anchors, cfg=cfg)

    oos_net_ret = stitched["net_ret"].copy()
    oos_gross_ret = stitched["gross_ret"].copy()

    if out_path is not None and persist_daily:
        ensure_dir(out_path / "reports")
        save_parquet(stitched, out_path / "reports" / "oos_daily.parquet")
        # Also save net returns only for convenience
        save_parquet(oos_net_ret.to_frame("net_ret"), out_path / "reports" / "oos_net_ret.parquet")
        save_parquet(oos_gross_ret.to_frame("gross_ret"), out_path / "reports" / "oos_gross_ret.parquet")
        save_yaml({"cfg": cfg.to_dict()}, out_path / "reports" / "config_snapshot.yaml")
        save_json(
            {"n_anchors": len(anchors), "stitch_rule": cfg.time.stitch_rule},
            out_path / "reports" / "walkforward_meta.json",
        )
        # ===== after finishing all anchors: stitch OOS =====
    # ===== after finishing all anchors: stitch OOS =====
    stitched = _stitch_first_step_only(engines, anchors, cfg=cfg)

    # ------------------------------------------------------------
    # Attach market columns needed for capacity (price / volume / adv)
    # (only if missing; avoid overlap errors)
    # ------------------------------------------------------------
    price_col = cfg.data.price_col
    vol_col = getattr(cfg.data, "volume_col", "volume")
    adv_col = cfg.data.adv_col

    # build ADV in tmp if missing but volume exists
    tmp = df_cont.copy()
    if adv_col not in tmp.columns and vol_col in tmp.columns:
        tmp[adv_col] = (
            tmp[vol_col]
            .astype(float)
            .rolling(window=20, min_periods=1)
            .mean()
        )

    # join only columns missing in stitched
    join_cols = []
    for c in [price_col, vol_col, adv_col]:
        if c in tmp.columns and c not in stitched.columns:
            join_cols.append(c)

    if join_cols:
        stitched = stitched.join(tmp[join_cols], how="left")

    missing_after = [c for c in [price_col, vol_col, adv_col] if c not in stitched.columns]
    if missing_after:
        print(f"[WARN] After join, missing columns in stitched: {missing_after}")

    # ------------------------------------------------------------------
    # NEW: unit-notional sleeve return (paper capacity input)
    #
    # Definition: same entry/exit decisions as the executed strategy,
    # but fixed notional = 1 when active. Use t-1 exposure convention.
    #
    # unit_active[t-1] = 1{ w_exec[t-1] != 0 }
    # unit_sleeve_ret[t] = unit_active[t-1] * r[t]
    # ------------------------------------------------------------------
    if "w_exec" not in stitched.columns:
        raise ValueError("stitched OOS daily missing required column 'w_exec' for unit sleeve")
    if "r" not in stitched.columns:
        raise ValueError("stitched OOS daily missing required column 'r' for unit sleeve")

    w_exec = stitched["w_exec"].astype(float)
    r = stitched["r"].astype(float)

    unit_active_prev = (w_exec.shift(1).fillna(0.0).abs() > 1e-12)
    stitched["unit_sleeve_active"] = unit_active_prev.astype(int)
    stitched["unit_sleeve_ret"] = unit_active_prev.astype(float) * r

    # Primary series
    oos_net_ret = stitched["net_ret"].copy()
    oos_gross_ret = stitched["gross_ret"].copy()

    if out_path is not None and persist_daily:
        ensure_dir(out_path / "reports")
        save_parquet(stitched, out_path / "reports" / "oos_daily.parquet")
        save_parquet(oos_net_ret.to_frame("net_ret"), out_path / "reports" / "oos_net_ret.parquet")
        save_parquet(oos_gross_ret.to_frame("gross_ret"), out_path / "reports" / "oos_gross_ret.parquet")
        save_yaml({"cfg": cfg.to_dict()}, out_path / "reports" / "config_snapshot.yaml")
        save_json(
            {"n_anchors": len(anchors), "stitch_rule": cfg.time.stitch_rule},
            out_path / "reports" / "walkforward_meta.json",
        )

    return WalkForwardResult(
        oos_daily=stitched,
        oos_net_ret=oos_net_ret,
        oos_gross_ret=oos_gross_ret,
        anchors=anchors,
        per_anchor=engines,
        frozen_params=frozen_params,
    )


