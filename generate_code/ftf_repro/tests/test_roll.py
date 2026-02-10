import pandas as pd
import pytest

from ftf.data.calendar import get_calendar
from ftf.data.futures_roll import build_continuous_front_month
from ftf.data.validation import validate_roll_rule
from ftf.utils.config import DataConfig


def _mk_contract_df(idx: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    # Deterministic OHLC bars; prices differ across contracts so splice is visible.
    close = pd.Series(base + pd.RangeIndex(len(idx)).astype(float).to_numpy() * 0.1, index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
            "adv": 2000.0,
        },
        index=idx,
    )
    return df


def test_roll_occurs_two_business_days_before_fnd():
    cal = get_calendar("NYSE")
    cfg = DataConfig(roll_bd_before_fnd=2)

    # Build a synthetic business-day index.
    idx = cal.bdays("2020-01-02", "2020-02-28")

    # Two sequential contracts.
    # Define FNDs such that they fall on business days within idx.
    fnd1 = pd.Timestamp("2020-02-10")
    fnd2 = pd.Timestamp("2020-03-10")  # outside sample; keeps contract2 valid

    # Contract bars: align to same calendar already.
    c1 = _mk_contract_df(idx, base=1500.0)
    c2 = _mk_contract_df(idx, base=1600.0)

    meta = pd.DataFrame({"contract": ["C1", "C2"], "fnd": [fnd1, fnd2]}).set_index("contract")

    res = build_continuous_front_month({"C1": c1, "C2": c2}, meta, cfg=cfg, start=idx[0], end=idx[-1])

    # Rule: active contract must satisfy date < (FND - 2 bdays)
    cutoff1 = cal.shift(fnd1, -cfg.roll_bd_before_fnd)

    # On cutoff day and after, C1 must not be active.
    assert (res.active_contract.loc[cutoff1:] != "C1").all()

    # Just before cutoff, it may still be C1 (given C2 eligible).
    prev_bd = cal.shift(cutoff1, -1)
    assert res.active_contract.loc[prev_bd] == "C1"

    # Validate with shared validator.
    n_viol, viol_df = validate_roll_rule(
        res.active_contract,
        meta["fnd"],
        calendar=cal,
        roll_bd_before_fnd=cfg.roll_bd_before_fnd,
    )
    assert n_viol == 0
    assert viol_df.empty


def test_continuous_splice_switches_prices_on_roll_day():
    cal = get_calendar("NYSE")
    cfg = DataConfig(roll_bd_before_fnd=2)
    idx = cal.bdays("2020-01-02", "2020-02-28")

    fnd1 = pd.Timestamp("2020-02-10")
    fnd2 = pd.Timestamp("2020-03-10")

    c1 = _mk_contract_df(idx, base=100.0)
    c2 = _mk_contract_df(idx, base=200.0)
    meta = pd.DataFrame({"contract": ["C1", "C2"], "fnd": [fnd1, fnd2]}).set_index("contract")

    res = build_continuous_front_month({"C1": c1, "C2": c2}, meta, cfg=cfg)
    cutoff1 = cal.shift(fnd1, -cfg.roll_bd_before_fnd)

    # On cutoff1 day (roll day) we should already be on contract2.
    assert res.active_contract.loc[cutoff1] == "C2"

    # Prices should come from contract2 on that day.
    assert res.df_cont.loc[cutoff1, "close"] == pytest.approx(c2.loc[cutoff1, "close"])

    # And just before should be contract1.
    prev_bd = cal.shift(cutoff1, -1)
    assert res.active_contract.loc[prev_bd] == "C1"
    assert res.df_cont.loc[prev_bd, "close"] == pytest.approx(c1.loc[prev_bd, "close"])
