from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import math


@dataclass(frozen=True)
class BondReturnConfig:
    maturity_years: float = 10.0
    coupon_frequency: int = 1
    trading_days_per_year: int = 252
    default_modified_duration: float | None = None


def load_yield_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"日期", "到期收益率_百分比", "到期收益率_小数"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"yield curve missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["日期"]),
            "yield_pct": pd.to_numeric(df["到期收益率_百分比"], errors="coerce"),
            "yield_decimal": pd.to_numeric(df["到期收益率_小数"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "yield_decimal"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("yield curve has no valid rows")
    return out


def par_bond_modified_duration(
    yield_decimal: float,
    maturity_years: float = 10.0,
    coupon_frequency: int = 1,
) -> float:
    if yield_decimal <= -0.99:
        raise ValueError("yield_decimal is too low to compute duration")
    periods = int(round(maturity_years * coupon_frequency))
    if periods <= 0:
        raise ValueError("maturity_years and coupon_frequency imply no periods")

    period_yield = yield_decimal / coupon_frequency
    period_coupon = yield_decimal / coupon_frequency
    discount = 1.0 + period_yield

    weighted_pv = 0.0
    pv_total = 0.0
    for period in range(1, periods + 1):
        cash_flow = period_coupon
        if period == periods:
            cash_flow += 1.0
        pv = cash_flow / (discount**period)
        time_years = period / coupon_frequency
        weighted_pv += time_years * pv
        pv_total += pv

    macaulay = weighted_pv / pv_total
    return macaulay / discount


def build_total_return_index(
    yield_curve: pd.DataFrame,
    config: BondReturnConfig | None = None,
) -> pd.DataFrame:
    cfg = config or BondReturnConfig()
    data = yield_curve.copy().sort_values("date").reset_index(drop=True)
    data["previous_yield_decimal"] = data["yield_decimal"].shift(1)
    data["yield_change_decimal"] = data["yield_decimal"].diff()
    data["yield_change_bp"] = data["yield_change_decimal"] * 10000.0

    if cfg.default_modified_duration is None:
        data["modified_duration"] = data["previous_yield_decimal"].apply(
            lambda y: par_bond_modified_duration(
                float(y),
                maturity_years=cfg.maturity_years,
                coupon_frequency=cfg.coupon_frequency,
            )
            if pd.notna(y)
            else math.nan
        )
    else:
        data["modified_duration"] = cfg.default_modified_duration

    data["modified_duration"] = pd.to_numeric(data["modified_duration"], errors="coerce")
    data["carry_return"] = pd.to_numeric(data["previous_yield_decimal"] / cfg.trading_days_per_year, errors="coerce")
    data["duration_pnl"] = pd.to_numeric(-data["modified_duration"] * data["yield_change_decimal"], errors="coerce")
    data["total_return"] = pd.to_numeric(data["carry_return"] + data["duration_pnl"], errors="coerce")
    data.loc[data.index[0], ["carry_return", "duration_pnl", "total_return"]] = 0.0
    data["nav"] = (1.0 + data["total_return"].fillna(0.0)).cumprod()
    data["benchmark_nav"] = data["nav"]
    return data
