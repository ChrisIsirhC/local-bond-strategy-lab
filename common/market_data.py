from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.bond_return import BondReturnConfig, build_total_return_index, load_yield_curve


LOCAL_GOV_10Y = "local_gov_10y"
GOV_10Y = "gov_10y"
DEFAULT_BENCHMARK_ID = LOCAL_GOV_10Y
TRADED_ASSET_ID = LOCAL_GOV_10Y
CONDITIONAL_BENCHMARK_ID = "conditional_gov_cash"
CONDITIONAL_BENCHMARK_NAME = "多头同仓位10Y国债 / 非多头现金"

CURVE_SPECS = {
    LOCAL_GOV_10Y: {
        "label": "10Y地方政府债",
        "file": "地方政府债到期收益率_10年_2020至最新.csv",
    },
    GOV_10Y: {
        "label": "10Y国债",
        "file": "中债国债到期收益率_10年_2020至最新.csv",
    },
}


def normalize_benchmark_id(benchmark_id: str | None) -> str:
    selected = benchmark_id or DEFAULT_BENCHMARK_ID
    if selected not in CURVE_SPECS:
        raise ValueError(f"不支持的基准: {selected}")
    return selected


def benchmark_label(benchmark_id: str | None) -> str:
    return str(CURVE_SPECS[normalize_benchmark_id(benchmark_id)]["label"])


def curve_path(root: Path, curve_id: str) -> Path:
    normalized = normalize_benchmark_id(curve_id)
    return root / "benchmark_data" / str(CURVE_SPECS[normalized]["file"])


def load_market_data(root: Path, benchmark_id: str | None = None) -> pd.DataFrame:
    selected = normalize_benchmark_id(benchmark_id)
    asset = build_total_return_index(load_yield_curve(curve_path(root, TRADED_ASSET_ID)), BondReturnConfig())
    benchmark = build_total_return_index(load_yield_curve(curve_path(root, selected)), BondReturnConfig())

    asset_columns = {column: f"asset_{column}" for column in asset.columns if column != "date"}
    asset = asset.rename(columns=asset_columns)
    market = asset.merge(benchmark, on="date", how="inner", validate="one_to_one")
    if market.empty:
        raise ValueError("交易标的与所选基准没有重叠日期")
    market.attrs["benchmark_id"] = selected
    market.attrs["benchmark_name"] = benchmark_label(selected)
    market.attrs["asset_name"] = benchmark_label(TRADED_ASSET_ID)
    return market.sort_values("date").reset_index(drop=True)


def market_date_bounds(root: Path, benchmark_id: str | None = None) -> tuple[object, object]:
    market = load_market_data(root, benchmark_id)
    return market["date"].min().date(), market["date"].max().date()
