from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategies.dashboard_signal_v1 import (
    DEFAULT_FACTOR_WINDOWS,
    DEFAULT_THRESHOLDS,
    DashboardThresholds,
    FactorWindowConfig,
    _bucket_multiplier,
    _nonbank_multiplier,
    _spread_change_multiplier,
    build_dashboard_factor_multipliers,
    signal_file_for_frequency,
)


# This is a separate research surface. None of these names are added to the
# production v1 signal or to the preserved 06/07 strategies.
BASE_FACTOR_COLUMNS = (
    "supply_amount",
    "supply_ratio",
    "supply_long",
    "bank_demand",
    "spread_gov",
    "spread_change",
    "spread_ncd",
    "nonbank_sentiment",
)
EXPANDED_FACTOR_COLUMNS = (
    "supply_amount_change",
    "bank_demand_change",
    "nonbank_sentiment_change",
    "spread_ncd_change",
    "spread_change_acceleration",
    "supply_bank_interaction",
    "supply_nonbank_interaction",
)
ALL_FACTOR_COLUMNS = BASE_FACTOR_COLUMNS + EXPANDED_FACTOR_COLUMNS

FACTOR_LABELS = {
    "supply_amount": "供给|发行量",
    "supply_ratio": "供给|发行占比",
    "supply_long": "供给|10Y以上发行",
    "bank_demand": "需求|银行总净买入",
    "spread_gov": "估值|地方债-国债利差",
    "spread_change": "估值|地方债-国债利差变化",
    "spread_ncd": "估值|地方债-NCD利差",
    "nonbank_sentiment": "情绪|非银情绪",
    "supply_amount_change": "供给|发行量变化",
    "bank_demand_change": "需求|银行总净买入变化",
    "nonbank_sentiment_change": "情绪|非银情绪变化",
    "spread_ncd_change": "估值|地方债-NCD利差变化",
    "spread_change_acceleration": "估值|地方债-国债利差变化加速度",
    "supply_bank_interaction": "供需交互|供给×银行承接",
    "supply_nonbank_interaction": "供需交互|供给×非银承接",
}

FACTOR_GROUPS = {
    "supply": ("supply_amount", "supply_ratio", "supply_long", "supply_amount_change"),
    "bank": ("bank_demand", "bank_demand_change"),
    "valuation": ("spread_gov", "spread_change", "spread_ncd", "spread_ncd_change", "spread_change_acceleration"),
    "sentiment": ("nonbank_sentiment", "nonbank_sentiment_change"),
    "interaction": ("supply_bank_interaction", "supply_nonbank_interaction"),
}

# Presentation order only.  Calculation and search still use ALL_FACTOR_COLUMNS
# so the legacy/economic factor ordering remains stable; tables group the same
# category together instead of splitting it into "original" and "new" blocks.
FACTOR_DISPLAY_COLUMNS = tuple(
    factor
    for group in FACTOR_GROUPS.values()
    for factor in group
)

_SOURCE_DATE_COLUMNS = {
    "supply": ("供给信息截至日期", "供给取数日期"),
    "bank": ("银行需求取数日期",),
    "valuation": ("利差取数日期",),
    "sentiment": ("非银情绪取数日期",),
}


def build_factor_expansion_inputs(root: Path, signal_frequency: str = "weekly") -> pd.DataFrame:
    """Return raw research inputs and their last-observed timestamps.

    A value released less frequently than the signal grid is intentionally
    carried forward with its original observation timestamp. Changes are
    measured only when that timestamp advances, never against a duplicated
    daily display row.
    """
    path = root / signal_file_for_frequency(signal_frequency)
    raw = pd.read_csv(path, encoding="utf-8-sig").copy()
    raw["signal_date"] = pd.to_datetime(raw["信号日期"], errors="coerce")
    if raw["signal_date"].isna().any() or not raw["signal_date"].is_monotonic_increasing:
        raise ValueError(f"{path.name} 的信号日期无效或未排序")

    out = pd.DataFrame({"signal_date": raw["signal_date"]})
    source_dates = {
        key: _source_dates(raw, candidates)
        for key, candidates in _SOURCE_DATE_COLUMNS.items()
    }
    for key, values in source_dates.items():
        out[f"{key}_observed_at"] = values

    columns = {
        "supply_amount_raw": "未来一周地方债发行量",
        "supply_ratio_raw": "未来一周地方债发行量/（国债发行量+地方债发行量）",
        "supply_long_raw": "地方债发行10年以上绝对发行量",
        "bank_demand_raw": "银行过去一周净买入金额",
        "spread_gov_raw": "10年好地区一般债-10年国债活跃券利差",
        "nonbank_sentiment_raw": "基煜纯债基金周度净申购情况",
        "spread_ncd_raw": "10年好地区一般债-1年国股行NCD利差",
        "spread_change_raw": "上述利差周度变化情况",
    }
    for output, source in columns.items():
        out[output] = pd.to_numeric(raw[source], errors="coerce")

    out["supply_amount_change_raw"] = _asof_observation_change(
        out["supply_amount_raw"], out["supply_observed_at"]
    )
    out["bank_demand_change_raw"] = _asof_observation_change(
        out["bank_demand_raw"], out["bank_observed_at"]
    )
    out["nonbank_sentiment_change_raw"] = _asof_observation_change(
        out["nonbank_sentiment_raw"], out["sentiment_observed_at"]
    )
    out["spread_ncd_change_raw"] = _asof_observation_change(
        out["spread_ncd_raw"], out["valuation_observed_at"]
    )
    out["spread_change_acceleration_raw"] = _asof_observation_change(
        out["spread_change_raw"], out["valuation_observed_at"]
    )
    return out


def build_factor_expansion_multipliers(
    root: Path,
    thresholds: DashboardThresholds | None = None,
    factor_windows: FactorWindowConfig | None = None,
    signal_frequency: str = "weekly",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the legacy and new factor multipliers without modifying v1."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    windows = factor_windows or DEFAULT_FACTOR_WINDOWS
    legacy = build_dashboard_factor_multipliers(
        root,
        thresholds=thresholds,
        factor_windows=windows,
        signal_frequency=signal_frequency,
    )
    inputs = build_factor_expansion_inputs(root, signal_frequency)
    if not legacy["signal_date"].equals(inputs["signal_date"]):
        raise ValueError("扩展因子与现有因子信号日期不一致")

    out = legacy.loc[:, ["signal_date", *BASE_FACTOR_COLUMNS]].copy()
    supply_change_pct = _observation_percentile(
        inputs["supply_amount_change_raw"], inputs["supply_observed_at"], windows.supply_months
    )
    bank_change_pct = _observation_percentile(
        inputs["bank_demand_change_raw"], inputs["bank_observed_at"], windows.bank_months
    )
    out["supply_amount_change"] = [
        _bucket_multiplier(value, thresholds.supply_low, thresholds.supply_high, high_is_bullish=False)
        for value in supply_change_pct
    ]
    out["bank_demand_change"] = [
        _bucket_multiplier(value, thresholds.demand_low, thresholds.demand_high, high_is_bullish=True)
        for value in bank_change_pct
    ]
    out["nonbank_sentiment_change"] = [
        _nonbank_multiplier(value) for value in inputs["nonbank_sentiment_change_raw"]
    ]
    out["spread_ncd_change"] = [
        _spread_change_multiplier(value, thresholds.spread_change_bp)
        for value in inputs["spread_ncd_change_raw"]
    ]
    out["spread_change_acceleration"] = [
        _spread_change_multiplier(value, thresholds.spread_change_bp)
        for value in inputs["spread_change_acceleration_raw"]
    ]
    # A high supply score means supply is easy to absorb; a high demand or
    # sentiment score means stronger buying. Products only reward both states.
    out["supply_bank_interaction"] = out["supply_amount"] * out["bank_demand"]
    out["supply_nonbank_interaction"] = out["supply_amount"] * out["nonbank_sentiment"]
    return out.loc[:, ["signal_date", *ALL_FACTOR_COLUMNS]], inputs


def factor_expansion_data_quality(inputs: pd.DataFrame, signal_frequency: str) -> pd.DataFrame:
    """Compact implementation checks for the extension report, not a factor screen."""
    raw_columns = {
        "supply_amount_change": ("supply_amount_change_raw", "supply_observed_at"),
        "bank_demand_change": ("bank_demand_change_raw", "bank_observed_at"),
        "nonbank_sentiment_change": ("nonbank_sentiment_change_raw", "sentiment_observed_at"),
        "spread_ncd_change": ("spread_ncd_change_raw", "valuation_observed_at"),
        "spread_change_acceleration": ("spread_change_acceleration_raw", "valuation_observed_at"),
    }
    rows: list[dict[str, object]] = []
    for factor, (column, date_column) in raw_columns.items():
        values = pd.to_numeric(inputs[column], errors="coerce")
        valid = values.notna()
        observations = pd.to_datetime(inputs[date_column], errors="coerce")
        rows.append(
            {
                "因子": FACTOR_LABELS[factor],
                "信号频率": "日频" if signal_frequency == "daily" else "周频",
                "有效信号数": int(valid.sum()),
                "首个有效信号日": inputs.loc[valid, "signal_date"].min(),
                "最新有效信号日": inputs.loc[valid, "signal_date"].max(),
                "底层更新次数": int(observations.nunique()),
                "底层观测是否早于等于信号日": bool((observations <= inputs["signal_date"]).all()),
            }
        )
    return pd.DataFrame(rows)


def supply_seasonality_feasibility(inputs: pd.DataFrame) -> dict[str, object]:
    """Describe why a week-of-year seasonal supply signal is deferred."""
    dates = pd.to_datetime(inputs["supply_observed_at"], errors="coerce").dropna()
    unique_dates = pd.Series(dates.unique()).sort_values()
    frame = pd.DataFrame({"date": unique_dates})
    frame["iso_week"] = frame["date"].dt.isocalendar().week.astype(int)
    counts = frame.groupby("iso_week").size()
    return {
        "底层供给观测数": int(len(frame)),
        "首个供给观测日": frame["date"].min().date().isoformat() if len(frame) else None,
        "最新供给观测日": frame["date"].max().date().isoformat() if len(frame) else None,
        "同一周序号的历史观测数_最小": int(counts.min()) if len(counts) else 0,
        "同一周序号的历史观测数_中位数": float(counts.median()) if len(counts) else 0.0,
        "同一周序号的历史观测数_最大": int(counts.max()) if len(counts) else 0,
        "结论": "同一周序号仅约4至5个年度观测，不纳入首批季节性供给模型。",
    }


def _source_dates(raw: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in raw:
            values = pd.to_datetime(raw[column], errors="coerce")
            if values.notna().all():
                return values
    return pd.to_datetime(raw["信号日期"], errors="coerce")


def _asof_observation_change(values: pd.Series, observed_at: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    dates = pd.to_datetime(observed_at, errors="coerce").reset_index(drop=True)
    result: list[float] = []
    previous_date: pd.Timestamp | None = None
    previous_value: float | None = None
    latest_change = float("nan")
    for value, observed in zip(numeric, dates):
        if pd.isna(value) or pd.isna(observed):
            result.append(latest_change)
            continue
        if previous_date is not None and observed < previous_date:
            raise ValueError("底层观测日期倒退，不能计算无未来信息的变化因子")
        if previous_date is None or observed > previous_date:
            latest_change = float("nan") if previous_value is None else float(value - previous_value)
            previous_date, previous_value = observed, float(value)
        result.append(latest_change)
    return pd.Series(result, index=values.index)


def _observation_percentile(values: pd.Series, observed_at: pd.Series, months: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    dates = pd.to_datetime(observed_at, errors="coerce").reset_index(drop=True)
    lookback = pd.Timedelta(days=int(months) * 30)
    result: list[float] = []
    history: list[tuple[pd.Timestamp, float]] = []
    previous_date: pd.Timestamp | None = None
    latest_percentile = float("nan")
    for value, observed in zip(numeric, dates):
        if pd.isna(observed):
            result.append(latest_percentile)
            continue
        if previous_date is not None and observed < previous_date:
            raise ValueError("底层观测日期倒退，不能计算无未来信息的滚动分位数")
        if previous_date is None or observed > previous_date:
            if pd.notna(value):
                window = [past for date, past in history if observed - lookback <= date < observed]
                latest_percentile = float(sum(past <= float(value) for past in window) / len(window) * 100.0) if window else float("nan")
                history.append((observed, float(value)))
            previous_date = observed
        result.append(latest_percentile)
    return pd.Series(result, index=values.index)
