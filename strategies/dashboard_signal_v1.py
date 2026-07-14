from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from strategies.position_policy import DEFAULT_POSITION_POLICY, DashboardPositionPolicy


SIGNAL_FILE = Path("data_processed") / "图表指标_周度宽表_统一日期.csv"


@dataclass(frozen=True)
class DashboardWeights:
    supply_amount: float = 10.0
    supply_ratio: float = 10.0
    supply_long: float = 10.0
    fly_penalty: float = -20.0
    bank_demand: float = 15.0
    spread_gov: float = 15.0
    spread_change: float = 10.0
    spread_ncd: float = 15.0
    nonbank_sentiment: float = 15.0

    def as_dict(self) -> dict[str, float]:
        return {
            "supply_amount": self.supply_amount,
            "supply_ratio": self.supply_ratio,
            "supply_long": self.supply_long,
            "fly_penalty": self.fly_penalty,
            "bank_demand": self.bank_demand,
            "spread_gov": self.spread_gov,
            "spread_change": self.spread_change,
            "spread_ncd": self.spread_ncd,
            "nonbank_sentiment": self.nonbank_sentiment,
        }


@dataclass(frozen=True)
class DashboardThresholds:
    supply_low: float = 25.0
    supply_high: float = 75.0
    demand_low: float = 25.0
    demand_high: float = 75.0
    spread_low: float = 20.0
    spread_high: float = 80.0
    ncd_low: float = 25.0
    ncd_high: float = 75.0
    spread_change_bp: float = 2.0

    def as_dict(self) -> dict[str, float]:
        return {
            "supply_low": self.supply_low,
            "supply_high": self.supply_high,
            "demand_low": self.demand_low,
            "demand_high": self.demand_high,
            "spread_low": self.spread_low,
            "spread_high": self.spread_high,
            "ncd_low": self.ncd_low,
            "ncd_high": self.ncd_high,
            "spread_change_bp": self.spread_change_bp,
        }


DEFAULT_WEIGHTS = DashboardWeights()
DEFAULT_THRESHOLDS = DashboardThresholds()
WEIGHT_COLUMNS = [
    "supply_amount",
    "supply_ratio",
    "supply_long",
    "fly_penalty",
    "bank_demand",
    "spread_gov",
    "spread_change",
    "spread_ncd",
    "nonbank_sentiment",
]


def _pct_to_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values
    return values * 100.0 if values.max() <= 1.5 else values


def _bucket_score(value: float, low: float, high: float, weight: float, high_is_bullish: bool) -> tuple[float, str]:
    if pd.isna(value):
        return weight / 2.0, "中性"
    if high_is_bullish:
        if value >= high:
            return weight, "利多"
        if value <= low:
            return 0.0, "利空"
    else:
        if value <= low:
            return weight, "利多"
        if value >= high:
            return 0.0, "利空"
    return weight / 2.0, "中性"


def _spread_change_score(value: float, weight: float = 10.0, threshold_bp: float = 2.0) -> tuple[float, str]:
    if pd.isna(value):
        return weight / 2.0, "中性"
    if value <= -threshold_bp:
        return weight, "利多"
    if value >= threshold_bp:
        return 0.0, "利空"
    return weight / 2.0, "中性"


def _nonbank_score(value: float, weight: float = 15.0) -> tuple[float, str]:
    if pd.isna(value) or abs(value) < 1e-12:
        return weight / 2.0, "中性"
    if value > 0:
        return weight, "利多"
    return 0.0, "利空"


def _bucket_multiplier(value: float, low: float, high: float, high_is_bullish: bool) -> float:
    if pd.isna(value):
        return 0.5
    if high_is_bullish:
        if value >= high:
            return 1.0
        if value <= low:
            return 0.0
    else:
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
    return 0.5


def _spread_change_multiplier(value: float, threshold_bp: float = 2.0) -> float:
    if pd.isna(value):
        return 0.5
    if value <= -threshold_bp:
        return 1.0
    if value >= threshold_bp:
        return 0.0
    return 0.5


def _nonbank_multiplier(value: float) -> float:
    if pd.isna(value) or abs(value) < 1e-12:
        return 0.5
    if value > 0:
        return 1.0
    return 0.0


def _conclusion(score: float) -> str:
    return DEFAULT_POSITION_POLICY.conclusion(score)


def _position(score: float) -> float:
    return DEFAULT_POSITION_POLICY.position(score)


def build_dashboard_factor_multipliers(
    root: Path,
    thresholds: DashboardThresholds | None = None,
) -> pd.DataFrame:
    t = thresholds or DEFAULT_THRESHOLDS
    path = root / SIGNAL_FILE
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["signal_date"] = pd.to_datetime(df["信号日期"])

    supply_amount_pct = _pct_to_100(df["未来一周地方债发行量_1年内滚动分位数"])
    supply_ratio_pct = _pct_to_100(df["未来一周地方债发行量/（国债发行量+地方债发行量）_1年内滚动分位数"])
    supply_long_pct = _pct_to_100(df["地方债发行10年以上绝对发行量_1年内滚动分位数"])
    bank_pct = _pct_to_100(df["银行过去一周净买入金额_1Y滚动分位数"])
    spread_pct = _pct_to_100(df["10年好地区一般债-10年国债活跃券利差_1年内滚动分位数"])
    ncd_pct = _pct_to_100(df["10年好地区一般债-1年国股行NCD利差_1年内滚动分位数"])

    out = pd.DataFrame({"signal_date": df["signal_date"]})
    out["supply_amount"] = [
        _bucket_multiplier(v, t.supply_low, t.supply_high, high_is_bullish=False) for v in supply_amount_pct
    ]
    out["supply_ratio"] = [
        _bucket_multiplier(v, t.supply_low, t.supply_high, high_is_bullish=False) for v in supply_ratio_pct
    ]
    out["supply_long"] = [
        _bucket_multiplier(v, t.supply_low, t.supply_high, high_is_bullish=False) for v in supply_long_pct
    ]
    out["fly_penalty"] = [
        1.0 if str(v).strip() == "是" else 0.0 for v in df["过去一周是否有地方债“发飞”"]
    ]
    out["bank_demand"] = [
        _bucket_multiplier(v, t.demand_low, t.demand_high, high_is_bullish=True) for v in bank_pct
    ]
    out["spread_gov"] = [
        _bucket_multiplier(v, t.spread_low, t.spread_high, high_is_bullish=True) for v in spread_pct
    ]
    out["spread_change"] = [
        _spread_change_multiplier(pd.to_numeric(v, errors="coerce"), t.spread_change_bp) for v in df["上述利差周度变化情况"]
    ]
    out["spread_ncd"] = [
        _bucket_multiplier(v, t.ncd_low, t.ncd_high, high_is_bullish=True) for v in ncd_pct
    ]
    out["nonbank_sentiment"] = [
        _nonbank_multiplier(pd.to_numeric(v, errors="coerce")) for v in df["基煜纯债基金周度净申购情况"]
    ]
    return out.sort_values("signal_date").reset_index(drop=True)


def build_dashboard_signal(
    root: Path,
    weights: DashboardWeights | None = None,
    thresholds: DashboardThresholds | None = None,
    position_policy: DashboardPositionPolicy | None = None,
) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    t = thresholds or DEFAULT_THRESHOLDS
    policy = position_policy or DEFAULT_POSITION_POLICY
    path = root / SIGNAL_FILE
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["signal_date"] = pd.to_datetime(df["信号日期"])

    supply_amount_pct = _pct_to_100(df["未来一周地方债发行量_1年内滚动分位数"])
    supply_ratio_pct = _pct_to_100(df["未来一周地方债发行量/（国债发行量+地方债发行量）_1年内滚动分位数"])
    supply_long_pct = _pct_to_100(df["地方债发行10年以上绝对发行量_1年内滚动分位数"])
    bank_pct = _pct_to_100(df["银行过去一周净买入金额_1Y滚动分位数"])
    spread_pct = _pct_to_100(df["10年好地区一般债-10年国债活跃券利差_1年内滚动分位数"])
    ncd_pct = _pct_to_100(df["10年好地区一般债-1年国股行NCD利差_1年内滚动分位数"])

    rows: list[dict[str, object]] = []
    for i, row in df.iterrows():
        factor_scores: dict[str, float] = {}
        factor_labels: dict[str, str] = {}

        factor_scores["供给_发行量"], factor_labels["供给_发行量"] = _bucket_score(
            supply_amount_pct.iloc[i], t.supply_low, t.supply_high, w.supply_amount, high_is_bullish=False
        )
        factor_scores["供给_发行占比"], factor_labels["供给_发行占比"] = _bucket_score(
            supply_ratio_pct.iloc[i], t.supply_low, t.supply_high, w.supply_ratio, high_is_bullish=False
        )
        factor_scores["供给_10Y以上发行"], factor_labels["供给_10Y以上发行"] = _bucket_score(
            supply_long_pct.iloc[i], t.supply_low, t.supply_high, w.supply_long, high_is_bullish=False
        )
        fly_penalty = w.fly_penalty if str(row["过去一周是否有地方债“发飞”"]).strip() == "是" else 0.0
        factor_scores["供给_发飞惩罚"] = fly_penalty
        factor_labels["供给_发飞惩罚"] = "利空" if fly_penalty < 0 else "中性"

        factor_scores["银行需求"], factor_labels["银行需求"] = _bucket_score(
            bank_pct.iloc[i], t.demand_low, t.demand_high, w.bank_demand, high_is_bullish=True
        )
        factor_scores["利差_地方债国债"], factor_labels["利差_地方债国债"] = _bucket_score(
            spread_pct.iloc[i], t.spread_low, t.spread_high, w.spread_gov, high_is_bullish=True
        )
        factor_scores["利差_周度变化"], factor_labels["利差_周度变化"] = _spread_change_score(
            pd.to_numeric(row["上述利差周度变化情况"], errors="coerce"), w.spread_change, t.spread_change_bp
        )
        factor_scores["利差_地方债NCD"], factor_labels["利差_地方债NCD"] = _bucket_score(
            ncd_pct.iloc[i], t.ncd_low, t.ncd_high, w.spread_ncd, high_is_bullish=True
        )
        factor_scores["非银情绪"], factor_labels["非银情绪"] = _nonbank_score(
            pd.to_numeric(row["基煜纯债基金周度净申购情况"], errors="coerce"), w.nonbank_sentiment
        )

        supply_bearish = (
            sum(factor_labels[name] == "利空" for name in ["供给_发行量", "供给_发行占比", "供给_10Y以上发行"]) >= 2
            or fly_penalty < 0
        )
        demand_bearish = factor_labels["银行需求"] == "利空"
        valuation_bearish = (
            sum(factor_labels[name] == "利空" for name in ["利差_地方债国债", "利差_周度变化", "利差_地方债NCD"]) >= 2
        )
        sentiment_bearish = factor_labels["非银情绪"] == "利空"
        bearish_module_count = sum([supply_bearish, demand_bearish, valuation_bearish, sentiment_bearish])

        raw_score = sum(factor_scores.values())
        score = min(max(raw_score, 0.0), 100.0)
        output = {
            "signal_date": row["signal_date"],
            "周期起始": row["周期起始"],
            "周期结束": row["周期结束"],
            "总分_raw": raw_score,
            "总分": score,
            "供给模块_利空": int(supply_bearish),
            "需求模块_利空": int(demand_bearish),
            "估值模块_利空": int(valuation_bearish),
            "情绪模块_利空": int(sentiment_bearish),
            "利空模块数": int(bearish_module_count),
            "供给或需求利空": int(supply_bearish or demand_bearish),
        }
        for name, value in factor_scores.items():
            output[f"{name}_得分"] = value
            output[f"{name}_定性"] = factor_labels[name]
        rows.append(output)

    out = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    scores = out["总分"].to_numpy(dtype=float)
    bearish_module_count = out["利空模块数"].to_numpy(dtype=int)
    supply_or_demand_bearish = out["供给或需求利空"].to_numpy(dtype=bool)
    out["结论"] = policy.vectorized_conclusions(scores, bearish_module_count, supply_or_demand_bearish)
    out["仓位"] = policy.vectorized_positions(scores, bearish_module_count, supply_or_demand_bearish)
    out.attrs["position_policy"] = policy.as_dict()
    out.attrs["weights"] = w.as_dict()
    out.attrs["thresholds"] = t.as_dict()
    return out
