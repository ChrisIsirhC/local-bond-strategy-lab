from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, product
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from common.config import ObjectiveConfig
from common.factor_expansion_features import (
    ALL_FACTOR_COLUMNS,
    BASE_FACTOR_COLUMNS,
    EXPANDED_FACTOR_COLUMNS,
    FACTOR_GROUPS,
    FACTOR_LABELS,
    build_factor_expansion_multipliers,
)
from common.market_data import GOV_10Y, load_market_data
from common.runner import _backtest_dashboard_signals
from common.trade_metrics import vectorized_capital_trade_metrics
from common.trade_metrics import capital_gain_trade_metrics
from strategies.dashboard_signal_v1 import DashboardThresholds, FactorWindowConfig
from strategies.position_policy import DashboardPositionPolicy


STEP = 5
MODULE_MAX = 50
MAX_FULL_WEIGHT_REFINEMENT_STEPS = 10
ROOT_POSITIONS = DashboardPositionPolicy(
    bullish_threshold=70.0, bearish_threshold=30.0,
    bullish_position=1.0, neutral_position=0.0, bearish_position=0.0,
    bearish_min_core_factors=0, bearish_require_supply_or_demand=0,
    bearish_confirmation_periods=1, take_profit_bp=0.0, stop_loss_bp=0.0,
)
ROOT_THRESHOLDS = DashboardThresholds()
OBJECTIVES = {
    "收益": ObjectiveConfig(capital_gain_bp_weight=1.0, capital_gain_excess_bp_weight=0.25, capital_trade_win_rate_weight=10.0),
    "胜率": ObjectiveConfig(capital_gain_bp_weight=0.2, capital_gain_excess_bp_weight=0.25, capital_trade_win_rate_weight=100.0, capital_gain_drawdown_bp_penalty=1.0),
    "综合": ObjectiveConfig(capital_gain_bp_weight=0.6, capital_gain_excess_bp_weight=0.25, capital_trade_win_rate_weight=50.0, capital_gain_drawdown_bp_penalty=0.5),
}


@dataclass(frozen=True)
class ExpansionResearchConfig:
    factor_version: str
    objective_name: str
    signal_frequency: str
    weights: dict[str, float]
    thresholds: DashboardThresholds = ROOT_THRESHOLDS
    positions: DashboardPositionPolicy = ROOT_POSITIONS
    factor_windows: FactorWindowConfig = FactorWindowConfig()
    objective_config: ObjectiveConfig | None = None
    enabled_factor_columns: tuple[str, ...] | None = None

    @property
    def objective(self) -> ObjectiveConfig:
        return self.objective_config or OBJECTIVES[self.objective_name]

    @property
    def factor_columns(self) -> tuple[str, ...]:
        # ``自选因子`` is a deliberately single-route study.  It uses exactly
        # the preselected universe instead of pretending that a partial subset
        # is an "original 8 factors" control group.
        universe = BASE_FACTOR_COLUMNS if self.factor_version == "原始因子" else ALL_FACTOR_COLUMNS
        if self.enabled_factor_columns is None:
            return universe
        return tuple(column for column in universe if column in self.enabled_factor_columns)


def root_research_config(
    factor_version: str,
    objective_name: str,
    signal_frequency: str,
    *,
    thresholds: DashboardThresholds | None = None,
    positions: DashboardPositionPolicy | None = None,
    factor_windows: FactorWindowConfig | None = None,
    objective_config: ObjectiveConfig | None = None,
    initial_weights: dict[str, float] | None = None,
    enabled_factor_columns: Iterable[str] | None = None,
) -> ExpansionResearchConfig:
    columns = BASE_FACTOR_COLUMNS if factor_version == "原始因子" else ALL_FACTOR_COLUMNS
    enabled = tuple(enabled_factor_columns) if enabled_factor_columns is not None else tuple(ALL_FACTOR_COLUMNS)
    unknown = set(enabled).difference(ALL_FACTOR_COLUMNS)
    if unknown:
        raise ValueError(f"存在无法识别的事前启用因子: {sorted(unknown)}")
    usable = tuple(column for column in columns if column in enabled)
    if not usable:
        raise ValueError(f"{factor_version}至少需要启用一个可用因子")
    usable_groups = {
        group
        for group, members in FACTOR_GROUPS.items()
        if any(column in members for column in usable)
    }
    return ExpansionResearchConfig(
        factor_version=factor_version, objective_name=objective_name, signal_frequency=signal_frequency,
        weights={column: float((initial_weights or {}).get(column, 0.0)) for column in usable},
        thresholds=thresholds or ROOT_THRESHOLDS,
        positions=positions or ROOT_POSITIONS,
        factor_windows=factor_windows or FactorWindowConfig(),
        objective_config=objective_config,
        enabled_factor_columns=enabled,
    )


@lru_cache(maxsize=64)
def _cached_factor_data(
    root_path: str,
    thresholds: DashboardThresholds,
    factor_windows: FactorWindowConfig,
    signal_frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache immutable factor inputs for repeated searches in one research run.

    Static and rolling search repeatedly evaluate new weights against identical
    factor values. The cache exists only in the Python process, so a later data
    refresh and a new research run always rebuild from the updated source files.
    """
    return build_factor_expansion_multipliers(
        Path(root_path), thresholds=thresholds, factor_windows=factor_windows,
        signal_frequency=signal_frequency,
    )


def _factor_data(
    root: Path,
    thresholds: DashboardThresholds,
    factor_windows: FactorWindowConfig,
    signal_frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _cached_factor_data(
        str(root.resolve()), thresholds, factor_windows, signal_frequency,
    )


def run_expansion_weight_search(
    root: Path,
    base: ExpansionResearchConfig,
    training_start: str | None,
    training_end: str,
    beam_width: int = 160,
) -> tuple[ExpansionResearchConfig, pd.DataFrame]:
    """Search only complete V2 weight configurations.

    Partial allocations cannot be judged under a score-entry threshold: before
    a partial sum can reach the long threshold, every candidate has the same
    zero-position result. The previous incremental beam pruned those ties and
    could discard valid zero-weight paths before evaluating them. This routine
    instead starts from complete 100-point allocations (including original-only
    allocations for the expanded universe), then refines them by 5-point weight
    transfers. Ties prefer fewer active factors.
    """
    if base.factor_version not in {"原始因子", "扩展因子", "自选因子"}:
        raise ValueError("未知因子版本")
    multipliers, _ = _factor_data(
        root, base.thresholds, base.factor_windows, base.signal_frequency,
    )
    market = _training_market(root, training_start, training_end)
    factor_columns = base.factor_columns
    matrix, prepared = _prepare_vectorized_market(multipliers, market, factor_columns)
    group_for_factor = _group_for_factor(factor_columns)
    survivor_count = min(max(int(beam_width), 32), 64)
    states = _full_weight_start_states(factor_columns, group_for_factor, survivor_count)
    baseline_state = tuple(int(round(float(base.weights.get(column, 0.0)))) for column in factor_columns)
    module_max = _module_max(group_for_factor)
    if (
        sum(baseline_state) == 100
        and all(value >= 0 and value % STEP == 0 for value in baseline_state)
        and all(
            sum(value for value, column in zip(baseline_state, factor_columns) if group_for_factor[column] == group) <= module_max
            for group in set(group_for_factor.values())
        )
    ):
        states.add(baseline_state)
    if base.factor_version == "扩展因子":
        base_factor_columns = tuple(column for column in BASE_FACTOR_COLUMNS if column in factor_columns)
        base_groups = _group_for_factor(base_factor_columns)
        base_states = _full_weight_start_states(base_factor_columns, base_groups, survivor_count)
        for base_state in base_states:
            base_weights = dict(zip(base_factor_columns, base_state))
            states.add(tuple(base_weights.get(column, 0) for column in factor_columns))
    trace: list[pd.DataFrame] = []
    visited: set[tuple[int, ...]] = set()
    frame = _score_weight_states(states, factor_columns, matrix, prepared, base.objective, base.positions)
    frame["搜索阶段"] = "完整起始候选"
    frame["搜索步数"] = 0
    trace.append(frame)
    visited.update(states)
    survivors = _select_weight_survivors(frame, factor_columns, survivor_count)
    states = {tuple(int(value) for value in row[list(factor_columns)]) for _, row in survivors.iterrows()}
    for step in range(1, MAX_FULL_WEIGHT_REFINEMENT_STEPS + 1):
        candidates = _transfer_weight_neighbors(states, factor_columns, group_for_factor) - visited
        if not candidates:
            break
        frame = _score_weight_states(candidates, factor_columns, matrix, prepared, base.objective, base.positions)
        frame["搜索阶段"] = "完整配置五点转移"
        frame["搜索步数"] = step
        trace.append(frame)
        visited.update(candidates)
        survivors = _select_weight_survivors(frame, factor_columns, survivor_count)
        states = {tuple(int(value) for value in row[list(factor_columns)]) for _, row in survivors.iterrows()}
    finalists = _select_weight_survivors(pd.concat(trace, ignore_index=True), factor_columns, 1)
    best = finalists.iloc[0]
    weights = {column: float(best[column]) for column in factor_columns}
    return replace(base, weights=weights), pd.concat(trace, ignore_index=True)


def run_expansion_threshold_search(
    root: Path,
    weighted: ExpansionResearchConfig,
    training_start: str | None,
    training_end: str,
) -> tuple[ExpansionResearchConfig, pd.DataFrame]:
    """Search effective score entry thresholds after weights are fixed."""
    multipliers, _ = _factor_data(
        root, weighted.thresholds, weighted.factor_windows, weighted.signal_frequency,
    )
    scores = _scores_from_multipliers(multipliers, weighted)
    market = _training_market(root, training_start, training_end)
    rows: list[dict[str, object]] = []
    bearish_values = (weighted.positions.bearish_threshold,)
    # With 1/0/0 positions, both neutral and bearish states are cash. Searching
    # their boundary would duplicate candidates, so only search an effective
    # long-entry threshold. Other position policies retain both dimensions.
    if weighted.positions.neutral_position != weighted.positions.bearish_position:
        bearish_values = (15.0, 20.0, 25.0, 30.0, 35.0)
    bullish_values = tuple(sorted({60.0, 65.0, 70.0, 75.0, float(weighted.positions.bullish_threshold)}))
    for bullish, bearish in product(bullish_values, bearish_values):
        if bearish >= bullish:
            continue
        policy = replace(weighted.positions, bullish_threshold=bullish, bearish_threshold=bearish)
        candidate = replace(weighted, positions=policy)
        signals = _signals_from_scores(multipliers["signal_date"], scores, policy)
        _, strategy, benchmark = _backtest_dashboard_signals(market, signals, policy)
        rows.append({
            "看多阈值": bullish, "看空阈值": bearish,
            **_metric_row(strategy, benchmark),
            "目标函数": _objective_value(strategy, benchmark, candidate.objective),
        })
    results = pd.DataFrame(rows).sort_values(["目标函数", "累计资本利得_BP", "资本利得超额_BP"], ascending=False).reset_index(drop=True)
    best = results.iloc[0]
    return replace(weighted, positions=replace(weighted.positions, bullish_threshold=float(best["看多阈值"]), bearish_threshold=float(best["看空阈值"]))), results


def expanded_original_candidate(original: ExpansionResearchConfig) -> ExpansionResearchConfig:
    """Embed an original-factor strategy exactly in the expanded universe."""
    return replace(
        original,
        factor_version="扩展因子",
        weights={
            column: float(original.weights.get(column, 0.0))
            for column in (original.enabled_factor_columns or ALL_FACTOR_COLUMNS)
        },
    )


def training_candidate_rank(
    config: ExpansionResearchConfig,
    strategy: dict[str, object],
    benchmark: dict[str, object],
) -> tuple[float, float, float, int]:
    """Use the search objective, then the same performance/sparsity tie-breaks."""
    row = _metric_row(strategy, benchmark)
    active_factors = sum(float(config.weights.get(column, 0.0)) > 0.0 for column in config.factor_columns)
    return (
        _objective_value(strategy, benchmark, config.objective),
        row["累计资本利得_BP"],
        row["资本利得超额_BP"],
        -active_factors,
    )


def evaluate_expansion_config(
    root: Path,
    config: ExpansionResearchConfig,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    signals = build_expansion_signals(root, config)
    market = load_market_data(root, GOV_10Y)
    if start:
        market = market.loc[market["date"] >= pd.Timestamp(start)]
    if end:
        market = market.loc[market["date"] <= pd.Timestamp(end)]
    daily, strategy, benchmark = _backtest_dashboard_signals(market.reset_index(drop=True), signals, config.positions)
    return daily, signals, strategy, benchmark


def build_expansion_signals(root: Path, config: ExpansionResearchConfig) -> pd.DataFrame:
    multipliers, inputs = _factor_data(
        root, config.thresholds, config.factor_windows, config.signal_frequency,
    )
    columns = config.factor_columns
    scores = _scores_from_multipliers(multipliers, config)
    positions = config.positions.vectorized_positions(scores)
    conclusions = config.positions.vectorized_conclusions(scores)
    out = pd.DataFrame({"signal_date": multipliers["signal_date"], "总分_raw": scores, "总分": scores, "仓位": positions, "结论": conclusions})
    for column in columns:
        out[f"{FACTOR_LABELS[column]}_得分"] = multipliers[column] * float(config.weights.get(column, 0.0))
        out[f"{FACTOR_LABELS[column]}_乘数"] = multipliers[column]
    for column in inputs.columns:
        if column != "signal_date":
            out[column] = inputs[column]
    out.attrs["factor_version"] = config.factor_version
    out.attrs["weights"] = config.weights
    out.attrs["thresholds"] = config.thresholds.as_dict()
    out.attrs["positions"] = config.positions.as_dict()
    return out


def _scores_from_multipliers(multipliers: pd.DataFrame, config: ExpansionResearchConfig) -> np.ndarray:
    columns = config.factor_columns
    values = multipliers.loc[:, columns].to_numpy(dtype=float)
    weights = np.array([float(config.weights.get(column, 0.0)) for column in columns])
    return np.clip(values @ weights, 0.0, 100.0)


def _signals_from_scores(signal_dates: pd.Series, scores: np.ndarray, policy: DashboardPositionPolicy) -> pd.DataFrame:
    return pd.DataFrame({
        "signal_date": signal_dates,
        "总分": scores,
        "结论": policy.vectorized_conclusions(scores),
        "仓位": policy.vectorized_positions(scores),
    })


def run_expansion_rolling_research(
    root: Path,
    base: ExpansionResearchConfig,
    minimum_training_months: int = 24,
    recalibration_months: int = 3,
    beam_width: int = 40,
    original_periods: pd.DataFrame | None = None,
    progress: Callable[[str, int | None, int | None], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window selection followed by next-period OOS execution.

    For an expanded-factor route, ``original_periods`` supplies the matching
    original-factor winner at each cutoff. That exact strategy, embedded with
    all added factors at zero, is the training-period fallback.
    """
    def check_cancelled() -> None:
        if cancel_requested is not None and cancel_requested():
            # Import lazily to avoid the normal rolling engine importing this
            # factor-search module at import time.
            from common.rolling_research import RollingResearchCancelled
            raise RollingResearchCancelled("已按请求终止因子滚动任务")

    market = load_market_data(root, GOV_10Y).sort_values("date").reset_index(drop=True)
    signal_dates = build_expansion_signals(root, base)["signal_date"]
    market = market.loc[market["date"].between(signal_dates.min(), signal_dates.max())].reset_index(drop=True)
    available_start, available_end = market["date"].min(), market["date"].max()
    desired_first_end = available_start + pd.DateOffset(months=minimum_training_months)
    eligible = market.loc[market["date"] >= desired_first_end, "date"]
    if eligible.empty:
        raise ValueError("不足以启动扩展因子滚动定参")
    train_end = pd.Timestamp(eligible.iloc[0])
    total_periods = 0
    count_end = train_end
    while count_end < available_end:
        next_start = _next_market_date(market["date"], count_end)
        next_end = _on_or_before(market["date"], count_end + pd.DateOffset(months=recalibration_months)) or available_end
        if next_start is None or next_start > next_end:
            break
        total_periods += 1
        count_end = next_end
    reported_total = progress_total if progress_total is not None else total_periods
    if progress:
        progress("正在准备因子滚动数据", progress_offset, reported_total)
    rows: list[dict[str, object]] = []
    daily_parts: list[pd.DataFrame] = []
    period = 0
    while train_end < available_end:
        check_cancelled()
        oos_start_values = market.loc[market["date"] > train_end, "date"]
        if oos_start_values.empty:
            break
        oos_start = pd.Timestamp(oos_start_values.iloc[0])
        requested_end = train_end + pd.DateOffset(months=recalibration_months)
        end_candidates = market.loc[(market["date"] <= requested_end) & (market["date"] >= oos_start), "date"]
        oos_end = pd.Timestamp(end_candidates.iloc[-1]) if not end_candidates.empty else available_end
        if progress:
            progress(
                f"第 {progress_offset + period + 1} 期：训练 {train_end:%Y-%m-%d}，正在搜索因子权重",
                progress_offset + period,
                reported_total,
            )
        weighted, _ = run_expansion_weight_search(root, base, None, train_end.date().isoformat(), beam_width=beam_width)
        check_cancelled()
        selected, _ = run_expansion_threshold_search(root, weighted, None, train_end.date().isoformat())
        check_cancelled()
        used_original_fallback = False
        if base.factor_version == "扩展因子" and set(BASE_FACTOR_COLUMNS).issubset(base.factor_columns):
            matching = pd.DataFrame()
            if original_periods is not None and not original_periods.empty:
                matching = original_periods.loc[
                    pd.to_datetime(original_periods["训练截止日"], errors="coerce").eq(train_end)
                ]
            if matching.empty:
                # Direct queued factor validation has no preceding static
                # original-factor route.  Produce the exact same-period
                # original candidate here, so the expanded universe still
                # contains the nested "all new factors = 0" safeguard.
                original_base = root_research_config(
                    "原始因子", base.objective_name, base.signal_frequency,
                    thresholds=base.thresholds, positions=base.positions,
                    factor_windows=base.factor_windows, objective_config=base.objective,
                    initial_weights=base.weights,
                    enabled_factor_columns=base.enabled_factor_columns,
                )
                if progress:
                    progress(
                        f"第 {progress_offset + period + 1} 期：正在搜索原始因子保底候选",
                        progress_offset + period,
                        reported_total,
                    )
                original_weighted, _ = run_expansion_weight_search(
                    root, original_base, None, train_end.date().isoformat(), beam_width=beam_width,
                )
                check_cancelled()
                original, _ = run_expansion_threshold_search(
                    root, original_weighted, None, train_end.date().isoformat(),
                )
                check_cancelled()
            else:
                original_row = matching.iloc[0]
                original_weights = {
                    str(key): float(value)
                    for key, value in json.loads(str(original_row["权重"])).items()
                }
                original = replace(
                    base,
                    factor_version="原始因子",
                    weights={column: float(original_weights.get(column, 0.0)) for column in BASE_FACTOR_COLUMNS},
                    positions=replace(
                        base.positions,
                        bullish_threshold=float(original_row["看多阈值"]),
                        bearish_threshold=float(original_row["看空阈值"]),
                    ),
                )
            fallback = expanded_original_candidate(original)
            _, _, selected_metrics, selected_benchmark = evaluate_expansion_config(
                root, selected, available_start.date().isoformat(), train_end.date().isoformat()
            )
            _, _, fallback_metrics, fallback_benchmark = evaluate_expansion_config(
                root, fallback, available_start.date().isoformat(), train_end.date().isoformat()
            )
            if training_candidate_rank(fallback, fallback_metrics, fallback_benchmark) >= training_candidate_rank(
                selected, selected_metrics, selected_benchmark
            ):
                selected = fallback
                used_original_fallback = True
        daily, _, strategy, benchmark = evaluate_expansion_config(root, selected, oos_start.date().isoformat(), oos_end.date().isoformat())
        daily["滚动训练截止日"] = train_end.date().isoformat()
        daily["滚动样本外起始日"] = oos_start.date().isoformat()
        daily["滚动样本外结束日"] = oos_end.date().isoformat()
        daily_parts.append(daily)
        period += 1
        rows.append({
            "期数": period,
            "训练起始日": available_start.date().isoformat(),
            "训练截止日": train_end.date().isoformat(),
            "样本外起始日": oos_start.date().isoformat(),
            "样本外结束日": oos_end.date().isoformat(),
            "权重": json_dumps_weights(selected.weights),
            "看多阈值": selected.positions.bullish_threshold,
            "看空阈值": selected.positions.bearish_threshold,
            "是否采用原始因子保底": used_original_fallback,
            **_metric_row(strategy, benchmark),
        })
        train_end = oos_end
        if progress:
            progress(
                f"第 {progress_offset + period} 期已完成；下一期将切换训练窗口。",
                progress_offset + period,
                reported_total,
            )
    if not daily_parts:
        raise ValueError("扩展因子滚动定参没有产生样本外区间")
    stitched = pd.concat(daily_parts, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    stitched = rebuild_expansion_stitched_cumulatives(stitched)
    return pd.DataFrame(rows), stitched


def json_dumps_weights(weights: dict[str, float]) -> str:
    import json
    return json.dumps(weights, ensure_ascii=False, sort_keys=True)


def rebuild_expansion_stitched_cumulatives(daily: pd.DataFrame) -> pd.DataFrame:
    """Recompute cumulative paths after independently evaluated OOS segments merge.

    Each rolling OOS segment is evaluated from a zero capital-gain anchor. Its
    day-level contributions are correct, but retaining those per-segment
    cumulative columns makes a stitched chart repeatedly return to zero.
    """
    out = daily.copy()
    bp_columns = {
        "strategy_capital_cum_bp": "strategy_capital_bp",
        "benchmark_capital_cum_bp": "benchmark_capital_bp",
        "capital_excess_cum_bp": "capital_excess_bp",
    }
    decimal_columns = {
        "strategy_capital_cum": "strategy_capital_return",
        "benchmark_capital_cum": "benchmark_capital_return",
        "capital_excess_cum": "capital_excess_return",
        "strategy_carry_cum": "strategy_carry_return",
        "benchmark_carry_cum": "benchmark_carry_return",
        "carry_excess_cum": "carry_excess_return",
    }
    for cumulative, contribution in bp_columns.items():
        if contribution in out:
            out[cumulative] = pd.to_numeric(out[contribution], errors="coerce").fillna(0.0).cumsum()
    for cumulative, contribution in decimal_columns.items():
        if contribution in out:
            out[cumulative] = pd.to_numeric(out[contribution], errors="coerce").fillna(0.0).cumsum()
    if "strategy_return" in out:
        out["strategy_nav"] = (1.0 + pd.to_numeric(out["strategy_return"], errors="coerce").fillna(0.0)).cumprod()
    if "total_return" in out:
        out["benchmark_nav_rebased"] = (1.0 + pd.to_numeric(out["total_return"], errors="coerce").fillna(0.0)).cumprod()
    if {"strategy_nav", "benchmark_nav_rebased"}.issubset(out.columns):
        out["excess_nav"] = out["strategy_nav"] / out["benchmark_nav_rebased"]
    return out


def _training_market(root: Path, start: str | None, end: str) -> pd.DataFrame:
    market = load_market_data(root, GOV_10Y)
    market = market.loc[market["date"] <= pd.Timestamp(end)]
    if start:
        market = market.loc[market["date"] >= pd.Timestamp(start)]
    if market.empty:
        raise ValueError("训练窗口没有市场数据")
    return market.reset_index(drop=True)


def _next_market_date(dates: pd.Series, value: pd.Timestamp) -> pd.Timestamp | None:
    selected = pd.to_datetime(dates, errors="coerce").dropna()
    selected = selected.loc[selected > value]
    return pd.Timestamp(selected.iloc[0]) if not selected.empty else None


def _on_or_before(dates: pd.Series, value: pd.Timestamp) -> pd.Timestamp | None:
    selected = pd.to_datetime(dates, errors="coerce").dropna()
    selected = selected.loc[selected <= value]
    return pd.Timestamp(selected.iloc[-1]) if not selected.empty else None


def _prepare_vectorized_market(multipliers: pd.DataFrame, market: pd.DataFrame, factor_columns: tuple[str, ...]):
    signal_for_merge = multipliers[["signal_date"]].copy()
    daily = pd.merge_asof(market.sort_values("date"), signal_for_merge.reset_index(names="signal_index").sort_values("signal_date"), left_on="date", right_on="signal_date", direction="backward").dropna(subset=["signal_date"]).reset_index(drop=True)
    index = daily["signal_index"].astype(int).to_numpy()
    matrix = multipliers.loc[:, factor_columns].to_numpy(dtype=float)
    prepared = {
        "signal_index": index,
        "asset_bp": -pd.to_numeric(daily["asset_yield_change_bp"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        "benchmark_bp": -pd.to_numeric(daily["yield_change_bp"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    }
    for value in prepared.values():
        if isinstance(value, np.ndarray) and len(value):
            value[0] = 0.0
    return matrix, prepared


def _module_max(groups: dict[str, str]) -> int:
    """Return the effective per-category cap for the active search universe.

    The normal V2 universe keeps each category at 50 points.  A deliberately
    narrow one-category experiment is also valid, so that category may carry
    the full 100 points; otherwise no complete candidate could be formed.
    """
    return 100 if len(set(groups.values())) <= 1 else MODULE_MAX


def _next_weight_states(states: set[tuple[int, ...]], columns: tuple[str, ...], groups: dict[str, str]) -> list[tuple[int, ...]]:
    next_states: set[tuple[int, ...]] = set()
    module_max = _module_max(groups)
    for state in states:
        for index, column in enumerate(columns):
            module = groups[column]
            spent = sum(value for value, other in zip(state, columns) if groups[other] == module)
            if spent >= module_max:
                continue
            updated = list(state)
            updated[index] += STEP
            next_states.add(tuple(updated))
    return sorted(next_states)


def _score_weight_states(states: Iterable[tuple[int, ...]], columns: tuple[str, ...], matrix: np.ndarray, prepared: dict[str, np.ndarray], objective: ObjectiveConfig, policy: DashboardPositionPolicy) -> pd.DataFrame:
    weights = np.asarray(list(states), dtype=float)
    scores = np.clip(weights @ matrix.T, 0.0, 100.0)
    positions = policy.vectorized_positions(scores)
    daily_positions = positions[:, prepared["signal_index"]]
    capital_bp = daily_positions * prepared["asset_bp"]
    benchmark_bp = np.clip(daily_positions, 0.0, None) * prepared["benchmark_bp"]
    stats = vectorized_capital_trade_metrics(daily_positions, capital_bp, np.zeros_like(daily_positions, dtype=bool))
    cumulative = np.cumsum(capital_bp, axis=1)
    drawdown = np.min(cumulative - np.maximum.accumulate(cumulative, axis=1), axis=1)
    values = pd.DataFrame(weights, columns=columns)
    values["总权重"] = values.loc[:, columns].sum(axis=1)
    values["有效因子数"] = values.loc[:, columns].gt(0.0).sum(axis=1)
    values["累计资本利得_BP"] = capital_bp.sum(axis=1)
    values["资本利得超额_BP"] = (capital_bp - benchmark_bp).sum(axis=1)
    values["资本利得交易胜率"] = np.nan_to_num(stats["trade_win_rate"], nan=0.0)
    values["平均每笔盈利_BP"] = np.nan_to_num(stats["average_win_bp"], nan=0.0)
    values["资本利得最大回撤_BP"] = drawdown
    values["已平仓交易数"] = stats["closed_trade_count"]
    values["目标函数"] = (
        objective.capital_gain_bp_weight * values["累计资本利得_BP"]
        + objective.capital_gain_excess_bp_weight * values["资本利得超额_BP"]
        + objective.capital_trade_win_rate_weight * values["资本利得交易胜率"]
        + objective.capital_gain_avg_win_bp_weight * values["平均每笔盈利_BP"]
        + objective.capital_gain_drawdown_bp_penalty * values["资本利得最大回撤_BP"]
    )
    return values


def _full_weight_start_states(columns: tuple[str, ...], groups: dict[str, str], target_count: int) -> set[tuple[int, ...]]:
    """Generate deterministic, feasible, full-weight starting points.

    Sparse two-factor starts guarantee that every factor, including the option
    to give all newly added factors zero weight, is evaluated before pruning.
    Random starts add broader feasible compositions without relying on factor
    column order.
    """
    module_max = _module_max(groups)
    units_per_module = module_max // STEP
    total_units = 100 // STEP
    states: set[tuple[int, ...]] = set()
    for left, right in combinations(range(len(columns)), 2):
        if groups[columns[left]] == groups[columns[right]]:
            continue
        state = [0] * len(columns)
        state[left] = units_per_module * STEP
        state[right] = units_per_module * STEP
        states.add(tuple(state))
    rng = np.random.default_rng(20260908 + len(columns) * 100 + target_count)
    desired = len(states) + max(target_count, 32)
    attempts = 0
    while len(states) < desired and attempts < desired * 100:
        attempts += 1
        state = [0] * len(columns)
        module_units = {group: 0 for group in set(groups.values())}
        for _ in range(total_units):
            eligible = [index for index, column in enumerate(columns) if module_units[groups[column]] < units_per_module]
            if not eligible:
                break
            index = int(rng.choice(eligible))
            state[index] += STEP
            module_units[groups[columns[index]]] += 1
        states.add(tuple(state))
    if not states:
        raise ValueError("无法生成完整权重候选")
    return states


def _transfer_weight_neighbors(states: set[tuple[int, ...]], columns: tuple[str, ...], groups: dict[str, str]) -> set[tuple[int, ...]]:
    """Move one 5-point unit between active/full configurations, retaining zeros."""
    result: set[tuple[int, ...]] = set()
    module_max = _module_max(groups)
    for state in states:
        module_totals = {
            group: sum(value for value, column in zip(state, columns) if groups[column] == group)
            for group in set(groups.values())
        }
        for source, value in enumerate(state):
            if value < STEP:
                continue
            for target, column in enumerate(columns):
                if source == target:
                    continue
                target_group = groups[column]
                if groups[columns[source]] != target_group and module_totals[target_group] >= module_max:
                    continue
                updated = list(state)
                updated[source] -= STEP
                updated[target] += STEP
                result.add(tuple(updated))
    return result


def _select_weight_survivors(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> pd.DataFrame:
    """Rank terminal candidates and use sparsity only as an exact-tie breaker."""
    ranking = frame.sort_values(
        ["目标函数", "累计资本利得_BP", "资本利得超额_BP", "有效因子数", *columns],
        ascending=[False, False, False, True, *([True] * len(columns))],
        kind="stable",
    )
    return ranking.head(count).copy()


def _group_for_factor(columns: tuple[str, ...]) -> dict[str, str]:
    result = {factor: group for group, factors in FACTOR_GROUPS.items() for factor in factors if factor in columns}
    if set(result) != set(columns):
        raise ValueError("因子缺少模块归属")
    return result


def _metric_row(strategy: dict[str, object], benchmark: dict[str, object]) -> dict[str, float]:
    total = float(strategy.get("capital_gain_total_bp", 0.0) or 0.0)
    benchmark_total = float(benchmark.get("capital_gain_total_bp", 0.0) or 0.0)
    return {
        "累计资本利得_BP": total,
        "资本利得超额_BP": total - benchmark_total,
        "资本利得交易胜率": float(strategy.get("capital_gain_trade_win_rate", 0.0) or 0.0),
        "平均每笔盈利_BP": float(strategy.get("capital_gain_avg_win_bp", 0.0) or 0.0),
        "资本利得最大回撤_BP": float(strategy.get("capital_gain_max_drawdown_bp", 0.0) or 0.0),
        "已平仓交易数": float(strategy.get("capital_gain_closed_trade_count", 0.0) or 0.0),
    }


def _objective_value(strategy: dict[str, object], benchmark: dict[str, object], objective: ObjectiveConfig) -> float:
    row = _metric_row(strategy, benchmark)
    return float(
        objective.capital_gain_bp_weight * row["累计资本利得_BP"]
        + objective.capital_gain_excess_bp_weight * row["资本利得超额_BP"]
        + objective.capital_trade_win_rate_weight * row["资本利得交易胜率"]
        + objective.capital_gain_avg_win_bp_weight * row["平均每笔盈利_BP"]
        + objective.capital_gain_drawdown_bp_penalty * row["资本利得最大回撤_BP"]
    )
