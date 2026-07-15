from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import DashboardStrategyConfig, ObjectiveConfig, load_strategy_config, save_strategy_config
from common.experiments import archive_dashboard_experiment
from common.market_data import load_market_data
from common.reporting import _build_period_diagnostics, write_strategy_outputs
from common.runner import run_dashboard_config
from common.trade_metrics import select_executed_weekly_positions, vectorized_capital_trade_metrics
from strategies.dashboard_signal_v1 import DashboardThresholds, DashboardWeights
from strategies.position_policy import DashboardPositionPolicy


OUTPUT_DIR = Path("backtest_outputs") / "阈值调参实验_v1"
BASE_CONFIG = Path("configs") / "experiments" / "阈值实验基线_原始权重_满仓多空.json"

MODULE_LABELS = {
    "supply": "供给分位阈值",
    "demand": "银行需求分位阈值",
    "spread": "地方债国债利差阈值",
    "ncd": "地方债NCD利差阈值",
    "spread_change": "利差周变化阈值",
}

EXPORT_COLUMNS = {
    "candidate_id": "实验编号",
    "stage": "实验阶段",
    "module": "调整模块",
    "candidate_label": "候选参数",
    "supply_low": "供给低分位",
    "supply_high": "供给高分位",
    "demand_low": "需求低分位",
    "demand_high": "需求高分位",
    "spread_low": "地方债国债利差低分位",
    "spread_high": "地方债国债利差高分位",
    "ncd_low": "地方债NCD利差低分位",
    "ncd_high": "地方债NCD利差高分位",
    "spread_change_bp": "利差周变化阈值_BP",
    "bearish_threshold": "看空总分阈值",
    "bearish_min_core_factors": "看空最少利空模块数",
    "bearish_require_supply_or_demand": "看空要求供给或需求利空",
    "bearish_confirmation_periods": "看空连续确认周数",
    "objective": "目标函数",
    "strategy_total_return": "策略累计收益率",
    "benchmark_total_return": "基准累计收益率",
    "excess_total_return": "累计超额收益率",
    "strategy_annual_return": "策略年化收益率",
    "strategy_sharpe": "策略夏普比率",
    "strategy_max_drawdown": "策略最大回撤",
    "capital_gain_total_bp": "累计资本利得_BP",
    "benchmark_capital_gain_bp": "基准累计资本利得_BP",
    "capital_gain_excess_bp": "资本利得超额_BP",
    "capital_trade_win_rate": "资本利得交易胜率",
    "capital_gain_trade_count": "开仓交易总数",
    "capital_gain_closed_trade_count": "已平仓交易数",
    "capital_gain_avg_trade_bp": "平均单笔资本利得_BP",
    "capital_gain_best_trade_bp": "最佳交易_BP",
    "capital_gain_worst_trade_bp": "最差交易_BP",
    "capital_gain_max_drawdown_bp": "资本利得最大回撤_BP",
    "capital_gain_open_trade_count": "未平仓交易数",
    "capital_gain_open_trade_bp": "未平仓浮动资本利得_BP",
    "signal_period_win_rate": "调仓周期胜率",
    "bearish_signal_count": "看空信号数",
    "changed_signal_count": "相对基线变化信号数",
    "strategy_carry_contribution": "策略Carry贡献",
    "strategy_capital_contribution": "策略资本利得贡献",
    "latest_score": "最新总分",
    "latest_position": "最新仓位",
}


def run_threshold_research(
    root: Path,
    objective_config: ObjectiveConfig | None = None,
    base_config_override: DashboardStrategyConfig | None = None,
) -> dict[str, object]:
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_base = base_config_override or load_strategy_config(root / BASE_CONFIG)
    base_config = DashboardStrategyConfig(
        name=selected_base.name,
        weights=selected_base.weights,
        thresholds=selected_base.thresholds,
        positions=selected_base.positions,
        objective=objective_config or selected_base.objective,
        backtest_start=None,
        backtest_end=None,
        benchmark_id=selected_base.benchmark_id,
    )
    context = _build_search_context(root, base_config.benchmark_id)
    module_options = _module_options(context["factor_values"], base_config.thresholds)

    stage1_candidates = [_candidate(base_config, "阈值原始基线", "基线", "全部")]
    for module, options in module_options.items():
        for label, changes in options:
            stage1_candidates.append(
                _candidate(base_config, f"{MODULE_LABELS[module]}_{label}", "单模块敏感性", MODULE_LABELS[module], changes)
            )
    stage1, baseline_positions = _evaluate_candidates(context, stage1_candidates, base_config.weights, base_config.objective)

    module_best_rows = _best_module_rows(stage1, base_config.thresholds)
    combined_changes: dict[str, float] = {}
    for module, row in module_best_rows.items():
        combined_changes.update(_module_changes_from_row(module, row))
    combined_candidate = _candidate(base_config, "单模块最优阈值组合", "组合基线", "全部", combined_changes)
    combined_result, _ = _evaluate_candidates(
        context, [combined_candidate], base_config.weights, base_config.objective, baseline_positions
    )

    score_rule_candidates: list[dict[str, object]] = []
    rules = [
        ("仅总分", 0, 0, 1),
        ("至少2个模块利空", 2, 0, 1),
        ("必须包含供给或需求利空", 0, 1, 1),
        ("至少2个模块且包含供需利空", 2, 1, 1),
        ("连续2周看空", 0, 0, 2),
    ]
    for bearish_threshold, (rule_name, min_modules, require_supply_demand, periods) in product(
        [15.0, 20.0, 25.0, 30.0, 35.0, 40.0], rules
    ):
        score_rule_candidates.append(
            _candidate(
                base_config,
                f"看空<{bearish_threshold:g}_{rule_name}",
                "看空触发条件",
                "总分与确认规则",
                combined_changes,
                {
                    "bearish_threshold": bearish_threshold,
                    "bearish_min_core_factors": min_modules,
                    "bearish_require_supply_or_demand": require_supply_demand,
                    "bearish_confirmation_periods": periods,
                },
            )
        )
    score_rules, _ = _evaluate_candidates(
        context, score_rule_candidates, base_config.weights, base_config.objective, baseline_positions
    )
    best_rule_row = score_rules.sort_values("objective", ascending=False).iloc[0]

    sensitivity = (
        stage1.loc[stage1["stage"] == "单模块敏感性"]
        .groupby("module")["objective"]
        .agg(lambda values: float(values.max() - values.min()))
        .sort_values(ascending=False)
    )
    top_modules_cn = sensitivity.head(2).index.tolist()
    reverse_labels = {value: key for key, value in MODULE_LABELS.items()}
    top_modules = [reverse_labels[name] for name in top_modules_cn]
    best_policy_changes = {
        "bearish_threshold": float(best_rule_row["bearish_threshold"]),
        "bearish_min_core_factors": int(best_rule_row["bearish_min_core_factors"]),
        "bearish_require_supply_or_demand": int(best_rule_row["bearish_require_supply_or_demand"]),
        "bearish_confirmation_periods": int(best_rule_row["bearish_confirmation_periods"]),
    }

    cross_candidates: list[dict[str, object]] = []
    if len(top_modules) == 2:
        left_module, right_module = top_modules
        for (left_label, left_changes), (right_label, right_changes) in product(
            module_options[left_module], module_options[right_module]
        ):
            changes = dict(combined_changes)
            changes.update(left_changes)
            changes.update(right_changes)
            cross_candidates.append(
                _candidate(
                    base_config,
                    f"{left_label}_x_{right_label}",
                    "关键模块交叉",
                    f"{MODULE_LABELS[left_module]} × {MODULE_LABELS[right_module]}",
                    changes,
                    best_policy_changes,
                )
            )
    cross_results, _ = _evaluate_candidates(
        context, cross_candidates, base_config.weights, base_config.objective, baseline_positions
    ) if cross_candidates else (pd.DataFrame(), np.empty((0, 0)))

    results = pd.concat([stage1, combined_result, score_rules, cross_results], ignore_index=True)
    results = results.sort_values(
        ["objective", "strategy_total_return", "excess_total_return"], ascending=False
    ).reset_index(drop=True)
    best_row = results.iloc[0]
    best_config = _config_from_result(base_config, best_row)
    best_config_path = root / "configs" / "experiments" / f"{_safe_config_name(best_config.name)}.json"
    save_strategy_config(
        best_config,
        best_config_path,
    )

    baseline_daily, baseline_signals, baseline_metrics, baseline_benchmark_metrics = run_dashboard_config(root, base_config)
    best_daily, best_signals, best_metrics, best_benchmark_metrics = run_dashboard_config(root, best_config)
    write_strategy_outputs(
        best_daily,
        best_signals,
        best_metrics,
        best_benchmark_metrics,
        output_dir / "最佳阈值策略报告",
    )
    export = _export_results(results)
    export.to_csv(output_dir / "全部阈值实验结果.csv", index=False, encoding="utf-8-sig")
    export.head(100).to_csv(output_dir / "阈值实验Top100.csv", index=False, encoding="utf-8-sig")
    module_summary = _module_summary(stage1, base_config.thresholds)
    module_summary.to_csv(output_dir / "单模块敏感性汇总.csv", index=False, encoding="utf-8-sig")
    diagnostics = _compare_diagnostics(
        _build_period_diagnostics(baseline_daily, baseline_signals),
        _build_period_diagnostics(best_daily, best_signals),
    )
    diagnostics.to_csv(output_dir / "最佳阈值相对基线_错判变化.csv", index=False, encoding="utf-8-sig")
    _write_html_report(
        results,
        module_summary,
        diagnostics,
        base_config,
        best_config,
        baseline_metrics,
        best_metrics,
        best_benchmark_metrics,
        output_dir / "阈值调参报告.html",
    )
    experiment_dir = archive_dashboard_experiment(
        root,
        best_config,
        best_daily,
        best_signals,
        best_metrics,
        best_benchmark_metrics,
        source="阈值向量化搜索",
    )
    return {
        **best_metrics,
        "candidate_count": int(len(results)),
        "best_config_path": str(best_config_path),
        "report_path": str(output_dir / "阈值调参报告.html"),
        "experiment_dir": str(experiment_dir),
    }


def _build_search_context(root: Path, benchmark_id: str) -> dict[str, object]:
    signal_path = root / "data_processed" / "图表指标_周度宽表_统一日期.csv"
    raw = pd.read_csv(signal_path, encoding="utf-8-sig")
    factor_values = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(raw["信号日期"]),
            "supply_amount": _pct_to_100(raw["未来一周地方债发行量_1年内滚动分位数"]),
            "supply_ratio": _pct_to_100(raw["未来一周地方债发行量/（国债发行量+地方债发行量）_1年内滚动分位数"]),
            "supply_long": _pct_to_100(raw["地方债发行10年以上绝对发行量_1年内滚动分位数"]),
            "fly": raw["过去一周是否有地方债“发飞”"].astype(str).str.strip().eq("是").to_numpy(),
            "bank": _pct_to_100(raw["银行过去一周净买入金额_1Y滚动分位数"]),
            "spread": _pct_to_100(raw["10年好地区一般债-10年国债活跃券利差_1年内滚动分位数"]),
            "spread_change": pd.to_numeric(raw["上述利差周度变化情况"], errors="coerce"),
            "ncd": _pct_to_100(raw["10年好地区一般债-1年国股行NCD利差_1年内滚动分位数"]),
            "nonbank": pd.to_numeric(raw["基煜纯债基金周度净申购情况"], errors="coerce"),
        }
    ).sort_values("signal_date").reset_index(drop=True)

    market = load_market_data(root, benchmark_id)
    mapping = factor_values[["signal_date"]].reset_index(names="signal_index")
    daily = pd.merge_asof(
        market.sort_values("date"),
        mapping.sort_values("signal_date"),
        left_on="date",
        right_on="signal_date",
        direction="backward",
    ).dropna(subset=["signal_date"]).reset_index(drop=True)
    return {
        "factor_values": factor_values,
        "daily": daily,
        "signal_index": daily["signal_index"].astype(int).to_numpy(),
    }


def _module_options(values: pd.DataFrame, base: DashboardThresholds) -> dict[str, list[tuple[str, dict[str, float]]]]:
    pairs = [(15.0, 85.0), (20.0, 80.0), (25.0, 75.0), (30.0, 70.0), (35.0, 65.0)]
    absolute_change = pd.to_numeric(values["spread_change"], errors="coerce").abs().dropna()
    bp_values = {float(base.spread_change_bp)}
    if not absolute_change.empty:
        for quantile in [0.40, 0.50, 0.60, 0.70, 0.80]:
            rounded = round(float(absolute_change.quantile(quantile)) * 2.0) / 2.0
            bp_values.add(max(0.5, rounded))
    bp_values.update([1.0, 1.5, 2.0, 2.5, 3.0])
    return {
        "supply": [(f"{low:g}/{high:g}", {"supply_low": low, "supply_high": high}) for low, high in pairs],
        "demand": [(f"{low:g}/{high:g}", {"demand_low": low, "demand_high": high}) for low, high in pairs],
        "spread": [(f"{low:g}/{high:g}", {"spread_low": low, "spread_high": high}) for low, high in pairs],
        "ncd": [(f"{low:g}/{high:g}", {"ncd_low": low, "ncd_high": high}) for low, high in pairs],
        "spread_change": [(f"±{value:g}BP", {"spread_change_bp": value}) for value in sorted(bp_values)],
    }


def _candidate(
    base: DashboardStrategyConfig,
    candidate_id: str,
    stage: str,
    module: str,
    threshold_changes: dict[str, float] | None = None,
    policy_changes: dict[str, float | int] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "stage": stage,
        "module": module,
        "candidate_label": candidate_id.split("_", 1)[-1],
        **base.thresholds.as_dict(),
        **base.positions.as_dict(),
    }
    row.update(threshold_changes or {})
    row.update(policy_changes or {})
    return row


def _evaluate_candidates(
    context: dict[str, object],
    candidates: list[dict[str, object]],
    weights: DashboardWeights,
    objective_config: ObjectiveConfig,
    baseline_weekly_positions: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not candidates:
        return pd.DataFrame(), np.empty((0, 0))
    params = pd.DataFrame(candidates)
    values = context["factor_values"]
    daily = context["daily"]
    signal_index = context["signal_index"]
    n = len(params)

    supply_amount = _bucket_matrix(values["supply_amount"], params["supply_low"], params["supply_high"], False)
    supply_ratio = _bucket_matrix(values["supply_ratio"], params["supply_low"], params["supply_high"], False)
    supply_long = _bucket_matrix(values["supply_long"], params["supply_low"], params["supply_high"], False)
    bank = _bucket_matrix(values["bank"], params["demand_low"], params["demand_high"], True)
    spread = _bucket_matrix(values["spread"], params["spread_low"], params["spread_high"], True)
    ncd = _bucket_matrix(values["ncd"], params["ncd_low"], params["ncd_high"], True)
    change_values = pd.to_numeric(values["spread_change"], errors="coerce").to_numpy(dtype=float)[None, :]
    change_threshold = params["spread_change_bp"].to_numpy(dtype=float)[:, None]
    spread_change = np.where(np.isnan(change_values), 0.5, np.where(change_values <= -change_threshold, 1.0, np.where(change_values >= change_threshold, 0.0, 0.5)))
    nonbank_values = pd.to_numeric(values["nonbank"], errors="coerce").to_numpy(dtype=float)[None, :]
    nonbank = np.where(np.isnan(nonbank_values) | (np.abs(nonbank_values) < 1e-12), 0.5, np.where(nonbank_values > 0, 1.0, 0.0))
    fly = values["fly"].to_numpy(dtype=bool)[None, :]

    scores = np.clip(
        weights.supply_amount * supply_amount
        + weights.supply_ratio * supply_ratio
        + weights.supply_long * supply_long
        + weights.fly_penalty * fly
        + weights.bank_demand * bank
        + weights.spread_gov * spread
        + weights.spread_change * spread_change
        + weights.spread_ncd * ncd
        + weights.nonbank_sentiment * nonbank,
        0.0,
        100.0,
    )
    supply_bearish = ((supply_amount == 0).astype(int) + (supply_ratio == 0).astype(int) + (supply_long == 0).astype(int) >= 2) | fly
    demand_bearish = bank == 0
    valuation_bearish = (spread == 0).astype(int) + (spread_change == 0).astype(int) + (ncd == 0).astype(int) >= 2
    sentiment_bearish = nonbank == 0
    module_count = supply_bearish.astype(int) + demand_bearish.astype(int) + valuation_bearish.astype(int) + sentiment_bearish.astype(int)
    supply_or_demand = supply_bearish | demand_bearish

    bearish = scores < params["bearish_threshold"].to_numpy(dtype=float)[:, None]
    min_modules = params["bearish_min_core_factors"].to_numpy(dtype=int)[:, None]
    bearish &= (min_modules == 0) | (module_count >= min_modules)
    require_supply_demand = params["bearish_require_supply_or_demand"].to_numpy(dtype=int)[:, None].astype(bool)
    bearish &= ~require_supply_demand | supply_or_demand
    confirmation_periods = params["bearish_confirmation_periods"].to_numpy(dtype=int)
    for periods in np.unique(confirmation_periods):
        periods = max(int(periods), 1)
        if periods == 1:
            continue
        selector = confirmation_periods == periods
        selected = bearish[selector]
        confirmed = np.zeros_like(selected, dtype=bool)
        if selected.shape[1] >= periods:
            window = np.ones_like(selected[:, periods - 1 :], dtype=bool)
            for offset in range(periods):
                window &= selected[:, offset : selected.shape[1] - periods + offset + 1]
            confirmed[:, periods - 1 :] = window
        bearish[selector] = confirmed

    weekly_positions = _candidate_position_matrix(scores, bearish, params)
    executed_weekly_positions, executed_signal_ids = select_executed_weekly_positions(
        weekly_positions, signal_index
    )
    daily_positions = weekly_positions[:, signal_index]
    daily_returns = pd.to_numeric(daily["asset_total_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    daily_capital_returns = pd.to_numeric(daily["asset_duration_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    benchmark_daily_returns = pd.to_numeric(daily["total_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    benchmark_daily_capital_returns = pd.to_numeric(daily["duration_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    asset_capital_bp = -pd.to_numeric(daily["asset_yield_change_bp"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    benchmark_capital_bp_daily = -pd.to_numeric(daily["yield_change_bp"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for values in [daily_returns, daily_capital_returns, benchmark_daily_returns, benchmark_daily_capital_returns, asset_capital_bp, benchmark_capital_bp_daily]:
        values[0] = 0.0
    returns = daily_positions * daily_returns[None, :]
    capital_returns = daily_positions * daily_capital_returns[None, :]
    capital_bp = daily_positions * asset_capital_bp[None, :]
    nav = np.cumprod(1.0 + returns, axis=1)
    benchmark_nav = np.cumprod(1.0 + benchmark_daily_returns)
    periods = max(len(daily_returns) - 1, 1)
    total_return = nav[:, -1] / nav[:, 0] - 1.0
    benchmark_total_return = float(benchmark_nav[-1] / benchmark_nav[0] - 1.0)
    annual_return = (nav[:, -1] / nav[:, 0]) ** (252 / periods) - 1.0
    annual_vol = returns.std(axis=1) * math.sqrt(252)
    sharpe = np.divide(annual_return - 0.014, annual_vol, out=np.zeros(n), where=annual_vol != 0)
    max_drawdown = np.min(nav / np.maximum.accumulate(nav, axis=1) - 1.0, axis=1)
    excess_total_return = total_return - benchmark_total_return

    period_return_columns = []
    for signal_id in np.unique(signal_index):
        mask = signal_index == signal_id
        period_return_columns.append(np.prod(1.0 + returns[:, mask], axis=1) - 1.0)
    period_returns = np.column_stack(period_return_columns)
    signal_win_rate = (period_returns > 0).mean(axis=1)
    capital_trade_columns = []
    for signal_id in np.unique(signal_index):
        capital_trade_columns.append(capital_bp[:, signal_index == signal_id].sum(axis=1))
    capital_trade_bp = np.column_stack(capital_trade_columns)
    capital_gain_bp = capital_bp.sum(axis=1)
    benchmark_capital_gain_bp = float(benchmark_capital_bp_daily.sum())
    capital_gain_excess_bp = capital_gain_bp - benchmark_capital_gain_bp
    trade_stats = vectorized_capital_trade_metrics(executed_weekly_positions, capital_trade_bp)
    capital_trade_win_rate = np.nan_to_num(trade_stats["trade_win_rate"], nan=0.0)
    capital_cumulative_bp = np.cumsum(capital_bp, axis=1)
    capital_gain_max_drawdown_bp = np.min(
        capital_cumulative_bp - np.maximum.accumulate(capital_cumulative_bp, axis=1), axis=1
    )
    objective = (
        objective_config.total_return_weight * total_return
        + objective_config.excess_return_weight * excess_total_return
        + objective_config.sharpe_weight * sharpe
        + objective_config.max_drawdown_penalty * max_drawdown
        + objective_config.signal_win_rate_weight * signal_win_rate
        + objective_config.capital_gain_bp_weight * capital_gain_bp
        + objective_config.capital_gain_excess_bp_weight * capital_gain_excess_bp
        + objective_config.capital_trade_win_rate_weight * capital_trade_win_rate
        + objective_config.capital_gain_drawdown_bp_penalty * capital_gain_max_drawdown_bp
    )
    carry = pd.to_numeric(daily["asset_carry_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    capital = pd.to_numeric(daily["asset_duration_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    result = params.copy()
    result["objective"] = objective
    result["strategy_total_return"] = total_return
    result["benchmark_total_return"] = benchmark_total_return
    result["excess_total_return"] = excess_total_return
    result["strategy_annual_return"] = annual_return
    result["strategy_sharpe"] = sharpe
    result["strategy_max_drawdown"] = max_drawdown
    result["signal_period_win_rate"] = signal_win_rate
    result["capital_gain_total_bp"] = capital_gain_bp
    result["benchmark_capital_gain_bp"] = benchmark_capital_gain_bp
    result["capital_gain_excess_bp"] = capital_gain_excess_bp
    result["capital_trade_win_rate"] = capital_trade_win_rate
    result["capital_gain_trade_count"] = trade_stats["trade_count"]
    result["capital_gain_closed_trade_count"] = trade_stats["closed_trade_count"]
    result["capital_gain_avg_trade_bp"] = trade_stats["average_trade_bp"]
    result["capital_gain_best_trade_bp"] = trade_stats["best_trade_bp"]
    result["capital_gain_worst_trade_bp"] = trade_stats["worst_trade_bp"]
    result["capital_gain_open_trade_count"] = trade_stats["open_trade_count"]
    result["capital_gain_open_trade_bp"] = trade_stats["open_trade_bp"]
    result["capital_gain_max_drawdown_bp"] = capital_gain_max_drawdown_bp
    result["bearish_signal_count"] = bearish[:, executed_signal_ids].sum(axis=1)
    result["strategy_carry_contribution"] = (daily_positions * carry[None, :]).sum(axis=1)
    result["strategy_capital_contribution"] = (daily_positions * capital[None, :]).sum(axis=1)
    result["latest_score"] = scores[:, executed_signal_ids[-1]]
    result["latest_position"] = executed_weekly_positions[:, -1]
    if baseline_weekly_positions is None:
        baseline_weekly_positions = executed_weekly_positions[0].copy()
    result["changed_signal_count"] = (
        executed_weekly_positions != baseline_weekly_positions[None, :]
    ).sum(axis=1)
    return result, baseline_weekly_positions


def _candidate_position_matrix(
    scores: np.ndarray,
    bearish: np.ndarray,
    params: pd.DataFrame,
) -> np.ndarray:
    bullish_threshold = params["bullish_threshold"].to_numpy(dtype=float)[:, None]
    bullish_position = params["bullish_position"].to_numpy(dtype=float)[:, None]
    neutral_position = params["neutral_position"].to_numpy(dtype=float)[:, None]
    bearish_position = params["bearish_position"].to_numpy(dtype=float)[:, None]
    positions = np.where(scores >= bullish_threshold, bullish_position, neutral_position)
    return np.where(bearish, bearish_position, positions)


def _bucket_matrix(values: pd.Series, low: pd.Series, high: pd.Series, high_is_bullish: bool) -> np.ndarray:
    matrix = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)[None, :]
    lows = low.to_numpy(dtype=float)[:, None]
    highs = high.to_numpy(dtype=float)[:, None]
    if high_is_bullish:
        return np.where(np.isnan(matrix), 0.5, np.where(matrix >= highs, 1.0, np.where(matrix <= lows, 0.0, 0.5)))
    return np.where(np.isnan(matrix), 0.5, np.where(matrix <= lows, 1.0, np.where(matrix >= highs, 0.0, 0.5)))


def _pct_to_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values * 100.0 if not values.dropna().empty and values.max() <= 1.5 else values


def _best_module_rows(stage1: pd.DataFrame, base: DashboardThresholds) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    reverse_labels = {value: key for key, value in MODULE_LABELS.items()}
    grouped = stage1.loc[stage1["stage"] == "单模块敏感性"].groupby("module")
    for module_label, group in grouped:
        module = reverse_labels[module_label]
        ranked = group.copy()
        ranked["_distance_to_base"] = _threshold_distance(ranked, module, base)
        max_objective = float(ranked["objective"].max())
        tied = ranked.loc[np.isclose(ranked["objective"], max_objective, atol=1e-12, rtol=0.0)]
        rows[module] = tied.sort_values(["_distance_to_base", "candidate_id"]).iloc[0]
    return rows


def _threshold_distance(frame: pd.DataFrame, module: str, base: DashboardThresholds) -> pd.Series:
    keys = {
        "supply": ["supply_low", "supply_high"],
        "demand": ["demand_low", "demand_high"],
        "spread": ["spread_low", "spread_high"],
        "ncd": ["ncd_low", "ncd_high"],
        "spread_change": ["spread_change_bp"],
    }[module]
    distance = pd.Series(0.0, index=frame.index)
    for key in keys:
        distance += (pd.to_numeric(frame[key], errors="coerce") - float(getattr(base, key))).abs()
    return distance


def _module_changes_from_row(module: str, row: pd.Series) -> dict[str, float]:
    keys = {
        "supply": ["supply_low", "supply_high"],
        "demand": ["demand_low", "demand_high"],
        "spread": ["spread_low", "spread_high"],
        "ncd": ["ncd_low", "ncd_high"],
        "spread_change": ["spread_change_bp"],
    }[module]
    return {key: float(row[key]) for key in keys}


def _config_from_result(base: DashboardStrategyConfig, row: pd.Series) -> DashboardStrategyConfig:
    thresholds = DashboardThresholds(**{key: float(row[key]) for key in base.thresholds.as_dict()})
    positions = replace(
        base.positions,
        bearish_threshold=float(row["bearish_threshold"]),
        bearish_min_core_factors=int(row["bearish_min_core_factors"]),
        bearish_require_supply_or_demand=int(row["bearish_require_supply_or_demand"]),
        bearish_confirmation_periods=int(row["bearish_confirmation_periods"]),
    )
    return DashboardStrategyConfig(
        name=f"阈值搜索最优_{base.name}",
        weights=base.weights,
        thresholds=thresholds,
        positions=positions,
        objective=base.objective,
        benchmark_id=base.benchmark_id,
    )


def _safe_config_name(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in name)


def _export_results(results: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in EXPORT_COLUMNS if column in results.columns]
    return results[columns].rename(columns=EXPORT_COLUMNS)


def _module_summary(stage1: pd.DataFrame, base: DashboardThresholds) -> pd.DataFrame:
    frame = stage1.loc[stage1["stage"] == "单模块敏感性"].copy()
    best_rows = _best_module_rows(stage1, base)
    reverse_labels = {value: key for key, value in MODULE_LABELS.items()}
    rows = []
    for module, group in frame.groupby("module"):
        module_key = reverse_labels[module]
        best = best_rows[module_key]
        distance = _threshold_distance(group, module_key, base)
        baseline = group.loc[distance.idxmin()]
        improvement = float(best["objective"] - baseline["objective"])
        rows.append(
            {
                "模块": module,
                "最佳候选": best["candidate_label"],
                "建议": "保留原始阈值" if abs(float(_threshold_distance(best.to_frame().T, module_key, base).iloc[0])) < 1e-12 else "采用候选阈值",
                "最佳目标函数": best["objective"],
                "最佳累计收益率": best["strategy_total_return"],
                "最佳超额收益率": best["excess_total_return"],
                "最佳累计资本利得_BP": best["capital_gain_total_bp"],
                "最佳资本利得超额_BP": best["capital_gain_excess_bp"],
                "最佳资本利得交易胜率": best["capital_trade_win_rate"],
                "相对原始阈值改善": improvement,
                "目标函数敏感区间": group["objective"].max() - group["objective"].min(),
                "收益率敏感区间": group["strategy_total_return"].max() - group["strategy_total_return"].min(),
                "资本利得敏感区间_BP": group["capital_gain_total_bp"].max() - group["capital_gain_total_bp"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values("目标函数敏感区间", ascending=False).reset_index(drop=True)


def _compare_diagnostics(baseline: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    left = baseline[["信号日期", "仓位", "周期策略资本利得_BP", "周期基准资本利得_BP", "周期资本利得超额_BP", "资本利得判断类型"]].rename(
        columns={"仓位": "基线仓位", "周期策略资本利得_BP": "基线策略资本利得_BP", "周期基准资本利得_BP": "基线基准资本利得_BP", "周期资本利得超额_BP": "基线资本利得超额_BP", "资本利得判断类型": "基线资本利得判断类型"}
    )
    right = best[["信号日期", "仓位", "周期策略资本利得_BP", "周期资本利得超额_BP", "资本利得判断类型"]].rename(
        columns={"仓位": "最优仓位", "周期策略资本利得_BP": "最优策略资本利得_BP", "周期资本利得超额_BP": "最优资本利得超额_BP", "资本利得判断类型": "最优资本利得判断类型"}
    )
    merged = left.merge(right, on="信号日期", how="outer")
    merged["仓位是否变化"] = np.where(merged["基线仓位"] != merged["最优仓位"], "是", "否")
    merged["资本利得判断是否变化"] = np.where(merged["基线资本利得判断类型"] != merged["最优资本利得判断类型"], "是", "否")
    merged["资本利得改善_BP"] = merged["最优策略资本利得_BP"] - merged["基线策略资本利得_BP"]
    return merged.sort_values("资本利得改善_BP", ascending=False).reset_index(drop=True)


def _write_html_report(
    results: pd.DataFrame,
    module_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    base_config: DashboardStrategyConfig,
    best_config: DashboardStrategyConfig,
    baseline_metrics: dict[str, object],
    best_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    path: Path,
) -> None:
    from pyecharts import options as opts
    from pyecharts.charts import Line, Scatter
    from pyecharts.globals import CurrentConfig

    dependencies: set[str] = set()
    chart_blocks: list[str] = []
    sensitivity = results.loc[results["stage"] == "单模块敏感性"]
    for module, group in sensitivity.groupby("module", sort=False):
        group = group.sort_values(["supply_low", "demand_low", "spread_low", "ncd_low", "spread_change_bp"])
        chart = (
            Line(init_opts=opts.InitOpts(width="100%", height="390px"))
            .add_xaxis(group["candidate_label"].astype(str).tolist())
            .add_yaxis("累计资本利得", group["capital_gain_total_bp"].round(3).tolist(), is_symbol_show=True, color="#bb654f")
            .add_yaxis("资本利得超额", group["capital_gain_excess_bp"].round(3).tolist(), is_symbol_show=True, color="#176b5b")
            .set_global_opts(
                title_opts=opts.TitleOpts(title=module),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                yaxis_opts=opts.AxisOpts(name="资本利得（BP）", type_="value", is_scale=True),
                xaxis_opts=opts.AxisOpts(type_="category"),
            )
        )
        dependencies.update(chart.js_dependencies.items)
        chart_blocks.append(chart.render_embed())

    scatter = (
        Scatter(init_opts=opts.InitOpts(width="100%", height="430px"))
        .add_xaxis(results["capital_gain_max_drawdown_bp"].round(3).tolist())
        .add_yaxis("全部候选", results["capital_gain_excess_bp"].round(3).tolist(), symbol_size=8, color="#bb654f")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="候选策略：资本利得回撤与资本利得超额"),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
            xaxis_opts=opts.AxisOpts(name="资本利得最大回撤（BP）", type_="value", is_scale=True),
            yaxis_opts=opts.AxisOpts(name="资本利得超额（BP）", type_="value", is_scale=True),
        )
    )
    dependencies.update(scatter.js_dependencies.items)
    chart_blocks.append(scatter.render_embed())

    top = _export_results(results.head(20)).copy()
    pct_columns = [column for column in top.columns if "收益率" in column or column in {"策略最大回撤", "调仓周期胜率", "资本利得交易胜率"}]
    for column in pct_columns:
        top[column] = pd.to_numeric(top[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    for column in ["累计资本利得_BP", "基准累计资本利得_BP", "资本利得超额_BP", "平均单笔资本利得_BP", "最佳交易_BP", "最差交易_BP", "资本利得最大回撤_BP"]:
        if column in top.columns:
            top[column] = pd.to_numeric(top[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    best = results.iloc[0]
    module_display = module_summary.copy()
    for column in ["最佳累计收益率", "最佳超额收益率", "收益率敏感区间"]:
        module_display[column] = pd.to_numeric(module_display[column], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    module_display["最佳目标函数"] = pd.to_numeric(module_display["最佳目标函数"], errors="coerce").map(lambda value: f"{value:.4f}")
    module_display["目标函数敏感区间"] = pd.to_numeric(module_display["目标函数敏感区间"], errors="coerce").map(lambda value: f"{value:.4f}")
    module_display["相对原始阈值改善"] = pd.to_numeric(module_display["相对原始阈值改善"], errors="coerce").map(lambda value: f"{value:.4f}")
    module_display["最佳资本利得交易胜率"] = pd.to_numeric(module_display["最佳资本利得交易胜率"], errors="coerce").map(lambda value: f"{value:.2%}")
    for column in ["最佳累计资本利得_BP", "最佳资本利得超额_BP", "资本利得敏感区间_BP"]:
        module_display[column] = pd.to_numeric(module_display[column], errors="coerce").map(lambda value: f"{value:.2f}")

    changed = diagnostics.loc[diagnostics["仓位是否变化"] == "是"].copy()
    for column in ["基线策略资本利得_BP", "基线基准资本利得_BP", "基线资本利得超额_BP", "最优策略资本利得_BP", "最优资本利得超额_BP", "资本利得改善_BP"]:
        changed[column] = pd.to_numeric(changed[column], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    comparison = pd.DataFrame(
        [
            {"指标": "累计资本利得", "原始阈值基线": baseline_metrics["capital_gain_total_bp"], "最佳方案": best_metrics["capital_gain_total_bp"], "长期持有基准": benchmark_metrics["capital_gain_total_bp"]},
            {"指标": "资本利得交易胜率", "原始阈值基线": baseline_metrics["capital_gain_trade_win_rate"], "最佳方案": best_metrics["capital_gain_trade_win_rate"], "长期持有基准": benchmark_metrics["capital_gain_trade_win_rate"]},
            {"指标": "平均单笔资本利得", "原始阈值基线": baseline_metrics["capital_gain_avg_trade_bp"], "最佳方案": best_metrics["capital_gain_avg_trade_bp"], "长期持有基准": benchmark_metrics["capital_gain_avg_trade_bp"]},
            {"指标": "最差交易", "原始阈值基线": baseline_metrics["capital_gain_worst_trade_bp"], "最佳方案": best_metrics["capital_gain_worst_trade_bp"], "长期持有基准": benchmark_metrics["capital_gain_worst_trade_bp"]},
            {"指标": "资本利得最大回撤", "原始阈值基线": baseline_metrics["capital_gain_max_drawdown_bp"], "最佳方案": best_metrics["capital_gain_max_drawdown_bp"], "长期持有基准": benchmark_metrics["capital_gain_max_drawdown_bp"]},
            {"指标": "累计收益", "原始阈值基线": baseline_metrics["total_return"], "最佳方案": best_metrics["total_return"], "长期持有基准": benchmark_metrics["total_return"]},
            {"指标": "年化收益", "原始阈值基线": baseline_metrics["annual_return"], "最佳方案": best_metrics["annual_return"], "长期持有基准": benchmark_metrics["annual_return"]},
            {"指标": "最大回撤", "原始阈值基线": baseline_metrics["max_drawdown"], "最佳方案": best_metrics["max_drawdown"], "长期持有基准": benchmark_metrics["max_drawdown"]},
            {"指标": "夏普比率", "原始阈值基线": baseline_metrics["sharpe"], "最佳方案": best_metrics["sharpe"], "长期持有基准": benchmark_metrics["sharpe"]},
            {"指标": "调仓周期胜率", "原始阈值基线": baseline_metrics["signal_period_win_rate"], "最佳方案": best_metrics["signal_period_win_rate"], "长期持有基准": benchmark_metrics["signal_period_win_rate"]},
        ]
    )
    comparison[["原始阈值基线", "最佳方案", "长期持有基准"]] = comparison[["原始阈值基线", "最佳方案", "长期持有基准"]].astype(object)
    for index in [0, 2, 3, 4]:
        for column in ["原始阈值基线", "最佳方案", "长期持有基准"]:
            comparison.loc[index, column] = f"{float(comparison.loc[index, column]):.2f} BP"
    for column in ["原始阈值基线", "最佳方案", "长期持有基准"]:
        value = comparison.loc[1, column]
        comparison.loc[1, column] = "暂无已平仓" if pd.isna(value) else f"{float(value):.2%}"
    for index in [5, 6, 7, 9]:
        for column in ["原始阈值基线", "最佳方案", "长期持有基准"]:
            comparison.loc[index, column] = f"{float(comparison.loc[index, column]):.2%}"
    comparison.loc[8, ["原始阈值基线", "最佳方案", "长期持有基准"]] = comparison.loc[8, ["原始阈值基线", "最佳方案", "长期持有基准"]].map(lambda value: f"{float(value):.3f}")

    threshold_rows = pd.DataFrame(
        [
            {"参数": "供给分位", "原始": f"{base_config.thresholds.supply_low:g}/{base_config.thresholds.supply_high:g}", "最佳": f"{best_config.thresholds.supply_low:g}/{best_config.thresholds.supply_high:g}"},
            {"参数": "银行需求分位", "原始": f"{base_config.thresholds.demand_low:g}/{base_config.thresholds.demand_high:g}", "最佳": f"{best_config.thresholds.demand_low:g}/{best_config.thresholds.demand_high:g}"},
            {"参数": "地方债-国债利差分位", "原始": f"{base_config.thresholds.spread_low:g}/{base_config.thresholds.spread_high:g}", "最佳": f"{best_config.thresholds.spread_low:g}/{best_config.thresholds.spread_high:g}"},
            {"参数": "地方债-NCD利差分位", "原始": f"{base_config.thresholds.ncd_low:g}/{base_config.thresholds.ncd_high:g}", "最佳": f"{best_config.thresholds.ncd_low:g}/{best_config.thresholds.ncd_high:g}"},
            {"参数": "利差周变化", "原始": f"±{base_config.thresholds.spread_change_bp:g}BP", "最佳": f"±{best_config.thresholds.spread_change_bp:g}BP"},
            {"参数": "看空总分", "原始": f"<{base_config.positions.bearish_threshold:g}", "最佳": f"<{best_config.positions.bearish_threshold:g}"},
            {"参数": "最少利空模块", "原始": base_config.positions.bearish_min_core_factors, "最佳": best_config.positions.bearish_min_core_factors},
            {"参数": "必须包含供给或需求利空", "原始": "否", "最佳": "是" if best_config.positions.bearish_require_supply_or_demand else "否"},
            {"参数": "连续确认周数", "原始": base_config.positions.bearish_confirmation_periods, "最佳": best_config.positions.bearish_confirmation_periods},
        ]
    )
    weights = pd.DataFrame(
        [{"因子": key, "固定权重": value} for key, value in base_config.weights.as_dict().items()]
    )
    if best_config.thresholds == base_config.thresholds:
        threshold_conclusion = "定性阈值保留原始设置；样本内改善主要来自看空总分或确认规则。"
    else:
        threshold_conclusion = "部分定性阈值在资本利得BP目标下发生变化，具体差异见下表。"
    scripts = "".join(f'<script src="{CurrentConfig.ONLINE_HOST}{dependency}.js"></script>' for dependency in sorted(dependencies))
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>地方债阈值调参实验</title>{scripts}
<style>
body{{margin:0;background:#f3f2ed;color:#18201d;font-family:Geist,"Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:48px 28px 80px}}h1{{font-size:38px;margin:0 0 14px}}h2{{margin-top:52px}}h3{{margin:28px 0 12px}}.lead{{color:#66716c;max-width:980px;line-height:1.8}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #cfd5d1;border-bottom:1px solid #cfd5d1;margin:30px 0}}.metric{{padding:22px 18px;border-right:1px solid #cfd5d1}}.metric:last-child{{border:0}}.metric b{{display:block;font-size:25px;color:#bb654f;margin-top:8px}}.chart{{background:#fbfaf6;margin:18px 0;padding:12px;border-radius:4px}}.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#d9ddd8;border:1px solid #d9ddd8}}.step{{background:#fbfaf6;padding:18px;line-height:1.65}}.step b{{display:block;color:#176b5b;margin-bottom:8px}}.callout{{border-left:4px solid #bb654f;background:#fbfaf6;padding:18px 22px;line-height:1.8}}table{{border-collapse:collapse;width:100%;font-size:12px;background:#fbfaf6}}th,td{{padding:8px;border-bottom:1px solid #d9ddd8;text-align:left;white-space:nowrap}}th{{background:#e8ece8;color:#176b5b;position:sticky;top:0}}.table-wrap{{overflow-x:auto;overflow-y:visible;border:1px solid #d9ddd8}}code{{color:#176b5b}}@media(max-width:800px){{.metrics,.steps{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>10Y地方债策略：阈值调参实验</h1>
<p class="lead">本报告是本次阈值研究的唯一说明文件。实验使用网页所选基线的九因子权重和多/中/空仓位，只检验定性分界、看空总分门槛和看空确认条件。资本利得BP按 -仓位 × YTM变化BP 计算，不乘久期。</p>
<div class="metrics"><div class="metric">候选组合<b>{len(results)}</b></div><div class="metric">累计资本利得<b>{best['capital_gain_total_bp']:.2f} BP</b></div><div class="metric">资本利得超额<b>{best['capital_gain_excess_bp']:.2f} BP</b></div><div class="metric">逐笔胜率<b>{best['capital_trade_win_rate']:.2%}</b></div></div>
<h2>实验设计</h2>
<div class="steps"><div class="step"><b>1. 单模块敏感性</b>分位阈值测试15/85至35/65，利差变化按0.5BP生成候选。</div><div class="step"><b>2. 看空总分</b>测试15至40；多/中/空仓位沿用所选基线。</div><div class="step"><b>3. 确认规则</b>测试模块数量、供需确认和连续两周确认。</div><div class="step"><b>4. 有限交叉</b>只对最敏感的两个模块组合，避免全参数暴力过拟合。</div></div>
<h3>固定权重</h3><div class="table-wrap">{weights.to_html(index=False, escape=False)}</div>
<h2>绩效对照</h2><div class="table-wrap">{comparison.to_html(index=False, escape=False)}</div>
<h2>研究结论</h2><div class="callout"><strong>{threshold_conclusion}</strong> 当前看空规则为：总分低于 {best_config.positions.bearish_threshold:g}，至少 {best_config.positions.bearish_min_core_factors} 个核心模块利空，{'且必须包含供给或银行需求利空' if best_config.positions.bearish_require_supply_or_demand else '不额外要求供给或需求确认'}，连续确认 {best_config.positions.bearish_confirmation_periods} 周。</div>
<h3>最佳参数与原始参数</h3><div class="table-wrap">{threshold_rows.to_html(index=False, escape=False)}</div>
<p class="lead">最佳候选来源：<code>{best['stage']} / {best['module']} / {best['candidate_label']}</code>。本次目标函数为 {base_config.objective.capital_gain_bp_weight:g} × 累计资本利得BP + {base_config.objective.capital_gain_excess_bp_weight:g} × 资本利得超额BP + {base_config.objective.capital_trade_win_rate_weight:g} × 已平仓交易胜率 + {base_config.objective.capital_gain_drawdown_bp_penalty:g} × 资本利得回撤BP + {base_config.objective.total_return_weight:g} × 累计收益 + {base_config.objective.excess_return_weight:g} × 超额收益 + {base_config.objective.sharpe_weight:g} × 夏普 + {base_config.objective.max_drawdown_penalty:g} × 最大回撤 + {base_config.objective.signal_win_rate_weight:g} × 调仓周期胜率。</p>
<h2>实际改变的调仓周期</h2><p class="lead">最佳方案相对原始阈值共有 {len(changed)} 个周期改变仓位。正改善代表新规则提高了该笔交易的资本利得BP。</p><div class="table-wrap">{changed.to_html(index=False, escape=False)}</div>
<h2>单模块敏感性</h2><div class="table-wrap">{module_display.to_html(index=False, escape=False)}</div>
{''.join(f'<div class="chart">{block}</div>' for block in chart_blocks)}
<h2>目标函数Top20</h2><div class="table-wrap">{top.to_html(index=False, escape=False)}</div>
<h2>如何解读</h2><p class="lead">单模块表同时展示资本利得BP、资本利得超额和逐笔胜率。目标函数改善为零的模块不应机械改阈值；只有形成稳定平台、而非单一尖峰的参数才值得保留。当前仍属于同一样本内研究，历史数据扩展完成前，不把结果视为样本外有效性证明。</p>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
