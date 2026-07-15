from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

from common.bond_return import BondReturnConfig, build_total_return_index, load_yield_curve
from common.config import DashboardStrategyConfig, ObjectiveConfig, save_strategy_config
from common.experiments import archive_dashboard_experiment
from common.performance import performance_metrics
from common.reporting import write_outputs, write_strategy_outputs
from common.trade_metrics import capital_gain_trade_metrics, select_executed_weekly_positions, vectorized_capital_trade_metrics
from strategies.dashboard_signal_v1 import DEFAULT_THRESHOLDS, DashboardThresholds, DashboardWeights, WEIGHT_COLUMNS, build_dashboard_factor_multipliers, build_dashboard_signal
from strategies.dashboard_weight_search_v1 import generate_weight_candidates
from strategies.position_policy import DEFAULT_POSITION_POLICY


def run_hold_10y_benchmark(root: Path) -> dict[str, object]:
    data_path = root / "benchmark_data" / "地方政府债到期收益率_10年_2024至最新.csv"
    output_dir = root / "backtest_outputs" / "hold_10y_benchmark"
    yield_curve = load_yield_curve(data_path)
    returns = build_total_return_index(yield_curve, BondReturnConfig())
    metrics = performance_metrics(returns)
    write_outputs(returns, metrics, output_dir)
    return metrics


def run_dashboard_signal_v1(root: Path) -> dict[str, object]:
    data_path = root / "benchmark_data" / "地方政府债到期收益率_10年_2024至最新.csv"
    output_dir = root / "backtest_outputs" / "dashboard_signal_v1"
    yield_curve = load_yield_curve(data_path)
    benchmark = build_total_return_index(yield_curve, BondReturnConfig())
    signals = build_dashboard_signal(root)

    daily, strategy_metrics, benchmark_metrics = _backtest_dashboard_signals(benchmark, signals)
    write_strategy_outputs(daily, signals, strategy_metrics, benchmark_metrics, output_dir)
    return strategy_metrics


def run_dashboard_config(
    root: Path,
    config: DashboardStrategyConfig,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    data_path = root / "benchmark_data" / "地方政府债到期收益率_10年_2024至最新.csv"
    yield_curve = load_yield_curve(data_path)
    benchmark = build_total_return_index(yield_curve, BondReturnConfig())
    benchmark = _apply_configured_backtest_window(benchmark, config)
    signals = build_dashboard_signal(root, config.weights, config.thresholds, config.positions)
    daily, strategy_metrics, benchmark_metrics = _backtest_dashboard_signals(benchmark, signals)
    if output_dir is not None:
        write_strategy_outputs(daily, signals, strategy_metrics, benchmark_metrics, output_dir)
    return daily, signals, strategy_metrics, benchmark_metrics


def _apply_configured_backtest_window(
    benchmark: pd.DataFrame,
    config: DashboardStrategyConfig,
) -> pd.DataFrame:
    start = pd.Timestamp(config.backtest_start) if config.backtest_start else None
    end = pd.Timestamp(config.backtest_end) if config.backtest_end else None
    if start is not None and end is not None and start > end:
        raise ValueError("回测起始日期不能晚于结束日期")
    selected = benchmark
    if start is not None:
        selected = selected.loc[selected["date"] >= start]
    if end is not None:
        selected = selected.loc[selected["date"] <= end]
    if selected.empty:
        raise ValueError("所选回测区间没有可用基准数据")
    return selected.reset_index(drop=True)


def _backtest_dashboard_signals(
    benchmark: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    daily = benchmark.copy()
    signal_for_merge = signals[["signal_date", "总分", "结论", "仓位"]].copy()
    daily = pd.merge_asof(
        daily.sort_values("date"),
        signal_for_merge.sort_values("signal_date"),
        left_on="date",
        right_on="signal_date",
        direction="backward",
    )
    daily = daily.dropna(subset=["signal_date"]).reset_index(drop=True)
    daily["strategy_return"] = pd.to_numeric(daily["仓位"] * daily["total_return"], errors="coerce")
    daily["total_return"] = pd.to_numeric(daily["total_return"], errors="coerce")
    daily["strategy_carry_return"] = pd.to_numeric(daily["仓位"] * daily["carry_return"], errors="coerce")
    daily["strategy_capital_return"] = pd.to_numeric(daily["仓位"] * daily["duration_pnl"], errors="coerce")
    daily["benchmark_carry_return"] = pd.to_numeric(daily["carry_return"], errors="coerce")
    daily["benchmark_capital_return"] = pd.to_numeric(daily["duration_pnl"], errors="coerce")
    daily["carry_excess_return"] = daily["strategy_carry_return"] - daily["benchmark_carry_return"]
    daily["capital_excess_return"] = daily["strategy_capital_return"] - daily["benchmark_capital_return"]
    daily["strategy_carry_cum"] = daily["strategy_carry_return"].fillna(0.0).cumsum()
    daily["strategy_capital_cum"] = daily["strategy_capital_return"].fillna(0.0).cumsum()
    daily["benchmark_carry_cum"] = daily["benchmark_carry_return"].fillna(0.0).cumsum()
    daily["benchmark_capital_cum"] = daily["benchmark_capital_return"].fillna(0.0).cumsum()
    daily["carry_excess_cum"] = daily["carry_excess_return"].fillna(0.0).cumsum()
    daily["capital_excess_cum"] = daily["capital_excess_return"].fillna(0.0).cumsum()
    daily["strategy_capital_bp"] = daily["strategy_capital_return"] * 10000.0
    daily["benchmark_capital_bp"] = daily["benchmark_capital_return"] * 10000.0
    daily["capital_excess_bp"] = daily["strategy_capital_bp"] - daily["benchmark_capital_bp"]
    daily["strategy_capital_cum_bp"] = daily["strategy_capital_bp"].fillna(0.0).cumsum()
    daily["benchmark_capital_cum_bp"] = daily["benchmark_capital_bp"].fillna(0.0).cumsum()
    daily["capital_excess_cum_bp"] = daily["capital_excess_bp"].fillna(0.0).cumsum()
    daily["strategy_nav"] = (1.0 + daily["strategy_return"].fillna(0.0)).cumprod()
    daily["benchmark_nav_rebased"] = (1.0 + daily["total_return"].fillna(0.0)).cumprod()
    daily["excess_nav"] = daily["strategy_nav"] / daily["benchmark_nav_rebased"]

    strategy_metrics = performance_metrics(daily, return_col="strategy_return", nav_col="strategy_nav")
    benchmark_metrics = performance_metrics(daily, return_col="total_return", nav_col="benchmark_nav_rebased")
    strategy_metrics.update(_signal_period_metrics(daily, "strategy_return"))
    benchmark_metrics.update(_signal_period_metrics(daily, "total_return"))
    strategy_metrics.update(capital_gain_trade_metrics(daily, "strategy_capital_return", position_col="仓位"))
    benchmark_metrics.update(capital_gain_trade_metrics(daily, "benchmark_capital_return", position_col=None))
    return daily, strategy_metrics, benchmark_metrics


def _signal_period_metrics(daily: pd.DataFrame, return_col: str) -> dict[str, object]:
    period_returns = (
        daily.groupby("signal_date")[return_col]
        .apply(lambda x: (1.0 + pd.to_numeric(x, errors="coerce").fillna(0.0)).prod() - 1.0)
        .dropna()
    )
    if period_returns.empty:
        return {
            "signal_period_count": 0,
            "winning_signal_periods": 0,
            "signal_period_win_rate": None,
            "avg_signal_period_return": None,
        }
    winning = int((period_returns > 0).sum())
    return {
        "signal_period_count": int(len(period_returns)),
        "winning_signal_periods": winning,
        "signal_period_win_rate": float(winning / len(period_returns)),
        "avg_signal_period_return": float(period_returns.mean()),
    }


def run_dashboard_weight_search_v1(
    root: Path,
    objective_config: ObjectiveConfig | None = None,
    base_config: DashboardStrategyConfig | None = None,
) -> dict[str, object]:
    data_path = root / "benchmark_data" / "地方政府债到期收益率_10年_2024至最新.csv"
    output_dir = root / "backtest_outputs" / "dashboard_weight_search_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield_curve = load_yield_curve(data_path)
    benchmark = build_total_return_index(yield_curve, BondReturnConfig())
    candidates = generate_weight_candidates()
    objective = objective_config or ObjectiveConfig()
    search_thresholds = base_config.thresholds if base_config is not None else DEFAULT_THRESHOLDS
    search_positions = base_config.positions if base_config is not None else DEFAULT_POSITION_POLICY
    results, best_weights = _run_weight_search_fast(
        root, benchmark, candidates, search_thresholds, search_positions, objective
    )

    results = results.sort_values(
        ["objective", "strategy_total_return", "excess_total_return"],
        ascending=False,
    )
    results.to_csv(output_dir / "all_results.csv", index=False, encoding="utf-8-sig")
    results.head(100).to_csv(output_dir / "top_configs.csv", index=False, encoding="utf-8-sig")
    _write_search_summary(
        results,
        output_dir / "权重搜索报告.html",
        search_thresholds,
        search_positions,
        objective,
    )

    if best_weights is None:
        raise RuntimeError("weight search produced no candidates")
    best_signals = build_dashboard_signal(root, best_weights, search_thresholds, search_positions)
    best_daily, best_strategy_metrics, best_benchmark_metrics = _backtest_dashboard_signals(benchmark, best_signals)
    best_config = DashboardStrategyConfig(
        name=_weight_search_strategy_name(best_weights),
        weights=best_weights,
        thresholds=search_thresholds,
        positions=search_positions,
        objective=objective,
    )
    save_strategy_config(best_config, root / "configs" / "experiments" / f"{best_config.name}.json")
    write_strategy_outputs(
        best_daily,
        best_signals,
        best_strategy_metrics,
        best_benchmark_metrics,
        output_dir / "best_config_report",
    )
    experiment_dir = archive_dashboard_experiment(
        root,
        best_config,
        best_daily,
        best_signals,
        best_strategy_metrics,
        best_benchmark_metrics,
        source="权重向量化搜索",
    )
    return {
        **best_strategy_metrics,
        "strategy_name": best_config.name,
        "experiment_dir": str(experiment_dir),
    }


def _weight_search_strategy_name(weights: DashboardWeights) -> str:
    values = weights.as_dict()
    number = lambda value: f"{float(value):g}"
    return (
        "权重搜索最优_"
        f"供给{number(values['supply_amount'])}-{number(values['supply_ratio'])}-{number(values['supply_long'])}_"
        f"发飞{number(values['fly_penalty'])}_需求{number(values['bank_demand'])}_"
        f"估值{number(values['spread_gov'])}-{number(values['spread_change'])}-{number(values['spread_ncd'])}_"
        f"情绪{number(values['nonbank_sentiment'])}"
    )


def _run_weight_search_fast(
    root: Path,
    benchmark: pd.DataFrame,
    candidates: list[DashboardWeights],
    thresholds: DashboardThresholds = DEFAULT_THRESHOLDS,
    position_policy = DEFAULT_POSITION_POLICY,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    chunk_size: int = 5000,
) -> tuple[pd.DataFrame, DashboardWeights | None]:
    multipliers = build_dashboard_factor_multipliers(root, thresholds)
    signal_for_merge = multipliers[["signal_date"]].copy()
    daily = pd.merge_asof(
        benchmark.sort_values("date"),
        signal_for_merge.sort_values("signal_date").reset_index(names="signal_index"),
        left_on="date",
        right_on="signal_date",
        direction="backward",
    )
    daily = daily.dropna(subset=["signal_date"]).reset_index(drop=True)
    signal_index = daily["signal_index"].astype(int).to_numpy()
    daily_returns = pd.to_numeric(daily["total_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    daily_capital_returns = pd.to_numeric(daily["duration_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    benchmark_metrics = performance_metrics(
        daily.assign(benchmark_nav_rebased=(1.0 + daily_returns).cumprod()),
        return_col="total_return",
        nav_col="benchmark_nav_rebased",
    )
    benchmark_total_return = float(benchmark_metrics["total_return"])

    factor_matrix = multipliers[WEIGHT_COLUMNS].to_numpy(dtype=float)
    periods = max(len(daily) - 1, 1)
    rf = 0.014
    rows: list[pd.DataFrame] = []
    best_objective = -math.inf
    best_weights: DashboardWeights | None = None
    policy = position_policy

    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        weights = np.array([[w.as_dict()[col] for col in WEIGHT_COLUMNS] for w in chunk], dtype=float)
        weekly_scores = np.clip(weights @ factor_matrix.T, 0.0, 100.0)
        weekly_positions = policy.vectorized_positions(weekly_scores)
        executed_weekly_positions, executed_signal_ids = select_executed_weekly_positions(
            weekly_positions, signal_index
        )
        positions = weekly_positions[:, signal_index]
        returns = positions * daily_returns
        capital_returns = positions * daily_capital_returns
        nav = np.cumprod(1.0 + returns, axis=1)
        total_return = nav[:, -1] / nav[:, 0] - 1.0
        final_nav = nav[:, -1]
        annual_return = (nav[:, -1] / nav[:, 0]) ** (252 / periods) - 1.0
        annual_vol = returns.std(axis=1) * math.sqrt(252)
        excess_annual_return = annual_return - rf
        sharpe = np.divide(
            excess_annual_return,
            annual_vol,
            out=np.zeros_like(excess_annual_return),
            where=annual_vol != 0,
        )
        running_max = np.maximum.accumulate(nav, axis=1)
        max_drawdown = np.min(nav / running_max - 1.0, axis=1)
        excess_total_return = total_return - benchmark_total_return
        capital_gain_bp = capital_returns.sum(axis=1) * 10000.0
        benchmark_capital_gain_bp = float(daily_capital_returns.sum() * 10000.0)
        capital_gain_excess_bp = capital_gain_bp - benchmark_capital_gain_bp
        capital_cumulative_bp = np.cumsum(capital_returns * 10000.0, axis=1)
        capital_running_max_bp = np.maximum.accumulate(capital_cumulative_bp, axis=1)
        capital_gain_max_drawdown_bp = np.min(capital_cumulative_bp - capital_running_max_bp, axis=1)
        capital_period_returns = []
        signal_period_returns = []
        for signal_id in np.unique(signal_index):
            period_mask = signal_index == signal_id
            capital_period_returns.append(capital_returns[:, period_mask].sum(axis=1) * 10000.0)
            signal_period_returns.append(np.prod(1.0 + returns[:, period_mask], axis=1) - 1.0)
        capital_trade_bp = np.column_stack(capital_period_returns)
        signal_period_returns_array = np.column_stack(signal_period_returns)
        signal_period_win_rate = (signal_period_returns_array > 0.0).mean(axis=1)
        trade_stats = vectorized_capital_trade_metrics(executed_weekly_positions, capital_trade_bp)
        capital_trade_win_rate = np.nan_to_num(trade_stats["trade_win_rate"], nan=0.0)
        objective = (
            objective_config.total_return_weight * total_return
            + objective_config.excess_return_weight * excess_total_return
            + objective_config.sharpe_weight * sharpe
            + objective_config.max_drawdown_penalty * max_drawdown
            + objective_config.signal_win_rate_weight * signal_period_win_rate
            + objective_config.capital_gain_bp_weight * capital_gain_bp
            + objective_config.capital_gain_excess_bp_weight * capital_gain_excess_bp
            + objective_config.capital_trade_win_rate_weight * capital_trade_win_rate
            + objective_config.capital_gain_drawdown_bp_penalty * capital_gain_max_drawdown_bp
        )

        latest_score = weekly_scores[:, executed_signal_ids[-1]]
        latest_position = executed_weekly_positions[:, -1]
        latest_conclusion = policy.vectorized_conclusions(latest_score)
        best_idx = int(np.argmax(objective))
        if float(objective[best_idx]) > best_objective:
            best_objective = float(objective[best_idx])
            best_weights = chunk[best_idx]

        chunk_rows = pd.DataFrame([w.as_dict() for w in chunk])
        chunk_rows["objective"] = objective
        chunk_rows["strategy_total_return"] = total_return
        chunk_rows["benchmark_total_return"] = benchmark_total_return
        chunk_rows["excess_total_return"] = excess_total_return
        chunk_rows["strategy_final_nav"] = final_nav
        chunk_rows["benchmark_final_nav"] = float(benchmark_metrics["final_nav"])
        chunk_rows["strategy_annual_return"] = annual_return
        chunk_rows["strategy_max_drawdown"] = max_drawdown
        chunk_rows["strategy_sharpe"] = sharpe
        chunk_rows["signal_period_win_rate"] = signal_period_win_rate
        chunk_rows["capital_gain_total_bp"] = capital_gain_bp
        chunk_rows["benchmark_capital_gain_bp"] = benchmark_capital_gain_bp
        chunk_rows["capital_gain_excess_bp"] = capital_gain_excess_bp
        chunk_rows["capital_trade_win_rate"] = capital_trade_win_rate
        chunk_rows["capital_gain_trade_count"] = trade_stats["trade_count"]
        chunk_rows["capital_gain_closed_trade_count"] = trade_stats["closed_trade_count"]
        chunk_rows["capital_gain_avg_trade_bp"] = trade_stats["average_trade_bp"]
        chunk_rows["capital_gain_best_trade_bp"] = trade_stats["best_trade_bp"]
        chunk_rows["capital_gain_worst_trade_bp"] = trade_stats["worst_trade_bp"]
        chunk_rows["capital_gain_open_trade_count"] = trade_stats["open_trade_count"]
        chunk_rows["capital_gain_open_trade_bp"] = trade_stats["open_trade_bp"]
        chunk_rows["capital_gain_max_drawdown_bp"] = capital_gain_max_drawdown_bp
        chunk_rows["latest_score"] = latest_score
        chunk_rows["latest_conclusion"] = latest_conclusion
        chunk_rows["latest_position"] = latest_position
        rows.append(chunk_rows)

    return pd.concat(rows, ignore_index=True), best_weights

def _write_search_summary(
    results: pd.DataFrame,
    path: Path,
    thresholds: DashboardThresholds,
    position_policy,
    objective_config: ObjectiveConfig,
) -> None:
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Scatter
    from pyecharts.globals import CurrentConfig

    labels = {
        "supply_amount": "供给/发行量",
        "supply_ratio": "供给/发行占比",
        "supply_long": "供给/10Y以上发行",
        "fly_penalty": "发飞惩罚",
        "bank_demand": "银行需求",
        "spread_gov": "地方债-国债利差",
        "spread_change": "利差周度变化",
        "spread_ncd": "地方债-NCD利差",
        "nonbank_sentiment": "非银情绪",
    }
    best = results.iloc[0]
    best_weights = pd.DataFrame(
        [{"因子": labels[column], "最佳权重": float(best[column])} for column in WEIGHT_COLUMNS]
    )
    threshold_table = pd.DataFrame(
        [
            {"模块": "供给", "低阈值": thresholds.supply_low, "高阈值": thresholds.supply_high},
            {"模块": "银行需求", "低阈值": thresholds.demand_low, "高阈值": thresholds.demand_high},
            {"模块": "地方债-国债利差", "低阈值": thresholds.spread_low, "高阈值": thresholds.spread_high},
            {"模块": "地方债-NCD利差", "低阈值": thresholds.ncd_low, "高阈值": thresholds.ncd_high},
            {"模块": "利差周变化", "低阈值": f"-{thresholds.spread_change_bp:g}BP", "高阈值": f"+{thresholds.spread_change_bp:g}BP"},
        ]
    )
    position_table = pd.DataFrame(
        [
            {"结论": "看多", "总分条件": f">={position_policy.bullish_threshold:g}", "仓位": position_policy.bullish_position},
            {"结论": "中性", "总分条件": f"{position_policy.bearish_threshold:g}至{position_policy.bullish_threshold:g}", "仓位": position_policy.neutral_position},
            {"结论": "看空", "总分条件": f"<{position_policy.bearish_threshold:g}", "仓位": position_policy.bearish_position},
        ]
    )
    summary = pd.DataFrame(
        [
            {"指标": "累计资本利得", "最佳权重策略": best["capital_gain_total_bp"], "长期持有基准": best["benchmark_capital_gain_bp"]},
            {"指标": "资本利得超额", "最佳权重策略": best["capital_gain_excess_bp"], "长期持有基准": 0.0},
            {"指标": "资本利得交易胜率", "最佳权重策略": best["capital_trade_win_rate"], "长期持有基准": np.nan},
            {"指标": "平均单笔资本利得", "最佳权重策略": best["capital_gain_avg_trade_bp"], "长期持有基准": np.nan},
            {"指标": "最差交易", "最佳权重策略": best["capital_gain_worst_trade_bp"], "长期持有基准": np.nan},
            {"指标": "资本利得最大回撤", "最佳权重策略": best["capital_gain_max_drawdown_bp"], "长期持有基准": np.nan},
            {"指标": "累计收益", "最佳权重策略": best["strategy_total_return"], "长期持有基准": best["benchmark_total_return"]},
            {"指标": "累计超额", "最佳权重策略": best["excess_total_return"], "长期持有基准": 0.0},
            {"指标": "最大回撤", "最佳权重策略": best["strategy_max_drawdown"], "长期持有基准": np.nan},
            {"指标": "夏普比率", "最佳权重策略": best["strategy_sharpe"], "长期持有基准": np.nan},
            {"指标": "最新总分", "最佳权重策略": best["latest_score"], "长期持有基准": np.nan},
            {"指标": "最新仓位", "最佳权重策略": best["latest_position"], "长期持有基准": 1.0},
        ]
    )
    summary[["最佳权重策略", "长期持有基准"]] = summary[["最佳权重策略", "长期持有基准"]].astype(object)
    for index in [0, 1, 3, 4, 5]:
        for column in ["最佳权重策略", "长期持有基准"]:
            value = summary.loc[index, column]
            summary.loc[index, column] = "" if pd.isna(value) else f"{float(value):.2f} BP"
    for column in ["最佳权重策略", "长期持有基准"]:
        value = summary.loc[2, column]
        summary.loc[2, column] = "" if pd.isna(value) else f"{float(value):.2%}"
    for index in [6, 7, 8]:
        for column in ["最佳权重策略", "长期持有基准"]:
            value = summary.loc[index, column]
            summary.loc[index, column] = "" if pd.isna(value) else f"{float(value):.2%}"
    for index in [9, 10, 11]:
        for column in ["最佳权重策略", "长期持有基准"]:
            value = summary.loc[index, column]
            summary.loc[index, column] = "" if pd.isna(value) else f"{float(value):.3f}"

    top = results.head(20).copy().rename(
        columns={
            **labels,
            "objective": "目标函数",
            "strategy_total_return": "策略累计收益率",
            "benchmark_total_return": "基准累计收益率",
            "excess_total_return": "累计超额收益率",
            "strategy_max_drawdown": "策略最大回撤",
            "strategy_sharpe": "策略夏普比率",
            "signal_period_win_rate": "调仓周期胜率",
            "latest_score": "最新总分",
            "latest_conclusion": "最新结论",
            "latest_position": "最新仓位",
            "capital_gain_total_bp": "累计资本利得_BP",
            "benchmark_capital_gain_bp": "基准资本利得_BP",
            "capital_gain_excess_bp": "资本利得超额_BP",
            "capital_trade_win_rate": "资本利得交易胜率",
            "capital_gain_trade_count": "开仓交易总数",
            "capital_gain_closed_trade_count": "已平仓交易数",
            "capital_gain_avg_trade_bp": "平均单笔_BP",
            "capital_gain_worst_trade_bp": "最差交易_BP",
            "capital_gain_max_drawdown_bp": "资本利得最大回撤_BP",
            "capital_gain_open_trade_count": "未平仓交易数",
            "capital_gain_open_trade_bp": "未平仓浮动资本利得_BP",
        }
    )
    keep = [*labels.values(), "目标函数", "累计资本利得_BP", "基准资本利得_BP", "资本利得超额_BP", "开仓交易总数", "已平仓交易数", "资本利得交易胜率", "平均单笔_BP", "最差交易_BP", "未平仓交易数", "未平仓浮动资本利得_BP", "资本利得最大回撤_BP", "策略累计收益率", "基准累计收益率", "累计超额收益率", "策略最大回撤", "策略夏普比率", "调仓周期胜率", "最新总分", "最新结论", "最新仓位"]
    top = top[keep]
    for column in ["策略累计收益率", "基准累计收益率", "累计超额收益率", "策略最大回撤"]:
        top[column] = pd.to_numeric(top[column], errors="coerce").map(lambda value: f"{value:.2%}")
    top["目标函数"] = pd.to_numeric(top["目标函数"], errors="coerce").map(lambda value: f"{value:.4f}")
    top["策略夏普比率"] = pd.to_numeric(top["策略夏普比率"], errors="coerce").map(lambda value: f"{value:.3f}")
    top["资本利得交易胜率"] = pd.to_numeric(top["资本利得交易胜率"], errors="coerce").map(lambda value: f"{value:.2%}")
    for column in ["累计资本利得_BP", "基准资本利得_BP", "资本利得超额_BP", "平均单笔_BP", "最差交易_BP", "未平仓浮动资本利得_BP", "资本利得最大回撤_BP"]:
        top[column] = pd.to_numeric(top[column], errors="coerce").map(lambda value: f"{value:.2f}")

    weights_chart = (
        Bar(init_opts=opts.InitOpts(width="100%", height="450px"))
        .add_xaxis(best_weights["因子"].tolist())
        .add_yaxis("权重", best_weights["最佳权重"].tolist(), color="#bb654f")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="最佳方案因子权重"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=24)),
            yaxis_opts=opts.AxisOpts(type_="value", name="分数"),
        )
    )
    if len(results) > 3000:
        scatter_sample = pd.concat(
            [results.head(1), results.iloc[1:].sample(n=2999, random_state=42)],
            ignore_index=True,
        )
    else:
        scatter_sample = results
    risk_chart = (
        Scatter(init_opts=opts.InitOpts(width="100%", height="440px"))
        .add_xaxis(scatter_sample["capital_gain_max_drawdown_bp"].round(3).tolist())
        .add_yaxis("候选策略", scatter_sample["capital_gain_excess_bp"].round(3).tolist(), symbol_size=6, color="#176b5b")
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"候选资本利得风险收益分布（展示{len(scatter_sample):,}个代表点）"),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
            xaxis_opts=opts.AxisOpts(type_="value", name="资本利得最大回撤（BP）", is_scale=True),
            yaxis_opts=opts.AxisOpts(type_="value", name="资本利得超额（BP）", is_scale=True),
        )
    )
    dependencies = set(weights_chart.js_dependencies.items) | set(risk_chart.js_dependencies.items)
    scripts = "".join(f'<script src="{CurrentConfig.ONLINE_HOST}{dependency}.js"></script>' for dependency in sorted(dependencies))
    formula = (
        f"{objective_config.capital_gain_bp_weight:g} × 累计资本利得BP + "
        f"{objective_config.capital_gain_excess_bp_weight:g} × 资本利得超额BP + "
        f"{objective_config.capital_trade_win_rate_weight:g} × 已平仓交易胜率 + "
        f"{objective_config.capital_gain_drawdown_bp_penalty:g} × 资本利得回撤BP + "
        f"{objective_config.total_return_weight:g} × 累计收益 + "
        f"{objective_config.excess_return_weight:g} × 超额收益 + "
        f"{objective_config.sharpe_weight:g} × 夏普 + "
        f"{objective_config.max_drawdown_penalty:g} × 最大回撤 + "
        f"{objective_config.signal_win_rate_weight:g} × 调仓周期胜率"
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>10Y地方债策略权重搜索</title>{scripts}
<style>
body{{margin:0;background:#f3f2ed;color:#18201d;font-family:Geist,"Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:48px 28px 80px}}h1{{font-size:38px;margin:0 0 14px}}h2{{margin-top:52px}}h3{{margin:28px 0 12px}}.lead{{color:#66716c;max-width:980px;line-height:1.8}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #cfd5d1;border-bottom:1px solid #cfd5d1;margin:30px 0}}.metric{{padding:22px 18px;border-right:1px solid #cfd5d1}}.metric:last-child{{border:0}}.metric b{{display:block;font-size:25px;color:#bb654f;margin-top:8px}}.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#d9ddd8;border:1px solid #d9ddd8}}.step{{background:#fbfaf6;padding:18px;line-height:1.65}}.step b{{display:block;color:#176b5b;margin-bottom:8px}}.chart{{background:#fbfaf6;margin:18px 0;padding:12px;border-radius:4px}}.callout{{border-left:4px solid #bb654f;background:#fbfaf6;padding:18px 22px;line-height:1.8}}table{{border-collapse:collapse;width:100%;font-size:12px;background:#fbfaf6}}th,td{{padding:8px;border-bottom:1px solid #d9ddd8;text-align:left;white-space:nowrap}}th{{background:#e8ece8;color:#176b5b;position:sticky;top:0}}.table-wrap{{overflow-x:auto;overflow-y:visible;border:1px solid #d9ddd8}}code{{color:#176b5b}}@media(max-width:800px){{.metrics,.steps{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>10Y地方债策略：因子权重搜索</h1>
<p class="lead">本报告汇总权重搜索的完整设计和结果。搜索只改变九个因子的赋分权重，定性阈值和仓位制度保持固定；全部候选使用NumPy矩阵分块计算，并以资本利得BP和逐笔胜率为主评价。</p>
<div class="metrics"><div class="metric">候选组合<b>{len(results):,}</b></div><div class="metric">累计资本利得<b>{best['capital_gain_total_bp']:.2f} BP</b></div><div class="metric">资本利得超额<b>{best['capital_gain_excess_bp']:.2f} BP</b></div><div class="metric">逐笔胜率<b>{best['capital_trade_win_rate']:.2%}</b></div></div>
<h2>搜索设计</h2><div class="steps"><div class="step"><b>权重步长</b>所有权重以5分为最小单位。</div><div class="step"><b>模块约束</b>供给20至40、银行10至25、估值25至45、非银5至25。</div><div class="step"><b>总分约束</b>正向权重合计固定100；发飞惩罚搜索0至-30。</div><div class="step"><b>排序目标</b>{formula}</div></div>
<h3>固定定性阈值</h3><div class="table-wrap">{threshold_table.to_html(index=False, escape=False)}</div>
<h3>固定仓位制度</h3><div class="table-wrap">{position_table.to_html(index=False, escape=False)}</div>
<h2>最佳结果</h2><div class="table-wrap">{summary.to_html(index=False, escape=False)}</div>
<div class="callout"><strong>目标函数只用于候选排序。</strong> 本次实际系数完整记录在“排序目标”中；零权重项目不影响排名。最终采用前仍需检查最差交易、资本利得回撤和样本外稳定性。</div>
<h3>最佳权重</h3><div class="table-wrap">{best_weights.to_html(index=False, escape=False)}</div>
<div class="chart">{weights_chart.render_embed()}</div><div class="chart">{risk_chart.render_embed()}</div>
<h2>目标函数Top20</h2><div class="table-wrap">{top.to_html(index=False, escape=False)}</div>
<h2>如何解读</h2><p class="lead">完整候选保存在 <code>all_results.csv</code>，前100保存在 <code>top_configs.csv</code>。为控制HTML体积，风险收益散点图使用固定随机种子展示最多3,000个代表点，并强制保留排名第一的候选；搜索、排序和CSV结果仍使用全部候选。本报告属于同一样本内权重选择，数据扩展后应使用滚动样本外验证。</p>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
