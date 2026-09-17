from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from common.config import DashboardStrategyConfig, ObjectiveConfig, save_strategy_config
from common.provenance import record_step
from common.experiments import archive_dashboard_experiment
from common.period_evaluation import evaluate_periods, period_summary_frame
from common.reporting import write_strategy_outputs
from common.runner import run_dashboard_config
from strategies.dashboard_signal_v1 import FactorWindowConfig


WINDOW_OPTIONS = (3, 6, 12)


def run_factor_window_research(
    root: Path,
    base_config: DashboardStrategyConfig,
    training_end: str,
    training_start: str | None = None,
    final_strategy_name: str | None = None,
) -> dict[str, object]:
    """Select percentile lookback windows on the training sample only."""
    base = replace(base_config, backtest_start=None, backtest_end=None)
    rows: list[dict[str, object]] = []

    for supply, bank, valuation in product(WINDOW_OPTIONS, repeat=3):
        windows = FactorWindowConfig(supply, bank, valuation)
        candidate = replace(base, factor_windows=windows)
        daily, signals, _, benchmark_metrics = run_dashboard_config(root, candidate)
        periods = evaluate_periods(daily, signals, training_end, benchmark_metrics["benchmark_name"])
        train, sample_out, recent, full = [item["strategy_metrics"] for item in periods.values()]
        rows.append(
            {
                "supply_months": supply,
                "bank_months": bank,
                "valuation_months": valuation,
                "objective": _objective_value(train, candidate.objective),
                "training_capital_gain_bp": train["capital_gain_total_bp"],
                "training_excess_bp": train["capital_gain_excess_bp"],
                "training_win_rate": train["capital_gain_trade_win_rate"],
                "training_avg_win_bp": train["capital_gain_avg_win_bp"],
                "training_drawdown_bp": train["capital_gain_max_drawdown_bp"],
                "oos_capital_gain_bp": sample_out["capital_gain_total_bp"],
                "oos_excess_bp": sample_out["capital_gain_excess_bp"],
                "oos_win_rate": sample_out["capital_gain_trade_win_rate"],
                "recent_capital_gain_bp": recent["capital_gain_total_bp"],
                "recent_excess_bp": recent["capital_gain_excess_bp"],
                "full_capital_gain_bp": full["capital_gain_total_bp"],
                "full_excess_bp": full["capital_gain_excess_bp"],
                "window_change_distance": abs(supply - base.factor_windows.supply_months)
                + abs(bank - base.factor_windows.bank_months)
                + abs(valuation - base.factor_windows.valuation_months),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["objective", "training_capital_gain_bp", "training_excess_bp", "window_change_distance"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = results.iloc[0]
    windows = FactorWindowConfig(
        int(best["supply_months"]),
        int(best["bank_months"]),
        int(best["valuation_months"]),
    )
    name = final_strategy_name or (
        f"因子窗口搜索_供给{windows.supply_months}M_银行{windows.bank_months}M_估值{windows.valuation_months}M"
    )
    best_config = replace(base, name=name, factor_windows=windows)
    daily, signals, strategy_metrics, benchmark_metrics = run_dashboard_config(root, best_config)
    periods = evaluate_periods(daily, signals, training_end, benchmark_metrics["benchmark_name"])

    frequency_label = "日频" if base.signal_frequency == "daily" else "周频"
    output_dir = root / "backtest_outputs" / "因子窗口搜索" / frequency_label
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "全部窗口组合.csv", index=False, encoding="utf-8-sig")
    results.head(20).to_csv(output_dir / "窗口组合Top20.csv", index=False, encoding="utf-8-sig")
    period_summary_frame(periods).to_csv(output_dir / "最优窗口分区间表现.csv", index=False, encoding="utf-8-sig")

    config_path = root / "configs" / "experiments" / f"{name}.json"
    best_config = record_step(
        base_config, best_config, root, "因子窗口搜索", output_path=config_path,
        training_end=training_end, search_version="3/6/12月窗口组合",
        entrypoint="common.factor_window_research.run_factor_window_research",
        arguments={"training_start": training_start, "training_end": training_end,
                   "final_strategy_name": final_strategy_name},
        note="供给、银行、估值各3/6/12月。当前实现训练起点为首个可用观察，training_start未参与筛选。",
    )
    save_strategy_config(best_config, config_path)
    write_strategy_outputs(
        daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        output_dir / "best_config_report",
    )
    experiment_dir = archive_dashboard_experiment(
        root,
        best_config,
        daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        source="因子窗口参数搜索",
        research_metadata={
            "训练起始日": training_start,
            "训练截止日": training_end,
            "样本外起始日": (pd.Timestamp(training_end) + pd.Timedelta(days=1)).date().isoformat(),
            "窗口候选": "供给、银行、估值各3M/6M/1Y",
            "候选数量": int(len(results)),
        },
    )
    period_summary_frame(periods).to_csv(experiment_dir / "period_evaluation.csv", index=False, encoding="utf-8-sig")
    return {
        **strategy_metrics,
        "strategy_name": name,
        "best_config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
        "candidate_count": int(len(results)),
        "factor_windows": windows.as_dict(),
        "results_path": str(output_dir / "全部窗口组合.csv"),
        "period_evaluation": period_summary_frame(periods).to_dict(orient="records"),
    }


def _objective_value(metrics: dict[str, object], objective: ObjectiveConfig) -> float:
    def value(name: str) -> float:
        raw = metrics.get(name)
        return 0.0 if raw is None or pd.isna(raw) else float(raw)

    return (
        objective.total_return_weight * value("total_return")
        + objective.excess_return_weight * value("excess_return")
        + objective.sharpe_weight * value("sharpe")
        + objective.signal_win_rate_weight * value("signal_period_win_rate")
        + objective.capital_gain_bp_weight * value("capital_gain_total_bp")
        + objective.capital_gain_excess_bp_weight * value("capital_gain_excess_bp")
        + objective.capital_trade_win_rate_weight * value("capital_gain_trade_win_rate")
        + objective.capital_gain_avg_win_bp_weight * value("capital_gain_avg_win_bp")
        + objective.capital_gain_drawdown_bp_penalty * value("capital_gain_max_drawdown_bp")
    )
