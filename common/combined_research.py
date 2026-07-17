from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

from common.config import DashboardStrategyConfig, ObjectiveConfig, load_strategy_config
from common.runner import run_dashboard_weight_search_v1
from common.threshold_research import run_threshold_research


COMBINED_SEARCH_SOURCE = "权重→阈值联合搜索"


def run_combined_search(
    root: Path,
    base_config: DashboardStrategyConfig,
    objective_config: ObjectiveConfig | None = None,
    training_end: str = "2025-06-30",
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    objective = objective_config or base_config.objective
    if progress is not None:
        progress("第一阶段：搜索因子权重")
    weight_metrics = run_dashboard_weight_search_v1(
        root,
        objective_config=objective,
        base_config=base_config,
        training_end=training_end,
        archive_result=False,
    )
    weight_config = load_strategy_config(Path(str(weight_metrics["best_config_path"])))
    frequency_label = "日频" if base_config.signal_frequency == "daily" else "周频"
    threshold_base = replace(
        weight_config,
        name=f"{frequency_label}联合搜索_权重阶段最优",
        thresholds=base_config.thresholds,
        positions=base_config.positions,
        objective=objective,
        backtest_start=None,
        backtest_end=None,
        signal_frequency=base_config.signal_frequency,
    )
    final_name = f"{frequency_label}联合搜索最优_权重后阈值"
    if progress is not None:
        progress("第二阶段：使用最优权重搜索定性阈值与看空规则")
    threshold_metrics = run_threshold_research(
        root,
        objective_config=objective,
        base_config_override=threshold_base,
        training_end=training_end,
        archive_source=COMBINED_SEARCH_SOURCE,
        final_strategy_name=final_name,
    )

    output_dir = root / "backtest_outputs" / ("联合搜索_v1_日频" if base_config.signal_frequency == "daily" else "联合搜索_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "信号频率": frequency_label,
        "训练截止日": training_end,
        "权重阶段最优配置": weight_metrics["best_config_path"],
        "权重阶段策略名称": weight_metrics["strategy_name"],
        "阈值阶段最优配置": threshold_metrics["best_config_path"],
        "最终实验目录": threshold_metrics["experiment_dir"],
        "最终累计资本利得_BP": threshold_metrics.get("capital_gain_total_bp"),
        "最终资本利得超额_BP": threshold_metrics.get("capital_gain_excess_bp"),
        "最终逐笔胜率": threshold_metrics.get("capital_gain_trade_win_rate"),
    }
    (output_dir / "联合搜索结果.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **threshold_metrics,
        "weight_stage": weight_metrics,
        "combined_summary_path": str(output_dir / "联合搜索结果.json"),
    }
