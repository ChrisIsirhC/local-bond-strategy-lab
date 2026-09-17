from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable, Literal

from common.config import DashboardStrategyConfig, ObjectiveConfig, load_strategy_config
from common.runner import run_dashboard_weight_search_v1, run_dashboard_weight_search_v2
from common.threshold_research import run_threshold_research


COMBINED_SEARCH_SOURCE = "权重→阈值联合搜索"
WeightSearchVersion = Literal["v1", "v2"]


def run_combined_search(
    root: Path,
    base_config: DashboardStrategyConfig,
    objective_config: ObjectiveConfig | None = None,
    training_start: str | None = None,
    training_end: str = "2025-06-30",
    progress: Callable[[str], None] | None = None,
    final_strategy_name: str | None = None,
    archive_result: bool = True,
    weight_search_version: WeightSearchVersion = "v2",
) -> dict[str, object]:
    if weight_search_version not in {"v1", "v2"}:
        raise ValueError(f"unsupported weight search version: {weight_search_version}")
    objective = objective_config or base_config.objective
    final_name = final_strategy_name or f"{_combined_strategy_name(base_config, training_end)}·{weight_search_version}"
    if progress is not None:
        progress(f"第一阶段：搜索因子权重 {weight_search_version}")
    weight_search = run_dashboard_weight_search_v1 if weight_search_version == "v1" else run_dashboard_weight_search_v2
    weight_metrics = weight_search(
        root,
        objective_config=objective,
        base_config=base_config,
        training_start=training_start,
        training_end=training_end,
        archive_result=False,
        persist_artifacts=archive_result,
    )
    weight_config = load_strategy_config(Path(str(weight_metrics["best_config_path"])))
    frequency_label = "日频" if base_config.signal_frequency == "daily" else "周频"
    threshold_base = replace(
        weight_config,
        name=f"{frequency_label}联合搜索_{weight_search_version}_权重阶段最优",
        thresholds=base_config.thresholds,
        positions=base_config.positions,
        objective=objective,
        backtest_start=None,
        backtest_end=None,
        signal_frequency=base_config.signal_frequency,
    )
    if progress is not None:
        progress("第二阶段：使用最优权重搜索定性阈值与看空规则")
    threshold_metrics = run_threshold_research(
        root,
        objective_config=objective,
        base_config_override=threshold_base,
        training_start=training_start,
        training_end=training_end,
        archive_source=COMBINED_SEARCH_SOURCE,
        final_strategy_name=final_name,
        archive_result=archive_result,
        research_metadata={"权重搜索版本": weight_search_version},
        persist_artifacts=archive_result,
    )

    output_dir = root / "backtest_outputs" / "联合搜索" / _safe_path_component(final_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "信号频率": frequency_label,
        "训练截止日": training_end,
        "训练起始日": training_start,
        "权重阶段最优配置": weight_metrics["best_config_path"],
        "权重阶段策略名称": weight_metrics["strategy_name"],
        "权重搜索版本": weight_search_version,
        "阈值阶段最优配置": threshold_metrics["best_config_path"],
        "最终策略名称": final_name,
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
        "weight_search_version": weight_search_version,
        "combined_summary_path": str(output_dir / "联合搜索结果.json"),
    }


def _combined_strategy_name(base_config: DashboardStrategyConfig, training_end: str) -> str:
    frequency = "日频" if base_config.signal_frequency == "daily" else "周频"
    positions = base_config.positions
    position_label = "/".join(
        f"{float(value):g}"
        for value in (
            positions.bullish_position,
            positions.neutral_position,
            positions.bearish_position,
        )
    )
    execution = []
    if positions.take_profit_bp > 0:
        execution.append(f"止盈{positions.take_profit_bp:g}BP")
    if positions.stop_loss_bp > 0:
        execution.append(f"止损{positions.stop_loss_bp:g}BP")
    if not execution:
        execution.append("无止盈止损")
    baseline = base_config.name.removeprefix("搜索基线_")
    cutoff = str(training_end).replace("-", "")
    return f"联合搜索·{baseline}·{frequency}·仓位{position_label}·{'_'.join(execution)}·训练至{cutoff}"


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")
