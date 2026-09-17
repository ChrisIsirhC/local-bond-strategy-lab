"""Queued, direct rolling validation for a selected factor universe.

This deliberately does not create a static factor-study batch first.  It
serializes the current factor universe and ordinary strategy settings into the
same single-worker queue used by rolling parameter research, then searches the
first and every following training window directly.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from common.config import DashboardStrategyConfig, ObjectiveConfig, load_strategy_config, save_strategy_config, strategy_config_from_dict
from common.experiments import _append_experiment_index, archive_dashboard_experiment
from common.factor_expansion_features import ALL_FACTOR_COLUMNS, BASE_FACTOR_COLUMNS, FACTOR_LABELS
from common.factor_expansion_research import (
    OBJECTIVES,
    evaluate_expansion_config,
    rebuild_expansion_stitched_cumulatives,
    root_research_config,
    run_expansion_rolling_research,
)
from common.performance import performance_metrics
from common.provenance import record_step
from common.reporting import write_strategy_outputs
from common.trade_metrics import capital_gain_trade_metrics
from strategies.dashboard_signal_v1 import DashboardWeights


ProgressCallback = Callable[[str, int | None, int | None], None]


def run_factor_rolling_task(
    root: Path,
    raw: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one factor universe directly and archive its rolling result.

    ``raw`` is intentionally a plain JSON-compatible payload: it is persisted
    in the queue before a worker starts, so page refreshes cannot alter the
    selected factors or execution rules midway through the run.
    """
    base_raw = raw.get("base_config")
    if not isinstance(base_raw, dict):
        raise ValueError("因子滚动任务缺少基线配置")
    base_config = strategy_config_from_dict(base_raw)
    enabled = tuple(str(value) for value in raw.get("enabled_factor_columns", ()) if str(value) in ALL_FACTOR_COLUMNS)
    if not enabled:
        raise ValueError("因子滚动任务至少需要一个启用因子")
    objective_name = str(raw.get("objective_name", "收益"))
    if objective_name not in OBJECTIVES:
        raise ValueError("因子滚动任务的搜索目标无效")
    frequency = str(raw.get("signal_frequency", base_config.signal_frequency))
    if frequency not in {"daily", "weekly"}:
        raise ValueError("因子滚动任务的信号频率无效")
    requested_version = str(raw.get("factor_version", ""))
    comparison_ready = set(BASE_FACTOR_COLUMNS).issubset(enabled) and bool(set(enabled).difference(BASE_FACTOR_COLUMNS))
    factor_version = "扩展因子" if comparison_ready else "自选因子"
    if requested_version == "自选因子":
        factor_version = "自选因子"
    minimum_months = int(raw.get("minimum_training_months", 24))
    recalibration_months = int(raw.get("recalibration_months", 3))
    if minimum_months <= 0 or recalibration_months <= 0:
        raise ValueError("最低训练长度和定参频率必须为正数")
    task_name = str(raw.get("task_name") or "").strip()
    factor_label = "扩展因子" if factor_version == "扩展因子" else "自选因子"
    display_frequency = "日频" if frequency == "daily" else "周频"
    task_name = task_name or f"因子完整滚动·{objective_name}优先·{display_frequency}·{factor_label}"
    objective = OBJECTIVES[objective_name]
    base = root_research_config(
        factor_version,
        objective_name,
        frequency,
        thresholds=base_config.thresholds,
        positions=base_config.positions,
        factor_windows=base_config.factor_windows,
        objective_config=objective,
        initial_weights=base_config.weights.as_dict(),
        enabled_factor_columns=enabled,
    )
    periods, daily = run_expansion_rolling_research(
        root,
        base,
        minimum_training_months=minimum_months,
        recalibration_months=recalibration_months,
        # This is the normal complete-configuration factor search, not the
        # faster factor-study rolling shortcut.
        beam_width=int(raw.get("beam_width", 160)),
        original_periods=None,
        progress=progress,
        cancel_requested=cancel_requested,
    )
    if periods.empty or daily.empty:
        raise ValueError("因子滚动任务未生成可归档的样本外结果")

    output_dir = root / "backtest_outputs" / "因子完整滚动" / (
        f"{_safe_component(task_name)}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    periods.to_csv(output_dir / "逐期定参与样本外表现.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "滚动定参样本外拼接_日度.csv", index=False, encoding="utf-8-sig")

    final_weights = json.loads(str(periods.iloc[-1]["权重"]))
    final_weights = {str(key): float(value) for key, value in final_weights.items()}
    first_period = periods.iloc[0]
    output_manifest = {
        "任务名称": task_name,
        "研究类型": "因子完整滚动定参",
        "因子版本": factor_version,
        "启用因子": list(enabled),
        "搜索目标": objective_name,
        "信号频率": frequency,
        "训练方式": "自首个共同可用数据日扩展",
        "最低训练长度月数": minimum_months,
        "定参频率月数": recalibration_months,
        "首次搜索期起始日": str(first_period["训练起始日"]),
        "首次搜索期结束日": str(first_period["训练截止日"]),
        "样本外起始日": str(periods.iloc[0]["样本外起始日"]),
        "样本外结束日": str(periods.iloc[-1]["样本外结束日"]),
        "说明": "直接入队的因子完整滚动验证；未先生成静态 F 研究归档。",
    }
    (output_dir / "滚动配置.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # The first parameter set is frozen in ``periods``.  Save the actual
    # training path it evaluated, just as the ordinary rolling engine does;
    # the stitched OOS file alone cannot represent this search period.
    search_daily, search_signals, _, _ = reconstruct_factor_rolling_search_period(
        root, base_config, output_dir, periods=periods, manifest=output_manifest,
    )
    search_daily.to_csv(output_dir / "滚动定参搜索期_日度.csv", index=False, encoding="utf-8-sig")
    search_signals.to_csv(output_dir / "滚动定参搜索期_信号.csv", index=False, encoding="utf-8-sig")

    oos_signals = (
        daily.loc[:, [column for column in ["signal_date", "总分", "结论", "目标仓位"] if column in daily.columns]]
        .rename(columns={"目标仓位": "仓位"})
        .drop_duplicates("signal_date", keep="last")
        .sort_values("signal_date")
        .reset_index(drop=True)
    )
    if "仓位" not in oos_signals:
        oos_signals["仓位"] = 0.0
    if "总分" not in oos_signals:
        oos_signals["总分"] = 50.0
    if "结论" not in oos_signals:
        oos_signals["结论"] = "中性"
    archive_daily = rebuild_expansion_stitched_cumulatives(
        pd.concat([search_daily, daily], ignore_index=True, sort=False)
        .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    )
    archive_daily["date"] = pd.to_datetime(archive_daily["date"], errors="coerce")
    archive_daily["signal_date"] = pd.to_datetime(archive_daily["signal_date"], errors="coerce")
    archive_daily = archive_daily.dropna(subset=["date", "signal_date"]).reset_index(drop=True)
    signals = (
        pd.concat([search_signals, oos_signals], ignore_index=True, sort=False)
        .assign(signal_date=lambda frame: pd.to_datetime(frame["signal_date"], errors="coerce"))
        .dropna(subset=["signal_date"])
        .drop_duplicates("signal_date", keep="last").sort_values("signal_date").reset_index(drop=True)
    )
    strategy_metrics = performance_metrics(archive_daily, return_col="strategy_return", nav_col="strategy_nav")
    benchmark_metrics = performance_metrics(archive_daily, return_col="total_return", nav_col="benchmark_nav_rebased")
    strategy_metrics.update(capital_gain_trade_metrics(archive_daily, "strategy_capital_bp", position_col="仓位"))
    benchmark_metrics.update(capital_gain_trade_metrics(archive_daily, "benchmark_capital_bp", position_col="comparison_position"))
    strategy_metrics["capital_gain_bp_definition"] = "收益率方向变动BP（不乘久期）"
    benchmark_metrics["benchmark_name"] = "多头同仓位10Y国债 / 非多头现金"
    legacy_weights = {key: final_weights.get(key, 0.0) for key in DashboardWeights().as_dict()}
    archive_config = DashboardStrategyConfig(
        name=task_name,
        weights=DashboardWeights(**legacy_weights),
        thresholds=base_config.thresholds,
        positions=base_config.positions,
        objective=objective,
        factor_windows=base_config.factor_windows,
        backtest_start=str(archive_daily["date"].min().date()),
        backtest_end=str(archive_daily["date"].max().date()),
        benchmark_id=base_config.benchmark_id,
        signal_frequency=frequency,
    )
    # Preserve the whole selected parent lineage before appending the direct
    # factor rolling contract.  This is deliberately the same mechanism used
    # by ordinary rolling research and makes all parent strategies linkable.
    archive_config = record_step(
        base_config,
        archive_config,
        root,
        "滚动定参",
        output_path=output_dir / "滚动配置.json",
        training_start=output_manifest["首次搜索期起始日"],
        training_end=output_manifest["首次搜索期结束日"],
        search_version="因子完整配置 V2 权重搜索 + 阈值搜索",
        entrypoint="common.factor_rolling_task.run_factor_rolling_task",
        arguments={
            "因子集合": "、".join(FACTOR_LABELS[column] for column in enabled),
            "训练窗口": "扩展窗口",
            "最低训练长度": f"{minimum_months}个月",
            "定参频率": f"每{recalibration_months}个月",
        },
        note="直接进入滚动任务队列；每期先在当前训练窗口搜索权重与阈值，再只在下一段样本外执行。",
    )
    provenance = dict(archive_config.research_provenance or {})
    notes = list(provenance.get("notes", [])) if isinstance(provenance.get("notes"), list) else []
    notes.append("未先生成静态因子研究归档；因子集合与搜索参数在入队时冻结。")
    provenance.update({
        "notes": list(dict.fromkeys(notes)),
        "研究类型": "因子完整滚动定参归档",
        "因子研究权重": final_weights,
        "因子集合": list(enabled),
        "因子版本": factor_version,
    })
    archive_config = replace(archive_config, research_provenance=provenance)
    experiment_dir = archive_dashboard_experiment(
        root,
        archive_config,
        archive_daily,
        signals,
        strategy_metrics,
        benchmark_metrics,
        source="因子滚动定参",
        research_metadata={
            "研究类型": "因子完整滚动定参",
            "训练起始日": output_manifest["首次搜索期起始日"],
            "训练截止日": output_manifest["首次搜索期结束日"],
            "样本外起始日": output_manifest["样本外起始日"],
            "样本外结束日": output_manifest["样本外结束日"],
            "滚动结果目录": str(output_dir.relative_to(root)),
            "定参期数": int(len(periods)),
        },
    )
    return {
        "output_dir": str(output_dir),
        "experiment_dir": str(experiment_dir),
        "task_name": task_name,
        "periods": periods,
}


def reconstruct_factor_rolling_search_period(
    root: Path,
    base_config: DashboardStrategyConfig,
    output_dir: Path,
    *,
    periods: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Rebuild the first frozen training path without running a new search.

    The first row of the rolling-period file is the selected configuration
    actually used after the initial search.  Replaying that configuration over
    its recorded training dates restores the diagnostic search path exactly in
    strategy terms, while preserving the original selected parameters.
    """
    if periods is None:
        periods = pd.read_csv(output_dir / "逐期定参与样本外表现.csv", encoding="utf-8-sig")
    if periods.empty:
        raise ValueError("因子完整滚动缺少首期定参记录")
    if manifest is None:
        manifest = json.loads((output_dir / "滚动配置.json").read_text(encoding="utf-8"))
    first = periods.iloc[0]
    try:
        raw_weights = json.loads(str(first["权重"]))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("因子完整滚动首期权重不可读取") from exc
    enabled = tuple(str(value) for value in manifest.get("启用因子", ()) if str(value) in ALL_FACTOR_COLUMNS)
    if not enabled:
        raise ValueError("因子完整滚动未登记启用因子")
    objective_name = str(manifest.get("搜索目标", "收益"))
    if objective_name not in OBJECTIVES:
        raise ValueError("因子完整滚动未登记有效搜索目标")
    frequency = str(manifest.get("信号频率", base_config.signal_frequency))
    factor_version = str(manifest.get("因子版本", "自选因子"))
    search_config = root_research_config(
        factor_version if factor_version in {"原始因子", "扩展因子", "自选因子"} else "自选因子",
        objective_name,
        frequency,
        thresholds=base_config.thresholds,
        positions=replace(
            base_config.positions,
            bullish_threshold=float(first["看多阈值"]),
            bearish_threshold=float(first["看空阈值"]),
        ),
        factor_windows=base_config.factor_windows,
        objective_config=base_config.objective,
        initial_weights={column: float(raw_weights.get(column, 0.0)) for column in enabled},
        enabled_factor_columns=enabled,
    )
    start = str(first["训练起始日"])
    end = str(first["训练截止日"])
    daily, signals, strategy, benchmark = evaluate_expansion_config(root, search_config, start, end)
    signals = signals.loc[pd.to_datetime(signals["signal_date"], errors="coerce") <= pd.Timestamp(end)].copy()
    return daily, signals, strategy, benchmark


def repair_factor_rolling_search_period_archives(root: Path) -> list[str]:
    """Backfill missing first-search artifacts for immutable factor roll archives.

    This is a migration, not a rerun: it consumes the archived first-period
    weights, thresholds, factor universe and training dates.  Existing OOS
    rows are retained verbatim; only their missing diagnostic prefix is added.
    """
    repaired: list[str] = []
    experiment_root = root / "backtest_outputs" / "experiments"
    for manifest_path in sorted(experiment_root.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        research = manifest.get("研究区间", {})
        if not isinstance(research, dict) or research.get("研究类型") != "因子完整滚动定参":
            continue
        experiment_dir = manifest_path.parent
        output_raw = research.get("滚动结果目录")
        output_dir = (root / Path(str(output_raw))).resolve() if output_raw else None
        if output_dir is None or not output_dir.is_dir():
            continue
        search_daily_path = output_dir / "滚动定参搜索期_日度.csv"
        search_signal_path = output_dir / "滚动定参搜索期_信号.csv"
        base_config = load_strategy_config(experiment_dir / "config.json")
        rolling_manifest = json.loads((output_dir / "滚动配置.json").read_text(encoding="utf-8"))
        periods = pd.read_csv(output_dir / "逐期定参与样本外表现.csv", encoding="utf-8-sig")
        training_start = pd.Timestamp(str(periods.iloc[0]["训练起始日"]))
        existing_daily_path = experiment_dir / "strategy_nav.csv"
        if search_daily_path.exists() and search_signal_path.exists() and existing_daily_path.exists():
            existing_frame = pd.read_csv(existing_daily_path, encoding="utf-8-sig", nrows=0)
            date_column = "date" if "date" in existing_frame.columns else "日期"
            existing_dates = pd.to_datetime(
                pd.read_csv(existing_daily_path, encoding="utf-8-sig", usecols=[date_column])[date_column],
                errors="coerce",
            )
            if not existing_dates.dropna().empty and existing_dates.min() <= training_start:
                continue
        search_daily, search_signals, _, _ = reconstruct_factor_rolling_search_period(
            root, base_config, output_dir, periods=periods, manifest=rolling_manifest,
        )
        search_daily.to_csv(search_daily_path, index=False, encoding="utf-8-sig")
        search_signals.to_csv(search_signal_path, index=False, encoding="utf-8-sig")
        oos_daily = pd.read_csv(output_dir / "滚动定参样本外拼接_日度.csv", encoding="utf-8-sig", parse_dates=["date"])
        archive_daily = rebuild_expansion_stitched_cumulatives(
            pd.concat([search_daily, oos_daily], ignore_index=True, sort=False)
            .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        )
        # Legacy OOS CSVs carry ISO strings while the replay produces
        # timestamps.  Reporting requires consistent datetimelike columns.
        archive_daily["date"] = pd.to_datetime(archive_daily["date"], errors="coerce")
        archive_daily["signal_date"] = pd.to_datetime(archive_daily["signal_date"], errors="coerce")
        archive_daily = archive_daily.dropna(subset=["date", "signal_date"]).reset_index(drop=True)
        oos_signals = pd.read_csv(experiment_dir / "signal_score.csv", encoding="utf-8-sig")
        archive_signals = (
            pd.concat([search_signals, oos_signals], ignore_index=True, sort=False)
            .assign(signal_date=lambda frame: pd.to_datetime(frame["signal_date"], errors="coerce"))
            .dropna(subset=["signal_date"])
            .drop_duplicates("signal_date", keep="last").sort_values("signal_date").reset_index(drop=True)
        )
        strategy = performance_metrics(archive_daily, return_col="strategy_return", nav_col="strategy_nav")
        benchmark = performance_metrics(archive_daily, return_col="total_return", nav_col="benchmark_nav_rebased")
        strategy.update(capital_gain_trade_metrics(archive_daily, "strategy_capital_bp", position_col="仓位"))
        benchmark.update(capital_gain_trade_metrics(archive_daily, "benchmark_capital_bp", position_col="comparison_position"))
        benchmark["benchmark_name"] = str(manifest.get("比较基准", "多头同仓位10Y国债 / 非多头现金"))
        write_strategy_outputs(archive_daily, archive_signals, strategy, benchmark, experiment_dir)
        repaired_config = replace(
            base_config,
            backtest_start=str(archive_daily["date"].min().date()),
            backtest_end=str(archive_daily["date"].max().date()),
        )
        save_strategy_config(repaired_config, experiment_dir / "config.json")
        manifest.update({
            "回测起始日期": strategy.get("start_date"),
            "回测结束日期": strategy.get("end_date"),
            "策略累计收益率": strategy.get("total_return"),
            "基准累计收益率": benchmark.get("total_return"),
            "累计超额收益率": float(strategy["total_return"]) - float(benchmark["total_return"]),
            "策略夏普比率": strategy.get("sharpe"),
            "策略最大回撤": strategy.get("max_drawdown"),
            "基准最大回撤": benchmark.get("max_drawdown"),
            "策略累计资本利得_BP": strategy.get("capital_gain_total_bp"),
            "基准累计资本利得_BP": benchmark.get("capital_gain_total_bp"),
            "资本利得超额_BP": float(strategy["capital_gain_total_bp"]) - float(benchmark["capital_gain_total_bp"]),
            "资本利得交易胜率": strategy.get("capital_gain_trade_win_rate"),
            "平均单笔资本利得_BP": strategy.get("capital_gain_avg_trade_bp"),
            "平均每笔盈利_BP": strategy.get("capital_gain_avg_win_bp"),
            "最差交易_BP": strategy.get("capital_gain_worst_trade_bp"),
            "平均每笔持有交易日": strategy.get("capital_gain_avg_holding_days"),
            "最长单笔持有交易日": strategy.get("capital_gain_max_holding_days"),
            "资本利得最大回撤_BP": strategy.get("capital_gain_max_drawdown_bp"),
            "基准资本利得最大回撤_BP": benchmark.get("capital_gain_max_drawdown_bp"),
        })
        artifacts = manifest.setdefault("产物", [])
        for name in ("滚动定参搜索期_日度.csv", "滚动定参搜索期_信号.csv"):
            if name not in artifacts:
                artifacts.append(name)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        _append_experiment_index(root, experiment_dir, manifest, strategy)
        repaired.append(str(manifest.get("short_id") or experiment_dir.name))
    return repaired


def _safe_component(value: str) -> str:
    return "".join("_" if char in '<>:"/\\|?*' else char for char in value).strip(". ") or "因子完整滚动"
