from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Callable, Literal
from html import escape
import math

import pandas as pd

from common.combined_research import WeightSearchVersion, run_combined_search
from common.config import DashboardStrategyConfig, ObjectiveConfig, load_strategy_config, save_strategy_config
from common.data_quality import assert_no_long_date_gaps
from common.market_data import GOV_10Y, load_market_data
from common.runner import _backtest_dashboard_signals, run_dashboard_config, run_dashboard_weight_search_v1, run_dashboard_weight_search_v2
from common.threshold_research import run_threshold_research
from common.trade_metrics import capital_gain_trade_metrics
from common.performance import performance_metrics
from common.experiments import archive_dashboard_experiment, archive_id_prefix
from common.provenance import display_provenance
from strategies.dashboard_signal_v1 import build_dashboard_signal


SearchMode = Literal["weight", "threshold", "combined"]
WindowMode = Literal["expanding", "rolling_months"]
IntervalUnit = Literal["days", "weeks", "months"]
ProgressCallback = Callable[[str, int | None, int | None], None]


class RollingResearchCancelled(RuntimeError):
    """Raised at a safe period boundary after a user cancels a rolling task."""


@dataclass(frozen=True)
class RollingResearchConfig:
    base_config: DashboardStrategyConfig
    search_mode: SearchMode
    training_mode: WindowMode
    first_training_end: str | None
    recalibration_interval: int
    recalibration_unit: IntervalUnit
    rolling_window_months: int | None = None
    min_objective_improvement: float = 0.0
    task_name: str | None = None
    weight_search_version: WeightSearchVersion = "v2"
    minimum_training_months: int | None = None
    resume_from: Path | None = None


def run_rolling_research(
    root: Path,
    config: RollingResearchConfig,
    progress: ProgressCallback | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Search only on each period's training window, then stitch its next OOS segment."""
    market = load_market_data(root, GOV_10Y).sort_values("date").reset_index(drop=True)
    if config.weight_search_version not in {"v1", "v2"}:
        raise ValueError(f"unsupported weight search version: {config.weight_search_version}")
    if config.search_mode not in {"weight", "threshold", "combined"}:
        raise ValueError("不支持的搜索模式")
    if config.training_mode not in {"expanding", "rolling_months"}:
        raise ValueError("不支持的训练窗口模式")
    if config.recalibration_unit not in {"days", "weeks", "months"}:
        raise ValueError("不支持的定参频率单位")
    if config.recalibration_interval <= 0:
        raise ValueError("定参频率必须为正数")
    if config.training_mode == "rolling_months" and (config.rolling_window_months or 0) <= 0:
        raise ValueError("固定滚动窗口需要填写训练窗口月数")
    minimum_training_months = _minimum_training_months(config)
    if not math.isfinite(config.min_objective_improvement) or config.min_objective_improvement < 0:
        raise ValueError("参数切换门槛必须是有限非负数")
    signals = build_dashboard_signal(
        root, weights=config.base_config.weights, thresholds=config.base_config.thresholds,
        position_policy=config.base_config.positions, factor_windows=config.base_config.factor_windows,
        signal_frequency=config.base_config.signal_frequency,
    )
    signal_dates = pd.to_datetime(signals["signal_date"], errors="raise")
    assert_no_long_date_gaps(market, "date", "滚动定参行情")
    assert_no_long_date_gaps(signals, "signal_date", "滚动定参信号",
                            max_gap_days=14 if config.base_config.signal_frequency == "daily" else 28)
    market = market.loc[market["date"].between(signal_dates.min(), signal_dates.max())].reset_index(drop=True)
    requested_first_end = pd.Timestamp(config.first_training_end) if config.first_training_end else None
    first_end = _first_eligible_training_end(market["date"], requested_first_end, config)
    if first_end is None or first_end >= market["date"].max():
        raise ValueError("数据不足以满足最低训练长度并留下下一段样本外区间")
    last_date = pd.Timestamp(market["date"].max())
    total_periods = _rolling_period_count(market["date"], first_end, last_date, config)
    if progress:
        progress("正在准备数据", 0, total_periods)
    _raise_if_cancelled(cancel_requested)

    task_name = config.task_name or _default_task_name(config)
    output_dir = root / "backtest_outputs" / "滚动定参" / _safe_component(task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_strategy_config(config.base_config, output_dir / "基线配置.json")

    periods: list[dict[str, object]] = []
    oos_signals: list[pd.DataFrame] = []
    current_config: DashboardStrategyConfig | None = None
    first_search_daily: pd.DataFrame | None = None
    first_search_signals: pd.DataFrame | None = None
    first_search_start: pd.Timestamp | None = None
    first_search_end: pd.Timestamp | None = None
    train_end = first_end
    reused_period_count = 0
    new_search_count = 0

    reusable = _load_reusable_periods(root, config, market["date"], last_date, progress)
    for source_row, selected in reusable:
        _raise_if_cancelled(cancel_requested)
        oos_start = pd.Timestamp(source_row["样本外起始日"])
        oos_end = pd.Timestamp(source_row["样本外结束日"])
        train_start = pd.Timestamp(source_row["训练起始日"])
        saved_train_end = pd.Timestamp(source_row["训练截止日"])
        selected_signals = build_dashboard_signal(
            root, weights=selected.weights, thresholds=selected.thresholds,
            position_policy=selected.positions, factor_windows=selected.factor_windows,
            signal_frequency=selected.signal_frequency,
        )
        if first_search_daily is None:
            search_market = market.loc[market["date"].between(train_start, saved_train_end)].copy()
            search_signals = selected_signals.loc[
                pd.to_datetime(selected_signals["signal_date"], errors="coerce") <= saved_train_end
            ].copy()
            first_search_daily, _, _ = _backtest_dashboard_signals(
                search_market, search_signals, selected.positions,
            )
            first_search_signals = search_signals
            first_search_start = train_start
            first_search_end = saved_train_end
        oos_signals.append(_segment_signals(selected_signals, oos_start, oos_end))
        period_number = len(periods) + 1
        config_file = output_dir / "periods" / f"{period_number:02d}_{saved_train_end:%Y%m%d}" / "参数配置.json"
        save_strategy_config(selected, config_file)
        row = dict(source_row)
        row.update({
            "期数": period_number,
            "训练起始日": train_start.date().isoformat(),
            "训练截止日": saved_train_end.date().isoformat(),
            "样本外起始日": oos_start.date().isoformat(),
            "样本外结束日": oos_end.date().isoformat(),
            "参数来源": "历史参数复用",
            "参数配置文件": str(config_file),
        })
        periods.append(row)
        current_config = selected
        train_end = oos_end
        reused_period_count += 1
        if progress:
            progress(
                f"第 {period_number} 期：复用已保存参数；样本外 "
                f"{oos_start:%Y-%m-%d} 至 {oos_end:%Y-%m-%d}",
                period_number,
                total_periods,
            )

    while train_end < last_date:
        _raise_if_cancelled(cancel_requested)
        train_start = _training_start(market["date"], train_end, config)
        next_cutoff = _advance(train_end, config.recalibration_interval, config.recalibration_unit)
        oos_start = _next_market_date(market["date"], train_end)
        oos_end = _on_or_before(market["date"], next_cutoff) or last_date
        if oos_start is None or oos_start > oos_end:
            break

        if progress:
            progress(
                f"第 {len(periods) + 1} 期：训练 {train_start:%Y-%m-%d} 至 {train_end:%Y-%m-%d}；样本外 {oos_start:%Y-%m-%d} 至 {oos_end:%Y-%m-%d}",
                len(periods),
                total_periods,
            )

        search_result = _search(root, config, train_start, train_end)
        _raise_if_cancelled(cancel_requested)
        new_search_count += 1
        candidate = load_strategy_config(Path(str(search_result["best_config_path"])))
        candidate = replace(candidate, objective=config.base_config.objective, benchmark_id=config.base_config.benchmark_id)
        if (candidate.positions.take_profit_bp, candidate.positions.stop_loss_bp) != (
            config.base_config.positions.take_profit_bp, config.base_config.positions.stop_loss_bp
        ):
            raise ValueError("滚动搜索不得改变固定止盈止损规则")
        candidate_score, _ = _score_config(root, candidate, train_start, train_end)
        if not math.isfinite(candidate_score):
            raise ValueError("候选目标函数非有限数，停止滚动定参")
        previous_score: float | None = None
        candidate_changed = False
        accepted = True
        selected = candidate
        if current_config is not None:
            previous_score, _ = _score_config(root, current_config, train_start, train_end)
            if not math.isfinite(previous_score):
                raise ValueError("上一期参数目标函数非有限数，停止滚动定参")
            candidate_changed = not _same_strategy_parameters(candidate, current_config)
            accepted = candidate_changed and candidate_score - previous_score >= config.min_objective_improvement
            if not accepted:
                selected = current_config

        selected_signals = build_dashboard_signal(
            root, weights=selected.weights, thresholds=selected.thresholds,
            position_policy=selected.positions, factor_windows=selected.factor_windows,
            signal_frequency=selected.signal_frequency,
        )
        if first_search_daily is None:
            # Keep the first training window as an explicit, reproducible
            # search-period artifact. It is diagnostic only and is never
            # mixed into the stitched out-of-sample execution path.
            # Reuse the already loaded market frame and selected signals. This
            # keeps the diagnostic artifact independent of a second data load
            # and makes mocked/unit-test data behave exactly like production.
            search_market = market.loc[market["date"].between(train_start, train_end)].copy()
            search_signals = selected_signals.loc[
                pd.to_datetime(selected_signals["signal_date"], errors="coerce") <= train_end
            ].copy()
            first_search_daily, _, _ = _backtest_dashboard_signals(
                search_market, search_signals, selected.positions,
            )
            first_search_signals = search_signals
            first_search_start = train_start
            first_search_end = train_end
        oos_signals.append(_segment_signals(selected_signals, oos_start, oos_end))
        switched = current_config is not None and accepted

        period_number = len(periods) + 1
        config_file = output_dir / "periods" / f"{period_number:02d}_{train_end:%Y%m%d}" / "参数配置.json"
        save_strategy_config(selected, config_file)
        periods.append({
            "期数": period_number,
            "权重搜索版本": config.weight_search_version,
            "训练起始日": train_start.date().isoformat(),
            "训练截止日": train_end.date().isoformat(),
            "样本外起始日": oos_start.date().isoformat(),
            "样本外结束日": oos_end.date().isoformat(),
            "搜索模式": config.search_mode,
            "候选目标函数": candidate_score,
            "上一期参数在当前窗口目标函数": previous_score,
            "目标函数改善": candidate_score - previous_score if previous_score is not None else None,
            "最小改善阈值": config.min_objective_improvement,
            "候选参数是否变化": candidate_changed,
            "首次定参": current_config is None,
            "是否切换新参数": switched,
            "参数来源": "本期重新搜索",
            "采用参数": selected.name,
            "参数配置文件": str(config_file),
        })
        current_config = selected
        train_end = oos_end
        if progress:
            progress(
                f"第 {period_number} 期已完成；下一期将切换训练窗口。",
                period_number,
                total_periods,
            )

    if not oos_signals:
        raise ValueError("没有可拼接的样本外区间，请提前首次训练截止日")
    _raise_if_cancelled(cancel_requested)
    # Execute the chronological target path once so returns and stop state survive boundaries.
    stitched, stitched_metrics = _execute_oos_path(market, oos_signals, first_end, config.base_config)
    for period in periods:
        mask = stitched["date"].between(pd.Timestamp(period["样本外起始日"]), pd.Timestamp(period["样本外结束日"]))
        segment = stitched.loc[mask]
        period["样本外资本利得_BP"] = float(segment["strategy_capital_bp"].sum())
        period["样本外超额_BP"] = float(segment["capital_excess_bp"].sum())
        segment_metrics = capital_gain_trade_metrics(segment, "strategy_capital_bp", position_col="仓位")
        period["样本外交易胜率"] = segment_metrics.get("capital_gain_trade_win_rate")
        stitched.loc[mask, "滚动训练截止日"] = period["训练截止日"]
        stitched.loc[mask, "滚动样本外起始日"] = period["样本外起始日"]
        stitched.loc[mask, "滚动参数已更新"] = period["是否切换新参数"]
    stitched.to_csv(output_dir / "滚动定参样本外拼接_日度.csv", index=False, encoding="utf-8-sig")
    if first_search_daily is not None and first_search_signals is not None:
        first_search_daily.to_csv(output_dir / "滚动定参搜索期_日度.csv", index=False, encoding="utf-8-sig")
        first_search_signals.to_csv(output_dir / "滚动定参搜索期_信号.csv", index=False, encoding="utf-8-sig")
    period_frame = pd.DataFrame(periods)
    period_frame.to_csv(output_dir / "逐期定参与样本外表现.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "任务名称": task_name,
        "运行ID": None,
        "搜索模式": config.search_mode,
        "训练方式": config.training_mode,
        "启动方式": "自动最早启动" if config.first_training_end is None else "指定日期",
        "共同可用数据起始日": market["date"].min().date().isoformat(),
        "首次训练截止日": first_end.date().isoformat(),
        "请求首次训练截止日": config.first_training_end,
        "实际首次训练截止日": first_end.date().isoformat(),
        "首次搜索期起始日": first_search_start.date().isoformat() if first_search_start is not None else None,
        "首次搜索期结束日": first_search_end.date().isoformat() if first_search_end is not None else None,
        "定参频率": f"{config.recalibration_interval} {config.recalibration_unit}",
        "固定训练窗口月数": config.rolling_window_months,
        "最低训练长度月数": minimum_training_months,
        "权重搜索版本": config.weight_search_version,
        "最小目标函数改善阈值": config.min_objective_improvement,
        "续跑来源": str(config.resume_from) if config.resume_from else None,
        "复用历史期数": reused_period_count,
        "本次新增搜索期数": new_search_count,
        "样本外起始日": str(stitched["date"].min().date()),
        "样本外结束日": str(stitched["date"].max().date()),
        "样本外逐笔胜率": stitched_metrics.get("capital_gain_trade_win_rate"),
        "执行口径": "逐期选参，连续执行；跨期延续持仓与止盈止损状态",
    }
    (output_dir / "滚动配置.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive_daily = stitched
    # Keep the complete per-period signal rows.  `stitched` is a daily return
    # path and intentionally only carries the execution fields, whereas
    # `oos_signals` also preserves signal generation dates, applicable dates,
    # data-as-of fields, factor inputs and factor scores for audit/display.
    archive_oos_signals = pd.concat(oos_signals, ignore_index=True, sort=False)
    if first_search_daily is not None and first_search_signals is not None:
        archive_daily = pd.concat([first_search_daily, stitched], ignore_index=True).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        archive_daily = _rebuild_stitched_paths(archive_daily)
        archive_signals = pd.concat([first_search_signals, archive_oos_signals], ignore_index=True, sort=False)
    else:
        archive_signals = archive_oos_signals
    if "目标仓位" in archive_signals and "仓位" not in archive_signals:
        archive_signals = archive_signals.rename(columns={"目标仓位": "仓位"})
    archive_signals = (
        archive_signals.drop_duplicates("signal_date", keep="last")
        .sort_values("signal_date")
        .reset_index(drop=True)
    )
    archive_signals.attrs["weights"] = config.base_config.weights.as_dict()
    archive_signals.attrs["thresholds"] = config.base_config.thresholds.as_dict()
    archive_signals.attrs["signal_frequency"] = config.base_config.signal_frequency
    archive_strategy_metrics = performance_metrics(archive_daily, return_col="strategy_return", nav_col="strategy_nav")
    archive_benchmark_metrics = performance_metrics(archive_daily, return_col="total_return", nav_col="benchmark_nav_rebased")
    archive_strategy_metrics.update(capital_gain_trade_metrics(archive_daily, "strategy_capital_bp", position_col="仓位"))
    archive_benchmark_metrics.update(capital_gain_trade_metrics(archive_daily, "benchmark_capital_bp", position_col="comparison_position"))
    archive_strategy_metrics["capital_gain_bp_definition"] = "收益率方向变动BP（不乘久期）"
    archive_benchmark_metrics["benchmark_name"] = "多头同仓位10Y国债 / 非多头现金"
    # A rolling run is a continuation of its selected strategy, not a new
    # research origin.  Retain the complete, documented parent lineage and
    # append only this run's rolling-search contract.
    rolling_provenance = display_provenance(config.base_config, root) or {}
    if not rolling_provenance:
        origin = {"label": config.base_config.name}
        if config.base_config.source_config_path:
            try:
                origin["path"] = str(Path(config.base_config.source_config_path).resolve().relative_to(root.resolve())).replace("\\", "/")
            except ValueError:
                origin["path"] = str(config.base_config.source_config_path)
        rolling_provenance = {"origin": origin, "earlier_history": "未登记", "steps": []}
    else:
        rolling_provenance = deepcopy(rolling_provenance)
        origin = rolling_provenance["origin"]

    prior_output = rolling_provenance["steps"][-1].get("output", origin) if rolling_provenance.get("steps") else origin
    window_description = (
        "扩展窗口"
        if config.training_mode == "expanding"
        else f"固定滚动窗口（近{config.rolling_window_months}个月）"
    )
    interval_label = {"days": "日", "weeks": "周", "months": "个月"}[config.recalibration_unit]
    search_label = _rolling_search_label(config)
    rolling_provenance.setdefault("steps", []).append({
        "operation": "滚动定参",
        "input": deepcopy(prior_output),
        "output": {"label": task_name, "path": str(output_dir.relative_to(root)).replace("\\", "/")},
        "training_start": manifest.get("首次搜索期起始日"),
        "training_end": manifest.get("首次搜索期结束日"),
        "signal_frequency": config.base_config.signal_frequency,
        "search_version": search_label,
        "arguments": {
            "训练窗口": window_description,
            "最低训练长度": f"{minimum_training_months}个月",
            "定参频率": f"每{config.recalibration_interval}{interval_label}",
            "参数切换门槛": config.min_objective_improvement,
        },
        "note": (
            f"{window_description}；最低训练长度{minimum_training_months}个月；"
            f"每{config.recalibration_interval}{interval_label}定参；参数切换门槛 {config.min_objective_improvement:g}。"
            f"复用 {reused_period_count} 期已保存参数，本次新增搜索 {new_search_count} 期；"
            "仅在跨过新的定参点时重新搜索，每段参数只在随后样本外执行。"
        ),
    })
    rolling_provenance.setdefault("notes", []).append(
        "滚动结果的首段搜索期和后续样本外区间均已保存；历史实验页按标准区间指标展示。"
    )
    archive_config = replace(config.base_config, name=task_name, backtest_start=str(archive_daily["date"].min().date()), backtest_end=str(archive_daily["date"].max().date()), research_provenance=rolling_provenance)
    experiment_dir = archive_dashboard_experiment(
        root, archive_config, archive_daily, archive_signals,
        archive_strategy_metrics, archive_benchmark_metrics,
        source="滚动定参",
        research_metadata={
            "研究类型": "滚动定参",
            "训练起始日": manifest.get("首次搜索期起始日"),
            "训练截止日": manifest.get("首次搜索期结束日"),
            "首次搜索期起始日": manifest.get("首次搜索期起始日"),
            "首次搜索期结束日": manifest.get("首次搜索期结束日"),
            "样本外起始日": manifest.get("样本外起始日"),
            "样本外结束日": manifest.get("样本外结束日"),
            "滚动结果目录": str(output_dir.relative_to(root)),
            "定参期数": len(periods),
        },
    )
    experiment_manifest = json.loads((experiment_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rolling_id = str(experiment_manifest.get("short_id", "")).strip()
    expected_prefix = archive_id_prefix("滚动定参", config.base_config)
    if not rolling_id.startswith(expected_prefix):
        raise ValueError(f"滚动定参历史归档未生成有效运行 ID（应为 {expected_prefix} 序列）")
    manifest["运行ID"] = rolling_id
    manifest["历史实验目录"] = str(experiment_dir.relative_to(root))
    (output_dir / "滚动配置.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_html_report(output_dir, manifest, period_frame, stitched)
    return {"output_dir": str(output_dir), "task_name": task_name, "periods": period_frame,
            "daily": stitched, "manifest": manifest, "metrics": stitched_metrics,
            "experiment_dir": str(experiment_dir)}


def _load_reusable_periods(
    root: Path,
    config: RollingResearchConfig,
    market_dates: pd.Series,
    last_date: pd.Timestamp,
    progress: ProgressCallback | None = None,
) -> list[tuple[dict[str, object], DashboardStrategyConfig]]:
    """Load saved period parameters and extend only the unfinished last period."""
    if config.resume_from is None:
        return []
    source_dir = Path(config.resume_from)
    if not source_dir.is_absolute():
        source_dir = root / source_dir
    period_path = source_dir / "逐期定参与样本外表现.csv"
    base_path = source_dir / "基线配置.json"
    if not period_path.exists() or not base_path.exists():
        if progress:
            progress("续跑来源缺少逐期参数，改为从第一期重新搜索。", None, None)
        return []
    source_base = load_strategy_config(base_path)
    if not _same_strategy_parameters(source_base, config.base_config):
        if progress:
            progress("当前策略参数已修改，与续跑来源不一致，改为从第一期重新搜索。", None, None)
        return []
    frame = pd.read_csv(period_path, encoding="utf-8-sig").sort_values("期数").reset_index(drop=True)
    loaded: list[tuple[dict[str, object], DashboardStrategyConfig]] = []
    fallback_configs = sorted((source_dir / "periods").glob("*/参数配置.json"))
    for index, row in frame.iterrows():
        oos_start = pd.Timestamp(row["样本外起始日"])
        if oos_start > last_date:
            break
        oos_end = min(pd.Timestamp(row["样本外结束日"]), last_date)
        raw_path = Path(str(row.get("参数配置文件", "")))
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(source_dir / raw_path)
        if index < len(fallback_configs):
            candidates.append(fallback_configs[index])
        config_path = next((path for path in candidates if str(path) and path.exists()), None)
        if config_path is None:
            if progress:
                progress(f"第 {index + 1} 期参数文件缺失，后续区间改为重新搜索。", None, None)
            break
        values = row.to_dict()
        values["样本外结束日"] = oos_end.date().isoformat()
        for key in ("首次定参", "候选参数是否变化", "是否切换新参数"):
            if key in values:
                values[key] = _stored_bool(values[key])
        loaded.append((values, load_strategy_config(config_path)))
    if not loaded:
        return []

    # A run commonly ends between two scheduled recalibration dates. New data
    # before the next date belongs to the existing last parameter period.
    final_row, final_config = loaded[-1]
    source_final_end = pd.Timestamp(final_row["样本外结束日"])
    source_train_end = pd.Timestamp(final_row["训练截止日"])
    scheduled_cutoff = _on_or_before(
        market_dates, _advance(source_train_end, config.recalibration_interval, config.recalibration_unit)
    )
    if scheduled_cutoff is None:
        scheduled_cutoff = last_date
    extension_end = min(scheduled_cutoff, last_date)
    if extension_end > source_final_end:
        final_row["样本外结束日"] = extension_end.date().isoformat()
        loaded[-1] = (final_row, final_config)
    return loaded


def _stored_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "是"}


def _segment_signals(signals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ordered = signals.sort_values("signal_date")
    prior = ordered.loc[ordered["signal_date"] <= start].tail(1).copy()
    if prior.empty:
        raise ValueError("样本外首日没有已生效的信号")
    prior["signal_date"] = start
    rest = ordered.loc[(ordered["signal_date"] > start) & (ordered["signal_date"] <= end)]
    result = pd.concat([prior, rest], ignore_index=True)
    result.attrs = {}
    return result


def _execute_oos_path(market, segments, anchor_date, base_config):
    signals = pd.concat(segments, ignore_index=True)
    anchor = signals.iloc[[0]].copy()
    anchor["signal_date"] = anchor_date
    anchor["仓位"] = 0.0
    anchor["总分"] = 50.0
    anchor["结论"] = "中性"
    end = market["date"].max()
    if signals["signal_date"].max() < end:
        tail = signals.iloc[[-1]].copy()
        tail["signal_date"] = end
        signals = pd.concat([signals, tail], ignore_index=True)
    signals = pd.concat([anchor, signals], ignore_index=True)
    daily, _, _ = _backtest_dashboard_signals(
        market.loc[market["date"] >= anchor_date].copy(), signals, base_config.positions,
    )
    daily = _rebuild_stitched_paths(daily.loc[daily["date"] > anchor_date])
    metrics = capital_gain_trade_metrics(daily, "strategy_capital_bp", position_col="仓位")
    return daily, metrics


def _search(root: Path, config: RollingResearchConfig, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    arguments = dict(root=root, objective_config=config.base_config.objective, training_start=start.date().isoformat(), training_end=end.date().isoformat())
    if config.search_mode == "weight":
        search = run_dashboard_weight_search_v1 if config.weight_search_version == "v1" else run_dashboard_weight_search_v2
        return search(base_config=config.base_config, archive_result=False, persist_artifacts=False, **arguments)
    if config.search_mode == "threshold":
        return run_threshold_research(base_config_override=config.base_config, archive_result=False, persist_artifacts=False, **arguments)
    return run_combined_search(
        root,
        base_config=config.base_config,
        objective_config=config.base_config.objective,
        training_start=start.date().isoformat(),
        training_end=end.date().isoformat(),
        archive_result=False,
        weight_search_version=config.weight_search_version,
    )


def _score_config(root: Path, config: DashboardStrategyConfig, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, dict[str, object]]:
    checked = replace(config, backtest_start=start.date().isoformat(), backtest_end=end.date().isoformat())
    daily, _, metrics, benchmark = run_dashboard_config(root, checked)
    objective = config.objective
    excess_return = float(metrics.get("total_return", 0.0) - benchmark.get("total_return", 0.0))
    capital_excess = float(metrics.get("capital_gain_total_bp", 0.0) - benchmark.get("capital_gain_total_bp", 0.0))
    score = (
        objective.total_return_weight * float(metrics.get("total_return", 0.0))
        + objective.excess_return_weight * excess_return
        + objective.sharpe_weight * float(metrics.get("sharpe", 0.0))
        + objective.max_drawdown_penalty * float(metrics.get("max_drawdown", 0.0))
        + objective.signal_win_rate_weight * float(metrics.get("signal_period_win_rate", 0.0) or 0.0)
        + objective.capital_gain_bp_weight * float(metrics.get("capital_gain_total_bp", 0.0))
        + objective.capital_gain_excess_bp_weight * capital_excess
        + objective.capital_trade_win_rate_weight * float(metrics.get("capital_gain_trade_win_rate", 0.0) or 0.0)
        + objective.capital_gain_avg_win_bp_weight * float(metrics.get("capital_gain_avg_win_bp", 0.0) or 0.0)
        + objective.capital_gain_drawdown_bp_penalty * float(metrics.get("capital_gain_max_drawdown_bp", 0.0) or 0.0)
    )
    return float(score), metrics


def _same_strategy_parameters(left: DashboardStrategyConfig, right: DashboardStrategyConfig) -> bool:
    """Compare tradable/search parameters, ignoring display name and backtest dates."""
    return (
        left.weights == right.weights
        and left.thresholds == right.thresholds
        and left.positions == right.positions
        and left.factor_windows == right.factor_windows
        and left.objective == right.objective
        and left.benchmark_id == right.benchmark_id
        and left.signal_frequency == right.signal_frequency
    )


def _training_start(dates: pd.Series, end: pd.Timestamp, config: RollingResearchConfig) -> pd.Timestamp:
    if config.training_mode == "expanding":
        return pd.Timestamp(dates.min())
    requested = end - pd.DateOffset(months=int(config.rolling_window_months or 1))
    if requested < pd.Timestamp(dates.min()):
        raise ValueError("历史数据不足以覆盖完整固定训练窗口")
    return _on_or_after(dates, requested)


def _minimum_training_months(config: RollingResearchConfig) -> int:
    value = config.minimum_training_months
    if value is None:
        value = config.rolling_window_months if config.training_mode == "rolling_months" else 24
    if value is None or int(value) != value or int(value) <= 0:
        raise ValueError("最低训练长度必须为正数")
    if config.training_mode == "rolling_months" and int(value) != config.rolling_window_months:
        raise ValueError("固定窗口的最低训练长度必须等于固定窗口长度")
    return int(value)


def _first_eligible_training_end(
    dates: pd.Series,
    requested_end: pd.Timestamp | None,
    config: RollingResearchConfig,
) -> pd.Timestamp | None:
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if dates.empty:
        return None
    minimum_end = pd.Timestamp(dates.iloc[0]) + pd.DateOffset(months=_minimum_training_months(config))
    if requested_end is None:
        return _on_or_after(dates, minimum_end)
    selected = _on_or_before(dates, requested_end)
    if selected is None or selected < minimum_end:
        raise ValueError(f"指定日期训练长度不足；至少需要训练至 {minimum_end.date()}，或选择自动最早启动")
    return selected


def _advance(value: pd.Timestamp, count: int, unit: IntervalUnit) -> pd.Timestamp:
    if unit == "days":
        return value + pd.Timedelta(days=count)
    if unit == "weeks":
        return value + pd.Timedelta(weeks=count)
    return value + pd.DateOffset(months=count)


def _next_market_date(dates: pd.Series, value: pd.Timestamp) -> pd.Timestamp | None:
    selected = dates.loc[dates > value]
    return pd.Timestamp(selected.iloc[0]) if not selected.empty else None


def _on_or_before(dates: pd.Series, value: pd.Timestamp) -> pd.Timestamp | None:
    selected = dates.loc[dates <= value]
    return pd.Timestamp(selected.iloc[-1]) if not selected.empty else None


def _on_or_after(dates: pd.Series, value: pd.Timestamp) -> pd.Timestamp | None:
    selected = dates.loc[dates >= value]
    return pd.Timestamp(selected.iloc[0]) if not selected.empty else None


def _rolling_period_count(
    dates: pd.Series,
    first_training_end: pd.Timestamp,
    last_date: pd.Timestamp,
    config: RollingResearchConfig,
) -> int:
    """Count scheduled training-window switches before costly searches begin."""
    train_end = pd.Timestamp(first_training_end)
    count = 0
    while train_end < last_date:
        next_cutoff = _advance(train_end, config.recalibration_interval, config.recalibration_unit)
        oos_start = _next_market_date(dates, train_end)
        oos_end = _on_or_before(dates, next_cutoff) or last_date
        if oos_start is None or oos_start > oos_end:
            break
        count += 1
        train_end = oos_end
    return count


def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise RollingResearchCancelled("已按请求终止滚动任务")


def _rebuild_stitched_paths(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True).copy()
    for column in ["strategy_capital_bp", "benchmark_capital_bp", "capital_excess_bp"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["strategy_capital_cum_bp"] = frame["strategy_capital_bp"].cumsum()
    frame["benchmark_capital_cum_bp"] = frame["benchmark_capital_bp"].cumsum()
    frame["capital_excess_cum_bp"] = frame["capital_excess_bp"].cumsum()
    for cumulative, contribution in {
        "strategy_capital_cum": "strategy_capital_return",
        "benchmark_capital_cum": "benchmark_capital_return",
        "capital_excess_cum": "capital_excess_return",
        "strategy_carry_cum": "strategy_carry_return",
        "benchmark_carry_cum": "benchmark_carry_return",
        "carry_excess_cum": "carry_excess_return",
    }.items():
        if contribution in frame:
            frame[cumulative] = pd.to_numeric(frame[contribution], errors="coerce").fillna(0.0).cumsum()
    if "strategy_return" in frame:
        frame["strategy_nav"] = (1.0 + pd.to_numeric(frame["strategy_return"], errors="coerce").fillna(0.0)).cumprod()
    if "total_return" in frame:
        frame["benchmark_nav_rebased"] = (1.0 + pd.to_numeric(frame["total_return"], errors="coerce").fillna(0.0)).cumprod()
    if {"strategy_nav", "benchmark_nav_rebased"}.issubset(frame.columns):
        frame["excess_nav"] = frame["strategy_nav"] / frame["benchmark_nav_rebased"]
    return frame


def _default_task_name(config: RollingResearchConfig) -> str:
    frequency = "日频" if config.base_config.signal_frequency == "daily" else "周频"
    window = "扩展窗口" if config.training_mode == "expanding" else f"近{config.rolling_window_months}个月"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"滚动定参·{config.base_config.name}·{frequency}·{config.search_mode}·{config.weight_search_version}·{window}·{stamp}"


def _rolling_objective_label(objective: ObjectiveConfig) -> str | None:
    """Name only the three registered objectives; custom objectives stay unnamed."""
    registered = {
        "收益": ObjectiveConfig(
            capital_gain_bp_weight=1.0, capital_gain_excess_bp_weight=0.25,
            capital_trade_win_rate_weight=10.0,
        ),
        "胜率": ObjectiveConfig(
            capital_gain_bp_weight=0.2, capital_gain_excess_bp_weight=0.25,
            capital_trade_win_rate_weight=100.0, capital_gain_drawdown_bp_penalty=1.0,
        ),
        "综合": ObjectiveConfig(
            capital_gain_bp_weight=0.6, capital_gain_excess_bp_weight=0.25,
            capital_trade_win_rate_weight=50.0, capital_gain_drawdown_bp_penalty=0.5,
        ),
    }
    return next((name for name, value in registered.items() if objective == value), None)


def _rolling_search_label(config: RollingResearchConfig) -> str:
    version = f"V{str(config.weight_search_version).lstrip('vV')}"
    objective = _rolling_objective_label(config.base_config.objective)
    suffix = f"（{objective}优先）" if objective else ""
    if config.search_mode == "combined":
        return f"{version} 权重搜索 + 阈值搜索{suffix}"
    if config.search_mode == "weight":
        return f"{version} 权重搜索{suffix}"
    return f"阈值搜索{suffix}"


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")


def _write_html_report(output_dir: Path, manifest: dict[str, object], periods: pd.DataFrame, stitched: pd.DataFrame) -> None:
    total = float(stitched["strategy_capital_cum_bp"].iloc[-1])
    excess = float(stitched["capital_excess_cum_bp"].iloc[-1])
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>滚动定参报告</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:40px;color:#17251f;background:#f6f8f5}}main{{max-width:1100px;margin:auto}}.cards{{display:flex;gap:16px}}.card{{background:#fff;padding:18px;border:1px solid #dce5dd;border-radius:8px;min-width:190px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:9px;border-bottom:1px solid #e4eae4;text-align:left}}th{{background:#eaf1eb}}code{{background:#eaf1eb;padding:2px 4px}}</style>
<main><h1>滚动定参结果</h1><p>每一期仅使用其训练窗口内数据选参；随后在下一段样本外执行，持仓与止盈止损状态跨期延续。新旧参数在当前同一训练窗口比较。</p>
<div class='cards'><div class='card'>样本外累计资本利得<br><b>{total:.2f} BP</b></div><div class='card'>样本外超额资本利得<br><b>{excess:.2f} BP</b></div><div class='card'>定参期数<br><b>{len(periods)}</b></div></div>
<h2>运行设置</h2><pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre><h2>逐期定参与样本外表现</h2>{periods.to_html(index=False, border=0)}</main></html>"""
    (output_dir / "滚动定参报告.html").write_text(html, encoding="utf-8")
