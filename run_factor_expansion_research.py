from __future__ import annotations

import json
from html import escape
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from common.config import DashboardStrategyConfig
from common.factor_expansion_features import (
    ALL_FACTOR_COLUMNS,
    BASE_FACTOR_COLUMNS,
    EXPANDED_FACTOR_COLUMNS,
    FACTOR_DISPLAY_COLUMNS,
    FACTOR_GROUPS,
    FACTOR_LABELS,
    factor_expansion_data_quality,
    build_factor_expansion_multipliers,
    supply_seasonality_feasibility,
)
from common.factor_expansion_research import (
    OBJECTIVES,
    ExpansionResearchConfig,
    evaluate_expansion_config,
    expanded_original_candidate,
    root_research_config,
    run_expansion_threshold_search,
    run_expansion_weight_search,
    training_candidate_rank,
)
from strategies.dashboard_signal_v1 import DashboardThresholds
from strategies.position_policy import DashboardPositionPolicy


ROOT = Path(__file__).resolve().parent
TRAINING_END = "2025-01-01"
BEAM_WIDTH = 160


def run_factor_expansion_static_research(
    root: Path = ROOT,
    training_end: str = TRAINING_END,
    beam_width: int = BEAM_WIDTH,
    output: Path | None = None,
    objectives: tuple[str, ...] | None = None,
    frequencies: tuple[str, ...] | None = None,
    research_scope: str = "全量六路线比较",
    base_config: DashboardStrategyConfig | None = None,
    enabled_factor_columns: tuple[str, ...] | None = None,
) -> Path:
    """Run a static factor-universe study for one route or the full six-route set."""
    root = Path(root).resolve()
    output = output or root / "backtest_outputs" / "因子增加研究" / f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    selected_objectives = tuple(objectives or tuple(OBJECTIVES))
    selected_frequencies = tuple(frequencies or ("daily", "weekly"))
    invalid_objectives = set(selected_objectives).difference(OBJECTIVES)
    invalid_frequencies = set(selected_frequencies).difference({"daily", "weekly"})
    if invalid_objectives or invalid_frequencies:
        raise ValueError("因子增加研究包含不支持的目标或信号频率")
    enabled_factors = tuple(enabled_factor_columns or ALL_FACTOR_COLUMNS)
    unknown_factors = set(enabled_factors).difference(ALL_FACTOR_COLUMNS)
    if unknown_factors:
        raise ValueError(f"存在无法识别的事前启用因子: {sorted(unknown_factors)}")
    if not enabled_factors:
        raise ValueError("请至少启用一个因子")
    # A valid original-vs-expanded comparison requires all eight original
    # factors plus at least one added factor.  Any other preselected universe
    # is still a valid strategy universe, but must be treated as a single
    # self-contained route rather than a misleading partial "original" control.
    comparison_mode = set(BASE_FACTOR_COLUMNS).issubset(enabled_factors) and bool(
        set(enabled_factors).intersection(EXPANDED_FACTOR_COLUMNS)
    )
    factor_versions = ("原始因子", "扩展因子") if comparison_mode else ("自选因子",)
    enabled_groups = {
        group for group, members in FACTOR_GROUPS.items()
        if any(column in enabled_factors for column in members)
    }
    module_cap_text = "单模块上限100（当前仅一个大类）" if len(enabled_groups) <= 1 else "单模块上限50"
    quality_rows: list[pd.DataFrame] = []
    seasonality: dict[str, object] = {}
    for frequency in selected_frequencies:
        multipliers, inputs = build_factor_expansion_multipliers(root, signal_frequency=frequency)
        multipliers.to_csv(output / f"{frequency}_因子乘数.csv", index=False, encoding="utf-8-sig")
        inputs.to_csv(output / f"{frequency}_新增因子底层输入.csv", index=False, encoding="utf-8-sig")
        quality_rows.append(factor_expansion_data_quality(inputs, frequency))
        seasonality[frequency] = supply_seasonality_feasibility(inputs)
    pd.concat(quality_rows, ignore_index=True).to_csv(output / "新增因子实现检查.csv", index=False, encoding="utf-8-sig")
    (output / "季节性供给可行性.json").write_text(json.dumps(seasonality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: list[dict[str, object]] = []
    original_winners: dict[tuple[str, str], ExpansionResearchConfig] = {}
    for factor_version in factor_versions:
        for objective_name in selected_objectives:
            for frequency in selected_frequencies:
                route_key = (objective_name, frequency)
                original_winner = original_winners.get(route_key)
                initial_weights = base_config.weights.as_dict() if base_config else None
                if factor_version == "扩展因子" and original_winner is not None:
                    # The exact original-factor winner, with all added factors
                    # set to zero, must enter the expanded search as a baseline.
                    initial_weights = original_winner.weights
                base = root_research_config(
                    factor_version,
                    objective_name,
                    frequency,
                    thresholds=base_config.thresholds if base_config else None,
                    positions=base_config.positions if base_config else None,
                    factor_windows=base_config.factor_windows if base_config else None,
                    initial_weights=initial_weights,
                    enabled_factor_columns=enabled_factors,
                )
                weighted, trace = run_expansion_weight_search(root, base, None, training_end, beam_width=beam_width)
                selected, thresholds = run_expansion_threshold_search(root, weighted, None, training_end)
                stem = f"{factor_version}_{objective_name}_{frequency}"
                trace.to_csv(output / f"{stem}_V2权重搜索轨迹.csv", index=False, encoding="utf-8-sig")
                thresholds.to_csv(output / f"{stem}_阈值搜索.csv", index=False, encoding="utf-8-sig")
                train_daily, train_signals, train_metrics, train_benchmark = evaluate_expansion_config(root, selected, None, training_end)
                used_original_fallback = False
                if factor_version == "扩展因子" and original_winner is not None:
                    fallback = expanded_original_candidate(original_winner)
                    fallback_daily, fallback_signals, fallback_metrics, fallback_benchmark = evaluate_expansion_config(
                        root, fallback, None, training_end
                    )
                    if training_candidate_rank(fallback, fallback_metrics, fallback_benchmark) >= training_candidate_rank(
                        selected, train_metrics, train_benchmark
                    ):
                        selected = fallback
                        train_daily, train_signals = fallback_daily, fallback_signals
                        train_metrics, train_benchmark = fallback_metrics, fallback_benchmark
                        used_original_fallback = True
                if factor_version == "原始因子":
                    original_winners[route_key] = selected
                oos_start = (pd.Timestamp(training_end) + pd.Timedelta(days=1)).date().isoformat()
                oos_daily, oos_signals, oos_metrics, oos_benchmark = evaluate_expansion_config(root, selected, oos_start, None)
                train_daily.to_csv(output / f"{stem}_训练期日度.csv", index=False, encoding="utf-8-sig")
                oos_daily.to_csv(output / f"{stem}_样本外日度.csv", index=False, encoding="utf-8-sig")
                train_signals.to_csv(output / f"{stem}_训练期信号.csv", index=False, encoding="utf-8-sig")
                oos_signals.to_csv(output / f"{stem}_样本外信号.csv", index=False, encoding="utf-8-sig")
                row = {
                    "因子版本": factor_version,
                    "目标": objective_name,
                    "频率": "日频" if frequency == "daily" else "周频",
                    "训练截止日": training_end,
                    "权重搜索候选数": int(len(trace)),
                    "权重": json.dumps(selected.weights, ensure_ascii=False),
                    "看多阈值": selected.positions.bullish_threshold,
                    "看空阈值": selected.positions.bearish_threshold,
                    "训练目标函数": training_candidate_rank(selected, train_metrics, train_benchmark)[0],
                    "新增因子入选数": sum(
                        float(selected.weights.get(column, 0.0)) > 0.0
                        for column in selected.factor_columns
                        if column not in original_winner.factor_columns
                    ) if factor_version == "扩展因子" and original_winner is not None else 0,
                    "有效因子数": sum(float(weight) > 0.0 for weight in selected.weights.values()),
                    "是否采用原始因子保底": used_original_fallback,
                    "事前停用因子": "、".join(FACTOR_LABELS[column] for column in ALL_FACTOR_COLUMNS if column not in enabled_factors) or "无",
                }
                for prefix, metrics, benchmark in (("训练", train_metrics, train_benchmark), ("样本外", oos_metrics, oos_benchmark)):
                    row[f"{prefix}累计资本利得_BP"] = metrics.get("capital_gain_total_bp")
                    row[f"{prefix}资本利得超额_BP"] = float(metrics.get("capital_gain_total_bp", 0.0)) - float(benchmark.get("capital_gain_total_bp", 0.0))
                    row[f"{prefix}交易胜率"] = metrics.get("capital_gain_trade_win_rate")
                    row[f"{prefix}已平仓交易数"] = metrics.get("capital_gain_closed_trade_count")
                    row[f"{prefix}最大回撤_BP"] = metrics.get("capital_gain_max_drawdown_bp")
                summary.append(row)
                (output / f"{stem}_最终配置.json").write_text(json.dumps({
                    "因子版本": factor_version, "目标": objective_name, "频率": frequency,
                    "训练截止日": training_end, "权重": selected.weights,
                    "阈值": selected.thresholds.as_dict(), "仓位": selected.positions.as_dict(),
                    "目标函数": selected.objective.as_dict(),
                    "启用因子": list(selected.factor_columns),
                    "事前停用因子": [column for column in ALL_FACTOR_COLUMNS if column not in enabled_factors],
                    "搜索说明": (
                        f"仅在事前启用因子中执行V2约束（总权重100、步长5、{module_cap_text}、可零权重）下的完整配置多起点搜索与五点权重转移；"
                        + (
                            "扩展路线强制纳入同路线原始因子最终策略（新增因子全为0）作为保底候选；"
                            if comparison_mode and factor_version == "扩展因子" else ""
                        )
                        + "完整候选同分时优先较少有效因子；随后搜索看多/看空阈值。"
                    ),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame(summary)
    frame.to_csv(output / "静态研究汇总.csv", index=False, encoding="utf-8-sig")
    write_factor_expansion_manifest(
        output,
        training_end=training_end,
        static_beam_width=beam_width,
        objectives=selected_objectives,
        frequencies=selected_frequencies,
        research_scope=research_scope,
        base_config=base_config,
        enabled_factor_columns=enabled_factors,
        study_mode="对照研究" if comparison_mode else "自选因子集合研究",
        factor_versions=factor_versions,
        root=root,
    )
    _write_report(output, frame, seasonality, training_end=training_end)
    print(output)
    return output


def write_factor_expansion_manifest(
    output: Path,
    training_end: str,
    static_beam_width: int,
    rolling: dict[str, object] | None = None,
    objectives: tuple[str, ...] | None = None,
    frequencies: tuple[str, ...] | None = None,
    research_scope: str = "全量六路线比较",
    base_config: DashboardStrategyConfig | None = None,
    root: Path = ROOT,
    enabled_factor_columns: tuple[str, ...] | None = None,
    study_mode: str | None = None,
    factor_versions: tuple[str, ...] | None = None,
) -> None:
    """Persist the small replay contract consumed by the factor-research page."""
    path = output / "研究清单.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    previous_static = current.get("静态搜索", {}) if isinstance(current.get("静态搜索"), dict) else {}
    selected_objectives = objectives or tuple(previous_static.get("目标", ())) or tuple(OBJECTIVES)
    selected_frequencies = frequencies or tuple(previous_static.get("频率", ())) or ("daily", "weekly")
    selected_scope = research_scope if research_scope != "全量六路线比较" or "研究范围" not in current else str(current["研究范围"])
    enabled_factors = tuple(enabled_factor_columns or previous_static.get("事前启用因子", ()) or ALL_FACTOR_COLUMNS)
    selected_versions = tuple(factor_versions or previous_static.get("因子版本", ()) or ("原始因子", "扩展因子"))
    selected_mode = study_mode or str(current.get("研究模式", "对照研究"))
    current.update({
        "研究类型": "因子增加研究",
        "研究范围": selected_scope,
        "研究引擎": "factor_expansion_full_configuration_v1",
        "研究模式": selected_mode,
        "训练截止日": str(training_end),
        "研究基线": base_config.as_dict() if base_config is not None else current.get("研究基线"),
        "静态搜索": {
            "因子版本": list(selected_versions),
            "目标": list(selected_objectives),
            "频率": list(selected_frequencies),
            "权重搜索": "完整配置多起点搜索与五点权重转移",
            "总权重": 100,
            "步长": 5,
            "单模块上限": 100 if len({group for group, members in FACTOR_GROUPS.items() if any(column in enabled_factors for column in members)}) <= 1 else 50,
            "允许零权重": True,
            "同分规则": "优先较少有效因子",
            "搜索宽度": int(static_beam_width),
            "事前启用因子": list(enabled_factors),
            "事前停用因子": [column for column in ALL_FACTOR_COLUMNS if column not in enabled_factors],
        },
    })
    # A study directory groups several strategy runs; F identifiers belong to
    # individually archived factor strategies, not to this batch container.
    current.pop("研究编号", None)
    if rolling is not None:
        current["滚动定参"] = rolling
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize_factor_expansion_static_details(root: Path, study_dir: Path) -> None:
    """Backfill display-only daily files from already selected static configs."""
    root = Path(root).resolve()
    study_dir = Path(study_dir).resolve()
    for config_path in study_dir.glob("*_最终配置.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        stem = config_path.name.removesuffix("_最终配置.json")
        if (study_dir / f"{stem}_训练期日度.csv").exists() and (study_dir / f"{stem}_样本外日度.csv").exists():
            continue
        config = ExpansionResearchConfig(
            factor_version=str(payload["因子版本"]),
            objective_name=str(payload["目标"]),
            signal_frequency=str(payload["频率"]),
            weights={str(key): float(value) for key, value in payload["权重"].items()},
            thresholds=DashboardThresholds(**payload.get("阈值", {})),
            positions=DashboardPositionPolicy(**payload.get("仓位", {})),
            enabled_factor_columns=tuple(payload.get("启用因子", ())) or None,
        )
        training_end = str(payload["训练截止日"])
        train_daily, _, _, _ = evaluate_expansion_config(root, config, None, training_end)
        oos_start = (pd.Timestamp(training_end) + pd.Timedelta(days=1)).date().isoformat()
        oos_daily, _, _, _ = evaluate_expansion_config(root, config, oos_start, None)
        train_daily.to_csv(study_dir / f"{stem}_训练期日度.csv", index=False, encoding="utf-8-sig")
        oos_daily.to_csv(study_dir / f"{stem}_样本外日度.csv", index=False, encoding="utf-8-sig")


def main() -> Path:
    return run_factor_expansion_static_research()


def _write_report(
    output: Path,
    frame: pd.DataFrame,
    seasonality: dict[str, object],
    training_end: str = TRAINING_END,
    root: Path = ROOT,
) -> None:
    """Write a human-readable report with links to frozen F-strategy archives.

    The research directory is a batch workspace.  Each static route is useful
    only when it is also represented by one immutable experiment archive.  Do
    not infer an identifier from the batch name: look it up from the archived
    route provenance instead.
    """
    root = Path(root).resolve()

    def archive_routes() -> dict[tuple[str, str, str], tuple[str, str]]:
        routes: dict[tuple[str, str, str], tuple[str, str]] = {}
        experiment_root = root / "backtest_outputs" / "experiments"
        if not experiment_root.exists():
            return routes
        for experiment_dir in experiment_root.iterdir():
            config_path = experiment_dir / "config.json"
            manifest_path = experiment_dir / "run_manifest.json"
            if not experiment_dir.is_dir() or not config_path.exists() or not manifest_path.exists():
                continue
            try:
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                provenance = config_payload.get("研究溯源", {})
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if not isinstance(provenance, dict) or provenance.get("研究类型") != "因子增加单一策略归档":
                continue
            if str(provenance.get("因子增加研究批次", "")) != output.name:
                continue
            route = provenance.get("因子增加研究路线", {})
            if not isinstance(route, dict):
                continue
            key = (str(route.get("目标", "")), str(route.get("频率", "")), str(route.get("因子版本", "")))
            if all(key):
                routes[key] = (str(manifest_payload.get("short_id", "未编号")), experiment_dir.name)
        return routes

    manifest_path = output / "研究清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    baseline = manifest.get("研究基线") if isinstance(manifest.get("研究基线"), dict) else None
    baseline_text = str(baseline.get("name", "统一1 / 0 / 0研究起点")) if baseline else "统一1 / 0 / 0研究起点"
    study_mode = str(manifest.get("研究模式", "对照研究"))
    enabled = set(manifest.get("静态搜索", {}).get("事前启用因子", ALL_FACTOR_COLUMNS))
    if study_mode == "自选因子集合研究":
        title = "自选因子集合研究"
        intro = f"研究对象为当前事前启用的{len(enabled)}个因子；未生成原始8因子对照。策略基线为{baseline_text}。"
        factor_status = lambda key: "本轮启用" if key in enabled else "事前停用"
    else:
        title = "因子增加研究"
        intro = f"研究对象是原始因子集合与扩展因子集合；策略基线为{baseline_text}。"
        factor_status = lambda key: "本轮新增" if key in EXPANDED_FACTOR_COLUMNS else "保留原始因子"
    seasonality_frequency = "weekly" if "weekly" in seasonality else next(iter(seasonality), None)
    if seasonality_frequency:
        observation_count = float(seasonality[seasonality_frequency]["同一周序号的历史观测数_中位数"])
        seasonality_text = f"未纳入首批。现有历史观测中位数为{observation_count:.0f}，不能构造稳定的季节性基准。"
    else:
        seasonality_text = "本次研究没有生成季节性供给检查。"
    route_archives = archive_routes()
    display = frame.copy()
    strategy_cells: list[str] = []
    for _, row in display.iterrows():
        frequency = str(row.get("频率", ""))
        normalized_frequency = "daily" if frequency == "日频" else "weekly" if frequency == "周频" else frequency
        factor_version = str(row.get("因子版本", ""))
        objective = str(row.get("目标", ""))
        archive = route_archives.get((objective, normalized_frequency, factor_version))
        route_label = f"{objective}优先 · {frequency} · {factor_version}"
        if archive:
            short_id, experiment_name = archive
            strategy_cells.append(
                f'<a class="strategy-link" href="http://localhost:8501/history?experiment={quote(experiment_name)}" '
                f'target="_blank" rel="noopener">[{escape(short_id)}] {escape(route_label)}</a>'
            )
        else:
            strategy_cells.append(f'<span class="unarchived">[未归档] {escape(route_label)}</span>')
    display.insert(0, "策略（历史结果）", strategy_cells)
    for column in display.columns:
        if "胜率" in column:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
        elif "BP" in column:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    rolling_path = output / "滚动定参" / "滚动定参汇总.csv"
    rolling_section = (
        "<p>滚动定参仍在运行；完成后将在此处追加逐期样本外汇总。</p>"
    )
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path, encoding="utf-8-sig")
        for column in rolling.columns:
            if "胜率" in column:
                rolling[column] = pd.to_numeric(rolling[column], errors="coerce").map(
                    lambda value: "" if pd.isna(value) else f"{value:.1%}"
                )
            elif "BP" in column:
                rolling[column] = pd.to_numeric(rolling[column], errors="coerce").map(
                    lambda value: "" if pd.isna(value) else f"{value:.2f}"
                )
        rolling_section = (
            "<p>每组从首个共同可用信号日累计训练24个月后启动；之后每3个月用截至当期的数据重新搜索权重和阈值，"
            "只执行随后一段样本外结果。下表是逐段样本外拼接后的总计，不参与静态选优。</p>"
            + rolling.to_html(index=False, escape=True)
        )
    factor_rows = pd.DataFrame([
        {
            "因子": FACTOR_LABELS[key],
            "状态": factor_status(key),
        }
        for key in FACTOR_DISPLAY_COLUMNS
    ])
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>因子增加研究</title>
<style>body{{margin:0;background:#f3f2ed;color:#18201d;font-family:Arial,'Microsoft YaHei',sans-serif}}main{{max-width:1180px;margin:0 auto;padding:54px 28px 80px}}h1{{font-size:34px;margin:0 0 8px}}h2{{font-size:20px;margin:42px 0 12px}}p,.note{{color:#66716c;line-height:1.7;max-width:900px}}.meta{{display:flex;gap:28px;margin:28px 0;padding:15px 0;border-top:1px solid #d9ddd8;border-bottom:1px solid #d9ddd8;font-size:14px}}table{{width:100%;border-collapse:collapse;background:#fbfaf6;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d9ddd8;vertical-align:top}}th{{color:#66716c;font-weight:600;white-space:nowrap}}code{{font-size:11px;word-break:break-all}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:600}}.strategy-link{{color:#176b5b;font-weight:700;text-decoration:none}}.strategy-link:hover{{text-decoration:underline}}.unarchived{{color:#8a8377}}</style></head><body><main>
<h1>{title}</h1><p>{intro}训练期截至{training_end}，之后数据只用于样本外展示。</p>
<div class='meta'><span>权重约束：总和100 / 步长5 / 单模块上限50 / 可归零</span><span>权重搜索：完整配置多起点 + 五点权重转移</span><span>同分时优先较少有效因子</span><span>阈值搜索：权重固定后进行</span></div>
<h2>因子集合</h2>{factor_rows.to_html(index=False, escape=True)}
<h2>静态搜索结果</h2><p>每条策略均以独立 F 编号标识；点击编号可在新标签页打开该策略的完整历史结果。</p>{display.drop(columns=['权重']).to_html(index=False, escape=False)}
<h2>季节性供给</h2><p>{seasonality_text}</p>
<h2>滚动定参</h2>{rolling_section}
<details><summary>实现与可用日期检查</summary><p>详见 <code>新增因子实现检查.csv</code>。新表达仅在底层观测日期推进时计算变化，日频展示中重复的低频数据不会被误当作新变化。</p></details>
</main></body></html>"""
    (output / "因子增加研究报告.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
