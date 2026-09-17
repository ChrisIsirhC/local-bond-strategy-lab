from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from common.config import DashboardStrategyConfig, strategy_config_from_dict


MISSING_PROVENANCE = "该策略尚未登记研究溯源。"
UNKNOWN_HISTORY = "更早历史来源未登记；当前基线是可复现起点。"


def config_parameters(config: DashboardStrategyConfig) -> dict[str, Any]:
    values = config.as_dict()
    values.pop("研究溯源", None)
    return values


def relative_path(root: Path, path: str | Path) -> str:
    target = Path(path)
    target = target if target.is_absolute() else root / target
    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(target.resolve())


def record_step(
    source: DashboardStrategyConfig,
    result: DashboardStrategyConfig,
    root: Path,
    operation: str,
    *,
    output_path: str | Path | None = None,
    training_start: str | None = None,
    training_end: str | None = None,
    search_version: str = "不适用",
    entrypoint: str = "",
    arguments: dict[str, Any] | None = None,
    changes: dict[str, Any] | None = None,
    note: str = "",
) -> DashboardStrategyConfig:
    """Append descriptive metadata only, after the algorithm has selected its result."""
    provenance = display_provenance(source, root)
    if not provenance:
        origin: dict[str, Any] = {"label": source.name}
        if source.source_config_path:
            origin["path"] = relative_path(root, source.source_config_path)
        else:
            origin["config"] = config_parameters(source)
        provenance = {"origin": origin, "earlier_history": "未登记", "steps": []}
    steps = provenance.setdefault("steps", [])
    output = {"label": result.name}
    if output_path:
        output["path"] = relative_path(root, output_path)
    step = {
        "operation": operation,
        "input": deepcopy(steps[-1]["output"] if steps else provenance["origin"]),
        "output": output,
        "training_start": training_start,
        "training_end": training_end,
        "signal_frequency": result.signal_frequency,
        "search_version": search_version,
        "entrypoint": entrypoint,
        "arguments": deepcopy(arguments or {}),
        "changes": deepcopy(changes or {}),
        "note": note,
    }
    steps.append(step)
    # Historical display aliases belong to the recorded result, not its descendants.
    provenance.pop("applies_to", None)
    return replace(result, research_provenance=provenance,
                   source_config_path=str(output_path) if output_path else None)


def record_manual_changes(
    source: DashboardStrategyConfig, result: DashboardStrategyConfig, root: Path
) -> DashboardStrategyConfig:
    before, after = config_parameters(source), config_parameters(result)
    changes = {key: {"before": before.get(key), "after": value}
               for key, value in after.items() if before.get(key) != value and key != "name"}
    if not changes:
        return replace(result, research_provenance=display_provenance(source, root),
                       source_config_path=source.source_config_path)
    labels = {"weights": "权重调整", "thresholds": "阈值调整", "positions": "仓位/执行规则调整",
              "factor_windows": "因子窗口调整", "objective": "搜索目标调整",
              "backtest": "回测区间调整", "signal_frequency": "信号频率调整", "benchmark": "基准调整"}
    if "positions" in changes:
        changed = [key for key, value in after["positions"].items() if before["positions"].get(key) != value]
        if set(changed) <= {"take_profit_bp", "stop_loss_bp"}:
            labels["positions"] = "止盈止损调整"
    return record_step(source, result, root, "、".join(labels.get(key, key) for key in changes),
                       entrypoint="dataclasses.replace / common.config.strategy_config_from_dict",
                       changes=changes, note="仅记录实际字段变化；按 changes 的 after 值更新输入配置。")


def _registered_objective_name(config: DashboardStrategyConfig) -> str | None:
    """Recognise only the three public presets; never guess a custom objective."""
    objective = config.objective
    registered = {
        "收益": {"capital_gain_bp_weight": 1.0, "capital_gain_excess_bp_weight": 0.25, "capital_trade_win_rate_weight": 10.0, "capital_gain_drawdown_bp_penalty": 0.0},
        "胜率": {"capital_gain_bp_weight": 0.2, "capital_gain_excess_bp_weight": 0.25, "capital_trade_win_rate_weight": 100.0, "capital_gain_drawdown_bp_penalty": 1.0},
        "综合": {"capital_gain_bp_weight": 0.6, "capital_gain_excess_bp_weight": 0.25, "capital_trade_win_rate_weight": 50.0, "capital_gain_drawdown_bp_penalty": 0.5},
    }
    for name, expected in registered.items():
        if objective == type(objective)(**expected):
            return name
    return None


def _rolling_search_label(manifest: dict[str, Any], base: DashboardStrategyConfig) -> str:
    mode = str(manifest.get("搜索模式", "")).strip()
    version = f"V{str(manifest.get('权重搜索版本', 'v2')).lstrip('vV')}"
    objective = _registered_objective_name(base)
    suffix = f"（{objective}优先）" if objective else ""
    if mode == "combined":
        return f"{version} 权重搜索 + 阈值搜索{suffix}"
    if mode == "weight":
        return f"{version} 权重搜索{suffix}"
    if mode == "threshold":
        return f"阈值搜索{suffix}"
    return "未登记"


def _enrich_legacy_rolling_provenance(
    provenance: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Repair only the *displayed* lineage of old rolling archives.

    Early rolling archives stored the selected historical experiment as a new
    origin.  The exact selected baseline was also saved beside the rolling
    periods, which lets us restore the parent lineage without inventing any
    research history or modifying a frozen archive.
    """
    steps = provenance.get("steps")
    if not isinstance(steps, list):
        return provenance
    rolling_index = next(
        (index for index, step in enumerate(steps)
         if isinstance(step, dict) and step.get("operation") == "滚动定参"),
        None,
    )
    if rolling_index is None:
        return provenance
    rolling_step = deepcopy(steps[rolling_index])
    output = rolling_step.get("output", {})
    output_path = output.get("path") if isinstance(output, dict) else None
    if not output_path:
        return provenance
    rolling_dir = root / str(output_path)
    base_path = rolling_dir / "基线配置.json"
    manifest_path = rolling_dir / "滚动配置.json"
    try:
        base_raw = json.loads(base_path.read_text(encoding="utf-8"))
        base = strategy_config_from_dict(base_raw)
        parent = base.research_provenance
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return provenance
    if not isinstance(parent, dict) or not parent.get("origin"):
        return provenance
    parent_steps = parent.get("steps", [])
    if not isinstance(parent_steps, list):
        return provenance
    # Newer archives already begin with the full inherited chain.
    if len(steps) >= len(parent_steps) and steps[:len(parent_steps)] == parent_steps:
        return provenance

    if isinstance(manifest, dict):
        training_mode = str(manifest.get("训练方式", ""))
        rolling_months = manifest.get("固定训练窗口月数")
        window = (
            "扩展窗口" if training_mode == "expanding"
            else f"固定滚动窗口（近{rolling_months}个月）" if training_mode == "rolling_months" and rolling_months
            else "未登记"
        )
        interval = str(manifest.get("定参频率", "")).replace(" months", "个月").replace(" weeks", "周").replace(" days", "日")
        if interval:
            interval = f"每{interval}" if not interval.startswith("每") else interval
        rolling_step["search_version"] = _rolling_search_label(manifest, base)
        rolling_step["arguments"] = {
            "训练窗口": window,
            "最低训练长度": f"{manifest.get('最低训练长度月数')}个月" if manifest.get("最低训练长度月数") else "未登记",
            "定参频率": interval or "未登记",
            "参数切换门槛": manifest.get("最小目标函数改善阈值", "未登记"),
        }
        rolling_step["note"] = (
            f"{window}；最低训练长度{manifest.get('最低训练长度月数')}个月；{interval}定参；"
            f"参数切换门槛 {manifest.get('最小目标函数改善阈值', '未登记')}。"
            f"复用 {manifest.get('复用历史期数', 0)} 期已保存参数，本次新增搜索 "
            f"{manifest.get('本次新增搜索期数', '未登记')} 期；仅在跨过新的定参点时重新搜索。"
        )

    rebuilt = deepcopy(parent)
    rebuilt["steps"] = [*deepcopy(parent_steps), rolling_step, *deepcopy(steps[rolling_index + 1:])]
    parent_notes = list(rebuilt.get("notes", [])) if isinstance(rebuilt.get("notes"), list) else []
    current_notes = list(provenance.get("notes", [])) if isinstance(provenance.get("notes"), list) else []
    rebuilt["notes"] = list(dict.fromkeys([*parent_notes, *current_notes]))
    return rebuilt


def _enrich_legacy_factor_provenance(
    provenance: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Display the selected baseline's full lineage for earlier factor archives.

    Factor-study archives originally kept only a text label for their selected
    baseline even though the exact baseline configuration is retained in the
    study manifest.  Rebuild the display lineage from that immutable manifest;
    this is read-only and never rewrites the historical archive.
    """
    research_type = str(provenance.get("研究类型", ""))
    study_name = str(provenance.get("因子增加研究批次", "")).strip()
    if not research_type.startswith("因子"):
        return provenance
    parent: dict[str, Any] | None = None
    if study_name:
        manifest_path = root / "backtest_outputs" / "因子增加研究" / study_name / "研究清单.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base_raw = manifest.get("研究基线")
            if isinstance(base_raw, dict):
                parent = display_provenance(strategy_config_from_dict(base_raw), root)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            parent = None

    # The earliest direct-to-queue factor rolling tasks only retained the
    # parent archive's label.  That label is enough to recover the immutable
    # parent archive, provided it is unique; never make up a parent when it is
    # ambiguous or absent.
    if parent is None:
        origin = provenance.get("origin")
        origin_label = str(origin.get("label", "")).strip() if isinstance(origin, dict) else ""
        if origin_label:
            matches: list[DashboardStrategyConfig] = []
            for config_path in sorted((root / "backtest_outputs" / "experiments").glob("*/config.json")):
                try:
                    candidate = strategy_config_from_dict(json.loads(config_path.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if candidate.name == origin_label:
                    matches.append(candidate)
            if len(matches) == 1:
                parent = display_provenance(matches[0], root)
    if not isinstance(parent, dict) or not parent.get("origin"):
        return provenance
    parent_steps = parent.get("steps", [])
    current_steps = provenance.get("steps", [])
    if not isinstance(parent_steps, list) or not isinstance(current_steps, list):
        return provenance
    if len(current_steps) >= len(parent_steps) and current_steps[:len(parent_steps)] == parent_steps:
        return provenance
    rebuilt = deepcopy(parent)
    rebuilt["steps"] = [*deepcopy(parent_steps), *deepcopy(current_steps)]
    parent_notes = list(rebuilt.get("notes", [])) if isinstance(rebuilt.get("notes"), list) else []
    current_notes = list(provenance.get("notes", [])) if isinstance(provenance.get("notes"), list) else []
    rebuilt["notes"] = list(dict.fromkeys([*parent_notes, *current_notes]))
    for key, value in provenance.items():
        if key not in {"origin", "earlier_history", "steps", "notes"}:
            rebuilt[key] = deepcopy(value)
    return rebuilt


def display_provenance(
    config: DashboardStrategyConfig, root: Path, experiment_dir: Path | None = None
) -> dict[str, Any] | None:
    if config.research_provenance:
        provenance = _enrich_legacy_rolling_provenance(deepcopy(config.research_provenance), root)
        return _enrich_legacy_factor_provenance(provenance, root)
    # Read-only enrichment: require a registered identity AND identical parameters.
    for path in sorted((root / "configs" / "baselines").glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not raw.get("研究溯源"):
                continue
            candidate = strategy_config_from_dict(raw)
            provenance = candidate.research_provenance
            aliases = provenance.get("applies_to", [])
            archive = relative_path(root, experiment_dir / "config.json") if experiment_dir else None
            source_path = relative_path(root, config.source_config_path) if config.source_config_path else None
            if config.name != candidate.name and archive not in aliases and source_path not in aliases:
                continue
            left, right = config_parameters(config), config_parameters(candidate)
            left.pop("name")
            right.pop("name")
            if left == right:
                return deepcopy(provenance)
        except (OSError, ValueError, TypeError):
            continue
    return None
