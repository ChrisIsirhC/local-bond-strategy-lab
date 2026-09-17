from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.archive_ids import archive_id_category, existing_short_archive_id, short_archive_id
from common.config import DashboardStrategyConfig, load_strategy_config, save_strategy_config
from common.provenance import record_step
from common.market_data import CONDITIONAL_BENCHMARK_ID, CONDITIONAL_BENCHMARK_NAME, GOV_10Y, TRADED_ASSET_ID, benchmark_label, curve_path
from common.period_evaluation import evaluate_period
from common.reporting import write_strategy_outputs
from common.trade_metrics import capital_gain_trade_metrics
from common.performance import performance_metrics
from strategies.dashboard_signal_v1 import signal_file_for_frequency


EXPERIMENT_DIR = Path("backtest_outputs") / "experiments"
EXPERIMENT_INDEX_FILE = Path("backtest_outputs") / "experiments_index.json"
_EXPERIMENTS_CACHE: dict[str, tuple[tuple[tuple[str, int, int], ...], pd.DataFrame]] = {}
_EXPERIMENT_INDEX_VERSION = 6


def archive_id_prefix(source: str, config: DashboardStrategyConfig | None = None) -> str:
    value = str(source or "")
    provenance = config.research_provenance if config and isinstance(config.research_provenance, dict) else {}
    # A strategy that originates in factor research retains the F namespace
    # even if a later operation is ordinary rolling parameter research.  The
    # identifier describes the research lineage, rather than only its last
    # execution method.
    factor_origin = bool(
        provenance.get("因子研究权重")
        or provenance.get("因子增加研究路线")
        or str(provenance.get("研究类型", "")).startswith("因子")
    )
    if "因子" in value or factor_origin:
        return "F"
    # Match the source label, not incidental words in a comparison study
    # name such as ``V2静态与滚动对照_静态候选``.
    if "滚动定参" in value or value.strip().startswith("滚动"):
        return "R"
    if any(token in value for token in ("搜索", "权重", "阈值", "静态候选", "联合")):
        return "S"
    return "B"


def _manifest_id_prefix(manifest: dict[str, object]) -> str:
    """Use the frozen prefix when available; never infer a new one from a renamed title."""
    saved = str(manifest.get("short_id_prefix", "")).upper().strip()
    return saved if len(saved) == 1 and saved.isalpha() else archive_id_prefix(str(manifest.get("运行来源", "")))


def _canonical_archive_path(root: Path, value: object) -> str:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return os.path.normcase(str(candidate.resolve()))


def _normalise_index_rows(root: Path, rows: list[object]) -> list[dict[str, object]]:
    """Collapse aliases of the same archive and reject actual ID collisions."""
    by_path: dict[str, dict[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        archive_value = raw.get("实验目录", "")
        if not str(archive_value).strip():
            continue
        normalized = dict(raw)
        normalized["实验目录"] = _canonical_archive_path(root, archive_value)
        # Later rows are fresher (for example after an archive repair), so
        # deliberately retain the last copy of an aliased index entry.
        by_path[normalized["实验目录"]] = normalized
    owners: dict[str, str] = {}
    for row in by_path.values():
        identifier = str(row.get("运行ID", "")).strip()
        if not identifier:
            continue
        archive_path = str(row["实验目录"])
        previous = owners.setdefault(identifier, archive_path)
        if previous != archive_path:
            raise ValueError(f"运行 ID 冲突：{identifier} 同时指向 {previous} 与 {archive_path}")
    return list(by_path.values())


def archive_dashboard_experiment(
    root: Path,
    config: DashboardStrategyConfig,
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    strategy_metrics: dict[str, object],
    benchmark_metrics: dict[str, object],
    source: str,
    research_metadata: dict[str, object] | None = None,
) -> Path:
    run_time = datetime.now()
    research_metadata = _enrich_research_metadata(config, research_metadata)
    run_id = f"{run_time:%Y%m%d_%H%M%S_%f}__{_safe_name(config.name)}"
    output_dir = _next_available_dir(root / EXPERIMENT_DIR / run_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix = archive_id_prefix(source, config)
    display_id = short_archive_id(root, archive_id_category(prefix), output_dir.name, prefix)

    write_strategy_outputs(daily, signals, strategy_metrics, benchmark_metrics, output_dir)
    config = record_step(
        config, config, root, "最终回测", output_path=output_dir / "config.json",
        entrypoint="common.runner.run_dashboard_config",
        arguments={"output_dir": str(output_dir.relative_to(root))},
        note=f"{source}；实际回测 {strategy_metrics.get('start_date')} 至 {strategy_metrics.get('end_date')}；不重新选参。",
    )
    save_strategy_config(config, output_dir / "config.json")

    asset_path = curve_path(root, TRADED_ASSET_ID)
    benchmark_path = curve_path(root, GOV_10Y)
    signal_path = root / signal_file_for_frequency(config.signal_frequency)
    # A search-derived strategy keeps its documented selection cutoff when it
    # is rerun.  Calculate the corresponding OOS slice now and store it in
    # the archive manifest as well as the lightweight history index.  This
    # makes the archive self-describing even if the index is rebuilt later.
    training_end = research_metadata.get("训练截止日")
    out_of_sample_metrics = _archived_out_of_sample_metrics(output_dir, training_end)
    manifest = {
        "run_id": output_dir.name,
        "short_id": display_id,
        "short_id_prefix": prefix,
        "策略名称": config.name,
        "策略版本": "dashboard_signal_v1",
        "信号频率": "日频" if config.signal_frequency == "daily" else "周频",
        "交易标的": benchmark_label(TRADED_ASSET_ID),
        "比较基准": str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME)),
        "基准ID": str(benchmark_metrics.get("benchmark_id", CONDITIONAL_BENCHMARK_ID)),
        "条件基准规则": "仓位>0时同仓位买入10Y国债；仓位<=0时持有现金",
        "资本利得BP口径": "收益率方向变动BP（不乘久期）",
        "运行来源": source,
        "研究区间": research_metadata,
        "运行时间": run_time.isoformat(timespec="seconds"),
        "回测起始日期": strategy_metrics.get("start_date"),
        "回测结束日期": strategy_metrics.get("end_date"),
        "策略累计收益率": strategy_metrics.get("total_return"),
        "基准累计收益率": benchmark_metrics.get("total_return"),
        "累计超额收益率": _difference(strategy_metrics.get("total_return"), benchmark_metrics.get("total_return")),
        "策略夏普比率": strategy_metrics.get("sharpe"),
        "策略最大回撤": strategy_metrics.get("max_drawdown"),
        "基准最大回撤": benchmark_metrics.get("max_drawdown"),
        "策略累计资本利得_BP": strategy_metrics.get("capital_gain_total_bp"),
        "基准累计资本利得_BP": benchmark_metrics.get("capital_gain_total_bp"),
        "资本利得超额_BP": _difference(strategy_metrics.get("capital_gain_total_bp"), benchmark_metrics.get("capital_gain_total_bp")),
        "资本利得交易胜率": strategy_metrics.get("capital_gain_trade_win_rate"),
        "平均单笔资本利得_BP": strategy_metrics.get("capital_gain_avg_trade_bp"),
        "平均每笔盈利_BP": strategy_metrics.get("capital_gain_avg_win_bp"),
        "最差交易_BP": strategy_metrics.get("capital_gain_worst_trade_bp"),
        "平均每笔持有交易日": strategy_metrics.get("capital_gain_avg_holding_days"),
        "最长单笔持有交易日": strategy_metrics.get("capital_gain_max_holding_days"),
        "资本利得最大回撤_BP": strategy_metrics.get("capital_gain_max_drawdown_bp"),
        "基准资本利得最大回撤_BP": benchmark_metrics.get("capital_gain_max_drawdown_bp"),
        "样本外定义": (
            f"训练截止日 {training_end} 后的首个可用交易日起"
            if training_end else "未登记训练截止日，不定义样本外区间"
        ),
        "样本外累计资本利得_BP": out_of_sample_metrics.get("capital_gain_total_bp"),
        "样本外交易胜率": out_of_sample_metrics.get("capital_gain_trade_win_rate"),
        "样本外最大回撤_BP": out_of_sample_metrics.get("capital_gain_max_drawdown_bp"),
        "输入数据": {
            "交易标的收益率曲线": _file_fingerprint(asset_path),
            "条件基准使用的10Y国债收益率曲线": _file_fingerprint(benchmark_path),
            "看板信号": _file_fingerprint(signal_path),
        },
        "代码文件": {
            "策略逻辑": _file_fingerprint(root / "strategies" / "dashboard_signal_v1.py"),
            "回测引擎": _file_fingerprint(root / "common" / "runner.py"),
            "收益模型": _file_fingerprint(root / "common" / "bond_return.py"),
        },
        "产物": [
            "config.json",
            "run_manifest.json",
            "performance_metrics.csv",
            "period_diagnostics.csv",
            "capital_gain_trades.csv",
            "signal_score.csv",
            "strategy_nav.csv",
            "performance_report.html",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _append_experiment_index(root, output_dir, manifest, strategy_metrics)
    return output_dir


def _enrich_research_metadata(
    config: DashboardStrategyConfig,
    research_metadata: dict[str, object] | None,
) -> dict[str, object]:
    """Carry forward a documented selection cutoff when a strategy is rerun.

    A normal hand-built backtest has no sample-out-of-sample definition and
    remains blank.  A search-derived configuration already records its cutoff
    in provenance, so losing it during a later web rerun would make the same
    strategy impossible to compare on the history page.
    """
    metadata = dict(research_metadata or {})
    if metadata.get("训练截止日"):
        return metadata
    provenance = config.research_provenance if isinstance(config.research_provenance, dict) else {}
    steps = provenance.get("steps", []) if isinstance(provenance, dict) else []
    if not isinstance(steps, list):
        return metadata
    for step in reversed(steps):
        if not isinstance(step, dict) or not step.get("training_end"):
            continue
        training_end = str(step["training_end"])
        metadata["训练截止日"] = training_end
        if step.get("training_start"):
            metadata.setdefault("训练起始日", str(step["training_start"]))
        try:
            metadata.setdefault("样本外起始日", (pd.Timestamp(training_end) + pd.Timedelta(days=1)).date().isoformat())
        except (TypeError, ValueError):
            pass
        metadata.setdefault("训练截止日来源", "继承策略研究溯源")
        break
    return metadata


def _archive_training_end(experiment_dir: Path, research_range: object) -> str | None:
    """Read the explicitly recorded cutoff, including pre-fix archive configs."""
    if isinstance(research_range, dict):
        for key in ("训练截止日", "首次搜索期结束日", "首期训练截止日"):
            if research_range.get(key):
                return str(research_range[key])
    try:
        config = load_strategy_config(experiment_dir / "config.json")
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return None
    return _enrich_research_metadata(config, {}).get("训练截止日") or None


def _archive_training_range(experiment_dir: Path, research_range: object) -> str | None:
    """Return the recorded training window without inventing an OOS definition.

    Older search archives often recorded a cutoff but not a calendar start.
    In that case their search contract was "from the first usable observation";
    preserving that wording is more accurate than manufacturing a date from a
    later backtest file.
    """
    training_end = _archive_training_end(experiment_dir, research_range)
    if not training_end:
        return None
    training_start: object | None = None
    if isinstance(research_range, dict):
        for key in ("训练起始日", "首次训练起始日", "搜索期起始日"):
            if research_range.get(key):
                training_start = research_range[key]
                break
    if not training_start:
        try:
            provenance = load_strategy_config(experiment_dir / "config.json").research_provenance or {}
            steps = provenance.get("steps", []) if isinstance(provenance, dict) else []
            for step in reversed(steps if isinstance(steps, list) else []):
                if not isinstance(step, dict):
                    continue
                if str(step.get("training_end", "")) == training_end and step.get("training_start"):
                    training_start = step["training_start"]
                    break
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            pass
    return f"{training_start or '首个可用观察'} 至 {training_end}"


def list_experiments(root: Path) -> pd.DataFrame:
    experiment_root = root / EXPERIMENT_DIR
    if not experiment_root.exists():
        return pd.DataFrame()
    # The archive index is derived data, not a replacement for individual
    # archive snapshots.  It prevents the history landing page from opening
    # hundreds of CSV files merely to paint the initial list.
    index_path = root / EXPERIMENT_INDEX_FILE
    try:
        cached_payload = json.loads(index_path.read_text(encoding="utf-8"))
        cached_rows = cached_payload.get("rows", []) if isinstance(cached_payload, dict) else []
        # Version 4 includes the complete trade-metric columns, a documented
        # training window, and the BP-denominated capital-gain drawdown for
        # the out-of-sample slice.
        # documented out-of-sample slice used by the history table.  Version 2
        # appended new archives with those OOS fields hard-coded to blank.
        # home snapshot.  Older indexes were append-only and left fields such
        # as annualized BP and winning/closed counts blank for newly archived
        # factor routes; rebuild them once rather than displaying false
        # "暂无" values.
        required_columns = {"年化资本利得_BP", "盈利交易数", "已平仓交易数", "样本训练区间", "样本外最大回撤_BP"}
        cache_version = cached_payload.get("version") if isinstance(cached_payload, dict) else None
        schema_ok = cache_version == _EXPERIMENT_INDEX_VERSION and all(
            isinstance(row, dict) and required_columns.issubset(row.keys()) for row in cached_rows
        )
        if isinstance(cached_rows, list) and schema_ok:
            normalized_rows = _normalise_index_rows(root, cached_rows)
            if len(normalized_rows) != len(cached_rows):
                _write_experiment_index(root, pd.DataFrame(normalized_rows))
            result = pd.DataFrame(normalized_rows)
            return result.sort_values("运行时间", ascending=False).reset_index(drop=True) if not result.empty else result
    except (OSError, json.JSONDecodeError):
        pass

    rows: list[dict[str, Any]] = []
    manifest_paths = sorted(experiment_root.glob("*/run_manifest.json"), key=lambda path: str(path))
    signature = tuple(
        (str(path), int(path.stat().st_mtime_ns), int(path.stat().st_size))
        for path in manifest_paths
        if path.exists()
    )
    cache_key = str(root.resolve())
    cached = _EXPERIMENTS_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1].copy(deep=True)
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        prefix = _manifest_id_prefix(manifest)
        saved_id = str(manifest.get("short_id", "")).strip()
        registered_id = existing_short_archive_id(root, archive_id_category(prefix), manifest_path.parent.name)
        # The registry is the immutable source of truth.  A manifest is an
        # archive snapshot and must never be allowed to silently redefine its
        # own visible identifier.
        if registered_id and saved_id and saved_id != registered_id:
            raise ValueError(
                f"归档 ID 已被篡改：{manifest_path.parent.name} 保存为 {saved_id}，注册表为 {registered_id}"
            )
        short_id = registered_id or saved_id
        trade_metrics = _archived_strategy_trade_metrics(manifest_path.parent)
        research_range = manifest.get("研究区间", {})
        training_end = _archive_training_end(manifest_path.parent, research_range)
        training_range = _archive_training_range(manifest_path.parent, research_range)
        out_of_sample_metrics = _archived_out_of_sample_metrics(manifest_path.parent, training_end)
        rows.append(
            {
                "运行ID": short_id,
                "运行时间": manifest.get("运行时间", ""),
                "策略名称": manifest.get("策略名称", ""),
                "运行来源": manifest.get("运行来源", ""),
                "信号频率": manifest.get("信号频率", "周频"),
                "比较基准": manifest.get("比较基准", "10Y地方政府债"),
                "BP口径": manifest.get("资本利得BP口径", "久期折算价格收益BP（旧口径）"),
                "回测起始日期": manifest.get("回测起始日期", ""),
                "回测结束日期": manifest.get("回测结束日期", ""),
                "回测区间": f"{manifest.get('回测起始日期', '')} 至 {manifest.get('回测结束日期', '')}",
                "样本训练区间": training_range,
                "策略累计收益率": manifest.get("策略累计收益率"),
                "基准累计收益率": manifest.get("基准累计收益率"),
                "累计超额收益率": manifest.get("累计超额收益率"),
                "夏普比率": manifest.get("策略夏普比率"),
                "最大回撤": manifest.get("策略最大回撤"),
                "基准最大回撤": trade_metrics.get("benchmark_max_drawdown", manifest.get("基准最大回撤")),
                "累计资本利得_BP": trade_metrics.get("capital_gain_total_bp", manifest.get("策略累计资本利得_BP")),
                "年化资本利得_BP": trade_metrics.get("capital_gain_annualized_bp"),
                "基准累计资本利得_BP": trade_metrics.get("benchmark_capital_gain_total_bp", manifest.get("基准累计资本利得_BP")),
                "资本利得超额_BP": manifest.get("资本利得超额_BP"),
                "资本利得交易胜率": trade_metrics.get("capital_gain_trade_win_rate", manifest.get("资本利得交易胜率")),
                "已平仓交易数": trade_metrics.get("capital_gain_closed_trade_count"),
                "盈利交易数": trade_metrics.get("capital_gain_winning_trades"),
                "亏损交易数": trade_metrics.get("capital_gain_losing_trades"),
                "平均单笔资本利得_BP": trade_metrics.get("capital_gain_avg_trade_bp", manifest.get("平均单笔资本利得_BP")),
                "平均每笔盈利_BP": trade_metrics.get("capital_gain_avg_win_bp", manifest.get("平均每笔盈利_BP")),
                "平均单笔亏损_BP": trade_metrics.get("capital_gain_avg_loss_bp"),
                "资本利得盈亏比": trade_metrics.get("capital_gain_profit_loss_ratio"),
                "最差交易_BP": trade_metrics.get("capital_gain_worst_trade_bp", manifest.get("最差交易_BP")),
                "平均每笔持有交易日": trade_metrics.get("capital_gain_avg_holding_days", manifest.get("平均每笔持有交易日")),
                "最长单笔持有交易日": trade_metrics.get("capital_gain_max_holding_days", manifest.get("最长单笔持有交易日")),
                "资本利得最大回撤_BP": trade_metrics.get("capital_gain_max_drawdown_bp", manifest.get("资本利得最大回撤_BP")),
                "资本利得最大回撤起点": trade_metrics.get("capital_gain_max_drawdown_start"),
                "资本利得最大回撤终点": trade_metrics.get("capital_gain_max_drawdown_end"),
                # Only research archives with a documented training cutoff have
                # an out-of-sample definition.  Ordinary backtests deliberately
                # remain blank here rather than retrospectively inventing one.
                "样本外累计资本利得_BP": out_of_sample_metrics.get("capital_gain_total_bp"),
                "样本外交易胜率": out_of_sample_metrics.get("capital_gain_trade_win_rate"),
                "样本外最大回撤_BP": out_of_sample_metrics.get("capital_gain_max_drawdown_bp"),
                "基准资本利得最大回撤_BP": trade_metrics.get("benchmark_capital_gain_max_drawdown_bp", manifest.get("基准资本利得最大回撤_BP")),
                "基准年化资本利得_BP": trade_metrics.get("benchmark_capital_gain_annualized_bp"),
                "基准资本利得最大回撤起点": trade_metrics.get("benchmark_capital_gain_max_drawdown_start"),
                "基准资本利得最大回撤终点": trade_metrics.get("benchmark_capital_gain_max_drawdown_end"),
                "实验目录": _canonical_archive_path(root, manifest_path.parent),
            }
        )
    rows = _normalise_index_rows(root, rows)
    result = pd.DataFrame(rows).sort_values("运行时间", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    _write_experiment_index(root, result)
    _EXPERIMENTS_CACHE[cache_key] = (signature, result.copy(deep=True))
    return result


def _write_experiment_index(root: Path, frame: pd.DataFrame) -> None:
    """Persist a rebuildable list index; detailed files remain in each archive."""
    path = root / EXPERIMENT_INDEX_FILE
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records") if not frame.empty else []
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": _EXPERIMENT_INDEX_VERSION, "rows": rows}, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_experiment_index(
    root: Path,
    output_dir: Path,
    manifest: dict[str, object],
    strategy_metrics: dict[str, object],
) -> None:
    """Add a just-created archive without triggering a history-wide rescan."""
    path = root / EXPERIMENT_INDEX_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
    except (OSError, json.JSONDecodeError):
        return
    # Let the next list read rebuild older indexes in one pass. Appending a
    # version-3 row to an older payload would otherwise preserve its stale OOS
    # blanks indefinitely while falsely claiming the new schema.
    if not isinstance(rows, list) or payload.get("version") != _EXPERIMENT_INDEX_VERSION:
        return
    archive_path = _canonical_archive_path(root, output_dir)
    rows = _normalise_index_rows(root, rows)
    rows = [row for row in rows if str(row.get("实验目录", "")) != archive_path]
    prefix = _manifest_id_prefix(manifest)
    registered_id = existing_short_archive_id(root, archive_id_category(prefix), Path(output_dir).name)
    manifest_id = str(manifest.get("short_id", "")).strip()
    if registered_id and manifest_id and registered_id != manifest_id:
        raise ValueError(f"归档 ID 已被篡改：{Path(output_dir).name} 保存为 {manifest_id}，注册表为 {registered_id}")
    archive_id = registered_id or manifest_id
    if archive_id:
        for row in rows:
            if str(row.get("运行ID", "")).strip() == archive_id:
                raise ValueError(f"运行 ID 冲突：{archive_id} 已属于 {row.get('实验目录')}")
    research_range = manifest.get("研究区间", {})
    training_end = _archive_training_end(output_dir, research_range)
    training_range = _archive_training_range(output_dir, research_range)
    out_of_sample_metrics = _archived_out_of_sample_metrics(output_dir, training_end)
    rows.append({
        "运行ID": archive_id, "运行时间": manifest.get("运行时间", ""),
        "策略名称": manifest.get("策略名称", ""), "运行来源": manifest.get("运行来源", ""),
        "信号频率": manifest.get("信号频率", "周频"), "比较基准": manifest.get("比较基准", "10Y地方政府债"),
        "BP口径": manifest.get("资本利得BP口径", "收益率方向变动BP（不乘久期）"),
        "回测起始日期": manifest.get("回测起始日期", ""), "回测结束日期": manifest.get("回测结束日期", ""),
        "回测区间": f"{manifest.get('回测起始日期', '')} 至 {manifest.get('回测结束日期', '')}",
        "样本训练区间": training_range,
        "策略累计收益率": manifest.get("策略累计收益率"), "基准累计收益率": manifest.get("基准累计收益率"),
        "累计超额收益率": manifest.get("累计超额收益率"), "夏普比率": manifest.get("策略夏普比率"),
        "最大回撤": manifest.get("策略最大回撤"), "基准最大回撤": manifest.get("基准最大回撤"),
        "累计资本利得_BP": strategy_metrics.get("capital_gain_total_bp"),
        "年化资本利得_BP": strategy_metrics.get("capital_gain_annualized_bp"),
        "基准累计资本利得_BP": manifest.get("基准累计资本利得_BP"), "资本利得超额_BP": manifest.get("资本利得超额_BP"),
        "资本利得交易胜率": strategy_metrics.get("capital_gain_trade_win_rate"),
        "已平仓交易数": strategy_metrics.get("capital_gain_closed_trade_count"),
        "盈利交易数": strategy_metrics.get("capital_gain_winning_trades"),
        "亏损交易数": strategy_metrics.get("capital_gain_losing_trades"),
        "平均单笔资本利得_BP": strategy_metrics.get("capital_gain_avg_trade_bp"),
        "平均每笔盈利_BP": strategy_metrics.get("capital_gain_avg_win_bp"),
        "平均单笔亏损_BP": strategy_metrics.get("capital_gain_avg_loss_bp"),
        "资本利得盈亏比": strategy_metrics.get("capital_gain_profit_loss_ratio"),
        "最差交易_BP": strategy_metrics.get("capital_gain_worst_trade_bp"),
        "平均每笔持有交易日": strategy_metrics.get("capital_gain_avg_holding_days"),
        "最长单笔持有交易日": strategy_metrics.get("capital_gain_max_holding_days"),
        "资本利得最大回撤_BP": strategy_metrics.get("capital_gain_max_drawdown_bp"),
        "样本外累计资本利得_BP": out_of_sample_metrics.get("capital_gain_total_bp"),
        "样本外交易胜率": out_of_sample_metrics.get("capital_gain_trade_win_rate"),
        "样本外最大回撤_BP": out_of_sample_metrics.get("capital_gain_max_drawdown_bp"),
        "实验目录": archive_path,
    })
    _write_experiment_index(root, pd.DataFrame(rows))
    _EXPERIMENTS_CACHE.pop(str(root.resolve()), None)


def load_experiment_result(
    experiment_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object], DashboardStrategyConfig]:
    config = load_strategy_config(experiment_dir / "config.json")
    nav = pd.read_csv(experiment_dir / "strategy_nav.csv", encoding="utf-8-sig")
    mapping = {
        "日期": "date", "信号日期": "signal_date", "目标仓位": "目标仓位", "止盈止损事件": "止盈止损事件",
        "条件基准仓位": "comparison_position", "交易标的到期收益率_百分比": "asset_yield_pct", "比较基准到期收益率_百分比": "yield_pct",
        "策略日收益率": "strategy_return", "基准日收益率": "total_return",
        "策略票息Carry收益": "strategy_carry_return", "策略资本利得收益": "strategy_capital_return",
        "基准票息Carry收益": "benchmark_carry_return", "基准资本利得收益": "benchmark_capital_return",
        "票息Carry超额": "carry_excess_return", "资本利得超额": "capital_excess_return",
        "策略累计票息Carry": "strategy_carry_cum", "策略累计资本利得": "strategy_capital_cum",
        "基准累计票息Carry": "benchmark_carry_cum", "基准累计资本利得": "benchmark_capital_cum",
        "策略净值": "strategy_nav", "基准净值": "benchmark_nav_rebased", "超额净值": "excess_nav",
        "策略资本利得_BP": "strategy_capital_bp", "基准资本利得_BP": "benchmark_capital_bp",
        "资本利得超额_BP": "capital_excess_bp", "策略累计资本利得_BP": "strategy_capital_cum_bp",
        "基准累计资本利得_BP": "benchmark_capital_cum_bp", "累计资本利得超额_BP": "capital_excess_cum_bp",
    }
    daily = nav.rename(columns=mapping)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["signal_date"] = pd.to_datetime(daily["signal_date"])
    if "strategy_capital_bp" not in daily:
        daily["strategy_capital_bp"] = pd.to_numeric(daily["strategy_capital_return"], errors="coerce").fillna(0.0) * 10000.0
        daily["benchmark_capital_bp"] = pd.to_numeric(daily["benchmark_capital_return"], errors="coerce").fillna(0.0) * 10000.0
        daily["capital_excess_bp"] = daily["strategy_capital_bp"] - daily["benchmark_capital_bp"]
        daily["strategy_capital_cum_bp"] = daily["strategy_capital_bp"].cumsum()
        daily["benchmark_capital_cum_bp"] = daily["benchmark_capital_bp"].cumsum()
        daily["capital_excess_cum_bp"] = daily["capital_excess_bp"].cumsum()
    if "carry_excess_cum" not in daily:
        daily["carry_excess_cum"] = pd.to_numeric(daily["carry_excess_return"], errors="coerce").fillna(0.0).cumsum()
    if "capital_excess_cum" not in daily:
        daily["capital_excess_cum"] = pd.to_numeric(daily["capital_excess_return"], errors="coerce").fillna(0.0).cumsum()

    signals = pd.read_csv(experiment_dir / "signal_score.csv", encoding="utf-8-sig")
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    signals.attrs["position_policy"] = config.positions.as_dict()
    provenance = config.research_provenance or {}
    expanded_weights = provenance.get("扩展因子权重") if isinstance(provenance, dict) else None
    # Factor-research archive configs keep legacy weights for compatibility;
    # surface the complete saved universe to result renderers instead of
    # silently reducing an expanded strategy to the eight-factor view.
    signals.attrs["weights"] = expanded_weights if isinstance(expanded_weights, dict) else config.weights.as_dict()
    signals.attrs["thresholds"] = config.thresholds.as_dict()
    signals.attrs["signal_frequency"] = config.signal_frequency

    metrics_frame = pd.read_csv(experiment_dir / "performance_metrics.csv", encoding="utf-8-sig")
    strategy_metrics = _metrics_from_archive(metrics_frame, "策略")
    benchmark_metrics = _metrics_from_archive(metrics_frame, "基准")
    manifest_path = experiment_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    benchmark_metrics["benchmark_id"] = manifest.get("基准ID", config.benchmark_id)
    benchmark_metrics["benchmark_name"] = manifest.get("比较基准", benchmark_label(config.benchmark_id))
    # Always rebuild trade-level metrics from the archived position path. Older
    # archives may contain the former weekly-period trade count even when their
    # daily NAV and position snapshots are otherwise complete.
    strategy_metrics.update(capital_gain_trade_metrics(daily, "strategy_capital_bp", position_col="仓位"))
    benchmark_position_col = "comparison_position" if "comparison_position" in daily else None
    benchmark_metrics.update(capital_gain_trade_metrics(daily, "benchmark_capital_bp", position_col=benchmark_position_col))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        definition = manifest.get("资本利得BP口径", "久期折算价格收益BP（旧口径）")
        strategy_metrics["capital_gain_bp_definition"] = definition
        benchmark_metrics["capital_gain_bp_definition"] = definition
    return daily, signals, strategy_metrics, benchmark_metrics, config


def _archived_strategy_trade_metrics(experiment_dir: Path) -> dict[str, object]:
    nav_path = experiment_dir / "strategy_nav.csv"
    if not nav_path.exists():
        return {}
    try:
        frame = pd.read_csv(nav_path, encoding="utf-8-sig")
        frame = frame.rename(
            columns={
                "日期": "date",
                "条件基准仓位": "comparison_position",
                "策略资本利得收益": "strategy_capital_return",
                "基准资本利得收益": "benchmark_capital_return",
                "策略资本利得_BP": "strategy_capital_bp",
                "基准资本利得_BP": "benchmark_capital_bp",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"])
        if "strategy_capital_bp" not in frame or "benchmark_capital_bp" not in frame:
            frame["strategy_capital_bp"] = pd.to_numeric(frame["strategy_capital_return"], errors="coerce").fillna(0.0) * 10000.0
            frame["benchmark_capital_bp"] = pd.to_numeric(frame["benchmark_capital_return"], errors="coerce").fillna(0.0) * 10000.0
        strategy_metrics = capital_gain_trade_metrics(frame, "strategy_capital_bp", position_col="仓位")
        benchmark_position_col = "comparison_position" if "comparison_position" in frame else None
        benchmark_metrics = capital_gain_trade_metrics(frame, "benchmark_capital_bp", position_col=benchmark_position_col)
        benchmark_nav = pd.to_numeric(frame.get("基准净值"), errors="coerce")
        benchmark_max_drawdown = None
        if benchmark_nav is not None and benchmark_nav.notna().any():
            benchmark_max_drawdown = float((benchmark_nav / benchmark_nav.cummax() - 1.0).min())
        return {
            **strategy_metrics,
            "benchmark_capital_gain_total_bp": benchmark_metrics.get("capital_gain_total_bp"),
            "benchmark_capital_gain_annualized_bp": benchmark_metrics.get("capital_gain_annualized_bp"),
            "benchmark_capital_gain_max_drawdown_bp": benchmark_metrics.get("capital_gain_max_drawdown_bp"),
            "benchmark_capital_gain_max_drawdown_start": benchmark_metrics.get("capital_gain_max_drawdown_start"),
            "benchmark_capital_gain_max_drawdown_end": benchmark_metrics.get("capital_gain_max_drawdown_end"),
            "benchmark_max_drawdown": benchmark_max_drawdown,
        }
    except (OSError, KeyError, ValueError):
        return {}


def _archived_out_of_sample_metrics(experiment_dir: Path, training_end: object) -> dict[str, object]:
    """Recalculate the documented out-of-sample slice from immutable archives.

    ``evaluate_period`` anchors the first available day at zero, which is the
    same period convention used throughout the research pages.  A missing or
    invalid cutoff means this is a normal backtest, not an incomplete OOS row.
    """
    if not training_end:
        return {}
    try:
        cutoff = pd.Timestamp(training_end)
        if pd.isna(cutoff):
            return {}
        nav = pd.read_csv(experiment_dir / "strategy_nav.csv", encoding="utf-8-sig")
        nav = nav.rename(columns={
            "日期": "date", "策略日收益率": "strategy_return", "策略资本利得收益": "strategy_capital_return",
            "策略资本利得_BP": "strategy_capital_bp", "策略净值": "strategy_nav", "仓位": "仓位",
        })
        nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
        nav = nav.dropna(subset=["date"]).sort_values("date")
        if nav.empty or nav["date"].max() <= cutoff:
            return {}
        frame = nav.loc[nav["date"] > cutoff].copy().reset_index(drop=True)
        if frame.empty or "strategy_return" not in frame:
            return {}
        frame["strategy_return"] = pd.to_numeric(frame["strategy_return"], errors="coerce").fillna(0.0)
        frame.loc[0, "strategy_return"] = 0.0
        frame["strategy_nav"] = (1.0 + frame["strategy_return"]).cumprod()
        if "strategy_capital_bp" not in frame:
            capital_return = frame["strategy_capital_return"] if "strategy_capital_return" in frame else pd.Series(0.0, index=frame.index)
            frame["strategy_capital_bp"] = pd.to_numeric(capital_return, errors="coerce").fillna(0.0) * 10000.0
        frame["strategy_capital_bp"] = pd.to_numeric(frame["strategy_capital_bp"], errors="coerce").fillna(0.0)
        frame.loc[0, "strategy_capital_bp"] = 0.0
        strategy_metrics = performance_metrics(frame, return_col="strategy_return", nav_col="strategy_nav")
        strategy_metrics.update(capital_gain_trade_metrics(frame, "strategy_capital_bp", position_col="仓位" if "仓位" in frame else None))
        return strategy_metrics
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError):
        return {}


def _metrics_from_archive(frame: pd.DataFrame, value_column: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for _, row in frame.iterrows():
        field = str(row["字段"])
        value = row.get(value_column)
        if pd.isna(value):
            metrics[field] = None
        elif field in {"start_date", "end_date", "max_drawdown_start", "max_drawdown_end"}:
            metrics[field] = str(value)
        else:
            try:
                metrics[field] = float(value)
            except (TypeError, ValueError):
                metrics[field] = value
    return metrics


def _file_fingerprint(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"路径": str(path), "存在": False}
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "路径": str(path.relative_to(path.parents[1])),
        "存在": True,
        "字节数": stat.st_size,
        "修改时间": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "SHA256": digest.hexdigest(),
    }


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return cleaned or "未命名策略"


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("同一秒内实验目录过多，请稍后重试。")


def _difference(left: object, right: object) -> float | None:
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
