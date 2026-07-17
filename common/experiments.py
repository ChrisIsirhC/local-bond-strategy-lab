from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.config import DashboardStrategyConfig, load_strategy_config, save_strategy_config
from common.market_data import CONDITIONAL_BENCHMARK_ID, CONDITIONAL_BENCHMARK_NAME, GOV_10Y, TRADED_ASSET_ID, benchmark_label, curve_path
from common.reporting import write_strategy_outputs
from common.trade_metrics import capital_gain_trade_metrics
from strategies.dashboard_signal_v1 import signal_file_for_frequency


EXPERIMENT_DIR = Path("backtest_outputs") / "experiments"


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
    run_id = f"{run_time:%Y%m%d_%H%M%S}__{_safe_name(config.name)}"
    output_dir = _next_available_dir(root / EXPERIMENT_DIR / run_id)
    output_dir.mkdir(parents=True, exist_ok=False)

    write_strategy_outputs(daily, signals, strategy_metrics, benchmark_metrics, output_dir)
    save_strategy_config(config, output_dir / "config.json")

    asset_path = curve_path(root, TRADED_ASSET_ID)
    benchmark_path = curve_path(root, GOV_10Y)
    signal_path = root / signal_file_for_frequency(config.signal_frequency)
    manifest = {
        "run_id": output_dir.name,
        "策略名称": config.name,
        "策略版本": "dashboard_signal_v1",
        "信号频率": "日频" if config.signal_frequency == "daily" else "周频",
        "交易标的": benchmark_label(TRADED_ASSET_ID),
        "比较基准": str(benchmark_metrics.get("benchmark_name", CONDITIONAL_BENCHMARK_NAME)),
        "基准ID": str(benchmark_metrics.get("benchmark_id", CONDITIONAL_BENCHMARK_ID)),
        "条件基准规则": "仓位>0时同仓位买入10Y国债；仓位<=0时持有现金",
        "资本利得BP口径": "收益率方向变动BP（不乘久期）",
        "运行来源": source,
        "研究区间": research_metadata or {},
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
    return output_dir


def list_experiments(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    experiment_root = root / EXPERIMENT_DIR
    if not experiment_root.exists():
        return pd.DataFrame()
    for manifest_path in experiment_root.glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trade_metrics = _archived_strategy_trade_metrics(manifest_path.parent)
        rows.append(
            {
                "运行时间": manifest.get("运行时间", ""),
                "策略名称": manifest.get("策略名称", ""),
                "运行来源": manifest.get("运行来源", ""),
                "信号频率": manifest.get("信号频率", "周频"),
                "比较基准": manifest.get("比较基准", "10Y地方政府债"),
                "BP口径": manifest.get("资本利得BP口径", "久期折算价格收益BP（旧口径）"),
                "回测起始日期": manifest.get("回测起始日期", ""),
                "回测结束日期": manifest.get("回测结束日期", ""),
                "回测区间": f"{manifest.get('回测起始日期', '')} 至 {manifest.get('回测结束日期', '')}",
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
                "基准资本利得最大回撤_BP": trade_metrics.get("benchmark_capital_gain_max_drawdown_bp", manifest.get("基准资本利得最大回撤_BP")),
                "基准年化资本利得_BP": trade_metrics.get("benchmark_capital_gain_annualized_bp"),
                "基准资本利得最大回撤起点": trade_metrics.get("benchmark_capital_gain_max_drawdown_start"),
                "基准资本利得最大回撤终点": trade_metrics.get("benchmark_capital_gain_max_drawdown_end"),
                "实验目录": str(manifest_path.parent),
            }
        )
    return pd.DataFrame(rows).sort_values("运行时间", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


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
    signals.attrs["weights"] = config.weights.as_dict()
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
