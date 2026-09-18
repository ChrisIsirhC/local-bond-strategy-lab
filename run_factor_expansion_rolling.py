from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from common.factor_expansion_research import (
    ALL_FACTOR_COLUMNS,
    OBJECTIVES,
    ExpansionResearchConfig,
    evaluate_expansion_config,
    root_research_config,
    run_expansion_rolling_research,
)
from common.config import DashboardStrategyConfig, strategy_config_from_dict
from common.trade_metrics import capital_gain_trade_metrics
from common.frame_store import frame_exists, read_frame, write_frame


ROOT = Path(__file__).resolve().parent
DEFAULT_MINIMUM_TRAINING_MONTHS = 24
DEFAULT_RECALIBRATION_MONTHS = 3
DEFAULT_BEAM_WIDTH = 40


def run_factor_expansion_rolling(
    static_dir: Path,
    root: Path = ROOT,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    base_config: DashboardStrategyConfig | None = None,
) -> Path:
    """Run every factor-version/objective/frequency rolling track for a study."""
    root = Path(root).resolve()
    static_dir = Path(static_dir).resolve()
    if not frame_exists(static_dir / "静态研究汇总.csv"):
        raise ValueError("static-dir 不包含已完成的静态研究汇总")
    static_summary = read_frame(static_dir / "静态研究汇总.csv")
    manifest_path = static_dir / "研究清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if base_config is None and isinstance(manifest.get("研究基线"), dict):
        base_config = strategy_config_from_dict(manifest["研究基线"])
    static_contract = manifest.get("静态搜索", {}) if isinstance(manifest.get("静态搜索"), dict) else {}
    enabled_factor_columns = tuple(static_contract.get("事前启用因子", ()) or ALL_FACTOR_COLUMNS)
    selected_objectives = static_summary["目标"].dropna().astype(str).drop_duplicates().tolist()
    selected_frequencies = {
        "日频": "daily",
        "周频": "weekly",
    }
    frequency_codes = [
        selected_frequencies[value]
        for value in static_summary["频率"].dropna().astype(str).drop_duplicates().tolist()
        if value in selected_frequencies
    ]
    if not selected_objectives or not frequency_codes:
        raise ValueError("静态研究汇总缺少目标或信号频率")
    output = static_dir / "滚动定参"
    output.mkdir(exist_ok=True)
    summary: list[dict[str, object]] = []
    factor_versions = tuple(static_summary["因子版本"].dropna().astype(str).drop_duplicates())
    if not factor_versions:
        raise ValueError("静态研究汇总缺少因子版本")
    original_period_results: dict[tuple[str, str], pd.DataFrame] = {}
    for factor_version in factor_versions:
        for objective_name in selected_objectives:
            for frequency in frequency_codes:
                label = f"{factor_version}_{objective_name}_{frequency}"
                print(f"开始滚动：{label}", flush=True)
                base = root_research_config(
                    factor_version,
                    objective_name,
                    frequency,
                    thresholds=base_config.thresholds if base_config else None,
                    positions=base_config.positions if base_config else None,
                    factor_windows=base_config.factor_windows if base_config else None,
                    initial_weights=base_config.weights.as_dict() if base_config else None,
                    enabled_factor_columns=enabled_factor_columns,
                )
                periods, daily = run_expansion_rolling_research(
                    root, base,
                    minimum_training_months=DEFAULT_MINIMUM_TRAINING_MONTHS,
                    recalibration_months=DEFAULT_RECALIBRATION_MONTHS,
                    beam_width=beam_width,
                    original_periods=original_period_results.get((objective_name, frequency)),
                )
                if factor_version == "原始因子":
                    original_period_results[(objective_name, frequency)] = periods.copy()
                write_frame(periods, output / f"{label}_逐期定参.csv")
                write_frame(daily, output / f"{label}_样本外日度.csv")
                if not periods.empty:
                    first = periods.iloc[0]
                    first_config = ExpansionResearchConfig(
                        factor_version=factor_version,
                        objective_name=objective_name,
                        signal_frequency=frequency,
                        weights={str(key): float(value) for key, value in json.loads(first["权重"]).items()},
                        thresholds=base.thresholds,
                        positions=replace(
                            base.positions,
                            bullish_threshold=float(first["看多阈值"]),
                            bearish_threshold=float(first["看空阈值"]),
                        ),
                        factor_windows=base.factor_windows,
                        enabled_factor_columns=enabled_factor_columns,
                    )
                    search_daily, search_signals, _, _ = evaluate_expansion_config(
                        root, first_config, str(first["训练起始日"]), str(first["训练截止日"])
                    )
                    write_frame(search_daily, output / f"{label}_搜索期日度.csv")
                    write_frame(search_signals, output / f"{label}_搜索期信号.csv")
                metrics = capital_gain_trade_metrics(daily, "strategy_capital_bp", position_col="仓位")
                summary.append({
                    "因子版本": factor_version,
                    "目标": objective_name,
                    "频率": "日频" if frequency == "daily" else "周频",
                    "最低训练长度月数": DEFAULT_MINIMUM_TRAINING_MONTHS,
                    "重定参间隔月数": DEFAULT_RECALIBRATION_MONTHS,
                    "beam宽度": beam_width,
                    "滚动期数": int(len(periods)),
                    "采用原始因子保底期数": int(
                        periods.get("是否采用原始因子保底", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
                    ),
                    "样本外起始日": daily["date"].min().date().isoformat(),
                    "样本外结束日": daily["date"].max().date().isoformat(),
                    "样本外累计资本利得_BP": metrics.get("capital_gain_total_bp"),
                    "样本外交易胜率": metrics.get("capital_gain_trade_win_rate"),
                    "样本外已平仓交易数": metrics.get("capital_gain_closed_trade_count"),
                    "样本外最大回撤_BP": metrics.get("capital_gain_max_drawdown_bp"),
                })
    frame = pd.DataFrame(summary)
    write_frame(frame, output / "滚动定参汇总.csv")
    rolling_contract = {
        "训练方式": "自首个可用数据日扩展",
        "最低训练长度月数": DEFAULT_MINIMUM_TRAINING_MONTHS,
        "重定参间隔月数": DEFAULT_RECALIBRATION_MONTHS,
        "权重搜索": (
            "每期在当前训练窗口做V2约束下的完整配置多起点搜索与五点权重转移；"
            + (
                "扩展路线强制纳入同一期原始最优策略（新增因子全为0）作为保底；"
                if "扩展因子" in factor_versions and "原始因子" in factor_versions else ""
            )
            + "完整候选同分时优先较少有效因子；随后做阈值搜索"
        ),
        "执行": "每段只使用当期选出的参数；逐期样本外结果拼接",
        "说明": "滚动结果不参与静态版本的权重或阈值选优。",
        "事前启用因子": list(enabled_factor_columns),
        "事前停用因子": [column for column in ALL_FACTOR_COLUMNS if column not in enabled_factor_columns],
    }
    (output / "滚动定参口径.json").write_text(json.dumps(rolling_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from run_factor_expansion_research import _write_report, write_factor_expansion_manifest

    seasonality = json.loads((static_dir / "季节性供给可行性.json").read_text(encoding="utf-8"))
    training_end = str(manifest.get("训练截止日", "2025-01-01"))
    static_width = int(manifest.get("静态搜索", {}).get("搜索宽度", 160))
    write_factor_expansion_manifest(
        static_dir, training_end, static_width,
        rolling=rolling_contract, base_config=base_config, root=root,
        enabled_factor_columns=enabled_factor_columns,
        study_mode=str(manifest.get("研究模式", "对照研究")),
        factor_versions=factor_versions,
    )
    _write_report(static_dir, static_summary, seasonality, training_end=training_end)
    print(output, flush=True)
    return output


def main() -> Path:
    parser = argparse.ArgumentParser(description="因子增加研究的滚动定参验证")
    parser.add_argument("--static-dir", type=Path, required=True, help="已完成静态研究的目录")
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    args = parser.parse_args()
    return run_factor_expansion_rolling(args.static_dir, ROOT, args.beam_width)


if __name__ == "__main__":
    main()
