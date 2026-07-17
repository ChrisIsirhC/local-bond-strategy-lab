from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from strategies.dashboard_signal_v1 import DashboardThresholds, DashboardWeights
from strategies.position_policy import DashboardPositionPolicy
from common.market_data import DEFAULT_BENCHMARK_ID, normalize_benchmark_id


CONFIG_DIR = Path("configs")
SIGNAL_FREQUENCIES = {"weekly", "daily"}


def normalize_signal_frequency(value: object) -> str:
    frequency = str(value or "weekly").strip().lower()
    if frequency not in SIGNAL_FREQUENCIES:
        raise ValueError(f"不支持的信号频率: {frequency}")
    return frequency


@dataclass(frozen=True)
class ObjectiveConfig:
    total_return_weight: float = 0.0
    excess_return_weight: float = 0.0
    sharpe_weight: float = 0.0
    max_drawdown_penalty: float = 0.0
    signal_win_rate_weight: float = 0.0
    capital_gain_bp_weight: float = 1.0
    capital_gain_excess_bp_weight: float = 0.25
    capital_trade_win_rate_weight: float = 10.0
    capital_gain_avg_win_bp_weight: float = 0.0
    capital_gain_drawdown_bp_penalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        values = asdict(self)
        values.pop("max_drawdown_penalty", None)
        return values


@dataclass(frozen=True)
class DashboardStrategyConfig:
    name: str
    weights: DashboardWeights
    thresholds: DashboardThresholds
    positions: DashboardPositionPolicy
    objective: ObjectiveConfig
    backtest_start: str | None = None
    backtest_end: str | None = None
    benchmark_id: str = DEFAULT_BENCHMARK_ID
    signal_frequency: str = "weekly"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": self.weights.as_dict(),
            "thresholds": self.thresholds.as_dict(),
            "positions": self.positions.as_dict(),
            "objective": self.objective.as_dict(),
            "backtest": {
                "start_date": self.backtest_start,
                "end_date": self.backtest_end,
            },
            "benchmark": self.benchmark_id,
            "signal_frequency": normalize_signal_frequency(self.signal_frequency),
        }


def load_strategy_config(path: Path) -> DashboardStrategyConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return strategy_config_from_dict(raw)


def save_strategy_config(config: DashboardStrategyConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strategy_config_from_dict(raw: dict[str, Any]) -> DashboardStrategyConfig:
    backtest = raw.get("backtest", {})
    objective_values = _float_dict(raw.get("objective", {}))
    if "capital_gain_avg_win_bp_weight" not in objective_values:
        objective_values["capital_gain_avg_win_bp_weight"] = objective_values.pop(
            "capital_gain_avg_trade_bp_weight", 0.0
        )
    else:
        objective_values.pop("capital_gain_avg_trade_bp_weight", None)
    return DashboardStrategyConfig(
        name=str(raw.get("name", "dashboard_signal_config")),
        weights=DashboardWeights(**_float_dict(raw.get("weights", {}))),
        thresholds=DashboardThresholds(**_float_dict(raw.get("thresholds", {}))),
        positions=DashboardPositionPolicy(**raw.get("positions", {})),
        objective=ObjectiveConfig(**objective_values),
        backtest_start=backtest.get("start_date"),
        backtest_end=backtest.get("end_date"),
        benchmark_id=normalize_benchmark_id(raw.get("benchmark")),
        signal_frequency=normalize_signal_frequency(raw.get("signal_frequency", raw.get("frequency"))),
    )


def _float_dict(raw: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in raw.items()}
