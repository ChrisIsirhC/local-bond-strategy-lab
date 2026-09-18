from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_strategy_config
from common.market_data import CONDITIONAL_BENCHMARK_ID, CONDITIONAL_BENCHMARK_NAME
from common.runner import _backtest_dashboard_signals, run_dashboard_config


ROOT = Path(__file__).resolve().parent


class CapitalGainBpTests(unittest.TestCase):
    def _config(self):
        return load_strategy_config(next((ROOT / "configs").glob("*.json")))

    def test_bp_uses_yield_change_without_duration(self) -> None:
        config = self._config()
        daily, _, strategy_metrics, benchmark_metrics = run_dashboard_config(ROOT, config)

        expected_strategy = -daily["仓位"] * daily["asset_yield_change_bp"]
        expected_benchmark = -daily["仓位"].clip(lower=0.0) * daily["yield_change_bp"]
        np.testing.assert_allclose(daily["strategy_capital_bp"], expected_strategy)
        np.testing.assert_allclose(daily["benchmark_capital_bp"], expected_benchmark)
        self.assertAlmostEqual(strategy_metrics["capital_gain_total_bp"], float(expected_strategy.sum()))
        self.assertAlmostEqual(benchmark_metrics["capital_gain_total_bp"], float(expected_benchmark.sum()))
        self.assertEqual(strategy_metrics["capital_gain_bp_definition"], "收益率方向变动BP（不乘久期）")

    def test_conditional_benchmark_for_long_flat_and_short_positions(self) -> None:
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        market = pd.DataFrame(
            {
                "date": dates,
                "asset_yield_change_bp": [9.0, -2.0, -4.0, 3.0, 5.0],
                "yield_change_bp": [8.0, -1.0, -2.0, 4.0, 6.0],
                "asset_carry_return": [0.0] * 5,
                "asset_duration_pnl": [0.0, 0.002, 0.004, -0.003, -0.005],
                "asset_total_return": [0.0, 0.002, 0.004, -0.003, -0.005],
                "carry_return": [0.0] * 5,
                "duration_pnl": [0.0, 0.001, 0.002, -0.004, -0.006],
                "total_return": [0.0, 0.001, 0.002, -0.004, -0.006],
            }
        )
        market.attrs["asset_name"] = "10Y地方政府债"
        signals = pd.DataFrame(
            {
                "signal_date": dates,
                "总分": [50.0] * 5,
                "结论": ["中性"] * 5,
                "仓位": [0.0, 1.0, 0.5, 0.0, -1.0],
            }
        )

        daily, strategy_metrics, benchmark_metrics = _backtest_dashboard_signals(market, signals)

        np.testing.assert_allclose(daily["comparison_position"], [0.0, 1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(daily["strategy_capital_bp"], [0.0, 2.0, 2.0, 0.0, 5.0])
        np.testing.assert_allclose(daily["benchmark_capital_bp"], [0.0, 1.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(daily["capital_excess_bp"], [0.0, 1.0, 1.0, 0.0, 5.0])
        self.assertEqual(benchmark_metrics["benchmark_id"], CONDITIONAL_BENCHMARK_ID)
        self.assertEqual(benchmark_metrics["benchmark_name"], CONDITIONAL_BENCHMARK_NAME)
        self.assertAlmostEqual(strategy_metrics["capital_gain_avg_holding_days"], 1.5)
        self.assertEqual(strategy_metrics["capital_gain_max_holding_days"], 2)

    def test_first_backtest_date_is_an_anchor(self) -> None:
        config = self._config()
        daily, _, _, _ = run_dashboard_config(ROOT, config)
        first = daily.iloc[0]
        for column in [
            "strategy_return", "total_return", "strategy_capital_return", "benchmark_capital_return",
            "strategy_capital_bp", "benchmark_capital_bp",
        ]:
            self.assertAlmostEqual(float(first[column]), 0.0)


if __name__ == "__main__":
    unittest.main()
