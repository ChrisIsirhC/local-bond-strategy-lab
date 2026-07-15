from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from common.config import load_strategy_config
from common.runner import run_dashboard_config


ROOT = Path(__file__).resolve().parent


class CapitalGainBpTests(unittest.TestCase):
    def _config(self):
        return load_strategy_config(next((ROOT / "configs").glob("*.json")))

    def test_bp_uses_yield_change_without_duration(self) -> None:
        config = self._config()
        daily, _, strategy_metrics, benchmark_metrics = run_dashboard_config(ROOT, config)

        expected_strategy = -daily["仓位"] * daily["asset_yield_change_bp"]
        expected_benchmark = -daily["yield_change_bp"]
        np.testing.assert_allclose(daily["strategy_capital_bp"], expected_strategy)
        np.testing.assert_allclose(daily["benchmark_capital_bp"], expected_benchmark)
        self.assertAlmostEqual(strategy_metrics["capital_gain_total_bp"], float(expected_strategy.sum()))
        self.assertAlmostEqual(benchmark_metrics["capital_gain_total_bp"], float(expected_benchmark.sum()))
        self.assertEqual(strategy_metrics["capital_gain_bp_definition"], "收益率方向变动BP（不乘久期）")

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
