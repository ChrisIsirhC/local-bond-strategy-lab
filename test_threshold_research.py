from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from common.threshold_research import _candidate_position_matrix
from common.trade_metrics import select_executed_weekly_positions, vectorized_capital_trade_metrics
from strategies.position_policy import DashboardPositionPolicy


class ThresholdSearchPositionTests(unittest.TestCase):
    def test_candidate_positions_match_strategy_policy(self) -> None:
        scores = np.array([[20.0, 50.0, 80.0], [20.0, 50.0, 80.0]])
        policies = [
            DashboardPositionPolicy(
                bullish_threshold=70.0,
                bearish_threshold=30.0,
                bullish_position=1.0,
                neutral_position=0.5,
                bearish_position=-1.0,
            ),
            DashboardPositionPolicy(
                bullish_threshold=75.0,
                bearish_threshold=25.0,
                bullish_position=1.2,
                neutral_position=0.7,
                bearish_position=0.0,
            ),
        ]
        params = pd.DataFrame([policy.as_dict() for policy in policies])
        bearish = np.vstack([policy.bearish_mask(row) for policy, row in zip(policies, scores)])

        actual = _candidate_position_matrix(scores, bearish, params)
        expected = np.vstack([policy.vectorized_positions(row) for policy, row in zip(policies, scores)])

        np.testing.assert_allclose(actual, expected)

    def test_future_signal_is_excluded_from_trade_metrics(self) -> None:
        weekly_positions = np.ones((2, 128), dtype=float)
        weekly_positions[1, 127] = -1.0
        daily_signal_index = np.arange(127, dtype=int)
        weekly_capital_bp = np.zeros((2, 127), dtype=float)

        executed, signal_ids = select_executed_weekly_positions(weekly_positions, daily_signal_index)
        metrics = vectorized_capital_trade_metrics(executed, weekly_capital_bp)

        self.assertEqual(executed.shape, (2, 127))
        self.assertEqual(signal_ids[-1], 126)
        np.testing.assert_array_equal(metrics["trade_count"], np.array([1, 1]))


if __name__ == "__main__":
    unittest.main()
