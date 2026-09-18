from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from common.threshold_research import _candidate_position_matrix, _training_module_options
from strategies.dashboard_signal_v1 import DashboardThresholds
from strategies.position_policy import DashboardPositionPolicy


class ThresholdSearchPositionTests(unittest.TestCase):
    def test_future_factor_values_do_not_change_training_grid(self):
        values = pd.DataFrame({"spread_change": [1.0, 1.1, 1.2, 1.3, 900.0]})
        context = {"factor_values": values, "signal_index": np.array([0, 1, 2, 3])}
        before = _training_module_options(context, DashboardThresholds())
        values.loc[4, "spread_change"] = 90000.0
        after = _training_module_options(context, DashboardThresholds())
        self.assertEqual(before, after)

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


if __name__ == "__main__":
    unittest.main()
