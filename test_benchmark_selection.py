from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from common.config import load_strategy_config
from common.market_data import DEFAULT_BENCHMARK_ID, load_market_data


ROOT = Path(__file__).resolve().parent


class BenchmarkSelectionTest(unittest.TestCase):
    def test_clean_curves_have_positive_yields(self) -> None:
        for filename in [
            "中债国债到期收益率_10年_2020至最新.csv",
            "地方政府债到期收益率_10年_2020至最新.csv",
        ]:
            frame = pd.read_csv(ROOT / "benchmark_data" / filename, encoding="utf-8-sig")
            self.assertFalse(frame.isna().any().any())
            self.assertTrue((pd.to_numeric(frame["到期收益率_百分比"]) > 0).all())

    def test_legacy_config_defaults_to_local_government_bond(self) -> None:
        source = json.loads(next((ROOT / "configs").glob("*.json")).read_text(encoding="utf-8"))
        source.pop("benchmark", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            config = load_strategy_config(path)
        self.assertEqual(config.benchmark_id, DEFAULT_BENCHMARK_ID)

    def test_benchmark_switch_keeps_traded_asset_unchanged(self) -> None:
        local = load_market_data(ROOT, "local_gov_10y")
        government = load_market_data(ROOT, "gov_10y")
        columns = ["date", "asset_total_return", "asset_duration_pnl"]
        pd.testing.assert_frame_equal(local[columns], government[columns])
        self.assertFalse(local["total_return"].equals(government["total_return"]))


if __name__ == "__main__":
    unittest.main()
