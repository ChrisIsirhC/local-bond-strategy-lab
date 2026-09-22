from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from common.experiments import EXPERIMENT_INDEX_FILE, list_experiments


class ExperimentIndexIdentityTests(unittest.TestCase):
    def test_legacy_index_upgrades_to_timestamp_uid_without_scanning_archives(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / EXPERIMENT_INDEX_FILE
            index_path.parent.mkdir(parents=True)
            (root / "backtest_outputs" / "experiments").mkdir()
            row = {
                "运行ID": "F001",
                "策略名称": "可改名但不参与索引",
                "archive_name": "20260922_102030_123456__任意标题",
                "实验目录": "backtest_outputs/experiments/F001",
                "运行时间": "2026-09-22T10:20:30",
                "年化资本利得_BP": None,
                "盈利交易数": None,
                "已平仓交易数": None,
                "样本训练区间": None,
                "样本外最大回撤_BP": None,
            }
            index_path.write_text(json.dumps({"version": 6, "rows": [row]}, ensure_ascii=False), encoding="utf-8")

            result = list_experiments(root)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["archive_uid"], "20260922_102030_123456")
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 9)
            self.assertNotIn("archive_name", payload["rows"][0])
            self.assertNotIn("archive_key", payload["rows"][0])


if __name__ == "__main__":
    unittest.main()
