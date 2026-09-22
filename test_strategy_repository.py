from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from common.strategy_repository import (
    archive_uid_for_strategy_id,
    ensure_strategy_ids,
    metadata_path,
    strategy_id_for_archive,
    strategy_id_for_archive_uid,
)
from prepare_public_archive_bridge import bridge


class StrategyRepositoryTests(unittest.TestCase):
    def test_strategy_ids_are_sqlite_unique_and_immutable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            assigned = ensure_strategy_ids(root, "experiments_F", ["archive_a", "archive_b"], "F")
            self.assertEqual(assigned, {"archive_a": "F001", "archive_b": "F002"})
            self.assertEqual(strategy_id_for_archive(root, "archive_a"), "F001")
            connection = sqlite3.connect(metadata_path(root))
            try:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("UPDATE strategy_identity SET strategy_id = 'F999' WHERE strategy_id = 'F001'")
            finally:
                connection.close()
            self.assertEqual(ensure_strategy_ids(root, "experiments_F", ["archive_a"], "F"), {"archive_a": "F001"})

    def test_timestamp_archive_uid_is_the_cross_machine_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = "20260922_102030_123456__名称可变"
            second = "20260922_102031__另一个名称"
            assigned = ensure_strategy_ids(root, "experiments_F", [first, second], "F")
            self.assertEqual(assigned[first], "F001")
            self.assertEqual(archive_uid_for_strategy_id(root, "F001"), "20260922_102030_123456")
            self.assertEqual(strategy_id_for_archive_uid(root, "20260922_102030_123456"), "F001")
            # A changed title with the same timestamp UID is the same archive,
            # never a new local strategy row.
            with self.assertRaisesRegex(ValueError, "唯一归档键冲突"):
                ensure_strategy_ids(root, "experiments_F", ["20260922_102030_123456__改名后"], "F")

    def test_public_bridge_allocates_target_id_on_cross_repository_collision(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            long_name = "20260922_102030_123456__本地策略"
            ensure_strategy_ids(root, "experiments_F", [long_name], "F")
            archive = root / "backtest_outputs" / "experiments" / "F001"
            archive.mkdir(parents=True)
            (archive / "config.json").write_text("{}", encoding="utf-8")
            (archive / "run_manifest.json").write_text(
                json.dumps({"short_id": "F001", "archive_uid": "20260922_102030_123456"}), encoding="utf-8"
            )
            row = {
                "运行ID": "F001", "strategy_id": "F001", "archive_uid": "20260922_102030_123456",
                "实验目录": "backtest_outputs/experiments/F001", "storage_path": "backtest_outputs/experiments/F001",
                "运行时间": "2026-09-22T10:20:30", "策略名称": "本地策略", "运行来源": "因子研究",
                "信号频率": "日频", "比较基准": "10Y", "BP口径": "BP", "回测起始日期": "", "回测结束日期": "",
                "回测区间": "", "样本训练区间": "", "年化资本利得_BP": None, "盈利交易数": None,
                "已平仓交易数": None, "样本外最大回撤_BP": None,
            }
            index = root / "backtest_outputs" / "experiments_index.json"
            index.write_text(json.dumps({"version": 9, "rows": [row]}, ensure_ascii=False), encoding="utf-8")
            remote_row = dict(row)
            remote_row["archive_uid"] = "20260922_102031_123456"
            base = root / "remote_index.json"
            base.write_text(json.dumps({"version": 9, "rows": [remote_row]}, ensure_ascii=False), encoding="utf-8")
            target = bridge(root, "F001", base_index=base)
            self.assertEqual(target.name, "F002")
            updated = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual({item["运行ID"] for item in updated["rows"]}, {"F001", "F002"})
            self.assertEqual(
                json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))["short_id"], "F002"
            )

    def test_public_bridge_keeps_custom_name_and_uid_presentation_state(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_name = "20260922_102030_123456__原始策略"
            archive_uid = "20260922_102030_123456"
            ensure_strategy_ids(root, "experiments_F", [archive_name], "F")
            archive = root / "backtest_outputs" / "experiments" / "F001"
            archive.mkdir(parents=True)
            (archive / "config.json").write_text("{}", encoding="utf-8")
            (archive / "run_manifest.json").write_text(
                json.dumps({"short_id": "F001", "archive_uid": archive_uid, "策略名称": "原始策略"}),
                encoding="utf-8",
            )
            row = {
                "运行ID": "F001", "strategy_id": "F001", "archive_uid": archive_uid,
                "实验目录": "backtest_outputs/experiments/F001", "storage_path": "backtest_outputs/experiments/F001",
                "策略名称": "原始策略", "运行时间": "", "运行来源": "因子研究", "信号频率": "日频",
                "比较基准": "10Y", "BP口径": "BP", "回测起始日期": "", "回测结束日期": "",
                "回测区间": "", "样本训练区间": "", "年化资本利得_BP": None,
                "盈利交易数": None, "已平仓交易数": None, "样本外最大回撤_BP": None,
            }
            output = root / "backtest_outputs"
            (output / "experiments_index.json").write_text(json.dumps({"version": 9, "rows": [row]}, ensure_ascii=False), encoding="utf-8")
            (output / "experiment_display_names.json").write_text(
                json.dumps({"version": 1, "names": {archive_name: "人工修改名称"}}, ensure_ascii=False), encoding="utf-8"
            )
            (output / "experiment_notes.json").write_text(
                json.dumps({"version": 1, "notes": {archive_name: "保留备注"}}, ensure_ascii=False), encoding="utf-8"
            )
            (output / "experiment_favorites.json").write_text(
                json.dumps({"experiments": [archive_name]}, ensure_ascii=False), encoding="utf-8"
            )

            bridge(root, "F001")

            manifest = json.loads((archive / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["策略名称"], "人工修改名称")
            self.assertEqual(manifest["原始策略名称"], "原始策略")
            self.assertEqual(
                json.loads((output / "experiment_display_names.json").read_text(encoding="utf-8"))["names"][archive_uid],
                "人工修改名称",
            )
            self.assertEqual(
                json.loads((output / "experiment_notes.json").read_text(encoding="utf-8"))["notes"][archive_uid],
                "保留备注",
            )
            self.assertEqual(
                json.loads((output / "experiment_favorites.json").read_text(encoding="utf-8"))["experiments"],
                [archive_uid],
            )


if __name__ == "__main__":
    unittest.main()
