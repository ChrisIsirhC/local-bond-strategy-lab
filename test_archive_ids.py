from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from common.archive_ids import ensure_short_archive_ids, short_archive_id


class ArchiveIdTests(unittest.TestCase):
    def test_ids_are_short_sequential_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ensure_short_archive_ids(root, "experiments", ["20260102", "20260101"], "E")
            self.assertEqual(first, {"20260102": "E002", "20260101": "E001"})
            self.assertEqual(short_archive_id(root, "experiments", "20260101", "E"), "E001")
            second = short_archive_id(root, "experiments", "20260103", "E")
            self.assertEqual(second, "E003")


if __name__ == "__main__":
    unittest.main()
