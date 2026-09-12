import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessfly.dataset import inventory, select_files, sha256_file, write_manifest


class DatasetTests(unittest.TestCase):
    def test_inventory_contains_only_required_first_stage_files(self):
        self.assertEqual(
            {item["key"] for item in inventory()},
            {"annotations", "transmitters", "weights"},
        )

    def test_all_selects_complete_inventory(self):
        self.assertEqual(len(select_files(("all",))), 3)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            select_files(("synapse-points",))

    def test_sha256_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(b"chessfly")
            self.assertEqual(
                sha256_file(path),
                "8f49d40398ce935c0471ebef6d6a664ae906928f901b408b714158687c494f9d",
            )

    def test_manifest_merges_incremental_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(({"key": "annotations", "sha256": "a"},), path)
            write_manifest(({"key": "weights", "sha256": "b"},), path)
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["key"] for item in payload["files"]],
                ["annotations", "weights"],
            )


if __name__ == "__main__":
    unittest.main()
