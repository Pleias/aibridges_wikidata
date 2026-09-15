"""summary.py reads scores.json and never re-scores."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

import summary  # noqa: E402


class SummaryTest(unittest.TestCase):
    def batch(self, rows, manifest=None):
        directory = Path(tempfile.mkdtemp())
        (directory / "scores.json").write_text(json.dumps(rows))
        if manifest is not None:
            (directory / "batch.json").write_text(json.dumps(manifest))
        return directory

    def test_totals_and_patterns(self):
        rows = [
            {"task": "LT_X_001", "exact": True, "content_exact": True,
             "status": "final", "iterations": 4, "reads": 10,
             "total_tokens": 100},
            {"task": "SST_Y_001", "exact": False, "content_exact": True,
             "status": "final", "iterations": 6, "reads": 20,
             "total_tokens": 300},
            {"task": "SST_Y_002", "exact": False, "content_exact": False,
             "status": "max_iters", "iterations": 10, "reads": 5,
             "total_tokens": None},
        ]
        report = summary.summarize(self.batch(rows, {"model": "m"}))
        self.assertEqual(report["overall"]["exact"], 1)
        self.assertEqual(report["overall"]["content_exact"], 2)
        self.assertEqual(report["overall"]["reached_final"], 2)
        self.assertEqual(report["overall"]["max_iters"], 1)
        self.assertEqual(report["overall"]["turns"], 20)
        self.assertEqual(report["overall"]["total_tokens"], 400)
        self.assertEqual(report["by_pattern"]["SST"]["tasks"], 2)
        self.assertNotIn("PST", report["by_pattern"])

    def test_a_batch_without_a_manifest_still_summarizes(self):
        rows = [{"task": "HARD_01", "exact": True, "content_exact": True,
                 "status": "final", "iterations": 3, "reads": 1,
                 "total_tokens": 1}]
        report = summary.summarize(self.batch(rows))
        self.assertIsNone(report["model"])
        self.assertEqual(report["by_pattern"]["HARD"]["exact"], 1)


if __name__ == "__main__":
    unittest.main()
