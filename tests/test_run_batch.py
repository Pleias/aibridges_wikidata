from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

import run_batch


class BatchCorpusTest(unittest.TestCase):
    def test_active_families_are_the_maintained_corpus(self):
        tasks = run_batch.discover_tasks(["cards", "hard", "nameonly"])

        self.assertEqual(len(tasks), 56)
        self.assertEqual(tasks, sorted(tasks))
        self.assertEqual(
            {path.parent.name for path in tasks}, {"cards", "hard", "nameonly"})

    def test_limit_is_applied_per_selected_family(self):
        tasks = run_batch.discover_tasks(["cards"], limit=3)

        self.assertEqual(len(tasks), 3)


if __name__ == "__main__":
    unittest.main()
