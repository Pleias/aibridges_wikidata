"""Graded scoring adds detail beside the exact verdict and never replaces it."""

import glob
import json
from unittest.mock import patch
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

from canonical import canonical_json          # noqa: E402
from grading import grade, row_identity       # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "bench"))


class RowIdentityTest(unittest.TestCase):
    def test_a_shared_unique_key_identifies_rows(self):
        rows = [{"qid": "Q1", "n": 1}, {"qid": "Q2", "n": 2}]
        self.assertEqual(row_identity(rows), "qid")

    def test_a_repeated_key_cannot_identify_rows(self):
        """Two rows with the same qid are distinct; merging them would
        silently drop one and inflate recall."""
        rows = [{"qid": "Q1", "n": 1}, {"qid": "Q1", "n": 2}]
        self.assertIsNone(row_identity(rows))

    def test_a_key_missing_from_one_row_does_not_identify(self):
        rows = [{"qid": "Q1"}, {"name": "x"}]
        self.assertIsNone(row_identity(rows))

    def test_scalar_rows_have_no_identity(self):
        self.assertIsNone(row_identity(["Q1", "Q2"]))


class GradeTest(unittest.TestCase):
    def test_a_perfect_answer_scores_one(self):
        want = {"a": 1, "items": [{"qid": "Q1"}, {"qid": "Q2"}]}
        self.assertEqual(grade(want, want)["set_f1"], 1.0)
        self.assertEqual(grade(want, want)["scalars_ok"], "1/1")

    def test_a_missing_row_is_named_not_just_counted(self):
        want = {"items": [{"qid": "Q1"}, {"qid": "Q2"}, {"qid": "Q3"}]}
        got = {"items": [{"qid": "Q1"}, {"qid": "Q2"}]}
        field = grade(want, got)["fields"]["items"]
        self.assertEqual(field["missing"], ["Q3"])
        self.assertEqual(field["precision"], 1.0)
        self.assertAlmostEqual(field["recall"], 0.6667, places=3)

    def test_an_invented_row_costs_precision(self):
        want = {"items": [{"qid": "Q1"}]}
        got = {"items": [{"qid": "Q1"}, {"qid": "Q9"}]}
        field = grade(want, got)["fields"]["items"]
        self.assertEqual(field["extra"], ["Q9"])
        self.assertEqual(field["recall"], 1.0)
        self.assertEqual(field["precision"], 0.5)

    def test_the_right_entity_with_wrong_evidence_still_matches(self):
        """A wrong description is not a missing entity. Blending the two would
        hide exactly the cross-linguistic signal these tasks exist to test."""
        want = {"m": [{"qid": "Q1", "evidence": [{"lang": "de", "text": "a"}]}]}
        got = {"m": [{"qid": "Q1", "evidence": [{"lang": "en", "text": "b"}]}]}
        field = grade(want, got)["fields"]["m"]
        self.assertEqual(field["f1"], 1.0)
        self.assertEqual(field["detail_ok"], "0/1")

    def test_rows_without_an_identity_fall_back_to_overlap(self):
        want = {"x": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
        got = {"x": [{"a": 1, "b": 2}]}
        field = grade(want, got)["fields"]["x"]
        self.assertEqual(field["identity"], "positional")
        self.assertAlmostEqual(field["f1"], 2 / 3, places=3)

    def test_a_wrong_scalar_reports_both_values(self):
        field = grade({"n": 114}, {"n": 1})["fields"]["n"]
        self.assertFalse(field["ok"])
        self.assertEqual((field["expected"], field["got"]), (114, 1))

    def test_an_absent_field_is_not_a_silent_pass(self):
        result = grade({"n": 1, "items": [{"qid": "Q1"}]}, {})
        self.assertFalse(result["fields"]["n"]["ok"])
        self.assertEqual(result["fields"]["items"]["f1"], 0.0)

    def test_set_f1_is_none_when_the_schema_has_no_list(self):
        self.assertIsNone(grade({"a": 1}, {"a": 1})["set_f1"])


class AgreementTest(unittest.TestCase):
    """The graded view must agree with the exact verdict at the extremes,
    across every real task schema — not just constructed examples."""

    def test_every_corpus_schema_grades_itself_perfectly(self):
        checked = 0
        for path in glob.glob(str(ROOT / "tasks" / "*" / "*.json")):
            expected = json.loads(Path(path).read_text())["answer"]["expected"]
            result = grade(expected, expected)
            checked += 1
            self.assertEqual(canonical_json(expected), canonical_json(expected))
            if result["set_f1"] is not None:
                self.assertEqual(result["set_f1"], 1.0, path)
            hits, total = result["scalars_ok"].split("/")
            self.assertEqual(hits, total, path)
        self.assertGreaterEqual(checked, 50)

    def test_every_corpus_schema_has_a_resolvable_row_identity(self):
        """Reported, not enforced: a field that falls back to positional
        overlap still scores, it just cannot name what went missing."""
        positional = []
        for path in glob.glob(str(ROOT / "tasks" / "*" / "*.json")):
            expected = json.loads(Path(path).read_text())["answer"]["expected"]
            if not isinstance(expected, dict):
                continue
            for name, value in expected.items():
                if isinstance(value, list) and value and row_identity(value) is None:
                    positional.append(f"{Path(path).stem}.{name}")
        # Measured at 5 of 32 list fields when this was written: 4 scalar-row
        # fields and one dict-row field with no shared unique key.
        self.assertLessEqual(len(positional), 8, positional)


if __name__ == "__main__":
    unittest.main()


class SweepCoverageTest(unittest.TestCase):
    """Enumeration is measured from the trajectory's own reads, not asked for.

    Nothing in coverage.py reads a task file. A question nobody has seen is
    scored by the same rule as one of the 56.
    """

    def setUp(self):
        from coverage import sweeps
        self.sweeps = sweeps

    def test_pages_of_one_slice_are_one_sweep(self):
        """A model that pages correctly must not be scored as partial sweeps."""
        log = [{"fn": "edges", "args": ["Q1", "in", "P108", 200],
                "ids": ["P108", "Q10", "Q11"]},
               {"fn": "edges", "args": ["Q1", "in", "P108", 200, 200],
                "ids": ["P108", "Q12"]}]
        got = self.sweeps(log)
        self.assertEqual(list(got), [("Q1", "in", "P108")])
        self.assertEqual(got[("Q1", "in", "P108")], {"Q10", "Q11", "Q12"})

    def test_the_anchor_and_predicate_are_not_counted_as_results(self):
        log = [{"fn": "edges", "args": ["Q1", "in", "P108", 200],
                "ids": ["P108", "Q1", "Q10"]}]
        self.assertEqual(self.sweeps(log)[("Q1", "in", "P108")], {"Q10"})

    def test_non_expanding_calls_are_ignored(self):
        log = [{"fn": "name", "args": ["Q1"], "ids": ["Q1"]},
               {"fn": "descriptions", "args": ["Q2"], "ids": ["Q2"]}]
        self.assertEqual(self.sweeps(log), {})

    def test_different_predicates_are_different_slices(self):
        log = [{"fn": "edges", "args": ["Q1", "in", "P108", 200], "ids": ["Q10"]},
               {"fn": "edges", "args": ["Q1", "out", "P27", 200], "ids": ["Q20"]}]
        self.assertEqual(len(self.sweeps(log)), 2)


class ContentExactTest(unittest.TestCase):
    """`content_exact` ignores what the search claims about itself.

    The field list is DECLARED by the task, never known to the harness, so a
    generated task carries its own and no hand-written category is needed.
    """

    def test_a_wrong_bookkeeping_field_does_not_fail_the_content(self):
        want = {"reviewed": 114, "matches": [{"qid": "Q1"}]}
        got = {"reviewed": 19, "matches": [{"qid": "Q1"}]}
        self.assertFalse(grade(want, got)["content_exact"])
        self.assertTrue(grade(want, got, ["reviewed"])["content_exact"])

    def test_a_wrong_answer_still_fails_with_bookkeeping_excluded(self):
        want = {"reviewed": 114, "matches": [{"qid": "Q1"}, {"qid": "Q2"}]}
        got = {"reviewed": 114, "matches": [{"qid": "Q1"}]}
        self.assertFalse(grade(want, got, ["reviewed"])["content_exact"])

    def test_declaring_nothing_leaves_the_verdict_unchanged(self):
        want = {"reviewed": 114, "matches": [{"qid": "Q1"}]}
        self.assertFalse(grade(want, dict(want, reviewed=19))["content_exact"])

    def test_every_declaring_task_names_a_field_it_actually_has(self):
        import glob
        declared = 0
        for path in glob.glob(str(ROOT / "tasks" / "*" / "*.json")):
            task = json.loads(Path(path).read_text())
            book = task["answer"].get("bookkeeping")
            if not book:
                continue
            declared += 1
            for name in book:
                self.assertIn(name, task["answer"]["expected"], task["id"])
        self.assertGreaterEqual(declared, 15)


class GroundingTest(unittest.TestCase):
    """A quotation is grounded only if the trajectory actually saw that text."""

    def setUp(self):
        from grounding import claims, seen_text
        self.claims, self.seen_text = claims, seen_text

    def test_edges_shows_labels_and_cannot_ground_a_description(self):
        """edges() returns each neighbour's LABEL. A model that only expanded
        a hub has not seen any description, so a quoted one came from
        somewhere else."""
        log = [{"fn": "edges", "args": ["Q9", "in", "P108", 200], "ids": ["Q2"]}]
        self.assertEqual(self.seen_text(log), {})

    def test_descriptions_exposes_every_language(self):
        log = [{"fn": "descriptions", "args": ["Q1"], "ids": ["Q1"]}]
        self.assertEqual(self.seen_text(log)["Q1"],
                         {"en", "fr", "de", "zh", "ar", "ru"})

    def test_a_single_language_call_grounds_only_that_language(self):
        log = [{"fn": "description", "args": ["Q1", "ar"], "ids": ["Q1"]}]
        self.assertEqual(self.seen_text(log)["Q1"], {"ar"})

    def test_search_results_carry_an_english_description(self):
        log = [{"fn": "search_entity", "args": ["berlin", 5], "ids": ["Q64"]}]
        self.assertEqual(self.seen_text(log)["Q64"], {"en"})

    def test_a_quotation_inherits_the_entity_it_is_nested_under(self):
        answer = {"matches": [{"qid": "Q1",
                               "evidence": [{"lang": "de", "text": "ein Autor"}]}]}
        self.assertEqual(list(self.claims(answer)), [("Q1", "de", "ein Autor")])

    def test_a_quotation_beside_its_entity_is_also_attributed(self):
        answer = {"matches": [{"qid": "Q2", "language": "ru",
                               "evidence": "писатель"}]}
        self.assertEqual(list(self.claims(answer)), [("Q2", "ru", "писатель")])


class NestedEntityAttributionTest(unittest.TestCase):
    """A quotation beside a nested subject is still attributable."""

    def setUp(self):
        from grounding import claims
        self.claims = claims

    def test_one_nested_entity_is_inherited(self):
        answer = {"entity": {"qid": "Q62604500", "name": "x"},
                  "language": "ar", "evidence": "قمة عربية"}
        self.assertEqual(list(self.claims(answer)),
                         [("Q62604500", "ar", "قمة عربية")])

    def test_two_nested_entities_are_ambiguous_and_not_guessed(self):
        """Attributing to the wrong entity is worse than declining to check."""
        answer = {"anchor_a": {"qid": "Q1"}, "anchor_b": {"qid": "Q2"},
                  "evidence": "text"}
        self.assertEqual(list(self.claims(answer)), [(None, None, "text")])

    def test_a_direct_qid_still_wins_over_a_nested_one(self):
        answer = {"qid": "Q9", "entity": {"qid": "Q1"}, "evidence": "text"}
        self.assertEqual(list(self.claims(answer))[0][0], "Q9")


class TypographyFoldingTest(unittest.TestCase):
    """A curly apostrophe and a straight one are the same character here."""

    def setUp(self):
        from canonical import canonical
        self.canonical = canonical

    def test_the_apostrophe_that_cost_a_task(self):
        stored = "This book explores over a century of Germany’s relations"
        model = "This book explores over a century of Germany's relations"
        self.assertEqual(self.canonical(stored), self.canonical(model))

    def test_quotes_dashes_and_hard_spaces_fold(self):
        self.assertEqual(self.canonical("“a” – b c"),
                         self.canonical('"a" - b c'))

    def test_accents_are_NOT_folded(self):
        """Six languages share this corpus; folding accents would merge real
        values, which is a worse error than the one being fixed."""
        self.assertNotEqual(self.canonical("Schroder"),
                            self.canonical("Schröder"))

    def test_case_is_not_folded_either(self):
        self.assertNotEqual(self.canonical("berlin"), self.canonical("Berlin"))

    def test_a_qid_is_still_normalised_by_case(self):
        self.assertEqual(self.canonical("q42"), "Q42")


class ManifestRecordsConfigTest(unittest.TestCase):
    """A reference run must carry its own configuration.

    batch.json held the git revision and worker count but not temperature,
    effort or the sub-query budget. Two variables moved between the 34/56
    baseline and the 8-worker run and the manifest could not say so, which
    is how that comparison was lost.
    """

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts" / "bench"))
        import run_batch
        self.settings = run_batch.generation_settings

    def test_it_captures_every_rlm_switch(self):
        with patch.dict(os.environ, {"RLM_TEMPERATURE": "0",
                                     "RLM_LLM_MAP": "1"}, clear=False):
            got = self.settings()
        self.assertEqual(got.get("RLM_TEMPERATURE"), "0")
        self.assertEqual(got.get("RLM_LLM_MAP"), "1")

    def test_it_captures_the_subquery_budget_and_server_batching(self):
        with patch.dict(os.environ, {"SUBQ_LIMIT": "80",
                                     "MAX_NUM_SEQS": "1"}, clear=False):
            got = self.settings()
        self.assertEqual(got.get("SUBQ_LIMIT"), "80")
        self.assertEqual(got.get("MAX_NUM_SEQS"), "1")

    def test_an_unrelated_variable_is_not_recorded(self):
        with patch.dict(os.environ, {"HOME": "/tmp/x"}, clear=False):
            self.assertNotIn("HOME", self.settings())

    def test_a_switch_added_later_needs_no_change_here(self):
        """Read from the environment, not from a hardcoded list."""
        with patch.dict(os.environ, {"RLM_SOMETHING_NEW": "7"}, clear=False):
            self.assertEqual(self.settings().get("RLM_SOMETHING_NEW"), "7")
