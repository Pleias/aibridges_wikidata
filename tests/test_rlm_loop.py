from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rlm_loop


class RlmLoopTest(unittest.TestCase):
    def task(self, **updates) -> dict:
        task = {
            "id": "test_task",
            "question": "What is recorded?",
            "taxonomy": {"category": "Q1", "type": "test"},
            "allowed_functions": ["claims", "references"],
            "limits": {"turns": 4},
            "answer": {
                "schema": "canonical",
                "format": "Call FINAL with JSON.",
                "expected": {"property": "P31", "values": ["Q5"]},
            },
        }
        task.update(updates)
        return task

    def test_task_file_is_the_single_input_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(self.task()))
            self.assertEqual(rlm_loop.load_task(path)["id"], "test_task")

            path.write_text(json.dumps({"id": "bad", "question": "missing tools"}))
            with self.assertRaisesRegex(ValueError, "allowed_functions"):
                rlm_loop.load_task(path)

    def test_single_entity_prompt_contains_only_relevant_guidance(self):
        prompt = rlm_loop.build_prompt(self.task())
        self.assertIn("claims(qid", prompt)
        self.assertIn("llm_query", prompt)
        self.assertNotIn("Graph navigation", prompt)
        self.assertNotIn("neighbors(qid", prompt)

    def test_navigation_guidance_is_opt_in(self):
        prompt = rlm_loop.build_prompt(self.task(
            allowed_functions=["edges"],
            limits={"turns": 8},
        ))
        self.assertIn("Graph navigation", prompt)

    def test_only_turns_may_be_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(self.task(
                limits={"reads": 3, "turns": 4},
            )))
            with self.assertRaisesRegex(ValueError, "only the turns limit"):
                rlm_loop.load_task(path)

    def test_repl_state_persists_and_final_is_structured(self):
        namespace = {"FINAL": rlm_loop.final}
        feedback, answer = rlm_loop.execute_code("x = 40\nprint(x)", namespace)
        self.assertIn("40", feedback)
        self.assertIsNone(answer)

        _, answer = rlm_loop.execute_code(
            'FINAL({"value": x + 2})', namespace
        )
        self.assertEqual(answer, '{"value": 42}')

    def test_llm_query_is_always_available_without_a_task_quota(self):
        class Env:
            read_log = []

            def repl_namespace(self, allowed):
                return {}

            def seen_ids(self):
                return set()

            def save_log(self, path):
                path.write_text("[]")

        responses = iter([
            "```python\n"
            "a = llm_query('first payload', 'classify it')\n"
            "b = llm_query('second payload', 'classify it')\n"
            "FINAL({'values': [a, b]})\n```",
            "first result",
            "second result",
        ])
        state = {
            "answer": None,
            "status": "max_iters",
            "iteration": 0,
            "sub_queries": 0,
            "sub_query_log": [],
            "shape_rejections": 0,
        }
        task = self.task(
            allowed_functions=[],
            limits={"turns": 1},
            answer={
                "schema": "test",
                "format": "Call FINAL with JSON.",
                "expected": {"values": []},
            },
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rlm_loop, "chat", side_effect=lambda *a, **k: next(responses)):
            rlm_loop.run_loop(
                object(), "model", task, Env(), Path(directory), [], state
            )
        self.assertEqual(state["sub_queries"], 2)
        self.assertEqual(
            [(entry["text"], entry["response"])
             for entry in state["sub_query_log"]],
            [("first payload", "first result"),
             ("second payload", "second result")],
        )
        self.assertEqual(state["status"], "final")
        self.assertEqual(
            json.loads(state["answer"]),
            {"values": ["first result", "second result"]},
        )

    def test_a_plan_cell_then_a_code_cell_is_not_a_valid_turn(self):
        # The protocol is one cell. Merging them let a model narrate a
        # computation across cells; reasoning belongs in prose before the block.
        content = """Plan.
```python
x = 40
```
Then finish.
```python
x += 2
FINAL({"value": x})
```
"""
        self.assertIsInstance(rlm_loop.extract_code(content), list)

    def test_fenced_fabricated_output_is_not_executed(self):
        # A model can fence its own invented stdout as Python. It does not
        # parse, so it is dropped and the message still counts as one cell.
        content = """```python
x = 40
FINAL({"value": x + 2})
```
Federated results:
```python
Processed 40 entities.
```"""
        code = rlm_loop.extract_code(content)
        self.assertIsInstance(code, str)
        self.assertNotIn("Processed", code)
        _, answer = rlm_loop.execute_code(code, {"FINAL": rlm_loop.final})
        self.assertEqual(answer, '{"value": 42}')

    def test_literal_final_after_real_cell_is_treated_as_an_artifact(self):
        content = """```python
result = {"value": 42}
print(result)
```
```python
FINAL({"value": 999})
```"""
        code = rlm_loop.extract_code(content)
        self.assertNotIn("999", code)
        namespace = {"FINAL": rlm_loop.final}
        feedback, answer = rlm_loop.execute_code(code, namespace)
        self.assertIn("42", feedback)
        self.assertIsNone(answer)
        self.assertEqual(namespace["result"], {"value": 42})

    def test_single_literal_final_remains_valid(self):
        code = rlm_loop.extract_code('```python\nFINAL({"value": 42})\n```')
        _, answer = rlm_loop.execute_code(code, {"FINAL": rlm_loop.final})
        self.assertEqual(answer, '{"value": 42}')

    def test_retry_policy_rejects_client_contract_errors(self):
        class Error(Exception):
            def __init__(self, status_code, message="error"):
                super().__init__(message)
                self.status_code = status_code

        self.assertFalse(rlm_loop.retryable_chat_error(Error(404)))
        self.assertTrue(rlm_loop.retryable_chat_error(Error(429)))
        self.assertTrue(rlm_loop.retryable_chat_error(Error(503)))
        self.assertTrue(rlm_loop.retryable_chat_error(
            Error(400, "ContentPolicyViolation")
        ))


class GroundingTest(unittest.TestCase):
    """A FINAL may only name identifiers the run read or the question gave."""

    class Env:
        def __init__(self, seen):
            self._seen = set(seen)

        def seen_ids(self):
            return self._seen

    def test_fabricated_identifiers_are_rejected(self):
        # HARD_01: the model invented Q21302873 by decrementing a QID from the
        # question and never called the environment at all.
        unread = rlm_loop.ungrounded_identifiers(
            '{"families": {"Q21302876": "Q21302873"}}',
            "report the family of Saturnus harpe (Q21302876)",
            self.Env(set()))
        self.assertEqual(unread, ["Q21302873"])

    def test_identifiers_from_the_question_are_grounded(self):
        self.assertEqual(rlm_loop.ungrounded_identifiers(
            '{"anchor": "Q42"}', "start from Q42", self.Env(set())), [])

    def test_a_property_recalled_from_memory_is_not_grounded(self):
        # A relation-discovery task is pointless if P22 can be asserted from
        # pre-training; the read log records every property a call returned.
        self.assertEqual(rlm_loop.ungrounded_identifiers(
            '{"relation": "P22"}', "follow the parent links", self.Env(set())),
            ["P22"])
        self.assertEqual(rlm_loop.ungrounded_identifiers(
            '{"relation": "P22"}', "follow the parent links",
            self.Env({"P22"})), [])

    def test_a_cell_may_not_name_identifiers_it_has_not_read(self):
        # The genealogy run's first cell was `if pid in ['P22','P25']`, written
        # before it had read anything: the premise was recalled, not observed.
        code = "parents = [e for e in edges(q)['out'] if e[0] in ['P22','P25']]"
        self.assertEqual(
            rlm_loop.recalled_identifiers(code, "who are the parents",
                                          self.Env(set())),
            ["P22", "P25"])
        self.assertEqual(
            rlm_loop.recalled_identifiers(code, "who are the parents",
                                          self.Env({"P22", "P25"})), [])

    def test_a_runtime_built_identifier_came_from_data(self):
        self.assertEqual(rlm_loop.recalled_identifiers(
            'target = f"Q{n}"', "find it", self.Env(set())), [])

    def test_identifiers_that_were_read_are_grounded(self):
        self.assertEqual(rlm_loop.ungrounded_identifiers(
            '{"family": "Q999"}', "start from Q42", self.Env({"Q999"})), [])

    def test_gene_symbols_are_not_parsed_as_property_ids(self):
        answer = '{"names": ["AP1S1", "ATP6V1B2", "PPP2R2A"]}'
        self.assertEqual(
            rlm_loop.ungrounded_identifiers(answer, "find molecular targets",
                                              self.Env(set())),
            [])


if __name__ == "__main__":
    unittest.main()


class ProtocolTest(unittest.TestCase):
    """One executable cell per turn, and a whole-run sub-query cap."""

    def test_two_real_cells_are_rejected_not_merged(self):
        content = """```python
x = 40
```
Now the rest.
```python
x += 2
FINAL({"value": x})
```"""
        self.assertEqual(rlm_loop.extract_code(content),
                         ['x = 40\n', 'x += 2\nFINAL({"value": x})\n'])

    def test_a_fabricated_trailing_final_still_leaves_one_cell(self):
        content = """```python
result = {"value": 42}
print(result)
```
```python
FINAL({"value": 999})
```"""
        code = rlm_loop.extract_code(content)
        self.assertIsInstance(code, str)
        self.assertNotIn("999", code)

    def test_sub_query_budget_stops_one_call_per_item(self):
        class Env:
            read_log = []

            def repl_namespace(self, allowed):
                return {}

            def seen_ids(self):
                return set()

            def save_log(self, path):
                path.write_text("[]")

        calls = {"n": 0}
        cell = ("```python\n"
                "for item in range(5):\n"
                "    llm_query(str(item), 'classify')\n"
                "FINAL({'values': []})\n```")

        def fake_chat(client, messages, model, usage_state=None, stop=None):
            if messages and messages[-1].get("role") == "user" \
                    and messages[0].get("role") == "system" \
                    and "helper inside an RLM program" in messages[0]["content"]:
                calls["n"] += 1
                return "yes"
            return cell

        state = {
            "answer": None,
            "status": "max_iters",
            "iteration": 0,
            "sub_queries": 0,
            "sub_query_log": [],
            "shape_rejections": 0,
        }
        task = RlmLoopTest().task(
            allowed_functions=[],
            limits={"turns": 1},
            answer={"schema": "test", "format": "Call FINAL with JSON.",
                    "expected": {"values": []}},
        )
        messages: list[dict] = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(rlm_loop, "chat", side_effect=fake_chat), \
                patch.object(rlm_loop, "SUBQ_LIMIT", 3):
            rlm_loop.run_loop(
                object(), "model", task, Env(), Path(directory), messages, state)
        # the cap stops the cell; the model is told to batch, not left guessing
        self.assertEqual(calls["n"], 3)
        self.assertEqual(state["sub_queries"], 3)
        self.assertIn("BUDGET EXHAUSTED", messages[-1]["content"])


class CellStopTest(unittest.TestCase):
    """Generation is cut at the closing fence, so a turn cannot carry two cells."""

    def test_the_stop_sequence_does_not_match_the_opening_fence(self):
        message = 'Plan.\n```python\nx = 1\n```\nMore prose.\n'
        first = min((message.index(s) for s in rlm_loop.CELL_END
                     if s in message), default=None)
        self.assertIsNotNone(first)
        self.assertGreater(first, message.index("```python"))

    def test_the_eaten_closing_fence_is_restored(self):
        class Response:
            class Choice:
                class Message:
                    content = 'Plan.\n```python\nx = 1'
                message = Message()
            choices = [Choice()]
            usage = None

        class Client:
            class Completions:
                def create(self, **kwargs):
                    assert kwargs["stop"] == rlm_loop.CELL_END
                    return Response()
            class Chat:
                completions = None
            chat = Chat()

        client = Client()
        client.chat.completions = Client.Completions()
        content = rlm_loop.chat(client, [], "m", stop=rlm_loop.CELL_END)
        self.assertTrue(content.endswith("```"))
        self.assertEqual(rlm_loop.extract_code(content).strip(), "x = 1")


class ThinkingTest(unittest.TestCase):
    """Thinking is opt-in, and a turn it destroys is counted, not absorbed."""

    def response(self, content: str, reasoning: str | None,
                 finish_reason: str = "stop"):
        class Response:
            class Choice:
                class Message:
                    pass
                message = Message()
            choices = [Choice()]
            usage = None

        choice = Response.choices[0]
        choice.finish_reason = finish_reason
        choice.message.content = content
        choice.message.reasoning_content = reasoning
        return Response()

    def client(self, response, seen: dict):
        class Completions:
            def create(_self, **kwargs):
                seen.update(kwargs)
                return response

        class Client:
            class Chat:
                completions = None
            chat = Chat()

        client = Client()
        client.chat.completions = Completions()
        return client

    def test_defaults_reproduce_the_measured_configuration(self):
        """An unset environment must not change an archived batch's meaning."""
        self.assertFalse(rlm_loop.ENABLE_THINKING)
        self.assertIsNone(rlm_loop.REASONING_EFFORT)
        self.assertEqual(rlm_loop.MAX_TOKENS, 8192)
        self.assertEqual(rlm_loop.TEMPERATURE, 0.1)
        seen: dict = {}
        rlm_loop.chat(self.client(self.response("ok", None), seen), [], "m")
        template = seen["extra_body"]["chat_template_kwargs"]
        self.assertIs(template["enable_thinking"], False)
        self.assertNotIn("reasoning_effort", seen)

    def test_a_fence_inside_thinking_is_counted_not_absorbed(self):
        """The stop sequence can fire inside <think>, leaving no cell.

        Empty content, non-empty reasoning and finish_reason 'stop' is that
        exact signature. Without the counter the turn looks like an ordinary
        empty reply and the cause is invisible in the archive.
        """
        state = {"model_calls": 0, "successful_model_calls": 0,
                 "input_tokens": 0, "output_tokens": 0,
                 "total_tokens": 0, "cached_tokens": 0,
                 "reasoning_tokens": 0}
        response = self.response("", "drafting:\n```python\nx = 1\n```")
        content = rlm_loop.chat(self.client(response, {}), [], "m",
                                usage_state=state, stop=rlm_loop.CELL_END)
        self.assertEqual(content, "")
        self.assertEqual(state["stopped_in_thinking"], 1)
        self.assertEqual(len(state["reasoning_log"]), 1)

    def test_reasoning_stays_out_of_the_returned_turn(self):
        """A root turn's SFT sample is what the root emitted, not its thinking."""
        state = {"model_calls": 0, "successful_model_calls": 0,
                 "input_tokens": 0, "output_tokens": 0,
                 "total_tokens": 0, "cached_tokens": 0,
                 "reasoning_tokens": 0}
        response = self.response("```python\nx = 1\n```", "hidden thoughts")
        content = rlm_loop.chat(self.client(response, {}), [], "m",
                                usage_state=state)
        self.assertNotIn("hidden thoughts", content)
        self.assertEqual(state["reasoning_log"][0]["reasoning"],
                         "hidden thoughts")
        self.assertEqual(state.get("stopped_in_thinking", 0), 0)


class TurnStopTest(unittest.TestCase):
    """The stop sequence and thinking cannot both be on."""

    def test_the_stop_sequence_applies_when_thinking_is_off(self):
        with patch.object(rlm_loop, "ENABLE_THINKING", False):
            self.assertEqual(rlm_loop.turn_stop(), rlm_loop.CELL_END)

    def test_thinking_drops_the_stop_sequence(self):
        """Measured, not assumed: with CELL_END active and thinking on, a
        probe produced a usable cell in 0 of 3 turns, and 3 of 3 without it.
        The stop matches the whole stream, and the model drafts code inside
        <think>."""
        with patch.object(rlm_loop, "ENABLE_THINKING", True):
            self.assertIsNone(rlm_loop.turn_stop())


class ExtraFieldTest(unittest.TestCase):
    """vLLM's additions must be read wherever the SDK version puts them."""

    def test_a_field_only_present_in_model_extra_is_found(self):
        """Without this, a thinking model archives no reasoning: every turn
        records an empty string."""
        class ExtraOnly:
            reasoning_content = None
            model_extra = {"reasoning_content": "hidden thoughts"}

        self.assertEqual(
            rlm_loop.extra_field(ExtraOnly(), "reasoning_content", ""),
            "hidden thoughts")

    def test_a_declared_attribute_still_wins(self):
        class Declared:
            reasoning_content = "hidden thoughts"

        self.assertEqual(
            rlm_loop.extra_field(Declared(), "reasoning_content", ""),
            "hidden thoughts")

    def test_an_absent_field_returns_the_default(self):
        class Absent:
            model_extra: dict = {}

        self.assertEqual(rlm_loop.extra_field(Absent(), "nope", ""), "")


class ReasoningFieldNameTest(unittest.TestCase):
    """vLLM 0.27.1 calls it `reasoning`; other servers `reasoning_content`."""

    def test_the_vllm_0_27_name_is_read(self):
        """Reading only reasoning_content archives nothing on this server.
        The endpoint's message carries ["content", "reasoning", "role"]."""
        class Message:
            reasoning = "the thinking"
            model_extra: dict = {}

        self.assertEqual(rlm_loop.reasoning_text(Message()), "the thinking")

    def test_the_older_name_still_works(self):
        class Message:
            reasoning = None
            model_extra = {"reasoning_content": "the thinking"}

        self.assertEqual(rlm_loop.reasoning_text(Message()), "the thinking")

    def test_no_thinking_is_an_empty_string(self):
        class Message:
            model_extra: dict = {}

        self.assertEqual(rlm_loop.reasoning_text(Message()), "")


class TrailingExpressionTest(unittest.TestCase):
    """The prompt calls this a REPL, so a bare last expression must echo."""

    def run_cell(self, code, namespace=None):
        return rlm_loop.execute_code(code, namespace if namespace is not None
                                     else {})

    def test_a_bare_trailing_call_shows_its_value(self):
        """A model that writes search_entity(...) without print and gets
        '(no output)' spends its remaining turns on other spellings of the
        name rather than suspecting the call worked."""
        ns = {"search_entity": lambda *a, **k: (2, [{"qid": "Q1"}])}
        feedback, answer = self.run_cell('search_entity("x")', ns)
        self.assertIsNone(answer)
        self.assertIn("Q1", feedback)
        self.assertNotIn("(no output)", feedback)

    def test_print_still_prints_once(self):
        feedback, _ = self.run_cell('print("hello")')
        self.assertEqual(feedback.count("hello"), 1)

    def test_a_statement_ending_cell_is_unchanged(self):
        feedback, _ = self.run_cell('x = 1 + 1')
        self.assertIn("(no output)", feedback)

    def test_final_as_the_last_expression_still_terminates(self):
        ns = {"FINAL": rlm_loop.final}
        feedback, answer = self.run_cell('FINAL({"a": 1})', ns)
        self.assertEqual(answer, '{"a": 1}')

    def test_a_syntax_error_is_still_reported(self):
        feedback, answer = self.run_cell('def (')
        self.assertIsNone(answer)
        self.assertIn("Error", feedback)




class AdditiveEnvChangeTest(unittest.TestCase):
    """count_edges and edges(offset=) must be inert until a task asks.

    These are gated by `allowed_functions`, which the task file already
    controls — so the guarantee is checkable, not a promise.
    """

    def test_llm_map_is_now_on_by_default(self):
        """llm_map is part of the reference configuration.
        RLM_LLM_MAP=0 removes it."""
        self.assertTrue(rlm_loop.LLM_MAP)
        self.assertIn("llm_map", rlm_loop.render_function_docs(["edges"]))

    def test_it_can_still_be_turned_off(self):
        with patch.object(rlm_loop, "LLM_MAP", False):
            self.assertNotIn("llm_map", rlm_loop.render_function_docs(["edges"]))

    def test_no_existing_task_sees_the_new_capability(self):
        import glob
        checked = 0
        for path in glob.glob(str(ROOT / "tasks" / "*" / "*.json")):
            task = json.loads(Path(path).read_text())
            docs = rlm_loop.render_function_docs(task["allowed_functions"])
            checked += 1
            self.assertNotIn("count_edges", docs, task["id"])
            self.assertNotIn("offset", docs, task["id"])
        self.assertGreaterEqual(checked, 50)

    def test_the_capability_appears_only_when_a_task_lists_it(self):
        docs = rlm_loop.render_function_docs(["edges", "count_edges"])
        self.assertIn("count_edges", docs)
        self.assertIn("offset", docs)

    def test_count_edges_is_documented_for_the_prompt(self):
        self.assertIn("count_edges", rlm_loop.FUNCTION_DOCS)


class FinalGuardTest(unittest.TestCase):
    """FINAL may refuse what the GRAPH contradicts, never what the gold does."""

    class FakeEnv:
        def __init__(self, objection=None):
            self.objection = objection
            self.asked = []

        def stored_text_objection(self, value):
            self.asked.append(value)
            return self.objection

    def test_off_by_default_so_no_existing_run_changes(self):
        env = self.FakeEnv("some objection")
        with self.assertRaises(rlm_loop.FinalAnswer):
            rlm_loop.make_final(env)({"a": 1})
        self.assertEqual(env.asked, [], "guard consulted while disabled")

    def test_prose_is_refused_with_a_usable_message(self):
        with patch.object(rlm_loop, "FINAL_GUARD", True):
            with self.assertRaises(ValueError) as caught:
                rlm_loop.make_final(self.FakeEnv())("the answer is Berlin")
        self.assertIn("not prose", str(caught.exception))

    def test_an_ungrounded_quotation_is_refused(self):
        with patch.object(rlm_loop, "FINAL_GUARD", True):
            with self.assertRaises(ValueError) as caught:
                rlm_loop.make_final(self.FakeEnv("REJECTED: bad evidence"))(
                    {"qid": "Q1", "evidence": "invented"})
        self.assertIn("REJECTED", str(caught.exception))

    def test_a_clean_answer_still_terminates_the_run(self):
        with patch.object(rlm_loop, "FINAL_GUARD", True):
            with self.assertRaises(rlm_loop.FinalAnswer):
                rlm_loop.make_final(self.FakeEnv())({"qid": "Q1"})


class EmptySearchHintTest(unittest.TestCase):
    """An empty search says which word failed, instead of returning silence."""

    class FakeIndex:
        def parse_query(self, text, fields, conjunction_by_default=True):
            return text

    class FakeSearcher:
        def __init__(self, counts):
            self.counts = counts

        def search(self, query, depth, count=True):
            return type("R", (), {"count": self.counts.get(query, 0)})()

    def explain(self, text, counts):
        import io
        import contextlib
        from wd_graph_env import WDGraphEnv
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            WDGraphEnv._explain_empty_search(
                object(), text, self.FakeIndex(),
                self.FakeSearcher(counts), ["label"])
        return buf.getvalue()

    def test_it_names_the_word_that_matches_nothing(self):
        out = self.explain("Shenyang University Press",
                           {"Shenyang": 47, "University": 0, "Press": 0})
        self.assertIn("University", out)
        self.assertIn("Press", out)
        self.assertIn("Shenyang=47", out)

    def test_all_words_matching_individually_is_reported_differently(self):
        """Every word exists but no entity carries them all -- a different
        problem from a wrong word, and it needs a different next step."""
        out = self.explain("Chinese fauna", {"Chinese": 900, "fauna": 30})
        self.assertIn("ALL", out)

    def test_a_single_word_gets_a_spelling_hint_not_a_word_list(self):
        out = self.explain("Xyzzy", {"Xyzzy": 0})
        self.assertIn("spelling", out)

    def test_it_is_off_by_default(self):
        import wd_graph_env
        self.assertFalse(wd_graph_env.SEARCH_HINTS)


class WrappedEvidenceTest(unittest.TestCase):
    """Telling a model its text is 'not stored' is useless when it wrapped
    the stored text in prose. Five of seven grounding failures were that."""

    def env(self, stored):
        import wd_graph_env

        class Env(wd_graph_env.WDGraphEnv):
            def __init__(self):
                pass

            def descriptions(self, qid, _free=False):
                return stored
        return Env()

    def test_a_wrapped_description_is_named_as_wrapped(self):
        env = self.env({"ru": "тайваньский писатель", "en": "Chinese author"})
        message = env.stored_text_objection(
            {"qid": "Q1",
             "evidence": "Russian description: 'тайваньский писатель' (note)"})
        self.assertIn("contains the stored ru description", message)
        self.assertIn("тайваньский писатель", message)

    def test_the_longest_contained_description_is_the_one_offered(self):
        env = self.env({"en": "author", "de": "deutscher Autor und Verleger"})
        message = env.stored_text_objection(
            {"qid": "Q1", "evidence": "x deutscher Autor und Verleger y"})
        self.assertIn("deutscher Autor und Verleger", message)

    def test_genuinely_absent_text_still_lists_what_is_stored(self):
        env = self.env({"en": "author"})
        message = env.stored_text_objection(
            {"qid": "Q1", "evidence": "born in Baghdad in 1919"})
        self.assertIn("is not one of", message)
        self.assertIn("author", message)

    def test_an_exact_quotation_raises_no_objection(self):
        env = self.env({"en": "author"})
        self.assertIsNone(
            env.stored_text_objection({"qid": "Q1", "evidence": "author"}))


class LlmMapTest(unittest.TestCase):
    """llm_map batches and returns one validated verdict per key."""

    def build(self, replies):
        """map_over bound to a stub query, so no graph or server is needed."""
        calls = []

        def fake_query(text, instruction):
            calls.append((text, instruction))
            return replies[len(calls) - 1]

        def bound(items, instruction, batch=10):
            return rlm_loop.map_over(fake_query, items, instruction, batch)
        return bound, calls

    def test_it_batches_at_the_requested_size(self):
        llm_map, calls = self.build(["a\tyes\nb\tno", "c\tyes"])
        out = llm_map({"a": "x", "b": "y", "c": "z"}, "judge", batch=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(out, {"a": "yes", "b": "no", "c": "yes"})

    def test_a_dropped_key_is_reported_not_missing(self):
        """A key the helper omits must not read as a negative verdict."""
        llm_map, _ = self.build(["a\tyes"])
        out = llm_map({"a": "x", "b": "y"}, "judge", batch=10)
        self.assertEqual(out["b"], "NO ANSWER")

    def test_an_empty_input_costs_no_call(self):
        llm_map, calls = self.build([])
        self.assertEqual(llm_map({}, "judge"), {})
        self.assertEqual(calls, [])

    def test_the_instruction_demands_one_line_per_item(self):
        llm_map, calls = self.build(["a\tyes"])
        llm_map({"a": "x"}, "judge")
        self.assertIn("one line per item", calls[0][1])
