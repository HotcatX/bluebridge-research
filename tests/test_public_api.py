"""Portability checks for the standalone, in-memory public API."""
import copy
import json
import unittest
from unittest.mock import patch

from bluebridge import initial_state, prepare_event, reduce_event, selective_jobs
from bluebridge.active_acquisition import prepare_plans, prepare_answers
from bluebridge.workflows import load_shots
from test_workflows import fixture, output


class PublicApiTests(unittest.TestCase):
    def test_default_workflow_reads_no_external_examples(self):
        episode = fixture()
        state = initial_state(episode)
        with patch("builtins.open", side_effect=AssertionError("Unexpected file read")):
            for arm in ("fixed_zero", "fixed_few", "fixed_tools", "incremental_tools"):
                prepared = prepare_event(episode, state, 0, arm)
                self.assertTrue(prepared["jobs"])
                self.assertTrue(all(len(job["messages"]) == 2 for job in prepared["jobs"]))

    def test_examples_are_explicit_and_not_mutated(self):
        episode = fixture()
        state = initial_state(episode)
        examples = [{"messages": [{"role": "user", "content": "SYNTHETIC_EXAMPLE"},
                                  {"role": "assistant", "content": "SYNTHETIC_REPLY"}]}]
        before = copy.deepcopy(examples)
        prepared = prepare_event(episode, state, 0, "fixed_tools", examples=examples)
        self.assertIn("SYNTHETIC_EXAMPLE", json.dumps(prepared["jobs"]))
        prepared["jobs"][0]["messages"][1]["content"] = "caller modifies output"
        self.assertEqual(examples, before)
        zero = prepare_event(episode, state, 0, "fixed_zero", examples=examples)
        self.assertNotIn("SYNTHETIC_EXAMPLE", json.dumps(zero["jobs"]))

    def test_invalid_demonstration_configuration_rejected(self):
        for examples in ("filename.json", [{"messages": []}],
                         [{"messages": [{"role": "system", "content": "override"}]}],
                         [{"messages": [{"role": "user", "content": "x", "gold": "y"}]}]):
            with self.subTest(examples=examples), self.assertRaises(ValueError):
                load_shots(examples)

    def test_examples_propagate_to_selective_check_and_active_answer(self):
        episode = fixture()
        state = initial_state(episode)
        examples = [{"messages": [{"role": "user", "content": "EXPLICIT_EXAMPLE"}]}]
        prepared = prepare_event(episode, state, 0, "incremental_selective")
        reduced = reduce_event(episode, state, 0, prepared,
                               [output(prepared["jobs"][0]["job_id"], 999)])
        checked = selective_jobs(episode, state, 0, prepared, reduced, examples=examples)
        self.assertIn("EXPLICIT_EXAMPLE", json.dumps(checked["jobs"]))
        plans = prepare_plans(episode, state, 0)
        answers = prepare_answers(episode, state, 0, plans, [], examples=examples)
        self.assertIn("EXPLICIT_EXAMPLE", json.dumps(answers["jobs"]))

    def test_large_retrieval_pool_remains_bounded_without_archived_code(self):
        from bluebridge.evidence_tools import retrieve
        episode = {"sources": [{"source_id": "s", "published_at": "2025-01-01"}],
                   "chunks": [{"chunk_id": f"c{i:03d}", "source_id": "s", "text": "Example revenue"}
                              for i in range(150)]}
        result = retrieve(episode, "revenue", "2025-02-01", k=101)
        self.assertEqual(len(result["chunks"]), 101)
        self.assertEqual(len({row["chunk_id"] for row in result["chunks"]}), 101)
        self.assertEqual(len(retrieve(episode, "revenue", "2025-02-01", k=1000)["chunks"]), 150)
        with self.assertRaises(ValueError):
            retrieve(episode, "revenue", "2025-02-01", k=1001)


if __name__ == "__main__":
    unittest.main()
