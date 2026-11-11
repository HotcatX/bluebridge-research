"""Whole-source map/reduce tests using newly authored public synthetic data."""
import copy
import json
import unittest
from unittest.mock import patch

from bluebridge import full_refresh as refresh
from bluebridge.workflows import initial_state, reduce_event


def fixture():
    table = """EXAMPLE FY2024 financial results
GAAP, USD in millions
Metric | FY2024 | FY2023
Revenue | 100 | 90
Footnote: segment figures exclude intersegment sales.
"""
    return {"episode_id": "synthetic", "issuer": "EXAMPLE", "initial_as_of": "2025-01-01",
            "sources": [{"source_id": "old", "published_at": "2025-01-01", "url": "https://example.test/old"},
                        {"source_id": "new", "published_at": "2025-04-01", "url": "https://example.test/new"},
                        {"source_id": "future", "published_at": "2025-07-01", "url": "https://example.test/future"}],
            "chunks": [{"chunk_id": "old_table", "source_id": "old", "text": table, "locator": "Table A"},
                       {"chunk_id": "old_guidance", "source_id": "old", "text": "Original Q1 revenue guidance is USD 105-115 million.", "locator": "Outlook"},
                       {"chunk_id": "new_actual", "source_id": "new", "text": "EXAMPLE Q1 actual revenue is USD 110 million; segment A revenue is USD 30 million.", "locator": "Results"},
                       {"chunk_id": "new_risk", "source_id": "new", "text": "Demand timing remains uncertain. " + "Context for source coverage. " * 20, "locator": "Risk"},
                       {"chunk_id": "future_value", "source_id": "future", "text": "FUTURE_SENTINEL results were USD 120 million.", "locator": "Future"}],
            "initial_claims": [{"claim_id": cid, "instruction": instruction, "text": "Own old note.", "citations": ["old_table"],
                                "metric_spec": {"temporal_policy": policy, "metrics": [{"metric": "revenue", "unit": "USD million", "basis": "GAAP", "scope": scope}]}}
                               for cid, instruction, policy, scope in [
                                   ("historical", "Preserve fixed historical FY2024 revenue.", "fixed_historical", "company"),
                                   ("rolling", "Update latest reported quarter revenue.", "latest_reported_quarter", "company"),
                                   ("guidance", "Compare actual revenue to original guidance.", "fixed_guidance_realization", "company"),
                                   ("segment", "Update segment A revenue only.", "latest_reported_quarter", "segment A")]],
            "events": [{"event_id": "e1", "as_of": "2025-04-01", "new_source_ids": ["new"]},
                       {"event_id": "e2", "as_of": "2025-07-01", "new_source_ids": ["future"]}]}


def outputs(prepared, claim_ids=None, *, empty=False, with_quotes=False):
    result = []
    for job in prepared["jobs"]:
        payload = json.loads(job["messages"][-1]["content"])
        ids = claim_ids or [claim["claim_id"] for claim in payload["claims"]]
        spans = [] if empty else [{"chunk_id": chunk["chunk_id"], "claim_ids": ids,
                                  **({"quote": chunk["text"]} if with_quotes else {})} for chunk in payload["evidence"]]
        result.append({"job_id": job["job_id"], "text": json.dumps({"evidence": spans}),
                       "input_tokens": 200, "output_tokens": 40, "elapsed_seconds": 1.5})
    return result


def small_messages(episode, claim, previous, event_index, evidence, arm, **kwargs):
    return [{"role": "system", "content": "Synthetic test answer instruction."},
            {"role": "user", "content": json.dumps({"claim_id": claim["claim_id"], "previous_state": previous,
                                                       "evidence": evidence, "arm": arm})}]


class SourceMapTests(unittest.TestCase):
    def test_intact_complete_source_coverage_and_source_first_batching(self):
        episode = fixture()
        prepared = refresh.prepare_maps(episode, initial_state(episode), 0)
        expected = {"old_table", "old_guidance", "new_actual", "new_risk"}
        actual = []
        for job in prepared["jobs"]:
            self.assertEqual(set(job), {"job_id", "messages", "max_new_tokens"})
            self.assertEqual(job["max_new_tokens"], 1024)
            payload = json.loads(job["messages"][-1]["content"])
            actual.extend(chunk["chunk_id"] for chunk in payload["evidence"])
            self.assertEqual({claim["claim_id"] for claim in payload["claims"]}, {"historical", "rolling", "guidance", "segment"})
            self.assertEqual(len({chunk["source_id"] for chunk in payload["evidence"]}), 1)
            self.assertLessEqual(sum(len(message["content"]) for message in job["messages"]), 14000)
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(prepared["cost"]["scheduled_source_characters"], sum(len(chunk["text"]) for chunk in episode["chunks"][:-1]))
        self.assertTrue(prepared["coverage_complete_at_preparation"])
        self.assertEqual(prepared["map_version"], "id_selection_v1")
        self.assertTrue(all("__id_selection_v1__" in job["job_id"] for job in prepared["jobs"]))
        self.assertNotIn('"quote":', prepared["jobs"][0]["messages"][0]["content"])


    def test_every_event_reprocesses_old_sources_and_cutoff_blocks_future(self):
        episode = fixture()
        state = initial_state(episode)
        first = refresh.prepare_maps(episode, state, 0)
        second = refresh.prepare_maps(episode, state, 1)
        self.assertEqual(len(first["coverage_receipts"]), 4)
        self.assertEqual(len(second["coverage_receipts"]), 5)
        self.assertNotIn("FUTURE_SENTINEL", json.dumps(first))
        self.assertIn("FUTURE_SENTINEL", json.dumps(second["jobs"]))
        episode["sources"][2]["published_at"] = "2025-08-01"
        self.assertEqual(len(refresh.prepare_maps(episode, state, 1)["coverage_receipts"]), 4)

    def test_no_hidden_fields_input_mutation_or_state_substitution(self):
        episode = fixture()
        state = initial_state(episode)
        state["rolling"]["text"] = "Own wrong previous number 777."
        episode["reference"] = {"answer": "PRIVATE_SENTINEL"}
        episode["chunks"][0]["expected_claim_ids"] = ["PRIVATE_SENTINEL"]
        episode["initial_claims"][0]["gold"] = "PRIVATE_SENTINEL"
        state["rolling"]["private_field"] = "PRIVATE_SENTINEL"
        before = copy.deepcopy((episode, state))
        prepared = refresh.prepare_maps(episode, state, 0)
        rendered = json.dumps(prepared["jobs"])
        self.assertNotIn("PRIVATE_SENTINEL", rendered)
        self.assertIn("Own wrong previous number 777", rendered)
        self.assertEqual((episode, state), before)
        prepared["jobs"][0]["messages"][1]["content"] = "changed"
        self.assertEqual((episode, state), before)

    def test_oversized_indivisible_chunks_are_receipted_without_trim(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0, char_cap=10)
        self.assertEqual(prepared["jobs"], [])
        self.assertEqual(len(prepared["errors"]), 4)
        self.assertTrue(all(row["code"] == "indivisible_chunk_exceeds_map_budget" for row in prepared["errors"]))
        self.assertTrue(all(row["status"] == "blocked_budget" for row in prepared["coverage_receipts"]))
        self.assertEqual(prepared["cost"]["scheduled_source_characters"], 0)
        self.assertGreater(prepared["cost"]["eligible_source_characters"], 0)

    def test_valid_map_keeps_originals_provenance_and_measured_usage(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        before = copy.deepcopy((episode, state, prepared))
        mapped = refresh.apply_maps(episode, state, 0, prepared, outputs(prepared))
        self.assertTrue(mapped["source_coverage_complete"], mapped)
        self.assertEqual(mapped["cost"]["accepted_spans"], 4)
        self.assertEqual(mapped["cost"]["processed_valid_source_characters"], prepared["cost"]["eligible_source_characters"])
        self.assertEqual(mapped["records"][0]["runner_usage"]["elapsed_seconds"], 1.5)
        self.assertTrue(all(row["source_sha256"] and row["map_job_id"] for row in mapped["evidence"]))
        self.assertFalse(mapped["semantic_relevance_checked"])
        self.assertTrue(all("quote" not in row for row in mapped["evidence"]))
        self.assertEqual((episode, state, prepared), before)

    def test_unknown_claim_future_and_other_input_chunks_fail(self):
        for attack in ("claim", "future", "other_map_input"):
            with self.subTest(attack=attack):
                episode = fixture()
                state = initial_state(episode)
                prepared = refresh.prepare_maps(episode, state, 0)
                response = outputs(prepared)
                payload = json.loads(response[0]["text"])
                if attack == "claim":
                    payload["evidence"][0]["claim_ids"] = ["unknown_claim"]
                elif attack == "future":
                    payload["evidence"][0]["chunk_id"] = "future_value"
                    payload["evidence"][0]["quote"] = episode["chunks"][-1]["text"]
                else:
                    payload["evidence"][0]["chunk_id"] = "new_actual"
                response[0]["text"] = json.dumps(payload)
                mapped = refresh.apply_maps(episode, state, 0, prepared, response)
                self.assertFalse(mapped["source_coverage_complete"])
                self.assertTrue(mapped["errors"])
                self.assertTrue(any(row["status"] == "map_failed_or_missing" for row in mapped["coverage_receipts"]))

    def test_optional_exact_quotes_are_validated_and_unique_whitespace_is_repaired(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        response = outputs(prepared, with_quotes=True)
        for output in response:
            payload = json.loads(output["text"])
            for selection in payload["evidence"]:
                if selection["chunk_id"] == "old_table":
                    selection["quote"] = " ".join(selection["quote"].split())
            output["text"] = json.dumps(payload)
        mapped = refresh.apply_maps(episode, state, 0, prepared, response)
        self.assertTrue(mapped["source_coverage_complete"], mapped)
        receipts = [receipt for row in mapped["records"] for receipt in row["quote_receipts"]]
        repaired = [receipt for receipt in receipts if receipt["status"] == "repaired_whitespace_only"]
        self.assertEqual(len(repaired), 1)
        self.assertIn(repaired[0]["canonical_quote"], episode["chunks"][0]["text"])
        self.assertEqual(" ".join(repaired[0]["original_quote"].split()), " ".join(repaired[0]["canonical_quote"].split()))
        self.assertTrue(any(receipt["status"] == "exact" for receipt in receipts))

    def test_bad_or_ambiguous_optional_quote_is_dropped_without_losing_valid_id_selection(self):
        for bad in ("Invented USD 999 million.", "Revenue 100", "Repeat phrase.", "", None):
            with self.subTest(bad=bad):
                episode = fixture()
                episode["chunks"][1]["text"] += " Repeat   phrase. Repeat\tphrase."
                state = initial_state(episode)
                prepared = refresh.prepare_maps(episode, state, 0)
                response = outputs(prepared)
                payload = json.loads(response[0]["text"])
                payload["evidence"][0]["quote"] = bad
                selected_id = payload["evidence"][0]["chunk_id"]
                response[0]["text"] = json.dumps(payload)
                mapped = refresh.apply_maps(episode, state, 0, prepared, response)
                self.assertTrue(mapped["source_coverage_complete"], mapped)
                self.assertTrue(mapped["warnings"])
                evidence = next(row for row in mapped["evidence"] if row["chunk_id"] == selected_id)
                self.assertNotIn("quote", evidence)
                receipt = mapped["records"][0]["quote_receipts"][0]
                self.assertEqual(receipt["status"], "dropped_invalid_optional_quote")
                self.assertFalse(receipt["accepted_as_evidence"])
                if bad == "Repeat phrase.":
                    self.assertEqual(receipt["reason"], "ambiguous_whitespace_match")

    def test_missing_failed_malformed_and_excessive_outputs_stay_failed(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        for bad in ({"error": "out_of_memory"}, {"text": "not json"}, {"text": '{"evidence":[],"answer":123}'},
                    {"text": json.dumps({"evidence": [{}] * 9})}):
            with self.subTest(bad=bad):
                response = outputs(prepared)
                response[0] = {"job_id": response[0]["job_id"], **bad}
                self.assertFalse(refresh.apply_maps(episode, state, 0, prepared, response)["source_coverage_complete"])
        self.assertFalse(refresh.apply_maps(episode, state, 0, prepared, [])["source_coverage_complete"])

    def test_state_source_input_tamper_and_duplicate_output_rejected(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        response = outputs(prepared)
        with self.assertRaises(ValueError):
            refresh.apply_maps(episode, state, 0, prepared, response + [response[0]])
        with self.assertRaises(ValueError):
            refresh.apply_maps(episode, state, 0, prepared, [{"job_id": "unknown", "text": "{}"}])
        state["rolling"]["text"] = "mutated"
        with self.assertRaises(ValueError):
            refresh.apply_maps(episode, state, 0, prepared, response)
        state = initial_state(episode)
        prepared["jobs"][0]["messages"][-1]["content"] = "changed"
        self.assertFalse(refresh.apply_maps(episode, state, 0, prepared, response)["source_coverage_complete"])


class FinalPackingTests(unittest.TestCase):
    def mapped(self, *, empty=False):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        return episode, state, refresh.apply_maps(episode, state, 0, prepared, outputs(prepared, empty=empty))

    def test_final_keeps_table_headers_footnotes_and_root_job_schema(self):
        episode, state, mapped = self.mapped()
        # A short row-only model quote must not strip original table context.
        for span in mapped["evidence"]:
            if span["chunk_id"] == "old_table":
                span["quote"] = "Revenue | 100 | 90"
        before = copy.deepcopy((episode, state, mapped))
        final = refresh.prepare_final(episode, state, 0, mapped)
        self.assertEqual(len(final["jobs"]), 4)
        self.assertTrue(all(set(job) == {"job_id", "messages", "max_new_tokens"} for job in final["jobs"]))
        payload = json.loads(final["jobs"][0]["messages"][-1]["content"])
        table = next(chunk for chunk in payload["evidence"] if chunk["chunk_id"] == "old_table")
        self.assertEqual(table["text"], episode["chunks"][0]["text"])
        self.assertIn("Footnote:", table["text"])
        self.assertTrue(payload["full_refresh_processing"]["all_eligible_sources_processed"])
        self.assertEqual((episode, state, mapped), before)

    def test_per_claim_packing_records_every_exclusion_and_actual_chars(self):
        episode, state, mapped = self.mapped()
        with patch.object(refresh, "messages", small_messages):
            final = refresh.prepare_final(episode, state, 0, mapped, char_cap=1800)
        self.assertEqual(len(final["jobs"]), 4)
        saw_exclusion = False
        for job, receipt in zip(final["jobs"], final["packing_receipts"]):
            payload = json.loads(job["messages"][-1]["content"])
            selected = {chunk["chunk_id"] for chunk in payload["evidence"]}
            all_receipts = receipt["candidates"]
            self.assertEqual({row["chunk_id"] for row in all_receipts}, {"old_table", "old_guidance", "new_actual", "new_risk"})
            excluded = {row["chunk_id"] for row in all_receipts if row["decision"] == "excluded_final_context_budget"}
            self.assertEqual(selected | excluded, {row["chunk_id"] for row in all_receipts})
            self.assertEqual(set(payload["full_refresh_processing"]["budget_excluded_candidate_ids"]), excluded)
            self.assertLessEqual(sum(len(message["content"]) for message in job["messages"]), 1800)
            saw_exclusion |= bool(excluded)
        self.assertTrue(saw_exclusion)
        self.assertEqual(final["shared_map_cost"], mapped["cost"])

    def test_empty_valid_maps_do_not_fall_back_to_unmapped_raw_sources(self):
        episode, state, mapped = self.mapped(empty=True)
        self.assertTrue(mapped["source_coverage_complete"])
        with patch.object(refresh, "messages", small_messages):
            final = refresh.prepare_final(episode, state, 0, mapped)
        self.assertTrue(all(not json.loads(job["messages"][-1]["content"])["evidence"] for job in final["jobs"]))
        for job in final["jobs"]:
            receipt = json.loads(job["messages"][-1]["content"])["full_refresh_processing"]
            self.assertEqual(receipt["evidence_selection_status"], "empty_model_selection_unknown")
            self.assertIn("do not infer that the original corpus omits", receipt["empty_packet_instruction"])

    def test_public_bm25_table_precedes_irrelevant_high_id_footer(self):
        episode = fixture()
        state = initial_state(episode)
        state["rolling"]["citations"] = []
        state["rolling"]["text"] = ""
        footer = {"chunk_id": "zzzz_footer", "source_id": "old", "text": "Contact investor relations. Copyright and privacy information.", "locator": "Footer"}
        episode["chunks"].append(footer)
        candidates = [episode["chunks"][0], footer]
        before = copy.deepcopy((episode, state, candidates))
        with patch.object(refresh, "ordinary_packet", wraps=refresh.ordinary_packet) as shared:
            ranked, cost = refresh._rank_candidates(episode, episode["initial_claims"][1], state["rolling"], 0, candidates)
        self.assertEqual([row["chunk_id"] for row in ranked], ["old_table", "zzzz_footer"])
        self.assertEqual(shared.call_args.kwargs["k"], len(candidates))
        self.assertEqual({row["chunk_id"] for row in shared.call_args.args[0]["chunks"]}, {"old_table", "zzzz_footer"})
        self.assertEqual(cost["ordinary_packet_cost"]["retrieval_calls"], len(cost["ordinary_packet_cost"]["tool_costs"]))
        self.assertGreaterEqual(cost["ordinary_packet_cost"]["retrieval_calls"], 2)
        self.assertEqual(cost["stable_unmatched_appended"], 1)
        self.assertEqual((episode, state, candidates), before)

    def test_shared_ranking_accepts_entire_197_chunk_candidate_pool(self):
        episode = fixture()
        episode["chunks"].extend({"chunk_id": f"extra_{index:03d}", "source_id": "old", "text": "Additional revenue discussion."}
                                 for index in range(193))
        state = initial_state(episode)
        candidates = refresh._available(episode, 0)
        self.assertEqual(len(candidates), 197)
        ranked, cost = refresh._rank_candidates(episode, episode["initial_claims"][1], state["rolling"], 0, candidates)
        self.assertEqual(len(ranked), 197)
        self.assertEqual({row["chunk_id"] for row in ranked}, {row["chunk_id"] for row in candidates})
        self.assertEqual(cost["candidate_chunks"], 197)

    def test_failed_maps_and_final_budget_block_carry_own_state_unresolved(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        mapped = refresh.apply_maps(episode, state, 0, prepared, [])
        final = refresh.prepare_final(episode, state, 0, mapped)
        self.assertEqual(final["jobs"], [])
        reduced = reduce_event(episode, state, 0, final, [])
        self.assertTrue(all(row["unresolved"] for row in reduced["state"].values()))
        self.assertEqual(reduced["state"]["rolling"]["text"], state["rolling"]["text"])
        episode, state, mapped = self.mapped()
        final = refresh.prepare_final(episode, state, 0, mapped, char_cap=10)
        self.assertEqual(final["jobs"], [])
        self.assertEqual(len(final["errors"]), 4)
        self.assertTrue(all(row["unresolved"] for row in reduce_event(episode, state, 0, final, [])["state"].values()))

    def test_final_source_changes_are_rejected(self):
        episode, state, mapped = self.mapped()
        episode["chunks"][0]["text"] += " changed"
        with self.assertRaises(ValueError):
            refresh.prepare_final(episode, state, 0, mapped)


class RawFallbackTests(unittest.TestCase):
    def setup_case(self, failure="malformed"):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0)
        response = outputs(prepared)
        for row in response:
            row.update(error=None, prompt_tokens=200)
        if failure == "malformed":
            response[0]["text"] = "not valid map JSON"
        elif failure in ("unknown", "cross_context", "future"):
            payload = json.loads(response[0]["text"])
            payload["evidence"][0]["chunk_id"] = {"unknown": "FORGED_CHUNK", "cross_context": "new_actual", "future": "future_value"}[failure]
            response[0]["text"] = json.dumps(payload)
        mapped = refresh.apply_maps(episode, state, 0, prepared, response)
        return episode, state, prepared, response, mapped


    def test_malformed_and_forged_ids_recover_only_original_failed_map_inputs(self):
        for failure in ("malformed", "unknown", "cross_context", "future"):
            with self.subTest(failure=failure):
                episode, state, prepared, response, mapped = self.setup_case(failure)
                self.assertFalse(mapped["source_coverage_complete"])
                before = copy.deepcopy((episode, state, prepared, response, mapped))
                recovered = refresh.recover_raw_sources(episode, state, 0, prepared, response, mapped)
                raw = [row for row in recovered["evidence"] if row.get("raw_source_fallback")]
                self.assertEqual({row["chunk_id"] for row in raw}, set(prepared["sidecars"][0]["chunk_ids"]))
                self.assertTrue(all(set(row["claim_ids"]) == {"historical", "rolling", "guidance", "segment"} for row in raw))
                self.assertNotIn("FORGED_CHUNK", {row["chunk_id"] for row in recovered["evidence"]})
                self.assertNotIn("future_value", {row["chunk_id"] for row in recovered["evidence"]})
                self.assertFalse(recovered["source_coverage_complete"])
                self.assertTrue(recovered["fallback_answer_pool_complete"])
                self.assertTrue(recovered["all_source_inputs_physically_executed"])
                self.assertEqual(recovered["errors"], mapped["errors"])
                self.assertEqual(recovered["records"], mapped["records"])
                self.assertEqual(recovered["cost"], mapped["cost"])
                self.assertEqual(recovered["recovery_cost"]["additional_model_calls"], 0)
                self.assertEqual(recovered["recovery_cost"]["original_map_calls_retained_in_cost"], len(prepared["jobs"]))
                self.assertEqual((episode, state, prepared, response, mapped), before)

    def test_missing_oom_and_no_positive_prompt_proof_never_claim_execution(self):
        for failure in ("missing", "oom", "no_prompt_tokens", "zero_prompt_tokens"):
            with self.subTest(failure=failure):
                episode, state, prepared, response, _ = self.setup_case()
                if failure == "missing":
                    response = response[1:]
                elif failure == "oom":
                    response[0]["error"] = "CUDA out of memory"
                elif failure == "no_prompt_tokens":
                    response[0].pop("prompt_tokens")
                else:
                    response[0]["prompt_tokens"] = 0
                mapped = refresh.apply_maps(episode, state, 0, prepared, response)
                recovered = refresh.recover_raw_sources(episode, state, 0, prepared, response, mapped)
                self.assertTrue(recovered["fallback_answer_pool_complete"])
                self.assertFalse(recovered["all_source_inputs_physically_executed"])
                self.assertEqual(set(recovered["physical_execution_coverage"]["unproven_chunk_ids"]), set(prepared["sidecars"][0]["chunk_ids"]))
                with patch.object(refresh, "messages", small_messages):
                    final = refresh.prepare_final(episode, state, 0, recovered)
                self.assertEqual(len(final["jobs"]), 4)
                payload = json.loads(final["jobs"][0]["messages"][-1]["content"])
                self.assertFalse(payload["full_refresh_processing"]["all_eligible_sources_processed"])
                self.assertFalse(payload["full_refresh_processing"]["all_source_inputs_physically_executed"])

    def test_final_uses_distinct_ids_keeps_errors_and_previous_state_flags(self):
        episode, state, prepared, response, mapped = self.setup_case("unknown")
        recovered = refresh.recover_raw_sources(episode, state, 0, prepared, response, mapped)
        with patch.object(refresh, "messages", small_messages):
            final = refresh.prepare_final(episode, state, 0, recovered)
        self.assertEqual(len(final["jobs"]), 4)
        self.assertFalse(final["source_coverage_complete"])
        self.assertEqual(final["strict_map_errors"], mapped["errors"])
        self.assertTrue(all(job["job_id"].endswith("__raw_map_fallback_v2_fixed_rerank") for job in final["jobs"]))
        payload = json.loads(final["jobs"][0]["messages"][-1]["content"])
        receipt = payload["full_refresh_processing"]
        self.assertFalse(receipt["valid_map_coverage_complete"])
        self.assertTrue(receipt["all_source_inputs_physically_executed"])
        self.assertTrue(receipt["fallback_answer_pool_complete"])
        self.assertIn("ALL of its original eligible input chunks", receipt["fallback_rule"])
        self.assertEqual(payload["previous_state"], state["historical"])
        carried = reduce_event(episode, state, 0, final, [])
        self.assertTrue(all(row["unresolved"] for row in carried["state"].values()))
        self.assertEqual(carried["state"]["rolling"]["text"], state["rolling"]["text"])


    def test_unscheduled_or_tampered_input_cannot_be_silently_recovered(self):
        episode = fixture()
        state = initial_state(episode)
        prepared = refresh.prepare_maps(episode, state, 0, char_cap=10)
        mapped = refresh.apply_maps(episode, state, 0, prepared, [])
        recovered = refresh.recover_raw_sources(episode, state, 0, prepared, [], mapped)
        self.assertFalse(recovered["fallback_answer_pool_complete"])
        self.assertFalse(recovered["all_source_inputs_physically_executed"])
        final = refresh.prepare_final(episode, state, 0, recovered)
        self.assertEqual(final["jobs"], [])
        self.assertTrue(all(row["unresolved"] for row in reduce_event(episode, state, 0, final, [])["state"].values()))
        episode, state, prepared, response, mapped = self.setup_case()
        prepared["jobs"][0]["messages"][-1]["content"] = "tampered"
        recovered = refresh.recover_raw_sources(episode, state, 0, prepared, response, mapped)
        self.assertFalse(recovered["fallback_answer_pool_complete"])
        self.assertTrue(recovered["recovery_errors"])


if __name__ == "__main__":
    unittest.main()
