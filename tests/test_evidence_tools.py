"""Synthetic public fixtures only; no V6 private labels or model outputs."""
import copy
import json
import unittest

from bluebridge.evidence_tools import (audit_proposal, calculate, deterministic_lookup,
                            eligible_chunks, is_available, retrieve)


AS_OF = "2025-11-01"
TABLE = """Acme Semiconductor — FY2025 Q4 results
GAAP; USD in millions
Metric | FY2025 Q4 | FY2024 Q4
Revenue | 1,200 | 1,000
Gross margin | 30% | 25%
Days of inventory | 45 | 40
"""


def episode():
    return {
        "sources": [{"source_id": "release", "published_at": "2025-10-30", "url": "https://example.test/release"},
                    {"source_id": "future", "published_at": "2025-11-02", "url": "https://example.test/future"}],
        "chunks": [{"chunk_id": "table", "source_id": "release", "text": TABLE, "locator": "Table 1"},
                   {"chunk_id": "decimal", "source_id": "release", "text": "Ratios 0.1 and 0.2; loss (5)."},
                   {"chunk_id": "future_table", "source_id": "future", "text": "Revenue | 9,999"}],
    }


def operand(name="current", value="1,200", **changes):
    result = {"name": name, "value": value, "scale": "million", "unit": "USD million",
              "chunk_id": "table", "quote": "Revenue | 1,200 | 1,000", "scale_quote": "GAAP; USD in millions"}
    result.update(changes)
    return result


def ratio(name, value):
    return {"name": name, "value": value, "scale": "one", "unit": "ratio",
            "chunk_id": "decimal", "quote": "Ratios 0.1 and 0.2; loss (5)."}


class EligibilityTests(unittest.TestCase):
    def test_cutoffs_and_timezone_boundary(self):
        self.assertTrue(is_available("2025-10-30", "2025-10-30"))
        self.assertFalse(is_available("2025-10-30", "2025-10-30T23:00:00Z"))
        self.assertTrue(is_available("2025-10-30T19:00:00-04:00", "2025-10-30T23:00:00Z"))
        self.assertFalse(is_available("2025-10-31T00:00:01Z", "2025-10-30"))
        self.assertFalse(is_available("2025-10-30T12:00:00", "2025-10-31"))
        self.assertFalse(is_available("2025-02-30", AS_OF))
        self.assertFalse(is_available(None, AS_OF))

    def test_source_visibility_and_no_future_evidence(self):
        self.assertEqual({c["chunk_id"] for c in eligible_chunks(episode(), AS_OF)}, {"table", "decimal"})
        self.assertEqual(eligible_chunks(episode(), AS_OF, source_ids=["future"]), [])
        self.assertEqual(eligible_chunks(episode(), AS_OF, source_ids=[]), [])

    def test_duplicate_unknown_and_invalid_ids_fail_closed(self):
        data = episode()
        data["sources"].append(copy.deepcopy(data["sources"][0]))
        self.assertEqual(eligible_chunks(data, AS_OF), [])
        data = episode()
        data["chunks"].append(copy.deepcopy(data["chunks"][0]))
        data["chunks"].append({"chunk_id": "bad", "source_id": [], "text": "untrusted"})
        self.assertEqual([c["chunk_id"] for c in eligible_chunks(data, AS_OF)], ["decimal"])

    def test_public_allowlist_and_independent_output(self):
        data = episode()
        data["reference"] = {"value": "PRIVATE_SENTINEL"}
        data["chunks"][0]["private_gold"] = "PRIVATE_SENTINEL"
        data["chunks"][0]["metadata"] = {"entity": "Acme Semiconductor", "gold": "PRIVATE_SENTINEL"}
        before = copy.deepcopy(data)
        result = eligible_chunks(data, AS_OF)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        result[1]["text"] = "changed"
        self.assertEqual(data, before)


class RetrievalTests(unittest.TestCase):
    def test_deterministic_rows_preserve_original_headers_and_charge_full_context(self):
        one = retrieve(episode(), "Acme Revenue FY2025 Q4", AS_OF, k=1)
        self.assertEqual(one, retrieve(episode(), "Acme Revenue FY2025 Q4", AS_OF, k=1))
        self.assertEqual(one["chunks"][0]["chunk_id"], "table")
        self.assertEqual(one["chunks"][0]["text"], TABLE)
        self.assertIn("Metric | FY2025 Q4 | FY2024 Q4", one["chunks"][0]["text"])
        self.assertEqual(one["cost"]["returned_characters"], len(TABLE))
        self.assertGreater(one["cost"]["retrieval_units_scored"], one["cost"]["eligible_chunks"])
        self.assertEqual(one["cost"]["model_calls"], 0)

    def test_empty_query_no_relevance_and_invalid_budgets(self):
        self.assertEqual(retrieve(episode(), "the and", AS_OF)["chunks"], [])
        self.assertEqual(retrieve(episode(), "Revenue", AS_OF, k=0)["chunks"], [])
        for k in (True, -1, 1001, 10000, 1.2):
            with self.subTest(k=k), self.assertRaises(ValueError):
                retrieve(episode(), "Revenue", AS_OF, k=k)


    def test_ties_are_resolved_by_public_id(self):
        data = {"sources": [{"source_id": "s", "published_at": "2025-01-01"}],
                "chunks": [{"chunk_id": key, "source_id": "s", "text": "Acme revenue"} for key in ("z", "a")]}
        self.assertEqual([c["chunk_id"] for c in retrieve(data, "revenue", AS_OF)["chunks"]], ["a", "z"])


class CalculatorTests(unittest.TestCase):
    def test_decimal_exactness_and_quoted_raw_literals(self):
        ops = [ratio("a", "0.1"), ratio("b", "0.2")]
        for expression in ("a+b", "0.1+0.2"):
            result = calculate(expression, ops, episode(), AS_OF)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["result"], "0.3")
        self.assertFalse(calculate("a+0.3", ops, episode(), AS_OF)["ok"])

    def test_comma_million_and_display_conversion(self):
        result = calculate("current-prior", [operand(), operand("prior", "1,000")], episode(), AS_OF, output_unit="USD million")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"], "200000000")
        self.assertEqual(result["unit"], "USD")
        self.assertEqual(result["display_result"], "200")
        self.assertEqual(result["display_unit"], "USD million")
        self.assertFalse(result["semantic_entailment_checked"])

    def test_growth_ratio_and_percent_output(self):
        result = calculate("(current-prior)/prior", [operand(), operand("prior", "1000")], episode(), AS_OF, output_unit="percent")
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["result"], result["display_result"]), ("0.2", "20"))
        self.assertEqual(result["unit"], "ratio")

    def test_percent_operands_require_explicit_grounded_scale(self):
        op = operand("margin", "30%", scale="percent", unit="percent", quote="Gross margin | 30% | 25%", scale_quote="")
        result = calculate("margin", [op], episode(), AS_OF, output_unit="percent")
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["result"], result["display_result"]), ("0.3", "30"))
        op["scale"] = "one"
        self.assertFalse(calculate("margin", [op], episode(), AS_OF)["ok"])

    def test_billion_to_million_conversion(self):
        data = episode()
        data["chunks"].append({"chunk_id": "billion", "source_id": "release", "text": "USD revenue was 1.2 billion."})
        op = operand("a", "1.2", unit="USD billion", scale="billion", chunk_id="billion", quote="USD revenue was 1.2 billion.", scale_quote="")
        result = calculate("a", [op], data, AS_OF, output_unit="USD million")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["display_result"], "1200")

    def test_negative_parentheses_unary_and_signed_literal(self):
        op = ratio("loss", "-5")
        for expression, expected in (("loss", "-5"), ("-loss", "5"), ("-5", "-5"), ("+loss", "-5")):
            result = calculate(expression, [op], episode(), AS_OF)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["result"], expected)

    def test_unsupported_code_and_resource_attacks(self):
        op = ratio("a", "0.1")
        attacks = ["__import__('os').system('touch /tmp/must_not_run')", "a.__class__", "abs(a)", "a**a", "a%a",
                   "[a][0]", "(lambda: a)()", "a if a else a", "True", "'0.1'", "1e9999", "a<<a", "a//a"]
        for expression in attacks:
            with self.subTest(expression=expression):
                self.assertFalse(calculate(expression, [op], episode(), AS_OF)["ok"])
        deep = "a"
        for _ in range(20):
            deep = "a+(" + deep + ")"
        self.assertFalse(calculate(deep, [op], episode(), AS_OF)["ok"])
        self.assertFalse(calculate("a" * 501, [op], episode(), AS_OF)["ok"])

    def test_numeric_binding_rejects_substrings_forged_quotes_and_future_data(self):
        for op in (operand(value="200"), operand(quote="Revenue | 200 | 1,000"),
                   operand(chunk_id="future_table", value="9999", quote="Revenue | 9,999"),
                   operand(value="NaN"), operand(value=True)):
            with self.subTest(op=op):
                self.assertFalse(calculate("current", [op], episode(), AS_OF)["ok"])

    def test_units_scales_and_numeric_literal_ambiguity(self):
        ops = [operand(), ratio("ratio", "0.1")]
        for expression, output_unit in (("current+ratio", None), ("current", "shares")):
            result = calculate(expression, ops, episode(), AS_OF, output_unit=output_unit)
            self.assertFalse(result["ok"], result)
        self.assertFalse(calculate("current", [operand(scale="billion")], episode(), AS_OF)["ok"])
        self.assertFalse(calculate("current", [operand(unit="shares million")], episode(), AS_OF)["ok"])
        data = episode()
        data["chunks"][0]["metadata"] = {"unit": "shares million"}
        self.assertFalse(calculate("current", [operand()], data, AS_OF)["ok"])
        # A scaled source literal cannot silently skip its scale; names are explicit.
        self.assertFalse(calculate("1200", [operand()], episode(), AS_OF)["ok"])
        data["chunks"][0]["metadata"] = {}
        ambiguous = [operand("a", "1000", scale="one", unit="USD"), operand("b", "1000", scale="one", unit="ratio")]
        self.assertFalse(calculate("1000", ambiguous, data, AS_OF)["ok"])

    def test_division_zero_duplicates_and_inputs_unchanged(self):
        data = episode()
        data["chunks"].append({"chunk_id": "zero", "source_id": "release", "text": "Ratio 0"})
        zero = ratio("z", "0")
        zero.update(chunk_id="zero", quote="Ratio 0")
        self.assertIn("division_by_zero", calculate("a/z", [ratio("a", "0.1"), zero], data, AS_OF)["flags"])
        self.assertFalse(calculate("a", [ratio("a", "0.1"), ratio("a", "0.2")], data, AS_OF)["ok"])
        ops = [ratio("a", "0.1")]
        before = copy.deepcopy((data, ops))
        result = calculate("a", ops, data, AS_OF)
        result["provenance"][0]["quote"] = "changed"
        self.assertEqual((data, ops), before)


class AuditTests(unittest.TestCase):
    def proposal(self):
        return {"text": "Revenue was USD 1,200 million.", "value": "1200", "unit": "USD million",
                "entity": "Acme Semiconductor", "period": "FY2025 Q4", "basis": "GAAP",
                "citations": ["table"], "evidence_quotes": [{"chunk_id": "table", "quote": "Revenue | 1,200 | 1,000"}]}

    def test_quotes_and_public_metadata_are_mechanical_only(self):
        proposal = self.proposal()
        result = audit_proposal(proposal, episode(), AS_OF, metric_spec={key: proposal[key] for key in ("entity", "period", "basis", "unit")})
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["semantic_entailment_checked"])
        # Deliberately wrong prose metric: matching numbers and metadata cannot
        # establish what the table row entails. The tool makes no such claim.
        proposal["text"] = "Gross margin was USD 1,200 million because demand improved."
        mechanical = audit_proposal(proposal, episode(), AS_OF)
        self.assertTrue(mechanical["ok"], mechanical)
        self.assertIn("no financial or narrative entailment", mechanical["scope"])
        self.assertFalse(mechanical["semantic_entailment_checked"])

    def test_unknown_numeric_metadata_and_missing_quotes_need_review(self):
        proposal = self.proposal()
        proposal["value"] = "9876"
        self.assertIn("proposal_value_not_quoted_or_calculated", audit_proposal(proposal, episode(), AS_OF)["flags"])
        proposal = self.proposal()
        result = audit_proposal(proposal, episode(), AS_OF, metric_spec={"entity": "Other Corp", "period": "FY2026 Q4", "basis": "non-GAAP"})
        self.assertFalse(result["ok"])
        self.assertIn("proposal_metadata_mismatch:period", result["flags"])
        self.assertIn("metadata_not_located:basis", result["flags"])
        proposal.pop("evidence_quotes")
        self.assertIn("citation_without_exact_quote:table", audit_proposal(proposal, episode(), AS_OF)["flags"])

    def test_forged_future_or_uncited_quotes_are_rejected(self):
        proposal = self.proposal()
        proposal["evidence_quotes"][0]["quote"] = "Revenue 9876"
        self.assertFalse(audit_proposal(proposal, episode(), AS_OF)["ok"])
        proposal["citations"] = ["future_table"]
        self.assertIn("unknown_or_ineligible_citation:future_table", audit_proposal(proposal, episode(), AS_OF)["flags"])
        proposal = self.proposal()
        proposal["citations"] = []
        self.assertIn("quote_chunk_not_cited:table", audit_proposal(proposal, episode(), AS_OF)["flags"])

    def test_verified_calculation_can_supply_output_percent(self):
        proposal = {"value": "20", "unit": "percent", "citations": ["table"],
                    "calculations": [{"expression": "(a-b)/b", "operands": [operand("a"), operand("b", "1000")], "output_unit": "percent"}]}
        result = audit_proposal(proposal, episode(), AS_OF, metric_spec={"unit": "percent"})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["calculations"][0]["display_result"], "20")
        self.assertEqual(result["cost"]["calculator_calls"], 1)
        data = episode()
        data["chunks"][0]["metadata"] = {"unit": "USD million"}
        self.assertTrue(audit_proposal(proposal, data, AS_OF, metric_spec={"unit": "percent"})["ok"])

    def test_citation_quote_and_operand_source_ids_must_match_the_chunk(self):
        proposal = self.proposal()
        proposal["citations"] = [{"chunk_id": "table", "source_id": "other", "quote": "Revenue | 1,200 | 1,000"}]
        result = audit_proposal(proposal, episode(), AS_OF)
        self.assertIn("citation_source_id_mismatch:table", result["flags"])
        self.assertIn("quote_source_id_mismatch:table", result["flags"])
        self.assertFalse(calculate("current", [operand(source_id="other")], episode(), AS_OF)["ok"])

    def test_conflicting_source_metadata_is_flagged(self):
        data = episode()
        data["chunks"][0]["metadata"] = {"basis": "non-GAAP", "entity": "Other Corp"}
        result = audit_proposal(self.proposal(), data, AS_OF)
        self.assertIn("source_metadata_conflict:basis", result["flags"])
        self.assertIn("source_metadata_conflict:entity", result["flags"])


class LookupBaselineTests(unittest.TestCase):
    def claim(self):
        return {"claim_id": "synthetic", "instruction": "Update Acme revenue.", "text": "Revenue was pending.",
                "metric_spec": {"metric": "Revenue", "entity": "Acme Semiconductor", "period": "FY2025 Q4",
                                "basis": "GAAP", "unit": "USD million", "scale": "million"}}

    def test_public_metric_lookup_returns_candidate_and_charged_review(self):
        claim = self.claim()
        before = copy.deepcopy(claim)
        result = deterministic_lookup(episode(), claim, AS_OF)
        self.assertEqual(result["status"], "candidate", result)
        self.assertEqual(result["value"], "1200")
        self.assertEqual(result["citations"], ["table"])
        self.assertEqual(result["disposition"], "review")
        self.assertEqual(result["cost"]["manual_review_required"], 1)
        self.assertGreater(result["cost"]["table_rows_examined"], 0)
        self.assertEqual(claim, before)

    def test_incomplete_missing_or_ambiguous_tables_abstain(self):
        claim = self.claim()
        claim.pop("metric_spec")
        self.assertEqual(deterministic_lookup(episode(), claim, AS_OF)["flags"], ["incomplete_public_metric_spec"])
        claim = self.claim()
        claim["metric_spec"]["period"] = "FY2026 Q4"
        self.assertEqual(deterministic_lookup(episode(), claim, AS_OF)["status"], "review")
        data = episode()
        data["chunks"].append({"chunk_id": "duplicate_table", "source_id": "release", "text": TABLE})
        result = deterministic_lookup(data, self.claim(), AS_OF)
        self.assertEqual(result["status"], "review")
        self.assertIn("ambiguous_lookup", result["flags"])
        self.assertEqual(result["candidate_count"], 2)

    def test_unobserved_entity_basis_unit_and_regex_like_patterns_do_not_fill(self):
        for field, value in (("entity", "Other Corp"), ("basis", "non-GAAP"), ("unit", "shares million"), ("row_pattern", ".*")):
            with self.subTest(field=field):
                claim = self.claim()
                claim["metric_spec"][field] = value
                self.assertEqual(deterministic_lookup(episode(), claim, AS_OF)["status"], "review")

    def test_private_fields_and_claim_ids_have_no_lookup_effect(self):
        one = self.claim()
        two = copy.deepcopy(one)
        two.update(claim_id="unknown-private-id", reference={"value": "9999", "source_id": "future"})
        self.assertEqual(deterministic_lookup(episode(), one, AS_OF), deterministic_lookup(episode(), two, AS_OF))

    def test_nested_public_metric_and_source_header_scale(self):
        claim = self.claim()
        claim["metric_spec"] = {"entity": "Acme Semiconductor", "period": "FY2025 Q4",
                                "metrics": [{"metric": "Revenue", "scope": "company", "basis": "GAAP", "unit": "USD billion"}]}
        result = deterministic_lookup(episode(), claim, AS_OF)
        self.assertEqual(result["status"], "candidate", result)
        self.assertEqual(result["value"], "1.2")


if __name__ == "__main__":
    unittest.main()
