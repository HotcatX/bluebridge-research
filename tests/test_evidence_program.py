"""Synthetic mechanical tests; no benchmark cases or reference answers."""
import copy
import json
import unittest

from bluebridge.evidence_program import MAX_FACTS, execute


AS_OF = "2025-08-01"


def episode(text="Revenue was $8 million.", period="FY2025 Q2"):
    return {"sources": [{"source_id": "release", "published_at": "2025-07-01", "period": period}],
            "chunks": [{"chunk_id": "revenue", "source_id": "release", "text": text}]}


def fact(identifier="main", chunk="revenue", quote="Revenue was $8 million.", value="8", **changes):
    return {"fact_id": identifier, "chunk_id": chunk, "quote": quote, "value": value,
            "unit": "USD", "scale": "million", "metric": "Revenue", "period": "FY2025 Q2",
            "scope": "", "basis": "", "qualifier": "exact", **changes}


def program(facts=None, **changes):
    return {"facts": [fact()] if facts is None else facts, "primary_fact_id": "main",
            "secondary_fact_ids": [], "operation": "report", "disposition": "update", **changes}


def claim(**changes):
    return {"instruction": "Report the requested metric using the supplied sources.",
            "metric_spec": {"metrics": [{"metric": "Revenue", "unit": "USD million"}],
                            "period": "FY2025 Q2", **changes}}


def comparison_case():
    data = {"sources": [
        {"source_id": "original", "published_at": "2025-04-01", "period": "FY2025 Q1"},
        {"source_id": "actual", "published_at": "2025-07-01", "period": "FY2025 Q2"}],
        "chunks": [
            {"chunk_id": "low_chunk", "source_id": "original", "text": "Revenue low guidance was $6 million for FY2025 Q2."},
            {"chunk_id": "high_chunk", "source_id": "original", "text": "Revenue high guidance was $9 million for FY2025 Q2."},
            {"chunk_id": "actual_chunk", "source_id": "actual", "text": "Revenue actual was $8 million for FY2025 Q2."}]}
    facts = [fact("actual", "actual_chunk", data["chunks"][2]["text"], "8"),
             fact("low", "low_chunk", data["chunks"][0]["text"], "6"),
             fact("high", "high_chunk", data["chunks"][1]["text"], "9")]
    prog = program(facts, primary_fact_id="actual", operation="compare_range",
                   comparison={"actual_fact_id": "actual", "low_fact_id": "low", "high_fact_id": "high"})
    request = claim(guidance_vintage="2025-04-01", temporal_policy="fixed_guidance_realization")
    return data, prog, request


class ReportTests(unittest.TestCase):
    def test_exact_report_and_explicit_disposition(self):
        result = execute(program(disposition="retain"), episode(), claim(), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["disposition"], "retain")
        self.assertEqual(result["proposal"]["value"], "8")
        self.assertIn("8 USD million [revenue]", result["proposal"]["patch"])
        self.assertEqual(result["proposal"]["citations"], ["revenue"])
        self.assertFalse(result["audit"]["semantic_entailment_checked"])
        self.assertTrue(result["audit"]["manual_review_required"])
        self.assertEqual(result["cost"]["model_calls"], 0)

    def test_conversion_updates_primary_scalar_and_render_together(self):
        result = execute(program(), episode(), claim(metrics=[{"metric": "Revenue", "unit": "USD billion"}]), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["value"], "0.008")
        self.assertIn("0.008 USD billion", result["proposal"]["patch"])
        self.assertTrue(result["audit"]["proposal_audit"]["ok"])

    def test_percent_conversion(self):
        data = episode("Margin was 17%.")
        selected = fact(quote=data["chunks"][0]["text"], value="17", unit="percent", scale="percent", metric="Margin")
        result = execute(program([selected]), data, claim(metrics=[{"metric": "Margin", "unit": "ratio"}]), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["value"], "0.17")
        self.assertIn("0.17 ratio", result["proposal"]["patch"])

    def test_each_nonexact_primary_has_null_scalar_and_qualifier(self):
        for cue, qualifier, rendered in (("over", "more_than", "more than"),
                                         ("at least", "at_least", "at least"),
                                         ("at most", "at_most", "at most"),
                                         ("less than", "less_than", "less than"),
                                         ("nearly", "approximate", "approximately"),
                                         ("exceeded", "more_than", "more than"),
                                         ("surpassed", "more_than", "more than")):
            with self.subTest(cue=cue):
                text = f"Revenue was {cue} $8 million."
                result = execute(program([fact(quote=text, qualifier=qualifier)]), episode(text), claim(), AS_OF)
                self.assertEqual(result["errors"], [])
                self.assertIsNone(result["proposal"]["value"])
                self.assertIn(f"{rendered} 8 USD million", result["proposal"]["patch"])

    def test_threshold_cue_outside_trimmed_quote_cannot_become_exact(self):
        result = execute(program([fact(quote="$8 million")]), episode("Revenue was over $8 million."), claim(), AS_OF)
        self.assertIn("source_qualifier_mismatch:main", result["errors"])
        self.assertIsNone(result["proposal"]["value"])
        self.assertTrue(result["proposal"]["unresolved"])

    def test_unsubstantiated_qualifier_is_rejected(self):
        result = execute(program([fact(qualifier="approximate")]), episode(), claim(), AS_OF)
        self.assertIn("source_qualifier_mismatch:main", result["errors"])

    def test_no_more_than_not_misread_as_more_than(self):
        text = "Revenue was no more than $8 million."
        result = execute(program([fact(quote=text, qualifier="at_most")]), episode(text), claim(), AS_OF)
        self.assertEqual(result["errors"], [])

    def test_table_scale_header_retained(self):
        text = "USD million\nMetric | FY2025 Q2\nRevenue | 8"
        selected = fact(quote="Revenue | 8", scale_quote="USD million")
        result = execute(program([selected]), episode(text), claim(), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertIn("table_column", " ".join(result["audit"]["warnings"]))

    def test_rich_metadata_does_not_certify_table_column(self):
        data = episode("USD million\nMetric | FY2025 Q1 | FY2025 Q2\nRevenue | 8 | 11")
        data["sources"][0]["metadata"] = {"unit": "USD", "basis": "GAAP"}
        selected = fact(quote="Revenue | 8 | 11", scale_quote="USD million", basis="GAAP")
        result = execute(program([selected]), data, claim(metrics=[{"metric": "Revenue", "unit": "USD million", "basis": "GAAP"}]), AS_OF)
        self.assertEqual(result["errors"], [])  # Deliberate mechanical limitation.
        self.assertFalse(result["audit"]["semantic_entailment_checked"])
        self.assertTrue(result["audit"]["manual_review_required"])
        self.assertIn("table_column", " ".join(result["audit"]["warnings"]))

    def test_missing_disposition_requires_review(self):
        prog = program()
        del prog["disposition"]
        result = execute(prog, episode(), claim(), AS_OF)
        self.assertIn("missing_or_invalid_disposition", result["errors"])
        self.assertEqual(result["proposal"]["status"], "review")

    def test_all_requested_secondary_metrics_required(self):
        request = claim(metrics=[{"metric": "Revenue", "unit": "USD million"}, {"metric": "Margin", "unit": "percent"}])
        result = execute(program(), episode(), request, AS_OF)
        self.assertIn("missing_requested_metric:1", result["errors"])

    def test_secondary_metric_is_rendered_and_cited(self):
        data = episode()
        data["chunks"].append({"chunk_id": "margin", "source_id": "release", "text": "Margin was 17%."})
        other = fact("margin_fact", "margin", "Margin was 17%.", "17", metric="Margin", unit="percent", scale="percent")
        request = claim(metrics=[{"metric": "Revenue", "unit": "USD million"}, {"metric": "Margin", "unit": "percent"}])
        result = execute(program([fact(), other], secondary_fact_ids=["margin_fact"]), data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["value"], "8")
        self.assertEqual(result["proposal"]["citations"], ["margin", "revenue"])
        self.assertIn("17 percent [margin]", result["proposal"]["patch"])


class TemporalTests(unittest.TestCase):
    def test_public_source_period_conflict_rejected(self):
        result = execute(program(), episode(period="FY2025 Q1"), claim(), AS_OF)
        self.assertIn("public_source_period_mismatch:main", result["errors"])

    def test_no_public_period_target_does_not_invent_one(self):
        request = claim()
        del request["metric_spec"]["period"]
        result = execute(program(), episode(period="FY2025 Q1"), request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertFalse(result["audit"]["semantic_entailment_checked"])

    def test_safe_quarter_alias(self):
        result = execute(program([fact(period="Q2 FY25")]), episode(period="FY2025Q2"), claim(), AS_OF)
        self.assertEqual(result["errors"], [])

    def test_latest_uses_newest_eligible_source_period(self):
        data = episode(period="FY2025 Q1")
        data["sources"].append({"source_id": "new", "published_at": "2025-07-15", "period": "FY2025 Q2"})
        data["chunks"].append({"chunk_id": "new_chunk", "source_id": "new", "text": "Revenue was $11 million."})
        request = claim(period="latest reported quarter", temporal_policy="latest_reported_quarter")
        result = execute(program([fact(period="FY2025 Q1")]), data, request, AS_OF)
        self.assertIn("requested_period_mismatch:main", result["errors"])
        self.assertIsNone(result["proposal"]["value"])
        selected = fact(chunk="new_chunk", quote="Revenue was $11 million.", value="11")
        current = execute(program([selected]), data, request, AS_OF)
        self.assertEqual(current["errors"], [])
        self.assertEqual(current["proposal"]["value"], "11")

    def test_future_or_undiscovered_source_does_not_change_latest(self):
        data = episode()
        data["sources"].append({"source_id": "future", "published_at": "2025-09-01", "period": "FY2025 Q3"})
        data["sources"].append({"source_id": "undiscovered", "published_at": "2025-07-15", "period": "FY2025 Q3"})
        request = claim(period="latest reported quarter", temporal_policy="latest_reported_quarter")
        result = execute(program(), data, request, AS_OF, source_ids=["release", "future"])
        self.assertEqual(result["errors"], [])

    def test_missing_newest_source_period_does_not_fall_back_to_old(self):
        data = episode()
        data["sources"].append({"source_id": "new", "published_at": "2025-07-15"})
        request = claim(period="latest reported quarter", temporal_policy="latest_reported_quarter")
        result = execute(program(), data, request, AS_OF)
        self.assertIn("latest_public_source_period_unavailable", result["errors"])

    def test_same_metric_can_have_distinct_explicit_period_requests(self):
        data = episode()
        data["sources"].append({"source_id": "old", "published_at": "2025-04-01", "period": "FY2025 Q1"})
        data["chunks"].append({"chunk_id": "old_chunk", "source_id": "old", "text": "Revenue was $6 million."})
        old = fact("old_fact", "old_chunk", "Revenue was $6 million.", "6", period="FY2025 Q1")
        request = claim(metrics=[{"metric": "Revenue", "unit": "USD million", "period": "FY2025 Q2"},
                                 {"metric": "Revenue", "unit": "USD million", "period": "FY2025 Q1"}])
        missing = execute(program(), data, request, AS_OF)
        self.assertIn("missing_requested_metric:1", missing["errors"])
        result = execute(program([fact(), old], secondary_fact_ids=["old_fact"]), data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["citations"], ["old_chunk", "revenue"])

    def test_explicit_historical_secondary_does_not_roll_with_latest_primary(self):
        data = episode()
        data["sources"].append({"source_id": "old", "published_at": "2025-04-01", "period": "FY2025 Q1"})
        data["chunks"].append({"chunk_id": "old_chunk", "source_id": "old", "text": "Revenue was $6 million."})
        old = fact("old_fact", "old_chunk", "Revenue was $6 million.", "6", period="FY2025 Q1")
        request = claim(period="latest reported quarter", temporal_policy="latest_reported_quarter",
                        metrics=[{"metric": "Revenue", "unit": "USD million"},
                                 {"metric": "Revenue", "unit": "USD million", "period": "FY2025 Q1"}])
        result = execute(program([fact(), old], secondary_fact_ids=["old_fact"]), data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertIn("FY2025 Q1", result["proposal"]["patch"])


class SplitHeaderTests(unittest.TestCase):
    def split_case(self):
        data = episode("Revenue | 8")
        data["chunks"].append({"chunk_id": "header", "source_id": "release", "text": "Table amounts in USD million."})
        selected = fact(quote="Revenue | 8", context_quotes=[{"chunk_id": "header", "quote": "Table amounts in USD million."}])
        return data, selected

    def test_same_source_split_header_binds_without_fabricating_original_text(self):
        data, selected = self.split_case()
        original = copy.deepcopy(data)
        result = execute(program([selected]), data, claim(), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["value"], "8")
        self.assertEqual(result["proposal"]["citations"], ["header", "revenue"])
        row = result["audit"]["facts"][0]
        self.assertFalse(row["context_binding"]["original_text_modified"])
        self.assertFalse(row["context_binding"]["header_to_row_semantic_link_checked"])
        self.assertNotEqual(row["calculation"]["provenance"][0]["chunk_id"], "revenue")
        self.assertEqual(data, original)
        self.assertEqual(result["cost"]["context_quote_checks"], 1)

    def test_conversion_with_context_has_explicit_audit_scope(self):
        data, selected = self.split_case()
        result = execute(program([selected]), data, claim(metrics=[{"metric": "Revenue", "unit": "USD billion"}]), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["value"], "0.008")
        self.assertIn("0.008 USD billion", result["proposal"]["patch"])
        audit = result["audit"]["proposal_audit"]
        self.assertEqual(audit["scalar_unit_validation"], "typed_calculator_with_explicit_augmented_context")
        self.assertEqual(audit["calculations"], [])
        self.assertEqual(result["proposal"]["calculations"][0]["operands"][0]["chunk_id"], "revenue")
        self.assertIn("context_quotes", result["proposal"]["calculations"][0]["operands"][0])

    def test_cross_source_header_is_rejected(self):
        data, selected = self.split_case()
        data["sources"].append({"source_id": "other", "published_at": "2025-07-01", "period": "FY2025 Q2"})
        data["chunks"][1]["source_id"] = "other"
        result = execute(program([selected]), data, claim(), AS_OF)
        self.assertIn("cross_source_context_quote:main", result["errors"])
        self.assertIsNone(result["proposal"]["value"])

    def test_ineligible_header_is_rejected(self):
        data, selected = self.split_case()
        data["sources"].append({"source_id": "future", "published_at": "2025-09-01", "period": "FY2025 Q3"})
        data["chunks"][1]["source_id"] = "future"
        result = execute(program([selected]), data, claim(), AS_OF)
        self.assertIn("unknown_or_ineligible_context_chunk:main", result["errors"])

    def test_header_quote_must_be_exact(self):
        data, selected = self.split_case()
        selected["context_quotes"][0]["quote"] = "Table amounts in USD billion."
        result = execute(program([selected]), data, claim(), AS_OF)
        self.assertIn("context_quote_missing:main", result["errors"])

    def test_context_cannot_fabricate_the_original_value_quote(self):
        data, selected = self.split_case()
        selected.update(quote="Revenue | 99", value="99")
        result = execute(program([selected]), data, claim(), AS_OF)
        self.assertIn("context_original_quote_missing:main", result["errors"])

    def test_at_most_three_header_quotes(self):
        data, selected = self.split_case()
        selected["context_quotes"] *= 4
        result = execute(program([selected]), data, claim(), AS_OF)
        self.assertIn("invalid_context_quotes:main", result["errors"])
        self.assertEqual(result["cost"]["calculator_calls"], 0)


class ComparisonTests(unittest.TestCase):
    def midpoint_case(self, relative=False):
        data, _, request = comparison_case()
        midpoint = "Revenue guidance midpoint was $20 million for FY2025 Q2."
        tolerance = ("Revenue guidance tolerance was plus or minus 15% for FY2025 Q2." if relative else
                     "Revenue guidance tolerance was plus or minus $3 million for FY2025 Q2.")
        actual = "Revenue actual was $22 million for FY2025 Q2."
        data["chunks"] = [{"chunk_id": "mid_chunk", "source_id": "original", "text": midpoint},
                          {"chunk_id": "tol_chunk", "source_id": "original", "text": tolerance},
                          {"chunk_id": "actual_chunk", "source_id": "actual", "text": actual}]
        tolerance_fact = fact("tolerance", "tol_chunk", tolerance, "15" if relative else "3",
                              metric="Revenue", unit="percent" if relative else "USD",
                              scale="percent" if relative else "million")
        prog = program([fact("actual", "actual_chunk", actual, "22"),
                        fact("mid", "mid_chunk", midpoint, "20"), tolerance_fact],
                       primary_fact_id="actual", operation="compare_range",
                       comparison={"actual_fact_id": "actual", "midpoint_fact_id": "mid",
                                   "tolerance_fact_id": "tolerance", "tolerance_mode": "relative" if relative else "absolute"})
        return data, prog, request

    def test_comparison_unions_all_operand_citations(self):
        data, prog, request = comparison_case()
        result = execute(prog, data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["citations"], ["actual_chunk", "high_chunk", "low_chunk"])
        self.assertIn("within the cited range", result["proposal"]["patch"])
        self.assertEqual(result["proposal"]["value"], "8")

    def test_later_outlook_cannot_replace_original_vintage(self):
        data, prog, request = comparison_case()
        data["sources"][0]["published_at"] = "2025-05-01"
        result = execute(prog, data, request, AS_OF)
        self.assertIn("original_guidance_vintage_mismatch:low", result["errors"])
        self.assertIn("original_guidance_vintage_mismatch:high", result["errors"])

    def test_comparison_different_units_are_normalized(self):
        data, prog, request = comparison_case()
        data["chunks"][1]["text"] = "Revenue high guidance was $0.009 billion for FY2025 Q2."
        prog["facts"][2].update(quote=data["chunks"][1]["text"], value="0.009", scale="billion")
        result = execute(prog, data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertIn("9 USD million", result["proposal"]["patch"])
        self.assertIn("within the cited range", result["proposal"]["patch"])

    def test_nonexact_comparison_operand_requires_review(self):
        data, prog, request = comparison_case()
        data["chunks"][0]["text"] = "Revenue low guidance was over $6 million for FY2025 Q2."
        prog["facts"][1].update(quote=data["chunks"][0]["text"], qualifier="more_than")
        result = execute(prog, data, request, AS_OF)
        self.assertIn("comparison_requires_exact_operands", result["errors"])

    def test_unknown_operand_id_fails_without_lookup(self):
        data, prog, request = comparison_case()
        prog["comparison"]["low_fact_id"] = "not_defined"
        result = execute(prog, data, request, AS_OF)
        self.assertIn("unknown_fact_id:not_defined", result["errors"])
        self.assertIsNone(result["proposal"]["value"])

    def test_reversed_range_fails(self):
        data, prog, request = comparison_case()
        prog["comparison"].update(low_fact_id="high", high_fact_id="low")
        result = execute(prog, data, request, AS_OF)
        self.assertIn("comparison_range_reversed", result["errors"])

    def test_midpoint_absolute_tolerance_derives_unquoted_bounds(self):
        data, prog, request = self.midpoint_case()
        result = execute(prog, data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["value"], "22")
        self.assertIn("Derived cited range: 17 to 23 USD million", result["proposal"]["patch"])
        self.assertIn("within the cited range", result["proposal"]["patch"])
        self.assertEqual(result["proposal"]["citations"], ["actual_chunk", "mid_chunk", "tol_chunk"])
        self.assertEqual(len(result["proposal"]["calculations"]), 3)
        self.assertFalse(any("17" in quote["quote"] or "23" in quote["quote"] for quote in result["proposal"]["evidence_quotes"]))

    def test_midpoint_relative_percent_tolerance_normalizes_ratio(self):
        data, prog, request = self.midpoint_case(relative=True)
        result = execute(prog, data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertIn("Derived cited range: 17 to 23 USD million", result["proposal"]["patch"])
        tolerance = next(row for row in result["audit"]["facts"] if row["fact_id"] == "tolerance")
        self.assertEqual(tolerance["calculation"]["result"], "0.15")
        self.assertEqual(tolerance["calculation"]["unit"], "ratio")

    def test_relative_tolerance_with_absolute_unit_requires_review(self):
        data, prog, request = self.midpoint_case()
        prog["comparison"]["tolerance_mode"] = "relative"
        result = execute(prog, data, request, AS_OF)
        self.assertIn("comparison_relative_tolerance_requires_ratio", result["errors"])
        self.assertIsNone(result["proposal"]["value"])

    def test_absolute_tolerance_with_relative_unit_requires_review(self):
        data, prog, request = self.midpoint_case(relative=True)
        prog["comparison"]["tolerance_mode"] = "absolute"
        result = execute(prog, data, request, AS_OF)
        self.assertTrue(any("incompatible_additive_units" in error for error in result["errors"]))

    def test_midpoint_vintage_applies_to_both_raw_guidance_facts(self):
        data, prog, request = self.midpoint_case()
        data["sources"][0]["published_at"] = "2025-05-01"
        result = execute(prog, data, request, AS_OF)
        self.assertIn("original_guidance_vintage_mismatch:mid", result["errors"])
        self.assertIn("original_guidance_vintage_mismatch:tolerance", result["errors"])

    def test_tolerance_formula_and_extra_keys_rejected(self):
        data, prog, request = self.midpoint_case()
        prog["comparison"]["formula"] = "__import__('os').system('false')"
        result = execute(prog, data, request, AS_OF)
        self.assertIn("invalid_comparison", result["errors"])

    def test_midpoint_and_percent_tolerance_accept_split_headers(self):
        data, prog, request = self.midpoint_case(relative=True)
        data["chunks"][0]["text"] = "Revenue guidance midpoint | 20"
        data["chunks"][1]["text"] = "Revenue guidance tolerance | 15"
        data["chunks"].extend([
            {"chunk_id": "money_header", "source_id": "original", "text": "USD million"},
            {"chunk_id": "pct_header", "source_id": "original", "text": "Percent (%)"}])
        prog["facts"][1].update(quote=data["chunks"][0]["text"], context_quotes=[{"chunk_id": "money_header", "quote": "USD million"}])
        prog["facts"][2].update(quote=data["chunks"][1]["text"], context_quotes=[{"chunk_id": "pct_header", "quote": "Percent (%)"}])
        result = execute(prog, data, request, AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertIn("Derived cited range: 17 to 23 USD million", result["proposal"]["patch"])
        self.assertEqual(result["proposal"]["citations"], ["actual_chunk", "mid_chunk", "money_header", "pct_header", "tol_chunk"])
        self.assertTrue(all(not identifier.startswith("__typed_") for identifier in result["proposal"]["citations"]))


class UnavailableAndBoundaryTests(unittest.TestCase):
    def test_unavailable_requires_explicit_model_reason(self):
        prog = program([], primary_fact_id=None, operation="unavailable")
        result = execute(prog, episode(), claim(), AS_OF)
        self.assertIn("explicit_missing_reason_required", result["errors"])

    def test_unavailable_preserves_ancillary_without_freeform_numeric_claim(self):
        prog = program([fact()], primary_fact_id=None, operation="unavailable", secondary_fact_ids=["main"],
                       missing_reason="Primary requested backlog missing; untrusted text says 999999.")
        result = execute(prog, episode(), claim(metrics=[{"metric": "Backlog", "unit": "USD million"}]), AS_OF)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["proposal"]["disposition"], "insufficient")
        self.assertIsNone(result["proposal"]["value"])
        self.assertIn("8 USD million", result["proposal"]["patch"])
        self.assertNotIn("999999", result["proposal"]["patch"])
        self.assertEqual(result["proposal"]["citations"], ["revenue"])
        self.assertTrue(result["proposal"]["unresolved"])

    def test_unknown_chunk_and_quote_number_and_unit_errors(self):
        cases = [({"chunk_id": "unknown"}, "unknown_or_ineligible_operand_chunk"),
                 ({"quote": "Revenue was $9 million."}, "operand_quote_missing"),
                 ({"value": "9"}, "operand_number_absent_from_quote"),
                 ({"unit": "EUR"}, "operand_unit_unsubstantiated")]
        for changes, expected in cases:
            with self.subTest(expected=expected):
                result = execute(program([fact(**changes)]), episode(), claim(), AS_OF)
                self.assertTrue(any(expected in error for error in result["errors"]))
                self.assertIsNone(result["proposal"]["value"])

    def test_code_injection_is_a_rejected_literal(self):
        malicious = "__import__('os').system('touch /tmp/evidence_program_should_never_execute')"
        result = execute(program([fact(value=malicious)]), episode(), claim(), AS_OF)
        self.assertTrue(any("invalid_operand_value_scale_or_unit" in error for error in result["errors"]))
        self.assertIsNone(result["proposal"]["value"])

    def test_arbitrary_rendered_patch_not_accepted(self):
        result = execute(program(patch="Revenue was 999999."), episode(), claim(), AS_OF)
        self.assertIn("unknown_program_field", result["errors"])
        self.assertNotIn("999999", result["proposal"]["patch"])

    def test_bounded_facts_are_rejected_before_calculation(self):
        result = execute(program([fact(identifier=f"fact_{i}") for i in range(MAX_FACTS + 1)]), episode(), claim(), AS_OF)
        self.assertIn("invalid_or_overlong_fact_list", result["errors"])
        self.assertEqual(result["cost"]["calculator_calls"], 0)

    def test_duplicate_fact_ids_fail(self):
        result = execute(program([fact(), fact()]), episode(), claim(), AS_OF)
        self.assertIn("duplicate_fact_id:main", result["errors"])

    def test_inputs_unmodified_and_result_json_serializable(self):
        data, prog, request = comparison_case()
        original = copy.deepcopy((data, prog, request))
        result = execute(prog, data, request, AS_OF)
        self.assertEqual((data, prog, request), original)
        json.dumps(result, allow_nan=False)

    def test_malformed_primary_ids_fail_without_raising(self):
        for identifier in ([], {}, 9, True):
            with self.subTest(identifier=identifier):
                result = execute(program(primary_fact_id=identifier), episode(), claim(), AS_OF)
                self.assertIn("primary_fact_id_required", result["errors"])
                self.assertTrue(result["proposal"]["unresolved"])

    def test_overlong_scale_quote_is_bounded(self):
        result = execute(program([fact(scale_quote="x" * 8001)]), episode(), claim(), AS_OF)
        self.assertIn("invalid_fact_fields:main", result["errors"])
        self.assertEqual(result["cost"]["calculator_calls"], 0)


if __name__ == "__main__":
    unittest.main()
