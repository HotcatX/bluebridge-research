"""Small public-only, typed financial evidence program prototype.

The model selects facts, roles and disposition; this backend never looks up a
replacement answer. It checks literal provenance and declared metadata, then
renders only typed facts. Passing these checks is not financial entailment:
table-column, metric, scope and accounting-basis alignment still need review.
No reference answers, claim IDs, previous model outputs or issuer rules are read.

``execute(program, episode, claim, as_of, source_ids=None)`` returns a V5 finish
``proposal`` together with ``audit``, ``errors`` and measured tool ``cost``.
Programs contain facts, primary_fact_id, secondary_fact_ids, operation and an
explicit model-selected disposition. compare_range additionally selects actual,
low and high fact IDs, or midpoint/tolerance IDs with tolerance_mode absolute or
relative; unavailable requires an explicit missing_reason. Facts
use fact_id/chunk_id/quote/value/unit/scale/metric/period/scope/basis/qualifier.
An optional scale_quote can retain a table's original unit header. Optional
context_quotes (at most three exact same-source chunk quotations) can supply
split headers. They are explicit augmented calculator context, never original
row text; their linkage to a row still requires semantic review. Values are
source literals; the calculator alone converts them to a requested output unit.

Supported public temporal declarations are per-metric/top-level period,
actual_period, guidance_period, guidance_vintage (source publication time), and
temporal_policy=latest_reported_quarter. Only that explicit policy resolves a
rolling target from the newest eligible SOURCE publication's public period.
Guidance bounds have a future fact period and need not match source.period.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from .evidence_tools import AUDIT_SCOPE, calculate, eligible_chunks, is_available
from .evidence_tools import audit_proposal


MAX_FACTS = 24
MAX_QUOTE = 8000
MAX_LABEL = 200
MAX_REASON = 1000
QUALIFIERS = ("exact", "at_least", "more_than", "at_most", "less_than", "approximate")
FACT_FIELDS = {"fact_id", "chunk_id", "quote", "value", "unit", "scale", "metric",
               "period", "scope", "basis", "qualifier", "scale_quote", "context_quotes"}
PROGRAM_FIELDS = {"facts", "primary_fact_id", "secondary_fact_ids", "operation",
                  "comparison", "missing_reason", "disposition"}
NUMBER = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.]|,\d)")
CUES = (
    ("at_most", r"(?:at\s+most|no\s+more\s+than|up\s+to)"),
    ("at_least", r"(?:at\s+least|no\s+less\s+than)"),
    ("more_than", r"(?:more\s+than|over|exceed(?:ed|s|ing)?|surpass(?:ed|es|ing)?)"),
    ("less_than", r"(?:less\s+than|under|below)"),
    ("approximate", r"(?:approximately|approx\.?|about|around|nearly|roughly|almost)"),
)
PREFIX = {"exact": "", "at_least": "at least ", "more_than": "more than ",
          "at_most": "at most ", "less_than": "less than ", "approximate": "approximately "}
UNSPECIFIED = {"", "unspecified", "not specified", "any"}


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _label(value: Any, *, empty=False) -> bool:
    return (isinstance(value, str) and len(value) <= MAX_LABEL
            and (empty or bool(value.strip()))
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _explicit(value: Any) -> bool:
    return isinstance(value, str) and _norm(value) not in UNSPECIFIED


def _quarter(value: str):
    text = re.sub(r"[\s_/-]", "", value.upper())
    match = re.fullmatch(r"(?:FY)?(\d{4}|\d{2})Q([1-4])", text)
    if not match:
        match = re.fullmatch(r"Q([1-4])(?:FY)?(\d{4}|\d{2})", text)
        return (match[2], match[1]) if match else None
    return match[1], match[2]


def _same_period(left: str, right: str) -> bool:
    if _norm(left) == _norm(right):
        return True
    a, b = _quarter(left), _quarter(right)
    if not a or not b or a[1] != b[1]:
        return False
    if len(a[0]) == len(b[0]):
        return a[0] == b[0]
    short, long = (a[0], b[0]) if len(a[0]) == 2 else (b[0], a[0])
    return long.startswith("20") and long.endswith(short)


def _publication(value: str) -> dt.datetime:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return dt.datetime.combine(dt.date.fromisoformat(value), dt.time.max, tzinfo=dt.timezone.utc)
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def _same_vintage(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected):
            return actual[:10] == expected and bool(dt.date.fromisoformat(expected))
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", actual):
            return False  # A publication day does not establish an exact time.
        parsed = dt.datetime.fromisoformat(expected.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and _publication(actual) == parsed.astimezone(dt.timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return False


def _public_sources(episode: dict, as_of: str, source_ids) -> dict:
    sources = episode.get("sources", [])
    if not isinstance(sources, list):
        return {}
    counts = Counter(source.get("source_id") for source in sources
                     if isinstance(source, dict) and isinstance(source.get("source_id"), str))
    selected = None if source_ids is None else set(source_ids)
    result = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        identifier = source.get("source_id")
        if (not isinstance(identifier, str) or counts[identifier] != 1
                or (selected is not None and identifier not in selected)
                or not is_available(source.get("published_at"), as_of)):
            continue
        metadata = source.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        result[identifier] = {"published_at": source["published_at"],
                              "period": source.get("period", metadata.get("period"))}
    return result


def _source_qualifiers(fact: dict, text: str, raw_value: str) -> set[str]:
    """Inspect original context, so trimming 'over' out of a quote cannot erase it.

    This is a deliberately small literal-cue check, not semantic qualification.
    Multiple matching quoted occurrences with conflicting cues fail closed.
    """
    wanted = Decimal(raw_value)
    positions = []
    for match in NUMBER.finditer(fact["quote"]):
        try:
            number = Decimal(match[0].replace(",", "").replace("−", "-"))
            before = fact["quote"][:match.start()].rstrip()
            after = fact["quote"][match.end():].lstrip()
            if before.endswith("(") and after.startswith(")"):
                number = -abs(number)
            if number == wanted:
                positions.append(match.start())
        except InvalidOperation:
            continue
    found, offset, occurrences = set(), 0, 0
    while True:
        start = text.find(fact["quote"], offset)
        if start < 0:
            break
        occurrences += 1
        if occurrences > 128:
            return {"ambiguous"}
        for position in positions:
            prefix = text[max(0, start + position - 80):start + position]
            cue = "exact"
            for qualifier, pattern in CUES:
                if re.search(r"\b" + pattern + r"\s*(?:(?:US\$|USD|EUR|\$|€)\s*)?$", prefix, re.I):
                    cue = qualifier
                    break
            found.add(cue)
        offset = start + 1
    return found


def _requested_metrics(claim: dict) -> tuple[dict, list[dict]]:
    raw = claim.get("metric_spec", {}) if isinstance(claim, dict) else {}
    spec = raw if isinstance(raw, dict) else {}
    metrics = spec.get("metrics", [])
    if not isinstance(metrics, list):
        metrics = []
    result = []
    for metric in metrics:
        if isinstance(metric, str):
            result.append({**spec, "metric": metric, "_metric_period_explicit": False,
                           "_metric_actual_period_explicit": False})
        elif isinstance(metric, dict):
            result.append({**spec, **metric, "_metric_period_explicit": "period" in metric,
                           "_metric_actual_period_explicit": "actual_period" in metric})
    if not result and _explicit(spec.get("metric", spec.get("name"))):
        result = [{**spec, "metric": spec.get("metric", spec.get("name"))}]
    return spec, result


def _request_for(fact: dict, metrics: list[dict]) -> dict:
    matches = [request for request in metrics
               if _explicit(request.get("metric")) and _norm(request["metric"]) == _norm(fact["metric"])]
    aligned = [request for request in matches
               if all(not _explicit(request.get(field)) or _norm(request[field]) == _norm(fact[field])
                      for field in ("scope", "basis"))]
    # Repeated metrics can request different historical periods. Declared fact
    # period is a selector, never a substitute for the subsequent source check.
    for request in aligned:
        if _explicit(request.get("period")) and _same_period(fact["period"], request["period"]):
            return request
    if aligned:
        return aligned[0]
    return matches[0] if len(matches) == 1 else {}


def _target_period(request: dict, spec: dict, latest_period, *, bound=False):
    if bound and _explicit(request.get("guidance_period", spec.get("guidance_period"))):
        return request.get("guidance_period", spec.get("guidance_period"))
    if not bound and request.get("_metric_actual_period_explicit"):
        return request.get("actual_period")
    if request.get("_metric_period_explicit"):
        return request.get("period")
    if not bound:
        return request.get("actual_period", spec.get("actual_period")) or latest_period or request.get("period", spec.get("period"))
    return request.get("period", spec.get("period"))


def _display_unit(fact: dict, request: dict) -> str:
    if _explicit(request.get("unit")):
        return request["unit"]
    if fact["scale"] == "percent":
        return "percent"
    if fact["scale"] in ("thousand", "million", "billion"):
        if not re.search(r"\b(?:thousand|million|billion)s?\b", fact["unit"], re.I):
            return fact["unit"] + " " + fact["scale"]
    return fact["unit"] or "ratio"


def _operand(fact: dict, name="fact") -> dict:
    return {"name": name, **{key: fact[key] for key in
            ("chunk_id", "quote", "value", "unit", "scale", "scale_quote", "context_quotes") if key in fact}}


def _render(fact: dict, checked: dict) -> str:
    context = "; ".join(fact[field] for field in ("period", "scope", "basis") if fact[field])
    context = f" ({context})" if context else ""
    return (f"{fact['metric']}{context}: {PREFIX[fact['qualifier']]}"
            f"{checked['display_result']} {checked['display_unit']} [{fact['chunk_id']}].")


def execute(program: Any, episode: dict, claim: dict, as_of: str, source_ids=None) -> dict:
    """Ground, check and render a bounded typed program without model inference.

    Errors produce insufficient/review with null scalar, never a substituted
    value. Nonexact primaries retain their bound/approximation in patch and use
    null value. Successful dispositions are the model's explicit selection.
    Every result requires semantic review even when mechanical checks pass.
    """
    errors, warnings, checks = [], [], []
    cost = {"program_calls": 1, "facts_checked": 0, "calculator_calls": 0,
            "operands_checked": 0, "source_characters_checked": 0,
            "context_quote_checks": 0, "model_calls": 0, "manual_review_required": 1}
    proposal = {"tool": "finish", "disposition": "insufficient", "value": None,
                "unit": None, "citations": [], "reason": "Evidence program requires review.",
                "patch": "The requested evidence program requires review.",
                "evidence_quotes": [], "calculations": [], "status": "review", "unresolved": True}

    def fail(code):
        errors.append(code)

    def finish(facts_audit=None, mechanical=None):
        audit = {"ok": not errors, "flags": sorted(set(errors)), "warnings": sorted(set(warnings)),
                 "facts": facts_audit or [], "checks": checks,
                 "semantic_entailment_checked": False, "manual_review_required": True,
                 "scope": AUDIT_SCOPE + " Typed rendering does not validate financial labels or table columns."}
        if mechanical is not None:
            audit["proposal_audit"] = mechanical
        if isinstance(program, dict) and program.get("operation") == "unavailable" and isinstance(program.get("missing_reason"), str):
            audit["unverified_model_missing_reason"] = program["missing_reason"][:MAX_REASON]
        return {"proposal": proposal, "audit": audit, "errors": sorted(set(errors)), "cost": cost}

    if not isinstance(program, dict):
        fail("invalid_program")
        return finish()
    if any(not isinstance(key, str) or key not in PROGRAM_FIELDS for key in program):
        fail("unknown_program_field")
    if not isinstance(episode, dict) or not isinstance(claim, dict):
        fail("invalid_public_input")
        return finish()
    if source_ids is not None and (not isinstance(source_ids, (list, tuple, set))
                                   or not all(isinstance(item, str) for item in source_ids)):
        fail("invalid_source_ids")
        return finish()
    raw_facts = program.get("facts")
    if not isinstance(raw_facts, list) or len(raw_facts) > MAX_FACTS:
        fail("invalid_or_overlong_fact_list")
        return finish()
    operation = program.get("operation")
    if operation not in ("report", "compare_range", "unavailable"):
        fail("invalid_operation")
    disposition = program.get("disposition")
    if operation != "unavailable" and disposition not in ("retain", "update", "insufficient"):
        fail("missing_or_invalid_disposition")
    reason = program.get("missing_reason")
    if operation == "unavailable" and (not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON):
        fail("explicit_missing_reason_required")
    comparison = program.get("comparison")
    roles, role_keys, tolerance_mode = {}, (), None
    literal_keys = ("actual_fact_id", "low_fact_id", "high_fact_id")
    tolerance_keys = ("actual_fact_id", "midpoint_fact_id", "tolerance_fact_id")
    if operation == "compare_range":
        if isinstance(comparison, dict) and set(comparison) == set(literal_keys):
            role_keys = literal_keys
        elif (isinstance(comparison, dict) and set(comparison) == {*tolerance_keys, "tolerance_mode"}
              and comparison.get("tolerance_mode") in ("absolute", "relative")):
            role_keys, tolerance_mode = tolerance_keys, comparison["tolerance_mode"]
        if not role_keys or not all(isinstance(comparison[key], str) for key in role_keys):
            fail("invalid_comparison")
        else:
            roles = {key: comparison[key] for key in role_keys}
            if len(set(roles.values())) != 3:
                fail("comparison_requires_distinct_facts")
    elif comparison is not None:
        fail("comparison_only_allowed_for_compare_range")
    chunks = {chunk["chunk_id"]: chunk for chunk in eligible_chunks(episode, as_of, source_ids)}
    sources = _public_sources(episode, as_of, source_ids)
    # Only this separate internal packet admits transparently synthetic binding
    # chunks. Public citations always refer to unchanged original source chunks.
    calculator_episode = {"sources": episode.get("sources", []),
                          "chunks": list(episode.get("chunks", [])) if isinstance(episode.get("chunks"), list) else []}
    bindings, contexts = {}, {}

    def bound_operand(fact, name="fact"):
        operand = bindings.get(fact["fact_id"], _operand(fact)).copy()
        operand["name"] = name
        return operand
    spec, requests = _requested_metrics(claim)
    latest_period = None
    if spec.get("temporal_policy") == "latest_reported_quarter":
        newest = max((_publication(source["published_at"]) for source in sources.values()), default=None)
        periods = [source.get("period") for source in sources.values()
                   if _publication(source["published_at"]) == newest]
        if not periods or not all(_explicit(period) for period in periods):
            fail("latest_public_source_period_unavailable")
        elif not all(_same_period(periods[0], period) for period in periods):
            fail("ambiguous_latest_public_source_period")
        else:
            latest_period = periods[0]
            checks.append({"field": "period", "expected": latest_period,
                           "resolution": "newest_eligible_public_source_publication"})
    facts, fact_audits, valid, calculated = {}, [], {}, {}
    for index, fact in enumerate(raw_facts):
        if not isinstance(fact, dict):
            fail(f"invalid_fact:{index}")
            continue
        identifier = fact.get("fact_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", identifier):
            fail(f"invalid_fact_id:{index}")
            continue
        if identifier in facts:
            fail("duplicate_fact_id:" + identifier)
            valid.pop(identifier, None)
            continue
        facts[identifier] = fact
        fields_ok = (not any(key not in FACT_FIELDS for key in fact)
                     and all(_label(fact.get(field), empty=field in ("period", "scope", "basis", "unit"))
                             for field in ("chunk_id", "metric", "period", "scope", "basis", "unit"))
                     and isinstance(fact.get("quote"), str) and 0 < len(fact["quote"]) <= MAX_QUOTE
                     and ("scale_quote" not in fact or (isinstance(fact["scale_quote"], str) and len(fact["scale_quote"]) <= MAX_QUOTE))
                     and isinstance(fact.get("scale"), str) and fact["scale"] in ("one", "percent", "thousand", "million", "billion")
                     and isinstance(fact.get("qualifier"), str) and fact["qualifier"] in QUALIFIERS)
        if not fields_ok:
            fail("invalid_fact_fields:" + identifier)
            continue
        supplied_context = fact.get("context_quotes", [])
        if not isinstance(supplied_context, list) or len(supplied_context) > 3:
            fail("invalid_context_quotes:" + identifier)
            continue
        original = chunks.get(fact["chunk_id"])
        context_ok, admitted = True, []
        for item in supplied_context:
            cost["context_quote_checks"] += 1
            if (not isinstance(item, dict) or set(item) != {"chunk_id", "quote"}
                    or not isinstance(item.get("chunk_id"), str)
                    or not isinstance(item.get("quote"), str) or not item["quote"].strip()
                    or len(item["quote"]) > MAX_QUOTE):
                fail("invalid_context_quote:" + identifier)
                context_ok = False
                continue
            header = chunks.get(item["chunk_id"])
            if not header:
                fail("unknown_or_ineligible_context_chunk:" + identifier)
                context_ok = False
            elif not original or header["source_id"] != original["source_id"]:
                fail("cross_source_context_quote:" + identifier)
                context_ok = False
            elif item["quote"] not in header["text"]:
                fail("context_quote_missing:" + identifier)
                context_ok = False
            else:
                admitted.append({"chunk_id": item["chunk_id"], "quote": item["quote"]})
        if not context_ok:
            continue
        binding_audit = None
        if admitted:
            if not original or fact["quote"] not in original["text"] or not fact["quote"].strip():
                fail("context_original_quote_missing:" + identifier)
                continue
            if fact.get("scale_quote") and fact["scale_quote"] not in original["text"]:
                fail("context_original_scale_quote_missing:" + identifier)
                continue
            internal_id = "__typed_augmented_context_" + str(index)
            while internal_id in chunks:
                internal_id += "_"
            scale_context = "\n".join(([fact["scale_quote"]] if fact.get("scale_quote") else [])
                                      + [item["quote"] for item in admitted])
            calculator_episode["chunks"].append({"chunk_id": internal_id, "source_id": original["source_id"],
                                                  "text": fact["quote"] + "\n" + scale_context,
                                                  "metadata": original.get("metadata", {}).copy()})
            bindings[identifier] = {**_operand(fact), "chunk_id": internal_id, "scale_quote": scale_context}
            contexts[identifier] = admitted
            binding_audit = {"kind": "explicit_augmented_calculator_context", "original_chunk_id": fact["chunk_id"],
                             "original_quote": fact["quote"], "context_quotes": admitted,
                             "internal_binding_chunk_id": internal_id, "original_text_modified": False,
                             "header_to_row_semantic_link_checked": False}
            warnings.append("header_to_row_context_link_requires_semantic_review:" + identifier)
        request = _request_for(fact, requests)
        checked = calculate("fact", [bound_operand(fact)], calculator_episode, as_of, source_ids=source_ids,
                            # A relative tolerance may name the same financial
                            # metric while having a different physical unit.
                            output_unit=_display_unit(fact, {} if identifier == roles.get("tolerance_fact_id") else request))
        cost["facts_checked"] += 1
        for key in ("calculator_calls", "operands_checked", "source_characters_checked"):
            cost[key] += checked["cost"].get(key, 0)
        row = {"fact_id": identifier, "calculation": checked, "qualifier": fact["qualifier"]}
        if binding_audit:
            row["context_binding"] = binding_audit
        fact_audits.append(row)
        if not checked["ok"]:
            errors.extend("fact:" + identifier + ":" + flag for flag in checked["flags"])
            continue
        cues = _source_qualifiers(fact, chunks[fact["chunk_id"]]["text"], checked["provenance"][0]["value"])
        row["source_literal_qualifiers"] = sorted(cues)
        if cues != {fact["qualifier"]}:
            fail("source_qualifier_mismatch:" + identifier)
            continue
        valid[identifier], calculated[identifier] = fact, checked
    primary = program.get("primary_fact_id")
    secondary = program.get("secondary_fact_ids", [])
    if not isinstance(secondary, list) or len(secondary) > MAX_FACTS or not all(isinstance(item, str) for item in secondary):
        fail("invalid_secondary_fact_ids")
        secondary = []
    if len(secondary) != len(set(secondary)):
        fail("duplicate_secondary_fact_id")
    if operation == "unavailable":
        if primary is not None:
            fail("unavailable_must_not_select_primary")
        order = list(dict.fromkeys(secondary))
    else:
        if not isinstance(primary, str):
            fail("primary_fact_id_required")
            primary = None
        order = list(dict.fromkeys(([primary] if isinstance(primary, str) else []) + secondary))
    if roles:
        if roles["actual_fact_id"] != primary:
            fail("comparison_primary_must_be_actual")
        order = list(dict.fromkeys([*(roles[key] for key in role_keys), *secondary]))
    for identifier in order:
        if identifier not in facts:
            fail("unknown_fact_id:" + identifier)
    for identifier in sorted(set(facts) - set(order)):
        warnings.append("unused_fact:" + identifier)
    bounds = {identifier for key, identifier in roles.items() if key != "actual_fact_id"}
    for identifier in order:
        if identifier not in valid:
            continue
        fact = valid[identifier]
        before = len(errors)
        request = _request_for(fact, requests)
        for field in ("scope", "basis"):
            if _explicit(request.get(field)) and _norm(request[field]) != _norm(fact[field]):
                fail("requested_" + field + "_mismatch:" + identifier)
        target = _target_period(request, spec, latest_period, bound=identifier in bounds)
        if _explicit(target):
            if re.search(r"\b(?:latest|current|most recent)\b", target, re.I):
                fail("unresolved_public_period_target:" + identifier)
            elif not _same_period(fact["period"], target):
                fail("requested_period_mismatch:" + identifier)
            source = sources[chunks[fact["chunk_id"]]["source_id"]]
            source_period = source.get("period")
            if identifier not in bounds and _explicit(source_period) and not _same_period(source_period, target):
                fail("public_source_period_mismatch:" + identifier)
            if identifier not in bounds and not _explicit(source_period):
                warnings.append("public_source_period_unavailable:" + identifier)
            checks.append({"fact_id": identifier, "field": "period", "expected": target,
                           "source_period": source_period, "guidance_bound": identifier in bounds})
        vintage = request.get("guidance_vintage", spec.get("guidance_vintage"))
        if identifier in bounds and _explicit(vintage):
            published = chunks[fact["chunk_id"]]["published_at"]
            if not _same_vintage(published, vintage):
                fail("original_guidance_vintage_mismatch:" + identifier)
            checks.append({"fact_id": identifier, "field": "guidance_vintage",
                           "expected": vintage, "source_published_at": published})
        metadata = chunks[fact["chunk_id"]].get("metadata", {})
        if _explicit(request.get("basis")) and _explicit(metadata.get("basis")) and _norm(request["basis"]) != _norm(metadata["basis"]):
            fail("public_source_basis_mismatch:" + identifier)
        if len(errors) != before:
            valid.pop(identifier, None)
    if operation != "unavailable" and primary in valid and requests:
        primary_request = requests[0]
        if (_explicit(primary_request.get("metric"))
                and _norm(valid[primary]["metric"]) != _norm(primary_request["metric"])):
            fail("primary_requested_metric_mismatch")
        for field in ("scope", "basis"):
            if _explicit(primary_request.get(field)) and _norm(primary_request[field]) != _norm(valid[primary][field]):
                fail("primary_requested_" + field + "_mismatch")
    if operation != "unavailable":
        for index, request in enumerate(requests):
            requested_period = _target_period(request, spec, latest_period)
            if _explicit(request.get("metric")) and not any(
                    identifier in valid and _norm(valid[identifier]["metric"]) == _norm(request["metric"])
                    and all(not _explicit(request.get(field)) or _norm(valid[identifier][field]) == _norm(request[field])
                            for field in ("scope", "basis"))
                    and (not _explicit(requested_period) or _same_period(valid[identifier]["period"], requested_period))
                    for identifier in order):
                fail(f"missing_requested_metric:{index}")
    relation, derived_range, range_calculations = None, None, []
    if roles and all(identifier in valid for identifier in roles.values()):
        actual_fact = valid[roles["actual_fact_id"]]
        for field in ("metric", "scope", "basis", "period"):
            aligned = [identifier for key, identifier in roles.items() if key != "actual_fact_id"
                       and not (field == "metric" and key == "tolerance_fact_id")]
            if any(not (_same_period(actual_fact[field], valid[identifier][field]) if field == "period"
                        else _norm(actual_fact[field]) == _norm(valid[identifier][field]))
                   for identifier in aligned):
                fail("comparison_declared_" + field + "_mismatch")
        if any(valid[identifier]["qualifier"] != "exact" for identifier in roles.values()):
            fail("comparison_requires_exact_operands")
        elif tolerance_mode:
            tolerance = calculated[roles["tolerance_fact_id"]]
            if Decimal(tolerance["result"]) < 0:
                fail("comparison_negative_tolerance")
            if tolerance_mode == "relative" and tolerance["unit"] != "ratio":
                fail("comparison_relative_tolerance_requires_ratio")
            operands = [bound_operand(valid[roles[key]], name) for key, name in
                        (("midpoint_fact_id", "midpoint"), ("tolerance_fact_id", "tolerance"))]
            public_operands = [_operand(valid[roles[key]], name) for key, name in
                               (("midpoint_fact_id", "midpoint"), ("tolerance_fact_id", "tolerance"))]
            delta = "tolerance" if tolerance_mode == "absolute" else "midpoint*tolerance"
            bounds_checked = []
            for expression in ("midpoint-(" + delta + ")", "midpoint+(" + delta + ")"):
                output_unit = calculated[roles["actual_fact_id"]]["display_unit"]
                checked = calculate(expression, operands, calculator_episode, as_of, source_ids=source_ids, output_unit=output_unit)
                for key in ("calculator_calls", "operands_checked", "source_characters_checked"):
                    cost[key] += checked["cost"].get(key, 0)
                errors.extend("comparison:" + flag for flag in checked["flags"])
                if checked["ok"]:
                    bounds_checked.append(checked)
                    range_calculations.append({"expression": expression, "operands": public_operands, "output_unit": output_unit})
            if len(bounds_checked) == 2:
                low, high = (Decimal(row["result"]) for row in bounds_checked)
                actual = Decimal(calculated[roles["actual_fact_id"]]["result"])
                if low > high:
                    fail("comparison_range_reversed")
                else:
                    relation = "below" if actual < low else "above" if actual > high else "within"
                    derived_range = {"low": bounds_checked[0]["display_result"],
                                     "high": bounds_checked[1]["display_result"],
                                     "unit": bounds_checked[0]["display_unit"], "tolerance_mode": tolerance_mode,
                                     "operand_fact_ids": [roles[key] for key in tolerance_keys],
                                     "source_bound_calculations": bounds_checked}
                    checks.append({"field": "derived_range", **derived_range})
        else:
            operands = [bound_operand(valid[roles[key]], name) for key, name in
                        (("actual_fact_id", "actual"), ("low_fact_id", "low"), ("high_fact_id", "high"))]
            differences = []
            for expression in ("high-low", "actual-low", "actual-high"):
                checked = calculate(expression, operands, calculator_episode, as_of, source_ids=source_ids)
                for key in ("calculator_calls", "operands_checked", "source_characters_checked"):
                    cost[key] += checked["cost"].get(key, 0)
                errors.extend("comparison:" + flag for flag in checked["flags"])
                if checked["ok"]:
                    differences.append(Decimal(checked["result"]))
            if len(differences) == 3:
                if differences[0] < 0:
                    fail("comparison_range_reversed")
                else:
                    relation = "below" if differences[1] < 0 else "above" if differences[2] > 0 else "within"
    warnings.append("metric_period_scope_basis_and_table_column_alignment_require_semantic_review")
    if operation == "unavailable":
        warnings.append("model_missing_reason_is_not_independent_proof_of_absence")
    selected = [identifier for identifier in order if identifier in valid]
    if errors:
        selected = [identifier for identifier in secondary if identifier in valid and identifier != primary]
    proposal["citations"] = sorted({valid[identifier]["chunk_id"] for identifier in selected}
                                   | {item["chunk_id"] for identifier in selected for item in contexts.get(identifier, [])})
    proposal["evidence_quotes"] = [{"chunk_id": valid[identifier]["chunk_id"], "quote": valid[identifier]["quote"]}
                                   for identifier in selected]
    proposal["evidence_quotes"].extend(item.copy() for identifier in selected for item in contexts.get(identifier, []))
    rendered = []
    for identifier in selected:
        label = ("Guidance midpoint: " if tolerance_mode and identifier == roles.get("midpoint_fact_id") else
                 "Guidance tolerance: " if tolerance_mode and identifier == roles.get("tolerance_fact_id") else "")
        rendered.append(label + _render(valid[identifier], calculated[identifier]))
    if errors:
        proposal["patch"] = "The requested evidence program requires review." + (" Ancillary source-linked facts: " + " ".join(rendered) if rendered else "")
        return finish(fact_audits)
    if operation == "unavailable":
        proposal.update(status="unavailable", unresolved=True,
                        reason="The model explicitly reports the primary fact unavailable; absence requires review.",
                        patch="The requested primary fact is unavailable from the supplied evidence." + (" Ancillary source-linked facts: " + " ".join(rendered) if rendered else ""))
    else:
        if primary not in valid:
            fail("primary_fact_not_valid")
            return finish(fact_audits)
        fact, checked = valid[primary], calculated[primary]
        proposal.update(disposition=disposition, status="mechanically_checked", unresolved=False,
                        value=checked["display_result"] if fact["qualifier"] == "exact" else None,
                        unit=checked["display_unit"], reason="Rendered from model-selected, mechanically checked typed facts.",
                        patch=" ".join(rendered),
                        calculations=[{"expression": "fact", "operands": [_operand(fact)], "output_unit": checked["display_unit"]},
                                      *range_calculations])
        if relation:
            ids = " ".join("[" + valid[roles[key]]["chunk_id"] + "]" for key in role_keys)
            if derived_range:
                proposal["patch"] += (f" Derived cited range: {derived_range['low']} to {derived_range['high']} "
                                       f"{derived_range['unit']} {ids}.")
            proposal["patch"] += f" The cited actual is {relation} the cited range {ids}."
    # Legacy audit sees only the primary output unit. Temporal checks above are
    # per role, preventing guidance source vintage from becoming an actual period.
    mechanical = None
    if proposal["citations"]:
        contextual = any(identifier in contexts for identifier in selected)
        # The legacy calculator accepts only one chunk per operand. A contextual
        # program therefore uses its public quote/citation audit, while the typed
        # scalar/unit checks above use transparent augmented bindings. Never pass
        # a fake original chunk or pretend the legacy audit checked split headers.
        audited_proposal = {**proposal, "calculations": [], "value": None, "unit": None} if contextual else proposal
        mechanical = audit_proposal(audited_proposal, episode, as_of, source_ids=source_ids,
                                    metric_spec={"unit": proposal["unit"]} if proposal["unit"] and not contextual else {})
        if contextual:
            mechanical["contextual_scope"] = "Public original citations and quotes only; typed scalar/unit grounding uses explicit augmented calculator bindings recorded per fact."
            mechanical["scalar_unit_validation"] = "typed_calculator_with_explicit_augmented_context"
        cost["audit_calls"] = mechanical["cost"]["audit_calls"]
        cost["calculator_calls"] += mechanical["cost"]["calculator_calls"]
        for checked in mechanical["calculations"]:
            for key in ("operands_checked", "source_characters_checked"):
                cost[key] += checked["cost"].get(key, 0)
        if not mechanical["ok"]:
            errors.extend("proposal_audit:" + flag for flag in mechanical["flags"])
            proposal.update(value=None, disposition="insufficient", status="review", unresolved=True,
                            reason="Evidence program requires review.", patch="The requested evidence program requires review.")
    return finish(fact_audits, mechanical)
