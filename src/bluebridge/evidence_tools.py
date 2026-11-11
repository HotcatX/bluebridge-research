"""Public-only evidence tools for the V6 research-note experiment.

These functions never inspect reference answers, split labels or claim families.
They return fresh JSON-compatible dictionaries and never mutate their inputs.
An audit verifies mechanical provenance and declared metadata, not entailment.
The traditional lookup baseline deliberately returns ``review`` when a literal
table lookup cannot establish a unique answer under the public metric request.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
import datetime as dt
from decimal import Decimal, DecimalException, localcontext
import math
import re
from typing import Any


AUDIT_SCOPE = (
    "Mechanical citation, quote, numeric-operand, unit and declared-metadata "
    "checks only; no financial or narrative entailment judgment."
)
MAX_EXPRESSION = 500
MAX_AST_NODES = 80
MAX_DEPTH = 12
MAX_OPERANDS = 32
MAX_ABS = Decimal("1e30")
SCALES = {
    "one": Decimal(1), "percent": Decimal("0.01"),
    "thousand": Decimal(1000), "million": Decimal(1000000),
    "billion": Decimal(1000000000),
}
SCALE_WORDS = {
    "percent": r"%|\bpercent(?:age)?\b", "thousand": r"\bthousands?\b",
    "million": r"\bmillions?\b", "billion": r"\bbillions?\b",
}
NUMBER = re.compile(
    r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?![\w.]|,\d)"
)
STOP = set("a an the of to in on and or for from by with as at is are was were "
           "be been what which how did does do please this that my our us it its".split())
PUBLIC_CHUNK_FIELDS = ("chunk_id", "source_id", "text", "locator")
METADATA_FIELDS = ("entity", "period", "basis", "unit")


def _time(value: Any) -> tuple[dt.datetime, bool]:
    if not isinstance(value, str):
        raise ValueError("A dated source and cutoff are required")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        day = dt.date.fromisoformat(value)
        return dt.datetime.combine(day, dt.time.max, tzinfo=dt.timezone.utc), True
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Intraday times require an explicit timezone")
    return parsed.astimezone(dt.timezone.utc), False


def is_available(published_at: Any, as_of: Any) -> bool:
    """Date-only publications are unavailable for intraday same-day cutoffs."""
    try:
        published, pub_day = _time(published_at)
        cutoff, cutoff_day = _time(as_of)
        return not (pub_day and not cutoff_day and published.date() == cutoff.date()) and published <= cutoff
    except (ValueError, TypeError, OverflowError):
        return False


def eligible_chunks(episode: dict, as_of: str, source_ids=None) -> list[dict]:
    """Explicit public allowlist; unknown/duplicate IDs fail closed.

    ``source_ids`` optionally restricts the already-discovered source universe;
    publication eligibility alone does not imply discovery by an event.
    """
    sources = episode.get("sources", [])
    chunks = episode.get("chunks", [])
    if not isinstance(sources, list) or not isinstance(chunks, list):
        return []
    counts = Counter(s.get("source_id") for s in sources if isinstance(s, dict) and isinstance(s.get("source_id"), str))
    selected = None if source_ids is None else set(source_ids)
    by_source = {
        s["source_id"]: s for s in sources
        if isinstance(s, dict) and isinstance(s.get("source_id"), str)
        and counts[s["source_id"]] == 1
        and (selected is None or s["source_id"] in selected)
        and is_available(s.get("published_at"), as_of)
    }
    chunk_counts = Counter(c.get("chunk_id") for c in chunks if isinstance(c, dict) and isinstance(c.get("chunk_id"), str))
    result = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
            continue
        if chunk_counts[chunk["chunk_id"]] != 1 or not isinstance(chunk.get("source_id"), str) or chunk["source_id"] not in by_source or not isinstance(chunk.get("text"), str):
            continue
        source = by_source[chunk["source_id"]]
        public = {key: copy.deepcopy(chunk[key]) for key in PUBLIC_CHUNK_FIELDS if key in chunk}
        public["published_at"] = source["published_at"]
        if isinstance(source.get("url"), str):
            public["url"] = source["url"]
        # Only explicitly public descriptive metadata is admitted, never tags.
        metadata = {}
        for obj in (source, chunk):
            supplied = obj.get("metadata", {})
            for field in METADATA_FIELDS:
                value = obj.get(field, supplied.get(field) if isinstance(supplied, dict) else None)
                if isinstance(value, str) and value:
                    metadata[field] = value
        if metadata:
            public["metadata"] = metadata
        result.append(public)
    return sorted(result, key=lambda c: c["chunk_id"])


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).casefold())


def _table_rows(text: str) -> list[str]:
    """Header-bearing retrieval units; returned evidence remains full chunks."""
    lines = text.splitlines()
    rows, header = [], []
    for line in lines:
        if "|" in line or "\t" in line:
            if not re.fullmatch(r"[\s|:\-+]+", line):
                if not NUMBER.search(line) or not header:
                    header = [*header[-2:], line]
                rows.append("\n".join([*header, line]))
        elif line.strip():
            header = [line]
    return rows


def retrieve(episode: dict, query: str, as_of: str, *, k=6, source_ids=None) -> dict:
    """Deterministic BM25 over chunks and header-bearing table rows.

    Whole original chunks are returned, so a high-scoring row never loses its
    table's headers or units. Cost counts the actual whole-chunk context.
    """
    if not isinstance(query, str) or len(query) > 4000:
        raise ValueError("Query must be a string of at most 4000 characters")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 1000:
        raise ValueError("k must be an integer from 0 through 1000")
    chunks = eligible_chunks(episode, as_of, source_ids)
    units = []
    for chunk in chunks:
        units.append((chunk["chunk_id"], "chunk", chunk["text"]))
        units.extend((chunk["chunk_id"], "table_row", row) for row in _table_rows(chunk["text"]))
    counts = [Counter(tokens(unit[2])) for unit in units]
    lengths = [sum(row.values()) for row in counts]
    avg = sum(lengths) / max(len(lengths), 1)
    query_terms = sorted(set(tokens(query)) - STOP)
    frequencies = {term: sum(term in row for row in counts) for term in query_terms}
    ranked = {}
    for (chunk_id, kind, text), row, length in zip(units, counts, lengths):
        score = 0.0
        for term in query_terms:
            frequency = row[term]
            if frequency:
                idf = math.log(1 + (len(units) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / max(avg, 1)))
        candidate = {"chunk_id": chunk_id, "score": round(score, 12), "matched_kind": kind, "matched_text": text}
        previous = ranked.get(chunk_id)
        if previous is None or (score, kind == "table_row", text) > (previous["score"], previous["matched_kind"] == "table_row", previous["matched_text"]):
            ranked[chunk_id] = candidate
    order = sorted(ranked.values(), key=lambda row: (-row["score"], row["chunk_id"]))
    # An empty/nonmatching query does not fabricate relevant evidence.
    order = [row for row in order if row["score"] > 0][:k]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    selected = [copy.deepcopy(by_id[row["chunk_id"]]) for row in order]
    return {
        "chunks": selected, "ranking": order,
        "cost": {"search_calls": 1, "eligible_chunks": len(chunks), "retrieval_units_scored": len(units),
                 "corpus_characters_scored": sum(len(unit[2]) for unit in units), "returned_chunks": len(selected),
                 "returned_characters": sum(len(chunk["text"]) for chunk in selected), "model_calls": 0},
    }


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Expected a finite decimal")
    text = str(value).strip().replace(",", "").replace("−", "-")
    if len(text) > 80 or not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?", text):
        raise ValueError("Invalid decimal literal")
    number = Decimal(text.removesuffix("%"))
    if not number.is_finite() or abs(number) > MAX_ABS or (number and abs(number) < Decimal("1e-30")):
        raise ValueError("Decimal outside calculator limits")
    return number


def _format(number: Decimal) -> str:
    if not number:
        return "0"
    return format(number, "f").rstrip("0").rstrip(".") if number.as_tuple().exponent < 0 else format(number, "f")


def _scale(value: Any) -> tuple[str, Decimal]:
    if isinstance(value, str) and value.casefold() in SCALES:
        key = value.casefold()
        return key, SCALES[key]
    try:
        number = _decimal(value)
    except (ValueError, DecimalException):
        raise ValueError("Unsupported scale") from None
    for name, factor in SCALES.items():
        if factor == number:
            return name, factor
    raise ValueError("Unsupported scale")


def _unit(value: Any) -> tuple[tuple, str, Decimal]:
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("A bounded explicit unit is required")
    text = " ".join(value.casefold().split())
    if text in ("", "ratio", "fraction", "dimensionless", "number"):
        return (), "ratio", Decimal(1)
    if text in ("%", "percent", "percentage", "percentage points", "pp"):
        return (), "ratio", SCALES["percent"]
    factor = Decimal(1)
    for name in ("thousand", "million", "billion"):
        if re.search(r"\b" + name + r"s?\b", text):
            factor = SCALES[name]
            text = re.sub(r"\b" + name + r"s?\b", "", text).strip()
            break
    aliases = {"$": "USD", "usd": "USD", "us$": "USD", "dollars": "USD", "us dollars": "USD",
               "eur": "EUR", "€": "EUR", "shares": "shares", "share": "shares",
               "units": "count", "unit": "count", "count": "count", "": "count"}
    base = aliases.get(text, text)
    return ((base, 1),), base, factor


def _dimension(left: tuple, right: tuple, sign=1) -> tuple:
    result = Counter(dict(left))
    for key, power in right:
        result[key] += sign * power
    return tuple(sorted((key, power) for key, power in result.items() if power))


def _unit_name(dimensions: tuple) -> str:
    if not dimensions:
        return "ratio"
    return " * ".join(key if power == 1 else f"{key}^{power}" for key, power in dimensions)


def _unit_in_context(unit: str, text: str) -> bool:
    dimensions, _, factor = _unit(unit)
    if not dimensions:
        return factor == 1 or bool(re.search(SCALE_WORDS["percent"], text, flags=re.IGNORECASE))
    base = dimensions[0][0]
    if base == "USD":
        located = "$" in text or bool(re.search(r"\b(?:USD|dollars?)\b", text, flags=re.IGNORECASE))
    else:
        located = _literal(base, text)
    return located and (factor == 1 or any(factor == SCALES[name] and re.search(pattern, text, flags=re.IGNORECASE)
                                          for name, pattern in SCALE_WORDS.items()))


def _number_matches(text: str, number: Decimal) -> list[tuple[int, int]]:
    found = []
    for match in NUMBER.finditer(text):
        try:
            raw = _decimal(match.group())
            before, after = text[:match.start()].rstrip(), text[match.end():].lstrip()
            if before.endswith("(") and after.startswith(")"):
                raw = -abs(raw)
            if raw == number:
                found.append((match.start(), match.end()))
        except (ValueError, DecimalException):
            pass
    return found


def _binding(operand: dict, by_id: dict) -> tuple[Decimal, tuple, dict]:
    if not isinstance(operand, dict):
        raise ValueError("invalid_operand")
    name = operand.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", name):
        raise ValueError("invalid_operand_name")
    chunk_id, quote = operand.get("chunk_id"), operand.get("quote")
    if not isinstance(chunk_id, str) or chunk_id not in by_id:
        raise ValueError(f"unknown_or_ineligible_operand_chunk:{name}")
    chunk = by_id[chunk_id]
    if operand.get("source_id") is not None and operand["source_id"] != chunk["source_id"]:
        raise ValueError(f"operand_source_id_mismatch:{name}")
    if not isinstance(quote, str) or not quote.strip() or len(quote) > 8000 or quote not in chunk["text"]:
        raise ValueError(f"operand_quote_missing:{name}")
    try:
        number = _decimal(operand.get("value"))
        scale_name, factor = _scale(operand.get("scale", "one"))
        dimensions, base, declared_factor = _unit(operand.get("unit", ""))
    except (ValueError, DecimalException):
        raise ValueError(f"invalid_operand_value_scale_or_unit:{name}") from None
    matches = _number_matches(quote, number)
    if not matches:
        raise ValueError(f"operand_number_absent_from_quote:{name}")
    scale_quote = operand.get("scale_quote", "")
    if not isinstance(scale_quote, str) or (scale_quote and scale_quote not in chunk["text"]):
        raise ValueError(f"scale_quote_missing:{name}")
    scale_context = quote + "\n" + scale_quote
    nearby_scales = set()
    for _, end in matches:
        after = quote[end:].lstrip()
        for label, pattern in SCALE_WORDS.items():
            if re.match("(?:" + pattern + ")", after, flags=re.IGNORECASE):
                nearby_scales.add(label)
    if nearby_scales and scale_name not in nearby_scales:
        raise ValueError(f"operand_scale_conflict:{name}")
    if scale_name != "one" and not re.search(SCALE_WORDS[scale_name], scale_context, flags=re.IGNORECASE):
        raise ValueError(f"operand_scale_unsubstantiated:{name}")
    if declared_factor != 1 and declared_factor != factor:
        raise ValueError(f"operand_unit_scale_conflict:{name}")
    metadata_unit = chunk.get("metadata", {}).get("unit")
    if metadata_unit is not None and _unit(metadata_unit)[0] != dimensions:
        raise ValueError(f"operand_source_unit_conflict:{name}")
    if operand.get("unit") and not _unit_in_context(operand["unit"], scale_context):
        if metadata_unit is None or _unit(metadata_unit) != _unit(operand["unit"]):
            raise ValueError(f"operand_unit_unsubstantiated:{name}")
    if str(operand.get("value", "")).strip().endswith("%") and scale_name != "percent":
        raise ValueError(f"operand_percent_scale_required:{name}")
    normalized = number * factor
    if abs(normalized) > MAX_ABS:
        raise ValueError(f"operand_out_of_bounds:{name}")
    return normalized, dimensions, {
        "name": name, "value": _format(number), "normalized_value": _format(normalized),
        "scale": scale_name, "unit": base, "chunk_id": chunk_id, "quote": quote,
        **({"scale_quote": scale_quote} if scale_quote else {}),
    }


def _calculate_grounded(expression: str, operands: list, chunk_provider, *, output_unit=None) -> dict:
    """Evaluate bounded decimal arithmetic after grounding every operand.

    Names denote normalized base-unit quantities. Numeric AST literals are
    accepted only if they exactly match a grounded raw operand and its unit is
    unambiguous. There are no implicit conversion constants or executable calls.
    Prefer ``(new-old)/old`` with ``output_unit='percent'`` for percentage growth.
    """
    result = {"ok": False, "result": None, "unit": None, "display_result": None,
              "display_unit": output_unit, "flags": [], "provenance": [],
              "semantic_entailment_checked": False, "scope": AUDIT_SCOPE,
              "cost": {"calculator_calls": 1, "operands_checked": 0, "source_characters_checked": 0, "model_calls": 0}}
    try:
        if not isinstance(expression, str) or not expression.strip() or len(expression) > MAX_EXPRESSION:
            raise ValueError("invalid_or_overlong_expression")
        if not isinstance(operands, list) or not 1 <= len(operands) <= MAX_OPERANDS:
            raise ValueError("invalid_operand_list")
        by_id = {chunk["chunk_id"]: chunk for chunk in chunk_provider()}
        bound = {}
        raw_literals = {}
        with localcontext() as context:
            context.prec = 40
            for operand in operands:
                value, dimensions, provenance = _binding(operand, by_id)
                name = provenance["name"]
                if name in bound:
                    raise ValueError("duplicate_operand_name:" + name)
                bound[name] = (value, dimensions)
                raw_literals.setdefault(_decimal(provenance["value"]), set()).add((value, dimensions))
                result["provenance"].append(provenance)
                result["cost"]["operands_checked"] += 1
                result["cost"]["source_characters_checked"] += len(provenance["quote"]) + len(provenance.get("scale_quote", ""))
            try:
                tree = ast.parse(expression, mode="eval")
            except (SyntaxError, RecursionError):
                raise ValueError("invalid_expression") from None
            if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
                raise ValueError("expression_too_complex")
            used = set()

            def source_literal(node, sign=1):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise ValueError("unsupported_literal")
                value = _decimal(ast.get_source_segment(expression, node)) * sign
                matches = raw_literals.get(value, set())
                if not matches:
                    raise ValueError("numeric_literal_not_source_bound")
                if len(matches) != 1:
                    raise ValueError("numeric_literal_unit_ambiguous")
                normalized, dimensions = next(iter(matches))
                if normalized != value:
                    raise ValueError("scaled_operand_requires_name")
                return value, dimensions

            def visit(node, depth=0):
                if depth > MAX_DEPTH:
                    raise ValueError("expression_too_deep")
                if isinstance(node, ast.Name):
                    if node.id not in bound:
                        raise ValueError("unknown_operand:" + node.id)
                    used.add(node.id)
                    return bound[node.id]
                if isinstance(node, ast.Constant):
                    # Parse the lexical literal, not Python's binary float.
                    return source_literal(node)
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    if isinstance(node.operand, ast.Constant):
                        return source_literal(node.operand, -1 if isinstance(node.op, ast.USub) else 1)
                    value, dimensions = visit(node.operand, depth + 1)
                    return (-value if isinstance(node.op, ast.USub) else value), dimensions
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    left, ld = visit(node.left, depth + 1)
                    right, rd = visit(node.right, depth + 1)
                    if isinstance(node.op, (ast.Add, ast.Sub)):
                        if ld != rd:
                            raise ValueError("incompatible_additive_units")
                        value, dimensions = (left + right if isinstance(node.op, ast.Add) else left - right), ld
                    elif isinstance(node.op, ast.Mult):
                        value, dimensions = left * right, _dimension(ld, rd)
                    else:
                        if not right:
                            raise ValueError("division_by_zero")
                        value, dimensions = left / right, _dimension(ld, rd, -1)
                    if not value.is_finite() or abs(value) > MAX_ABS or (value and abs(value) < Decimal("1e-30")):
                        raise ValueError("arithmetic_out_of_bounds")
                    return value, dimensions
                raise ValueError("unsupported_expression_node:" + type(node).__name__)

            value, dimensions = visit(tree.body)
            display = value
            if output_unit is not None:
                requested_dimensions, _, requested_scale = _unit(output_unit)
                if requested_dimensions != dimensions:
                    raise ValueError("output_unit_mismatch")
                display = value / requested_scale
            result.update(ok=True, result=_format(value), unit=_unit_name(dimensions),
                          display_result=_format(display), display_unit=output_unit or _unit_name(dimensions))
            result["unused_operands"] = sorted(set(bound) - used) if used else []
    except (ValueError, TypeError, KeyError, DecimalException, RecursionError, OverflowError) as error:
        result["flags"].append(str(error) or type(error).__name__)
    return result


def calculate(expression: str, operands: list, episode: dict, as_of: str, *, source_ids=None, output_unit=None) -> dict:
    """Legacy dated-episode calculator; discovery and publication gates retained.

    Arithmetic and source-quote binding are performed by the shared grounded
    engine. Its output shape and chronology behavior remain the original API.
    """
    return _calculate_grounded(expression, operands,
        lambda: eligible_chunks(episode, as_of, source_ids), output_unit=output_unit)


def calculate_prevalidated_chunks(expression: str, operands: list, chunks: list, *, output_unit=None) -> dict:
    """Calculate on an explicit visible chunk set whose access/time gates are external.

    This low-level interface DOES NOT establish chronology, discovery, financial
    meaning, column/scope binding or numeric qualifier entailment. The caller
    must independently validate that these exact immutable chunks were already
    presented and eligible. No source publication dates are invented here.
    """
    def checked():
        if not isinstance(chunks, list):
            raise ValueError("invalid_prevalidated_chunk_list")
        ids = set()
        for chunk in chunks:
            if (not isinstance(chunk, dict)
                    or any(not isinstance(chunk.get(key), str) or not chunk[key]
                           for key in ("chunk_id", "source_id", "text"))):
                raise ValueError("invalid_prevalidated_chunk")
            if chunk["chunk_id"] in ids:
                raise ValueError("duplicate_prevalidated_chunk")
            ids.add(chunk["chunk_id"])
        return chunks
    result = _calculate_grounded(expression, operands, checked, output_unit=output_unit)
    result["source_eligibility_checked"] = False
    result["eligibility_boundary"] = "Caller must validate immutable visible evidence and source availability; this arithmetic core does not."
    return result


def _literal(needle: Any, text: str) -> bool:
    return isinstance(needle, str) and bool(needle.strip()) and " ".join(needle.casefold().split()) in " ".join(text.casefold().split())


def audit_proposal(proposal: dict, episode: dict, as_of: str, *, metric_spec=None, source_ids=None) -> dict:
    """Audit supplied quotations and arithmetic without claiming entailment."""
    result = {"ok": False, "flags": [], "calculations": [], "checks": [],
              "semantic_entailment_checked": False, "scope": AUDIT_SCOPE,
              "cost": {"audit_calls": 1, "citation_checks": 0, "quote_checks": 0,
                       "calculator_calls": 0, "model_calls": 0}}
    if not isinstance(proposal, dict):
        result["flags"].append("invalid_proposal")
        return result
    by_id = {chunk["chunk_id"]: chunk for chunk in eligible_chunks(episode, as_of, source_ids)}
    citations = proposal.get("citations", [])
    if not isinstance(citations, list):
        citations = []
        result["flags"].append("invalid_citation_list")
    ids, quotes = [], []
    for citation in citations:
        if isinstance(citation, str):
            chunk_id = citation
        elif isinstance(citation, dict):
            chunk_id = citation.get("chunk_id")
            if "quote" in citation:
                quotes.append(citation)
        else:
            result["flags"].append("invalid_citation")
            continue
        if not isinstance(chunk_id, str):
            result["flags"].append("invalid_citation_id")
            continue
        ids.append(chunk_id)
        result["cost"]["citation_checks"] += 1
        if chunk_id not in by_id:
            result["flags"].append("unknown_or_ineligible_citation:" + chunk_id)
        elif isinstance(citation, dict) and citation.get("source_id") is not None and citation["source_id"] != by_id[chunk_id]["source_id"]:
            result["flags"].append("citation_source_id_mismatch:" + chunk_id)
    for key in ("evidence_quotes", "quotes"):
        supplied = proposal.get(key, [])
        if not isinstance(supplied, list):
            result["flags"].append("invalid_" + key)
        else:
            quotes.extend(supplied)
    calculations = proposal.get("calculations", [])
    if not isinstance(calculations, list) or len(calculations) > 10:
        result["flags"].append("invalid_calculation_list")
        calculations = []
    for calculation in calculations:
        if not isinstance(calculation, dict):
            result["flags"].append("invalid_calculation")
            continue
        operands = calculation.get("operands", [])
        if isinstance(operands, list):
            quotes.extend(op for op in operands if isinstance(op, dict))
        checked = calculate(calculation.get("expression"), operands, episode, as_of, source_ids=source_ids,
                            output_unit=calculation.get("output_unit", proposal.get("unit")))
        result["calculations"].append(checked)
        result["cost"]["calculator_calls"] += 1
        result["flags"].extend("calculation:" + flag for flag in checked["flags"])
    quoted = set()
    valid_quotes = []
    for item in quotes:
        result["cost"]["quote_checks"] += 1
        if not isinstance(item, dict):
            result["flags"].append("invalid_quote")
            continue
        chunk_id, quote = item.get("chunk_id"), item.get("quote")
        if not isinstance(chunk_id, str) or chunk_id not in by_id:
            result["flags"].append("unknown_or_ineligible_quote_chunk:" + str(chunk_id))
        elif not isinstance(quote, str) or not quote.strip() or len(quote) > 8000 or quote not in by_id[chunk_id]["text"]:
            result["flags"].append("quote_not_in_chunk:" + chunk_id)
        elif item.get("source_id") is not None and item["source_id"] != by_id[chunk_id]["source_id"]:
            result["flags"].append("quote_source_id_mismatch:" + chunk_id)
        else:
            quoted.add(chunk_id)
            valid_quotes.append((chunk_id, quote))
            if chunk_id not in ids:
                result["flags"].append("quote_chunk_not_cited:" + chunk_id)
    for chunk_id in sorted(set(ids) - quoted):
        result["flags"].append("citation_without_exact_quote:" + chunk_id)
    if not ids:
        result["flags"].append("no_citations")
    spec = metric_spec if isinstance(metric_spec, dict) else {}
    cited = [by_id[chunk_id] for chunk_id in sorted(set(ids)) if chunk_id in by_id]
    text = "\n".join(chunk["text"] for chunk in cited)
    for field in METADATA_FIELDS:
        expected = spec.get(field, proposal.get(field))
        declared = proposal.get(field)
        if isinstance(expected, str) and expected and expected.casefold() not in ("unspecified", "not specified", "any"):
            if declared is not None and (not isinstance(declared, str) or " ".join(declared.casefold().split()) != " ".join(expected.casefold().split())):
                result["flags"].append("proposal_metadata_mismatch:" + field)
            observed = [chunk.get("metadata", {}).get(field) for chunk in cited]
            structured = any(isinstance(value, str) and value.casefold() == expected.casefold() for value in observed)
            conflicting = any(isinstance(value, str) and value.casefold() != expected.casefold() for value in observed)
            literal = _literal(expected, text)
            calculated_unit = field == "unit" and any(row["ok"] and _unit(row["display_unit"]) == _unit(expected) for row in result["calculations"])
            if conflicting and not structured and not calculated_unit:
                result["flags"].append("source_metadata_conflict:" + field)
            if field == "unit":
                literal = _unit_in_context(expected, text)
            if not structured and not literal and not calculated_unit:
                result["flags"].append("metadata_not_located:" + field)
            result["checks"].append({"field": field, "expected": expected,
                                     "structured_match": structured, "literal_match": literal,
                                     "calculation_unit_match": calculated_unit,
                                     "semantic_match_checked": False})
    # If a scalar was offered, at least one exact quotation must contain it or
    # one verified calculation must produce it in the stated display unit.
    value = proposal.get("value")
    if value is not None and not (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)):
        try:
            number = _decimal(value)
            present = any(_number_matches(quote, number) for _, quote in valid_quotes)
            calculated = any(row["ok"] and _decimal(row["display_result"]) == number for row in result["calculations"])
            if not present and not calculated:
                result["flags"].append("proposal_value_not_quoted_or_calculated")
        except (ValueError, DecimalException):
            result["flags"].append("invalid_proposal_value")
    result["flags"] = sorted(set(result["flags"]))
    result["ok"] = not result["flags"]
    return result


def _cells(line: str) -> list[str]:
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    if "|" in line:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]
    return []


def deterministic_lookup(episode: dict, claim: dict, as_of: str, *, source_ids=None, k=8) -> dict:
    """Ordinary BM25 + literal table lookup, with charged manual review.

    Public metric_spec accepts metric/name or row_pattern, header_pattern,
    column_header, period, entity, basis, unit and scale. Patterns are literals.
    Exact row/chunk/source mappings are never read from reference fields.
    A lone scalar is a candidate, not automatic proof of a prose claim's meaning.
    """
    spec = claim.get("metric_spec", {}) if isinstance(claim, dict) else {}
    spec = spec if isinstance(spec, dict) else {}
    if isinstance(spec.get("metrics"), list) and len(spec["metrics"]) == 1 and isinstance(spec["metrics"][0], dict):
        spec = {**spec, **spec["metrics"][0]}
    metric = spec.get("row_pattern") or spec.get("metric") or spec.get("name")
    if not metric and isinstance(spec.get("metrics"), list) and len(spec["metrics"]) == 1:
        metric = spec["metrics"][0]
    query = " ".join(str(value) for value in [metric, spec.get("entity"), spec.get("period"), spec.get("basis"),
                                                 claim.get("instruction", "") if isinstance(claim, dict) else ""] if value)
    retrieved = retrieve(episode, query, as_of, k=k, source_ids=source_ids)
    cost = copy.deepcopy(retrieved["cost"])
    cost.update(lookup_calls=1, table_rows_examined=0, manual_review_required=1)
    base = {"status": "review", "disposition": "review", "value": None,
            "unit": spec.get("unit"), "citations": [], "evidence_quotes": [],
            "reason": "Public intent does not define a unique supported literal table lookup.",
            "flags": [], "semantic_entailment_checked": False, "scope": AUDIT_SCOPE, "cost": cost}
    if not isinstance(metric, str) or not metric.strip() or not all(isinstance(spec.get(key), str) and spec[key].strip() for key in ("period", "unit")):
        base["flags"].append("incomplete_public_metric_spec")
        return base
    candidates = []
    for chunk in retrieved["chunks"]:
        lines = chunk["text"].splitlines()
        for index, row in enumerate(lines):
            cost["table_rows_examined"] += 1
            if not _literal(metric, row):
                continue
            cells = _cells(row)
            if not cells:
                continue
            # Context never extends across the next table; only preceding
            # header/unit prose is eligible to identify the selected column.
            context = "\n".join(lines[max(0, index - 8):index])
            header_pattern = spec.get("header_pattern")
            if header_pattern and not _literal(header_pattern, context):
                continue
            column = spec.get("column_header") or spec["period"]
            headers = [(line, _cells(line)) for line in lines[max(0, index - 8):index] if _cells(line)]
            positions = {position for _, header in headers for position, cell in enumerate(header) if _literal(column, cell)}
            if len(positions) != 1:
                continue
            position = next(iter(positions))
            if position >= len(cells) or not _literal(spec["period"], context):
                continue
            try:
                dimension, base_unit, unit_scale = _unit(spec["unit"])
                if "scale" in spec:
                    scale_name, factor = _scale(spec["scale"])
                else:
                    # Infer the table's stated scale from public header text,
                    # independently of the requested display-unit scale.
                    header_scales = [name for name, pattern in SCALE_WORDS.items()
                                     if re.search(pattern, context, flags=re.IGNORECASE)]
                    if len(header_scales) > 1:
                        continue
                    scale_name = header_scales[0] if header_scales else "one"
                    factor = SCALES[scale_name]
            except (ValueError, DecimalException):
                continue
            # Unit context is required even when the candidate number is unique.
            unit_literal = _literal(spec["unit"], context)
            currency_ok = bool(dimension) and dimension[0][0] == "USD" and ("$" in context or _literal("USD", context) or _literal("dollars", context))
            scale_ok = factor == 1 or bool(re.search(SCALE_WORDS[scale_name], context, flags=re.IGNORECASE))
            if not (unit_literal or (currency_ok and scale_ok)):
                continue
            if any(not (_literal(spec[field], chunk["text"]) or _literal(spec[field], chunk.get("metadata", {}).get(field, "")))
                   for field in ("entity", "basis") if spec.get(field)):
                continue
            numeric = list(NUMBER.finditer(cells[position]))
            if len(numeric) != 1:
                continue
            number = _decimal(numeric[0].group())
            if cells[position].strip().startswith("(") and cells[position].strip().endswith(")"):
                number = -abs(number)
            header_text = "\n".join(lines[max(0, index - 8):index])
            operand = {"name": "observed", "value": _format(number), "scale": scale_name,
                       "unit": base_unit, "chunk_id": chunk["chunk_id"], "quote": row, "scale_quote": header_text}
            calculation = calculate("observed", [operand], episode, as_of, source_ids=source_ids, output_unit=spec["unit"])
            if calculation["ok"]:
                candidates.append((chunk, row, calculation))
    # Different source locations remain ambiguous even when their scalars agree;
    # silent precedence by publication date would overwrite historical context.
    if len(candidates) != 1:
        base["flags"].append("ambiguous_lookup" if candidates else "no_supported_literal_row")
        base["candidate_count"] = len(candidates)
        return base
    chunk, row, calculation = candidates[0]
    base.update(status="candidate", value=calculation["display_result"], citations=[chunk["chunk_id"]],
                evidence_quotes=[{"chunk_id": chunk["chunk_id"], "quote": row}],
                reason="A unique literal table candidate was located; human review is required before changing prose.",
                calculations=[calculation], flags=["prose_disposition_requires_review"])
    # Keep retain/update decisions with a reviewer: original prose does not
    # carry a trustworthy machine-readable old value or comparison semantics.
    return base
