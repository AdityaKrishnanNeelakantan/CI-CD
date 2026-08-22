"""Compiles layout regions into a DocumentTemplateSpec: the portable,
source-free description of a document's structure that later
generation/rendering stages will consume - reusable static wording and
region positions, plus each variable field's inferred data role and a
masked preview, but never a real filled-in value.

A region's content is either:
- static template wording (headings, paragraph boilerplate) - persisted
  almost verbatim, since that IS the template content to re-render later,
  but still PII-scanned and redacted first (a paragraph can still contain
  an inline value, e.g. "Contact: grace@example.com.").
- one or more variable fields (a detected "label: value" pair, or a
  table's per-row cells) - only the label and the value's inferred type
  and a masked preview are ever persisted, never the value itself.
"""

from __future__ import annotations

import re
from typing import Any

from synth_platform.engine.documents.pdf.entities import extract_entities
from synth_platform.engine.documents.pdf.pii import detect_pii
from synth_platform.engine.documents.pdf.shape import infer_shape_pattern

_FIELD_LINE_PATTERN = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 /&.'-]{1,60}):\s*(?P<value>.+)$")
_CURRENCY_PATTERN = re.compile(r"^[$€£]\s?-?\d[\d,]*\.?\d*$")
_NUMERIC_PATTERN = re.compile(r"^-?\d[\d,]*\.?\d*$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$")

# Paragraph/heading text is persisted close to verbatim (it's the reusable
# static template wording), so on top of pii.detect_pii's narrow PII
# categories it also needs its own sweep for source-specific values that
# aren't strictly "PII" but are still real source data that must never
# appear in a portable template - account/reference numbers and amounts
# narrated inline in running text (e.g. "Account #12345678", "SERVICE
# CHARGE 12.00"), which a bank/invoice statement's boilerplate is full of.
# Biased towards over-redacting: a false-positive reference number masked
# in static wording is harmless, a real account number leaking is not.
_INLINE_CURRENCY_PATTERN = re.compile(r"[$€£]\s?\d[\d,]*\.\d{2}")
_INLINE_DECIMAL_AMOUNT_PATTERN = re.compile(r"\b\d+\.\d{2}\b")
_INLINE_LONG_DIGIT_RUN_PATTERN = re.compile(r"\d{5,}")


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


#: A field/cell's classification is only trusted from a PII/entity match
#: that covers most of the value, not a fragment of it. Table cells built
#: from OCR-derived layout occasionally end up holding a whole misclustered
#: sentence rather than one atomic value (real example: an OCR'd bank
#: statement's narrative boilerplate landed inside a "table" cell) - the
#: PERSON pattern only needs 2-4 title-case words *anywhere* in the string
#: to fire, so without this guard a long sentence containing an incidental
#: title-case phrase ("Relationship Checking") gets typed PERSON, and the
#: entire cell is later replaced by one fabricated Faker name at
#: generation time - actively misleading, not just an unreadable shape-fill.
_WHOLE_VALUE_MATCH_MIN_COVERAGE = 0.7

#: Longest reconstructed length for a generic (unclassified) "text" cell.
#: Chosen well above any realistic single-field value (a label, a short
#: description) and well below what actually overflowed a table cell in
#: testing (~90-100 characters).
_MAX_GENERIC_TEXT_SHAPE_LENGTH = 30


def _classify_value(raw_value: str) -> dict[str, Any]:
    value = raw_value.strip()
    # shape_pattern is a structural fingerprint only (letter-case/digit/
    # punctuation positions, see src/documents/shape.py) - it carries zero
    # characters of the real value, so it's safe to persist even though
    # it's computed here, on the raw value, before the raw value itself is
    # discarded in favour of value_type + masked_preview below.
    shape_pattern = infer_shape_pattern(value)

    def _covers_most_of_value(finding: dict[str, Any]) -> bool:
        if not value:
            return False
        return (finding["end"] - finding["start"]) / len(value) >= _WHOLE_VALUE_MATCH_MIN_COVERAGE

    pii_findings = [f for f in detect_pii(value) if _covers_most_of_value(f)]
    if pii_findings:
        top = pii_findings[0]
        return {
            "value_type": top["type"],
            "confidence": top["confidence"],
            "masked_preview": top["redaction_preview"],
            "shape_pattern": shape_pattern,
        }

    entity_findings = [f for f in extract_entities(value) if _covers_most_of_value(f)]
    if entity_findings:
        top = entity_findings[0]
        # Weak Title-Case PERSON matches (confidence 0.6, no honorific) fire
        # on ordinary headers and phrases ("Medication Summary", "Grocery
        # Mart", "Routine Diabetes Check-up"). Promoting those to value_type
        # PERSON breaks table-header detection and makes whole columns
        # generate Faker names. Only strong entity evidence may win here;
        # labelled fields still resolve person_name via label keywords.
        if top["type"] in {"PERSON", "ORG"} and top["confidence"] < 0.8:
            pass
        else:
            # entities.py's own type constant is upper-case ("DATE"), but this
            # module already had its own, older lower-case "date" value_type
            # (from the _DATE_PATTERN fallback below) that every downstream
            # consumer (field_semantics._VALUE_TYPE_TO_ROLE,
            # binding_engine.GENERATOR_STRATEGY_BY_ROLE) keys off - normalise
            # here so entities.py detecting a numeric-shaped date (e.g.
            # "24/05/1977", added so headings/paragraphs stop leaking it - see
            # _redact_inline_sensitive_values) doesn't silently produce an
            # unrecognised value_type for a field/cell going through this path
            # instead.
            value_type = "date" if top["type"] == "DATE" else top["type"]
            return {
                "value_type": value_type,
                "confidence": top["confidence"],
                "masked_preview": top["redaction_preview"],
                "shape_pattern": shape_pattern,
            }

    if _CURRENCY_PATTERN.match(value):
        return {"value_type": "currency", "confidence": 0.9, "masked_preview": _mask(value), "shape_pattern": shape_pattern}
    if _DATE_PATTERN.match(value):
        return {"value_type": "date", "confidence": 0.9, "masked_preview": _mask(value), "shape_pattern": shape_pattern}
    if _NUMERIC_PATTERN.match(value):
        return {"value_type": "numeric", "confidence": 0.85, "masked_preview": _mask(value), "shape_pattern": shape_pattern}

    # A single table cell/field genuinely exceeding this length is rare;
    # in practice this branch is what OCR/layout misclustering falls
    # into (a whole run of boilerplate merged into one "cell" - see the
    # _covers_most_of_value guard above, which exists for the same root
    # cause). Capping the *reconstructed* length here, not the original
    # shape_pattern's informativeness, is what stops that from turning
    # into a fixed-width table cell overflowing at render time - a real
    # regression caught by re-running this exact fixture after the guard
    # above started correctly declining to mislabel these cells PERSON.
    if len(shape_pattern) > _MAX_GENERIC_TEXT_SHAPE_LENGTH:
        shape_pattern = shape_pattern[:_MAX_GENERIC_TEXT_SHAPE_LENGTH]
    return {"value_type": "text", "confidence": 0.4, "masked_preview": _mask(value), "shape_pattern": shape_pattern}


def _redact_inline_sensitive_values(
    text: str, known_values: list[tuple[str, str]] = ()
) -> tuple[str, list[dict[str, Any]]]:
    """Redact every sensitive span in static heading/paragraph text, same
    as before, but also return where each one was and its shape - so a
    later stage can generate a fresh, plausible replacement value for
    that exact position instead of the rendered twin showing a literal
    masked placeholder ("69*96") in what should be brand-new synthetic
    content. Masking is length-preserving (_mask() below never changes a
    span's length), so a span's (start, end) computed against the
    original text remains a valid index into the redacted text too.

    known_values (see _collect_known_sensitive_values) is an additional,
    literal sweep for real values already identified elsewhere in the
    SAME document as an explicit "Label: Value" field - this is what
    catches a patient's name/DOB repeated in a page's running header or
    narrative summary, which the entity/PII detectors above miss on their
    own (a bare two-word name with no title prefix scores only 0.6,
    deliberately below the 0.8 bar used here, to avoid mismarking generic
    headings like "Summary of Your Account" - real regression found by
    inspecting an actual generated synthetic twin whose header leaked the
    real patient identity verbatim next to correctly-faked demographics).
    """
    replacements: list[tuple[int, int, str]] = []
    variable_spans: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    # PERSON/ORG/DATE entities (e.g. a real name in a letterhead or
    # signature block) are exactly as sensitive here as PII - a template's
    # static wording must never carry a real one through either. Only
    # high-confidence entity findings qualify: extract_entities' weakest
    # signal ("capitalised_sequence" alone, no title prefix) fires on
    # completely generic capitalised headings ("Summary of Your Account"),
    # which would otherwise mangle static template wording that was never
    # sensitive in the first place.
    strong_entities = [f for f in extract_entities(text) if f["confidence"] >= 0.8]
    for finding in [*detect_pii(text), *strong_entities]:
        start, end = finding["start"], finding["end"]
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        occupied.append((start, end))
        replacements.append((start, end, finding["redaction_preview"]))
        # entities.py's own type constant is upper-case ("DATE"); every
        # downstream consumer of a variable span's value_type
        # (field_semantics._VALUE_TYPE_TO_ROLE, in turn
        # binding_engine.GENERATOR_STRATEGY_BY_ROLE) keys off the
        # lower-case "date" instead - without normalising here a
        # heading/paragraph DOB falls through to the generic_text/
        # shape-filling strategy at generation time instead of
        # bounded_date, producing an invalid date (e.g. month "43").
        value_type = "date" if finding["type"] == "DATE" else finding["type"]
        variable_spans.append(
            {
                "start": start,
                "end": end,
                "value_type": value_type,
                "shape_pattern": infer_shape_pattern(text[start:end]),
            }
        )

    for pattern, value_type in (
        (_INLINE_CURRENCY_PATTERN, "currency"),
        (_INLINE_DECIMAL_AMOUNT_PATTERN, "numeric"),
        (_INLINE_LONG_DIGIT_RUN_PATTERN, "numeric"),
    ):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue
            occupied.append((start, end))
            matched_text = match.group(0)
            replacements.append((start, end, _mask(matched_text)))
            variable_spans.append(
                {
                    "start": start,
                    "end": end,
                    "value_type": value_type,
                    "shape_pattern": infer_shape_pattern(matched_text),
                }
            )

    # known_values is pre-sorted longest-first (see
    # _collect_known_sensitive_values) so a full name is matched whole
    # before any shorter/coincidental substring could be.
    for value, value_type in known_values:
        start_at = 0
        while True:
            start = text.find(value, start_at)
            if start == -1:
                break
            end = start + len(value)
            start_at = end
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue
            occupied.append((start, end))
            replacements.append((start, end, _mask(value)))
            variable_spans.append(
                {
                    "start": start,
                    "end": end,
                    "value_type": value_type,
                    "shape_pattern": infer_shape_pattern(text[start:end]),
                }
            )

    redacted = text
    for start, end, preview in sorted(replacements, key=lambda r: r[0], reverse=True):
        redacted = redacted[:start] + preview + redacted[end:]

    variable_spans.sort(key=lambda s: s["start"])
    return redacted, variable_spans


# Value types trusted as "known sensitive" when they come from an explicit
# field's own Label: Value shape - a field is much stronger evidence of
# sensitivity than a bare inline match, so these are swept for in
# heading/paragraph text regardless of the stricter standalone entity-
# confidence bar in _redact_inline_sensitive_values.
_KNOWN_SENSITIVE_VALUE_TYPES = {"PERSON", "ORG", "date"}
_MIN_KNOWN_VALUE_LENGTH = 3

# Label-driven roles whose raw values must be swept for narrative recurrence
# even when weak PERSON/ORG entity confidence no longer wins _classify_value
# (Title-Case headers were false-positive PERSON matches).
_KNOWN_SENSITIVE_ROLES = {
    "person_name": "PERSON",
    "organization": "ORG",
    "date": "date",
    "street_address": "street_address",
    "email_address": "email",
    "phone_number": "phone_number",
}


def _collect_known_sensitive_values(layout: dict[str, Any]) -> list[tuple[str, str]]:
    """First pass over every field region's raw value, across the whole
    document, before it's ever discarded - collects (raw_value,
    value_type) for every field classified PERSON/ORG/date, so a second
    pass over heading/paragraph text (_redact_inline_sensitive_values)
    can catch that exact same value recurring elsewhere - e.g. a
    patient's name or DOB repeated in a running page header - even where
    it fails the stricter standalone entity-confidence bar used there.
    Never persisted itself; used only to drive the redaction sweep below.
    """
    from synth_platform.engine.documents.pdf.field_semantics import infer_field_semantics

    known: dict[str, str] = {}
    for page in layout["pages"]:
        for region in page["regions"]:
            if region["region_type"] != "field":
                continue
            label, raw_value = _extract_field_label_value(region["lines"][0])
            value = raw_value.strip()
            if len(value) < _MIN_KNOWN_VALUE_LENGTH:
                continue
            classification = _classify_value(value)
            if classification["value_type"] in _KNOWN_SENSITIVE_VALUE_TYPES:
                known[value] = classification["value_type"]
                continue
            role = infer_field_semantics(label, classification["value_type"])["semantic_role"]
            mapped = _KNOWN_SENSITIVE_ROLES.get(role)
            if mapped:
                known[value] = mapped
    return sorted(known.items(), key=lambda kv: len(kv[0]), reverse=True)


def _extract_field_label_value(line: dict[str, Any]) -> tuple[str | None, str]:
    """Same label/value extraction rule _compile_field uses to build its
    persisted output, factored out so a first pass can also read the raw
    value (see _collect_known_sensitive_values) without _compile_field
    itself ever persisting it.
    """
    cells = line["cells"]
    if len(cells) >= 2 and cells[0].rstrip().endswith(":"):
        label = cells[0].rstrip()[:-1].strip()
        value = " ".join(cells[1:]).strip()
        return label, value

    match = _FIELD_LINE_PATTERN.match(line["text"])
    if match:
        return match.group("label").strip(), match.group("value").strip()

    return None, line["text"].strip()


def _compile_field(line: dict[str, Any]) -> dict[str, Any]:
    label, value = _extract_field_label_value(line)
    return {"label": label, **_classify_value(value)}


def _is_header_shaped(row: list[dict[str, Any]]) -> bool:
    """A header row - static, reusable column labels ("Date",
    "Description", "Amount") - looks structurally different from a data
    row: every cell classifies as plain "text" with no PII/entity/
    currency/date/numeric signal at all (that's what reaching the "text"
    fallback in _classify_value already establishes per cell). Requiring
    *every* cell to qualify, not just some, keeps this from misfiring on
    a genuine data row that merely happens to have one non-numeric
    column - a data row for a real customer virtually always has at
    least one currency/date/numeric cell somewhere in it.
    """
    return bool(row) and all(cell["value_type"] == "text" for cell in row)


def _leading_header_row_count(
    rows: list[list[dict[str, Any]]], raw_lines: list[dict[str, Any]] | None = None
) -> int:
    """How many rows, starting from row 0, form an unbroken preamble of
    header/title-shaped rows - a real scanned statement often has more
    than one (a section title, then a sub-label, then the actual column
    header), not just row 0. Stops at the first row containing any real
    data signal (numeric/date/currency/PII/entity), so a data row deep
    in the table can never be swept in by this - the run must be
    unbroken from the very start.

    All-text tables (medications, reasons) used to keep consuming every
    row but the last as "header", which both leaked source values as
    static_label and left only one row to synthesise. Allow any number of
    ALL-CAPS section-title rows, but at most one mixed/Title-Case label
    row (the real column headers).
    """
    count = 0
    seen_label_row = False
    for index, row in enumerate(rows):
        if not _is_header_shaped(row):
            break
        raw_cells = (raw_lines[index]["cells"] if raw_lines and index < len(raw_lines) else [])
        joined = " ".join(str(c) for c in raw_cells).strip()
        if joined.isupper():
            count += 1
            continue
        if seen_label_row:
            break
        seen_label_row = True
        count += 1
    # Never treat the entire table as header - at least one data row
    # must remain, or there is nothing to distinguish "header" from.
    return min(count, len(rows) - 1)


def _compile_table(lines: list[dict[str, Any]], known_values: list[tuple[str, str]] = ()) -> dict[str, Any]:
    rows = [[_classify_value(cell) for cell in line["cells"]] for line in lines]
    header_row_count = _leading_header_row_count(rows, lines) if len(rows) > 1 else 0
    for row_index in range(header_row_count):
        header_cells = []
        for raw_cell in lines[row_index]["cells"]:
            redacted_text, _ = _redact_inline_sensitive_values(raw_cell, known_values)
            header_cells.append({"value_type": "static_label", "text": redacted_text})
        rows[row_index] = header_cells
    return {
        "row_count": len(rows),
        "column_count": max((len(row) for row in rows), default=0),
        "rows": rows,
        "header_row_count": header_row_count,
    }


def compile_document_template(layout: dict[str, Any]) -> dict[str, Any]:
    known_values = _collect_known_sensitive_values(layout)
    pages_out = []
    for page in layout["pages"]:
        regions_out = []
        for region in page["regions"]:
            entry: dict[str, Any] = {
                "region_id": region["region_id"],
                "region_type": region["region_type"],
                "bbox": region["bbox"],
                "reading_order_index": region["reading_order_index"],
            }

            if region["region_type"] == "field":
                entry.update(_compile_field(region["lines"][0]))
            elif region["region_type"] == "table":
                entry.update(_compile_table(region["lines"], known_values))
            else:  # heading / paragraph
                joined = " ".join(ln["text"] for ln in region["lines"])
                redacted_text, variable_spans = _redact_inline_sensitive_values(joined, known_values)
                entry["text"] = redacted_text
                if variable_spans:
                    entry["inline_variable_spans"] = variable_spans

            regions_out.append(entry)
        page_out: dict[str, Any] = {"page_number": page["page_number"], "regions": regions_out}
        if "width_pt" in page and "height_pt" in page:
            page_out["width_pt"] = page["width_pt"]
            page_out["height_pt"] = page["height_pt"]
        pages_out.append(page_out)

    return {"page_count": layout["page_count"], "pages": pages_out}
