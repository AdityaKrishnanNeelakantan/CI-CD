from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.template_compiler import compile_document_template

pytestmark = pytest.mark.unit


def _layout(regions, page_count=1):
    return {"page_count": page_count, "pages": [{"page_number": 1, "regions": regions}]}


def _region(region_type, lines, region_id="p1_r0", index=0):
    return {
        "region_id": region_id,
        "region_type": region_type,
        "bbox": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 0.1},
        "reading_order_index": index,
        "lines": lines,
    }


def test_field_with_colon_cells_extracts_label_and_masks_value():
    region = _region("field", [{"text": "Account Number: 8823471", "cells": ["Account Number:", "8823471"]}])
    template = compile_document_template(_layout([region]))
    out = template["pages"][0]["regions"][0]
    assert out["label"] == "Account Number"
    assert out["value_type"] == "numeric"
    assert "8823471" not in out["masked_preview"]


def test_field_without_cell_split_falls_back_to_regex():
    region = _region("field", [{"text": "Total Due: 401.50", "cells": ["Total Due: 401.50"]}])
    template = compile_document_template(_layout([region]))
    out = template["pages"][0]["regions"][0]
    assert out["label"] == "Total Due"
    assert "401.50" not in str(out)


def test_table_region_masks_every_cell_and_reports_dimensions():
    lines = [
        {"text": "Beginning balance 69.96", "cells": ["Beginning balance", "69.96"]},
        {"text": "Ending balance 5340.43", "cells": ["Ending balance", "5340.43"]},
        {"text": "Fees 12.00", "cells": ["Fees", "12.00"]},
    ]
    region = _region("table", lines)
    template = compile_document_template(_layout([region]))
    out = template["pages"][0]["regions"][0]
    assert out["row_count"] == 3
    assert out["column_count"] == 2
    serialized = str(out)
    assert "69.96" not in serialized
    assert "5340.43" not in serialized
    assert "12.00" not in serialized


def test_long_misclustered_blob_is_not_mistyped_person_from_an_embedded_title_case_phrase():
    """Regression: found running the pipeline against a real scanned bank
    statement - OCR/layout clustering occasionally merges a whole run of
    boilerplate into one "table cell" instead of a paragraph. The PERSON
    entity pattern only needs 2-4 consecutive title-case words *anywhere*
    in the string to fire, so a long cell merely containing an incidental
    title-case phrase ("Relationship Checking") used to get typed PERSON
    wholesale - meaning the entire cell would be replaced by one
    fabricated Faker name at generation time, actively misleading rather
    than just unreadable.
    """
    long_blob = (
        "Activity for Relationship Checking - Account continued Account "
        "Transactions by type with detailed description Deposits and "
        "Other Credits Date Description PREAUTHORIZED CREDIT"
    )
    region = _region("table", [{"text": long_blob, "cells": [long_blob]}])
    template = compile_document_template(_layout([region]))
    cell = template["pages"][0]["regions"][0]["rows"][0][0]
    assert cell["value_type"] == "text"


def test_genuine_short_person_name_cell_is_still_classified_person_with_strong_evidence():
    """Weak Title-Case pairs are no longer typed PERSON (they break header
    detection). Strong evidence — honorific prefix — still is.
    """
    region = _region("table", [{"text": "Dr. Donald Booth", "cells": ["Dr. Donald Booth"]}])
    template = compile_document_template(_layout([region]))
    cell = template["pages"][0]["regions"][0]["rows"][0][0]
    assert cell["value_type"] == "PERSON"


def test_title_case_header_phrase_is_not_mistyped_person():
    """Regression: 'Medication Summary' used to classify as PERSON, which
    prevented header-row detection and caused header cells to be shape-
    scrambled into gibberish at generation time.
    """
    region = _region("table", [{"text": "Medication Summary", "cells": ["Medication Summary"]}])
    template = compile_document_template(_layout([region]))
    cell = template["pages"][0]["regions"][0]["rows"][0][0]
    assert cell["value_type"] == "text"


def test_medication_table_headers_are_preserved_not_scrambled():
    lines = [
        {
            "text": "Medication Summary Dosage Frequency",
            "cells": ["Medication Summary", "Dosage", "Frequency"],
        },
        {"text": "Metformin 500mg Twice daily", "cells": ["Metformin", "500mg", "Twice daily"]},
        {"text": "Lisinopril 10mg Once daily", "cells": ["Lisinopril", "10mg", "Once daily"]},
    ]
    template = compile_document_template(_layout([_region("table", lines)]))
    region = template["pages"][0]["regions"][0]
    assert region["header_row_count"] == 1
    assert [c["text"] for c in region["rows"][0]] == ["Medication Summary", "Dosage", "Frequency"]
    assert all(c["value_type"] == "static_label" for c in region["rows"][0])


def test_long_generic_text_cell_shape_is_capped_to_avoid_render_overflow():
    """Regression, found immediately after the fix above: correctly no
    longer mistyping a long misclustered blob PERSON meant it fell into
    the generic "text" bucket and its *entire* shape (~140 characters)
    was reconstructed at generation time and rendered into a table cell
    sized for ordinary short values - a real, measured render overflow
    (0 -> 10 overflow failures) on the same real statement fixture.
    """
    long_blob = "X" * 140
    region = _region("table", [{"text": long_blob, "cells": [long_blob]}])
    template = compile_document_template(_layout([region]))
    cell = template["pages"][0]["regions"][0]["rows"][0][0]
    assert cell["value_type"] == "text"
    assert len(cell["shape_pattern"]) <= 30


def test_table_header_row_is_preserved_verbatim_not_shape_scrambled():
    """A leading all-text row ("Date", "Description", "Amount") is a
    static, reusable column header - the same wording on every render -
    not per-row data to reconstruct from a shape_pattern. Requested
    feature: without this, every table's headers rendered as
    unreadable scrambled letters even though they carry no real
    information to protect.
    """
    lines = [
        {"text": "Date Description Amount", "cells": ["Date", "Description", "Amount"]},
        {"text": "01/02/2024 Groceries 45.20", "cells": ["01/02/2024", "Groceries", "45.20"]},
        {"text": "01/05/2024 Fuel 30.10", "cells": ["01/05/2024", "Fuel", "30.10"]},
    ]
    template = compile_document_template(_layout([_region("table", lines)]))
    region = template["pages"][0]["regions"][0]
    assert region["header_row_count"] == 1
    header_row = region["rows"][0]
    assert [c["text"] for c in header_row] == ["Date", "Description", "Amount"]
    assert all(c["value_type"] == "static_label" for c in header_row)
    # Data rows are unaffected - still shape-classified, never verbatim.
    assert region["rows"][1][0]["value_type"] == "date"


def test_multiple_leading_preamble_rows_are_all_preserved_until_real_data_appears():
    """Regression: a real scanned bank statement had a section title, a
    sub-label, and the actual column-header row all before the first
    real data row - restricting detection to row 0 only fixed the title
    and left the genuine column header (Date/Debit/Credit/Balance)
    still scrambled. The leading run must extend through every
    consecutive all-text row, not just the first one.
    """
    lines = [
        # All-caps, to stay clear of the PERSON pattern's own Title-Case
        # requirement (two Title-Case words, e.g. "Account Activity",
        # would otherwise be a false-positive PERSON match in its own
        # right - a separate, pre-existing precision limit of that
        # pattern, not something this header-detection feature changes).
        {"text": "TRANSACTION HISTORY", "cells": ["TRANSACTION HISTORY"]},
        {"text": "Date Debit Credit Balance", "cells": ["Date", "Debit", "Credit", "Balance"]},
        {"text": "01/02/2024 45.20 0.00 100.00", "cells": ["01/02/2024", "45.20", "0.00", "100.00"]},
    ]
    template = compile_document_template(_layout([_region("table", lines)]))
    region = template["pages"][0]["regions"][0]
    assert region["header_row_count"] == 2
    assert region["rows"][0][0]["text"] == "TRANSACTION HISTORY"
    assert [c["text"] for c in region["rows"][1]] == ["Date", "Debit", "Credit", "Balance"]
    assert region["rows"][2][0]["value_type"] == "date"


def test_header_detection_never_consumes_the_entire_table():
    """A table where every row happens to be all-text (no numeric/date/
    currency cell anywhere) must still leave at least one row as data -
    a "header-only" table with nothing to actually be a header for is a
    sign this isn't really a table, not licence to preserve every row
    verbatim."""
    lines = [
        {"text": "Alpha Beta", "cells": ["Alpha", "Beta"]},
        {"text": "Gamma Delta", "cells": ["Gamma", "Delta"]},
    ]
    template = compile_document_template(_layout([_region("table", lines)]))
    region = template["pages"][0]["regions"][0]
    assert region["header_row_count"] == 1
    assert region["rows"][1][0]["value_type"] != "static_label"


def test_header_detection_never_fires_on_a_genuine_data_row_with_no_leading_run():
    """A table whose row 0 already contains a data-shaped cell (e.g. a
    currency amount) must never be treated as a header, protecting against a
    genuine data row being preserved near-verbatim by this heuristic.
    """
    lines = [
        {"text": "Donald Booth 45.20", "cells": ["Donald Booth", "45.20"]},
        {"text": "Grace Hopper 12.10", "cells": ["Grace Hopper", "12.10"]},
    ]
    template = compile_document_template(_layout([_region("table", lines)]))
    region = template["pages"][0]["regions"][0]
    assert region["header_row_count"] == 0
    assert region["rows"][0][1]["value_type"] in {"currency", "numeric"}


def test_header_row_pii_is_still_redacted_not_shown_verbatim():
    """A header/preamble row is preserved close to verbatim, but must go
    through the same PII/entity redaction sweep as ordinary paragraph
    text - a section title inline with a real account number
    ("Account #12345678 Summary") must still have that number masked.
    """
    lines = [
        {"text": "Account #88234671 Summary", "cells": ["Account #88234671 Summary"]},
        {"text": "01/02/2024 45.20", "cells": ["01/02/2024", "45.20"]},
    ]
    template = compile_document_template(_layout([_region("table", lines)]))
    region = template["pages"][0]["regions"][0]
    assert region["header_row_count"] == 1
    assert "88234671" not in region["rows"][0][0]["text"]


def test_paragraph_text_is_preserved_when_not_sensitive():
    region = _region("paragraph", [{"text": "This section describes the terms.", "cells": ["irrelevant"]}])
    template = compile_document_template(_layout([region]))
    assert template["pages"][0]["regions"][0]["text"] == "This section describes the terms."


def test_paragraph_email_is_redacted():
    region = _region(
        "paragraph", [{"text": "Contact us at grace@example.com for help.", "cells": ["irrelevant"]}]
    )
    template = compile_document_template(_layout([region]))
    text = template["pages"][0]["regions"][0]["text"]
    assert "grace@example.com" not in text


def test_paragraph_long_digit_run_is_redacted():
    region = _region("paragraph", [{"text": "Account # 12345678 is active.", "cells": ["irrelevant"]}])
    template = compile_document_template(_layout([region]))
    text = template["pages"][0]["regions"][0]["text"]
    assert "12345678" not in text


def test_paragraph_currency_amount_is_redacted():
    region = _region("paragraph", [{"text": "A fee of $1,845.20 was charged.", "cells": ["irrelevant"]}])
    template = compile_document_template(_layout([region]))
    text = template["pages"][0]["regions"][0]["text"]
    assert "1,845.20" not in text


def test_heading_is_preserved_verbatim_when_generic():
    region = _region("heading", [{"text": "Summary of Your Account", "cells": ["irrelevant"]}])
    template = compile_document_template(_layout([region]))
    assert template["pages"][0]["regions"][0]["text"] == "Summary of Your Account"


def test_paragraph_street_address_is_redacted():
    """Regression test: a real scanned bank statement's letterhead address
    ("1765 SHERIDAN DRIVE") leaked through verbatim before pii.py gained a
    street_address pattern - caught via smoke testing against actual data.
    """
    region = _region(
        "paragraph",
        [{"text": "JAMES C. MORRISON 1765 SHERIDAN DRIVE YOUR CITY, USA 03087", "cells": ["irrelevant"]}],
    )
    template = compile_document_template(_layout([region]))
    text = template["pages"][0]["regions"][0]["text"]
    assert "SHERIDAN" not in text
    assert "MORRISON" not in text


def test_paragraph_never_contains_raw_high_confidence_entity():
    region = _region(
        "paragraph", [{"text": "Please contact Dr. Grace Hopper for questions.", "cells": ["irrelevant"]}]
    )
    template = compile_document_template(_layout([region]))
    text = template["pages"][0]["regions"][0]["text"]
    assert "Grace Hopper" not in text


def test_known_field_name_recurring_in_a_heading_is_still_redacted():
    """Regression: found by inspecting an actual generated synthetic
    twin - a patient's name, correctly masked/regenerated where it
    appeared as an explicit "Name:" field, still leaked verbatim in the
    document's running header, because a bare two-word name with no
    title prefix scores only 0.6 confidence on its own (see
    entities.py/_PERSON_PATTERN), below the 0.8 bar
    _redact_inline_sensitive_values requires to avoid mismarking generic
    headings. A field's own explicit "Label: Value" shape is much
    stronger evidence, so any value already classified PERSON/ORG/date
    there must also be swept for and masked wherever it recurs elsewhere
    in the document, regardless of that bar.
    """
    heading = _region(
        "heading",
        [{"text": "Healthcare Kimberly Lawrence 24/05/1977", "cells": ["irrelevant"]}],
        region_id="p1_r0",
        index=0,
    )
    field = _region(
        "field",
        [{"text": "Name: Kimberly Lawrence", "cells": ["Name:", "Kimberly Lawrence"]}],
        region_id="p1_r1",
        index=1,
    )
    template = compile_document_template(_layout([heading, field]))
    heading_text = template["pages"][0]["regions"][0]["text"]
    assert "Kimberly Lawrence" not in heading_text
    assert template["pages"][0]["regions"][0]["inline_variable_spans"]


def test_known_field_value_sweep_does_not_touch_unrelated_generic_headings():
    """A known field value's sweep must never over-redact unrelated
    static template wording it merely happens to share no text with -
    guards against a sweep implementation that's overly broad.
    """
    heading = _region("heading", [{"text": "Patient Demographics", "cells": ["irrelevant"]}], region_id="p1_r0", index=0)
    field = _region(
        "field", [{"text": "Name: Kimberly Lawrence", "cells": ["Name:", "Kimberly Lawrence"]}], region_id="p1_r1", index=1
    )
    template = compile_document_template(_layout([heading, field]))
    assert template["pages"][0]["regions"][0]["text"] == "Patient Demographics"
