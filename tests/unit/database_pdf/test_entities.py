from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.entities import (
    ENTITY_TYPE_DATE,
    ENTITY_TYPE_ORG,
    ENTITY_TYPE_PERSON,
    extract_entities,
)

pytestmark = pytest.mark.unit


def test_iso_date_is_found_with_exact_offsets():
    text = "The report was filed on 2024-05-01 for review."
    findings = extract_entities(text)
    date_findings = [f for f in findings if f["type"] == ENTITY_TYPE_DATE]
    assert len(date_findings) == 1
    assert text[date_findings[0]["start"] : date_findings[0]["end"]] == "2024-05-01"


def test_long_form_date_is_found():
    text = "Signed on January 5, 2024 by the parties."
    findings = extract_entities(text)
    date_findings = [f for f in findings if f["type"] == ENTITY_TYPE_DATE]
    assert len(date_findings) == 1
    assert text[date_findings[0]["start"] : date_findings[0]["end"]] == "January 5, 2024"


def test_numeric_slash_date_is_found():
    """Regression: found by inspecting an actual generated synthetic twin
    whose running header/narrative repeated the patient's DOB in this
    exact numeric shape ("24/05/1977") on every page - neither the ISO
    nor long-form date pattern ever fires on it, so it leaked through
    completely unmasked while the identical value, printed as an explicit
    "DOB:"-labelled field elsewhere on the same page, was correctly
    masked and regenerated.
    """
    text = "Healthcare Kimberly Lawrence     24/05/1977"
    findings = extract_entities(text)
    date_findings = [f for f in findings if f["type"] == ENTITY_TYPE_DATE]
    assert len(date_findings) == 1
    assert text[date_findings[0]["start"] : date_findings[0]["end"]] == "24/05/1977"


def test_organization_with_suffix_is_found():
    text = "This agreement is between Acme Corp and the client."
    findings = extract_entities(text)
    org_findings = [f for f in findings if f["type"] == ENTITY_TYPE_ORG]
    assert len(org_findings) == 1
    assert text[org_findings[0]["start"] : org_findings[0]["end"]] == "Acme Corp"


def test_organization_is_never_also_reported_as_a_person():
    text = "Acme Corp signed the contract yesterday."
    findings = extract_entities(text)
    assert not any(f["type"] == ENTITY_TYPE_PERSON for f in findings)
    assert any(f["type"] == ENTITY_TYPE_ORG for f in findings)


def test_person_name_is_found_with_exact_offsets():
    text = "The invoice was approved by Ada Lovelace last week."
    findings = extract_entities(text)
    person_findings = [f for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert len(person_findings) == 1
    assert text[person_findings[0]["start"] : person_findings[0]["end"]] == "Ada Lovelace"


def test_title_prefixed_name_gets_higher_confidence():
    text = "Please contact Dr. Grace Hopper for questions."
    findings = extract_entities(text)
    person_findings = [f for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert len(person_findings) == 1
    f = person_findings[0]
    assert text[f["start"] : f["end"]] == "Grace Hopper"
    assert "title_prefix" in f["evidence"][0]
    assert f["confidence"] == 0.85


def test_sentence_starting_capitalised_phrase_is_not_a_false_person():
    text = "The Quick Brown Fox jumps over the lazy dog."
    findings = extract_entities(text)
    assert not any(f["type"] == ENTITY_TYPE_PERSON for f in findings)


def test_multiple_entity_types_in_one_document():
    text = "On 2024-01-15, Ada Lovelace signed a contract with Acme Corp."
    findings = extract_entities(text)
    found_types = {f["type"] for f in findings}
    assert found_types == {ENTITY_TYPE_DATE, ENTITY_TYPE_PERSON, ENTITY_TYPE_ORG}
    starts = [f["start"] for f in findings]
    assert starts == sorted(starts)


def test_findings_never_contain_raw_matched_text():
    text = "Contact Ada Lovelace, who works at Acme Corp, on 2024-01-15."
    findings = extract_entities(text)
    for finding in findings:
        assert set(finding) == {"type", "start", "end", "confidence", "redaction_preview", "evidence"}
    serialized = str(findings)
    assert "Ada Lovelace" not in serialized
    assert "Acme Corp" not in serialized


def test_empty_text_returns_no_findings():
    assert extract_entities("") == []


def test_name_at_end_of_line_is_not_merged_into_next_line_label():
    """Regression test: a name immediately followed by a newline and then a
    labeled field ("Ada Lovelace\\nEmail:") must not have \\s+ span the
    newline and merge into "Ada Lovelace Email", which would then get
    rejected as a form label and lose the real name entirely - caught by
    a real multi-line PDF smoke test.
    """
    text = "Customer: Ada Lovelace\nEmail: ada.lovelace@example.com\n"
    findings = extract_entities(text)
    person_texts = [text[f["start"] : f["end"]] for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert person_texts == ["Ada Lovelace"]


def test_all_caps_name_with_middle_initial_is_found():
    """Regression test: a real scanned bank statement printed the account
    holder's name in all caps with a middle initial ("JAMES C. MORRISON"),
    which the Title-Case-only _PERSON_PATTERN never matches - caught via
    smoke testing against actual data, not a hand-written fixture.
    """
    text = "JAMES C. MORRISON 1765 SHERIDAN DRIVE"
    findings = extract_entities(text)
    person_texts = [text[f["start"] : f["end"]] for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert person_texts == ["JAMES C. MORRISON"]


def test_title_case_name_with_middle_initial_is_found():
    text = "The invoice was signed by James C. Morrison yesterday."
    findings = extract_entities(text)
    person_texts = [text[f["start"] : f["end"]] for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert person_texts == ["James C. Morrison"]


def test_form_labels_are_not_misdetected_as_person_names():
    """Regression test: 'Invoice Number:', 'Bill To:', 'Total Due:' are
    structurally identical to a two-word name (two Title-Case words) but
    are form labels, not people - caught by a real PDF smoke test.
    """
    text = "Invoice Number: INV-1001\nBill To: Acme Corp\nTotal Due: $540.00"
    findings = extract_entities(text)
    person_texts = [text[f["start"] : f["end"]] for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert person_texts == []


def test_citation_style_author_list_is_detected_as_person_with_high_confidence():
    """Regression test: a real academic-report PDF (health_report.pdf)
    listed reference-list authors as 'Surname Initial, Surname Initial,
    ...' (e.g. 'Di Gregorio C, Frattini M, Maffei S, Ponti G.') - a shape
    the plain _PERSON_PATTERN never matches, and which previously survived
    into rendered/redacted twins because entities.py never reported it at
    all. It must now be reported as a single PERSON span at confidence
    >= 0.8 so template_compiler.py's strong_entities redaction filter
    picks it up.
    """
    text = "References: Di Gregorio C, Frattini M, Maffei S, Ponti G. Study of colorectal cancer."
    findings = extract_entities(text)
    person_findings = [f for f in findings if f["type"] == ENTITY_TYPE_PERSON]
    assert len(person_findings) == 1
    finding = person_findings[0]
    # The leading name-particle "Di" is not itself part of the
    # Surname-Initial unit shape, so the redacted span starts at
    # "Gregorio" - the identifying surname/initial chain is still fully
    # covered, which is what matters for leakage protection.
    assert text[finding["start"] : finding["end"]] == "Gregorio C, Frattini M, Maffei S, Ponti G"
    assert finding["confidence"] >= 0.8


def test_single_surname_initial_unit_alone_is_not_treated_as_citation_list():
    """Guard against false positives: a single 'Surname Initial'-shaped
    unit (e.g. table/figure references like 'Table A', 'Figure B') must
    NOT be treated as a citation author list - only 2+ consecutive units
    chained by list-separator punctuation qualify.
    """
    text = "See Table A for details and Figure B for the chart."
    findings = extract_entities(text)
    person_texts = [
        text[f["start"] : f["end"]]
        for f in findings
        if f["type"] == ENTITY_TYPE_PERSON and f["evidence"] == ["citation_author_list_pattern"]
    ]
    assert person_texts == []


def test_generic_report_headings_are_not_misdetected_as_citation_author_lists():
    """Guard against false positives on generic financial/report headings
    that superficially resemble Title-Case name pairs (e.g.
    'Balance Sheet', 'Cash Flow Statement') - none of these should ever be
    picked up by the citation-author-list heuristic.
    """
    text = "Balance Sheet\nCash Flow Statement\nFinancial Overview\nLoss Statement"
    findings = extract_entities(text)
    citation_texts = [
        text[f["start"] : f["end"]]
        for f in findings
        if f["type"] == ENTITY_TYPE_PERSON and f["evidence"] == ["citation_author_list_pattern"]
    ]
    assert citation_texts == []
