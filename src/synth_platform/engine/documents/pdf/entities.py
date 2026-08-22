"""Rule-based named entity extraction: PERSON, ORGANIZATION, DATE.

Deliberately pattern-based rather than a black-box NER model, consistent
with this project's approach everywhere else (src/inference/engine.py,
src/documents/pii.py): confidence must trace back to an explicit,
inspectable rule. This trades recall on unusual real-world text for being
fully explainable and dependency-light - acceptable for a baseline, and
documented as a known limitation rather than hidden.

Known limitation: also US/English-centric by construction, like
src/documents/pii.py (_MONTH_NAMES are English month names, _TITLE_PREFIXES
are English honorifics, and _NAME_WORD assumes Latin-alphabet Title-Case
names). Non-English names, dates, and organisation naming conventions will
generally not be detected.

Like PII findings, entity findings never carry the raw matched text -
offsets into the normalised document plus a masked preview only.
"""

from __future__ import annotations

import re
from typing import Any

ENTITY_TYPE_DATE = "DATE"
ENTITY_TYPE_ORG = "ORG"
ENTITY_TYPE_PERSON = "PERSON"

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_ORG_SUFFIXES = (
    "Inc|LLC|L\\.L\\.C\\.|Corp|Corporation|Ltd|Co|Company|University|Institute|"
    "Foundation|Group|Association|Partners"
)
_TITLE_PREFIXES = {"mr.", "mrs.", "ms.", "dr.", "prof."}
_NON_NAME_STARTERS = {
    "the", "this", "that", "these", "those", "dear", "sincerely", "please",
    "thank", "regards", "best", "hello", "hi", "attention", "re",
}

# Ordered by specificity: dates and orgs are checked before the generic
# person heuristic so e.g. "Acme Corp" is never misread as a two-word name.
# The slash/dash numeric form (DD/MM/YYYY, MM-DD-YYYY, ...) was a real gap
# found via smoke-testing against an actual medical record whose running
# header/narrative repeated the patient's DOB in this exact numeric shape
# ("24/05/1977") - the ISO and long-form patterns above never fire on it,
# so it leaked through every page unmasked while the same DOB, expressed
# identically, was correctly masked wherever it appeared as an explicit
# "DOB:"-labelled field (template_compiler._classify_value has its own,
# separate numeric-date pattern that already covered the field case).
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{4}}\b"),
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+\d{{4}}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
]
_ORG_PATTERN = re.compile(
    rf"\b(?:[A-Z][a-zA-Z&]*[ \t]+){{0,4}}[A-Z][a-zA-Z&]*[ \t]+(?:{_ORG_SUFFIXES})\.?\b"
)
# [ \t]+ (not \s+) between words deliberately: \s+ also matches newlines,
# which let a name at the end of a line merge into the next line's first
# word (e.g. "Ada Lovelace\nEmail" as one match) - real bug caught by
# testing against an actual multi-line PDF, not a hand-written fixture.
# _NAME_SUFFIX (?:[-'][A-Za-z]+)* on each word allows hyphenated/apostrophe
# surnames ("Garcia-Lopez", "O'Brien") without weakening the base
# Title-Case requirement that keeps this from matching generic capitalised
# phrases. Shared by both patterns below so the hyphen/apostrophe rule
# only needs to change in one place.
_NAME_SUFFIX = r"(?:[-'][A-Za-z]+)*"
_NAME_WORD = rf"[A-Z][a-z]+{_NAME_SUFFIX}"
_PERSON_PATTERN = re.compile(rf"\b{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{1,3}}\b")
# A middle initial ("James C. Morrison") is a strong, specific signal on
# its own - rare in generic boilerplate - so this one also matches
# ALL-CAPS names ("JAMES C. MORRISON"), which _PERSON_PATTERN above never
# catches (it requires lowercase letters after each initial capital).
# Real gap found via smoke-testing against an actual scanned bank
# statement, where the account holder's name is printed in all caps.
_NAME_WORD_ANY_CASE = rf"[A-Z][A-Za-z]+{_NAME_SUFFIX}"
_PERSON_PATTERN_WITH_MIDDLE_INITIAL = re.compile(
    rf"\b{_NAME_WORD_ANY_CASE}[ \t]+[A-Z]\.[ \t]+{_NAME_WORD_ANY_CASE}\b"
)

# Bibliographic citation-list author names ("Di Gregorio C, Frattini M,
# Maffei S, ... Ponz de Leon M.") use a "Surname Initial" shape that
# neither _PERSON_PATTERN (needs 2+ full Title-Case words) nor
# _PERSON_PATTERN_WITH_MIDDLE_INITIAL (needs "First I. Last", period
# required) matches - real gap found via smoke-testing against an actual
# multi-page medical/genomics report, whose reference list's author names
# leaked verbatim into the rendered synthetic twin. A single "Surname
# Initial" unit alone is deliberately NOT enough to fire (too easily
# confused with things like "Table A", "Figure B", "Appendix C"); only
# a run of 2+ such units chained by list separators ("Name I, Name I, ...")
# is treated as a citation author list, since that specific repetition
# is not a shape ordinary headings/prose ever produce.
_CITATION_AUTHOR_UNIT_PATTERN = re.compile(r"\b(?:[a-z]+\s+)?[A-Z][a-z]+(?:[-'][A-Za-z]+)*\s+[A-Z]\b")
_CITATION_LIST_SEPARATOR_PATTERN = re.compile(r"^[,\s]{1,4}(?:and\s+)?$")


def _find_citation_author_lists(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    run_start: int | None = None
    run_end: int | None = None
    run_count = 0
    for match in _CITATION_AUTHOR_UNIT_PATTERN.finditer(text):
        if run_end is not None and _CITATION_LIST_SEPARATOR_PATTERN.match(text[run_end : match.start()]):
            run_end = match.end()
            run_count += 1
        else:
            if run_count >= 2:
                spans.append((run_start, run_end))
            run_start, run_end, run_count = match.start(), match.end(), 1
    if run_count >= 2:
        spans.append((run_start, run_end))
    return spans


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _looks_like_a_name(matched_text: str) -> bool:
    first_word = matched_text.split()[0].lower()
    return first_word not in _NON_NAME_STARTERS


def _is_form_label(text: str, end: int) -> bool:
    # A capitalised phrase immediately followed by a colon ("Invoice
    # Number:", "Bill To:", "Total Due:") is a form/table label, not a
    # person's name - this generalises far better than trying to enumerate
    # every possible label word.
    return text[end : end + 2].lstrip().startswith(":")


def extract_entities(text: str) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    def try_add(entity_type: str, start: int, end: int, confidence: float, evidence: str) -> None:
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            return
        occupied.append((start, end))
        accepted.append(
            {
                "type": entity_type,
                "start": start,
                "end": end,
                "confidence": confidence,
                "redaction_preview": _mask(text[start:end]),
                "evidence": [evidence],
            }
        )

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            try_add(ENTITY_TYPE_DATE, match.start(), match.end(), 0.9, "date_pattern")

    for match in _ORG_PATTERN.finditer(text):
        try_add(ENTITY_TYPE_ORG, match.start(), match.end(), 0.8, "org_suffix_pattern")

    for start, end in _find_citation_author_lists(text):
        try_add(ENTITY_TYPE_PERSON, start, end, 0.85, "citation_author_list_pattern")

    for match in _PERSON_PATTERN.finditer(text):
        matched_text = match.group(0)
        if not _looks_like_a_name(matched_text):
            continue
        if _is_form_label(text, match.end()):
            continue
        start = match.start()
        preceding_text = text[max(0, start - 6) : start].strip().lower()
        has_title_prefix = any(preceding_text.endswith(prefix) for prefix in _TITLE_PREFIXES)
        confidence = 0.85 if has_title_prefix else 0.6
        evidence = "title_prefix+capitalised_sequence" if has_title_prefix else "capitalised_sequence"
        try_add(ENTITY_TYPE_PERSON, start, match.end(), confidence, evidence)

    for match in _PERSON_PATTERN_WITH_MIDDLE_INITIAL.finditer(text):
        matched_text = match.group(0)
        if not _looks_like_a_name(matched_text):
            continue
        if _is_form_label(text, match.end()):
            continue
        start = match.start()
        preceding_text = text[max(0, start - 6) : start].strip().lower()
        has_title_prefix = any(preceding_text.endswith(prefix) for prefix in _TITLE_PREFIXES)
        evidence = "title_prefix+middle_initial_pattern" if has_title_prefix else "middle_initial_pattern"
        try_add(ENTITY_TYPE_PERSON, start, match.end(), 0.85, evidence)

    accepted.sort(key=lambda finding: finding["start"])
    return accepted
