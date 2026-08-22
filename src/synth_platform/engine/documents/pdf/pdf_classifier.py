"""PDF structural classification: text / scanned / form / mixed.

Distinct from src/documents/classifier.py's content classification
(invoice/contract/letter/...), which needs actual extracted text and runs
after extraction. This one runs on the cheap structural signals from
run_pdf_preflight(), before extraction, purely to decide whether native
extraction is sufficient or an OCR fallback is required. Same
proposal/confidence/evidence/status shape used throughout this project
(see src/documents/language.py).
"""

from __future__ import annotations

from typing import Any

PDF_TYPES = frozenset({"text", "scanned", "form", "mixed"})
STATUS_PROPOSED = "proposed"


def classify_pdf_type(preflight: dict[str, Any]) -> dict[str, Any]:
    native_char_count = preflight["native_char_count"]
    has_images = preflight["has_images"]
    has_form_fields = preflight["has_form_fields"]
    evidence: list[str] = []

    if has_form_fields:
        evidence.append("has_form_fields")
        return {"pdf_type": "form", "status": STATUS_PROPOSED, "confidence": 0.9, "evidence": evidence}

    if native_char_count == 0:
        # No embedded text at all - a form/report with no static text and no
        # signal either way would already have returned above via
        # has_form_fields, so zero native text here means the page content
        # exists only as an image (or the page is genuinely blank).
        evidence.append("native_char_count=0")
        if has_images:
            evidence.append("has_images")
        return {"pdf_type": "scanned", "status": STATUS_PROPOSED, "confidence": 0.9, "evidence": evidence}

    evidence.append(f"native_char_count={native_char_count}")
    if has_images:
        evidence.append("has_images")
        return {"pdf_type": "mixed", "status": STATUS_PROPOSED, "confidence": 0.7, "evidence": evidence}

    return {"pdf_type": "text", "status": STATUS_PROPOSED, "confidence": 0.95, "evidence": evidence}
