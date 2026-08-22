"""PDF preflight: cheap structural signals gathered before deciding how to
extract a document's content - page count, how much native text is
embedded, whether pages contain images, and whether the file is a form.

This is the signal, not the decision (see src/documents/pdf_classifier.py
for the decision) - the same layering used by src/profiling and
src/inference for structured tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pypdf

from synth_platform.engine.documents.pdf.errors import ExtractionError
from synth_platform.engine.documents.pdf.native_layout import extract_acroform_field_values

# US Letter, in PDF points (1/72 inch) - only used as a fallback when a
# page's own mediabox reports a zero/degenerate size, matching the
# fallback native_layout.py already uses for the same edge case.
_LETTER_WIDTH_PT = 612.0
_LETTER_HEIGHT_PT = 792.0


def run_pdf_preflight(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    try:
        reader = pypdf.PdfReader(str(file_path))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF for preflight: {exc}") from exc

    is_encrypted = reader.is_encrypted
    if is_encrypted:
        # Only ever try an empty password, same rule as PDFDocumentAdapter -
        # never attempt to guess a real one.
        try:
            reader.decrypt("")
        except Exception:
            pass
        is_encrypted = reader.is_encrypted

    try:
        page_count = len(reader.pages)
        page_dimensions = [
            {
                "page_number": index,
                "width_pt": float(page.mediabox.width) or _LETTER_WIDTH_PT,
                "height_pt": float(page.mediabox.height) or _LETTER_HEIGHT_PT,
            }
            for index, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        # Reached when decrypt("") above didn't unlock the file and pypdf
        # can't even determine its encryption filter/cipher (e.g. an
        # AES-encrypted PDF with the optional `cryptography` package not
        # installed) - accessing .pages then raises a raw dependency error
        # instead of the clean "still encrypted" case already handled
        # above. Same fail-closed contract as every other stage: never let
        # a raw third-party exception escape uncaught.
        raise ExtractionError(f"Could not read PDF pages for preflight: {exc}") from exc

    native_char_count = 0
    has_images = False

    if not is_encrypted:
        for page in reader.pages:
            try:
                native_char_count += len((page.extract_text() or "").strip())
            except Exception:
                pass
            try:
                if page.images:
                    has_images = True
            except Exception:
                pass

    has_form_fields = False
    if not is_encrypted:
        try:
            has_form_fields = bool(reader.get_fields())
        except Exception:
            has_form_fields = False
        if has_form_fields:
            # AcroForm field values live in the form's own value objects,
            # never in a page's content stream - page.extract_text() above
            # cannot see them (verified empirically: a real filled field
            # extracts as completely empty text), which used to make a
            # genuinely data-bearing fillable form look like an empty
            # document and misroute to a useless OCR fallback.
            try:
                for field in extract_acroform_field_values(file_path):
                    native_char_count += len(field["value"])
            except ExtractionError:
                pass

    return {
        "page_count": page_count,
        "native_char_count": native_char_count,
        "has_images": has_images,
        "is_encrypted": is_encrypted,
        "has_form_fields": has_form_fields,
        "page_dimensions": page_dimensions,
    }
