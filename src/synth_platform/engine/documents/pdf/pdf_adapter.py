"""PDF implementation of DocumentAdapter, using pypdf for text extraction.

All PDF-specific behaviour (pypdf calls, encryption handling, per-page
extraction) is isolated in this module.
"""

from __future__ import annotations

from pathlib import Path

import pypdf

from synth_platform.engine.documents.pdf.base import DocumentAdapter
from synth_platform.engine.documents.pdf.errors import ExtractionError
from synth_platform.engine.documents.pdf.native_layout import (
    extract_acroform_field_values,
    extract_pdf_info_metadata,
)


class PDFDocumentAdapter(DocumentAdapter):
    source_format = "pdf"
    supported_extensions = frozenset({".pdf"})

    def _extract_pages(self, path: Path) -> list[str]:
        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open PDF: {exc}") from exc

        if reader.is_encrypted:
            # Only ever try an empty password (some PDFs are "encrypted"
            # with no real password set) - never attempt to guess a real one.
            try:
                reader.decrypt("")
            except Exception:
                pass
            if reader.is_encrypted:
                raise ExtractionError(f"PDF is encrypted and could not be opened: {path}")

        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                # A single unparseable page must not fail the whole document -
                # record it as an empty page rather than aborting extraction.
                pages.append("")

        # AcroForm field values are never part of a page's content stream
        # (page.extract_text() cannot see them - verified empirically: a
        # real filled field extracts as completely empty text), so without
        # this a fillable form's actual data would be entirely invisible
        # to PII scanning, entity detection, and the document's text_hash.
        try:
            for field in extract_acroform_field_values(path):
                index = field["page_number"] - 1
                if 0 <= index < len(pages):
                    pages[index] = f"{pages[index]}\n{field['name']}: {field['value']}".strip()
        except ExtractionError:
            pass

        # The PDF "Info dictionary" (Author/Title/Subject/Keywords) is a
        # real metadata-leakage vector distinct from page content - Office
        # exporters routinely auto-populate /Author with the OS username or
        # the actual author's name (see extract_pdf_info_metadata's own
        # docstring). Merged into page 1's text purely so PII/entity
        # scanning and the text_hash cover it too; the rendered PDF twin
        # never reads the source file at all, so this can never reach the
        # twin itself - only this adapter's own profiling output.
        try:
            if pages:
                for entry in extract_pdf_info_metadata(path):
                    pages[0] = f"{pages[0]}\n{entry['name']}: {entry['value']}".strip()
        except ExtractionError:
            pass

        return pages
