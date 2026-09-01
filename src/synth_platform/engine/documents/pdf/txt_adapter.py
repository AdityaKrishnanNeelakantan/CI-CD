"""Plain-text implementation of DocumentAdapter - concrete proof that the
plugin architecture in src/documents/base.py needs no redesign to support
a second document type beyond PDF (see DocumentAdapter(ABC)'s own
docstring: "all file-format-specific behaviour ... must live only inside
a concrete DocumentAdapter implementation"). Everything downstream
(text_stats, language, pii, entities, document_classifier) already
consumes only the generic NormalisedDocument dict extract() returns, so
none of it needs any change to support this adapter.
"""

from __future__ import annotations

from pathlib import Path

from synth_platform.engine.documents.pdf.base import DocumentAdapter


class TXTDocumentAdapter(DocumentAdapter):
    source_format = "txt"
    supported_extensions = frozenset({".txt"})

    def _extract_pages(self, path: Path) -> list[str]:
        content = path.read_text(encoding="utf-8", errors="replace")
        # Plain text has no native page concept; the form-feed character
        # (\f) is the de facto convention some editors/exporters use as a
        # page break - honor it if present, otherwise treat the whole
        # file as a single page.
        if "\f" in content:
            return content.split("\f")
        return [content]
