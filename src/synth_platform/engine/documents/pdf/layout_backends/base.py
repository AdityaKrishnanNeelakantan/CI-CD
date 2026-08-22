"""Pluggable span-extraction backend contract for template compilation.

A LayoutBackend's only job is producing positioned text spans - the same
{text, x0, y0, x1, y1, page_number, confidence} shape src/documents/
layout_engine.py already treats as backend-agnostic (see that module's
docstring). Everything downstream of extract_spans() - layout_engine.py,
template_compiler.py, binding_engine.py, etc. - must never know or care
which backend produced the spans.

Adding a new backend (e.g. a Docling- or ML-model-based layout extractor)
means adding one class here and registering it in registry.py; it must
never require changes to src/documents/template_service.py, which only
ever calls the generic LayoutBackend interface. Mirrors the isolation
src/synthesis/base.py provides for synthesizer adapters and
src/adapters/base.py provides for source adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LayoutBackend(ABC):
    #: Registry key, e.g. "native", "ocr" - must match the extraction_method
    #: strings src/documents/extraction_router.py already produces, since
    #: template compilation is always invoked with the extraction_method
    #: run_document_profiling() already decided.
    backend_name: str

    @abstractmethod
    def extract_spans(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Return positioned text spans for the given file.

        Each span is {text, x0, y0, x1, y1, page_number, confidence} with
        fractional 0-1 coordinates and a top-left origin.
        """
