from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.docling_engine import extract_with_docling
from synth_platform.engine.documents.pdf.layout_backends.base import LayoutBackend


class DoclingLayoutBackend(LayoutBackend):
    """Extract positioned spans through Docling's PDF document model."""

    backend_name = "docling"

    def extract_spans(self, file_path: str | Path) -> list[dict[str, Any]]:
        return extract_with_docling(file_path)["positioned_spans"]
