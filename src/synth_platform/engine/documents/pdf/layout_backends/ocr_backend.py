from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.layout_backends.base import LayoutBackend
from synth_platform.engine.documents.pdf.ocr_engine import run_ocr


class OcrLayoutBackend(LayoutBackend):
    """Wraps the existing pytesseract-based extractor - no behavior change."""

    backend_name = "ocr"

    def extract_spans(self, file_path: str | Path) -> list[dict[str, Any]]:
        return run_ocr(file_path)["positioned_spans"]
