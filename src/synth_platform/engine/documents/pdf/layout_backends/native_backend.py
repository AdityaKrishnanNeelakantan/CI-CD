from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.layout_backends.base import LayoutBackend
from synth_platform.engine.documents.pdf.native_layout import extract_positioned_spans


class NativeLayoutBackend(LayoutBackend):
    """Wraps the existing pypdf-based extractor - no behavior change."""

    backend_name = "native"

    def extract_spans(self, file_path: str | Path) -> list[dict[str, Any]]:
        return extract_positioned_spans(file_path)
