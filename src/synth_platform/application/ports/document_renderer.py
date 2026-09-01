"""Port for rendering source-free synthetic documents."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DocumentRenderer(Protocol):
    def render(self, artifact, tables, output_path: str | Path,
               generation_run_id: str, document_id: str | None = None): ...
