"""Generic document-access contract.

Mirrors SourceAdapter's role for structured sources (src/adapters/base.py):
all file-format-specific behaviour (PDF, plain text, ...) must live only
inside a concrete DocumentAdapter implementation. Everything downstream
(text statistics, language detection, PII/entity detection, document
classification) consumes only the generic NormalisedDocument dict returned
by extract() and must never know or care what the source file format was.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.errors import (
    DocumentAdapterError,
    ExtractionError,
    UnsupportedFileTypeError,
)

DEFAULT_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class DocumentAdapter(ABC):
    """Turns one supported file into a NormalisedDocument."""

    #: Short identifier for this adapter, e.g. "pdf".
    source_format: str
    #: Lower-cased file extensions this adapter accepts, e.g. {".pdf"}.
    supported_extensions: frozenset[str]

    def __init__(self, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        self._max_file_size_bytes = max_file_size_bytes

    @abstractmethod
    def _extract_pages(self, path: Path) -> list[str]:
        """Return extracted text per page/section. Format-specific only."""

    def _validate(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise DocumentAdapterError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"{file_path.suffix!r} is not supported by the {self.source_format} adapter "
                f"(expected one of {sorted(self.supported_extensions)})"
            )
        size = file_path.stat().st_size
        if size == 0:
            raise ExtractionError(f"File is empty: {file_path}")
        if size > self._max_file_size_bytes:
            raise DocumentAdapterError(
                f"File {file_path} is {size} bytes, exceeds max {self._max_file_size_bytes} bytes"
            )

    def extract(self, path: str | Path) -> dict[str, Any]:
        """Validate a file and extract it into the generic NormalisedDocument contract.

        This method is intentionally concrete (not overridable per adapter):
        genericity comes from every adapter implementing the same
        `_extract_pages`, not from each adapter formatting its own output.
        """
        file_path = Path(path)
        self._validate(file_path)

        try:
            pages = self._extract_pages(file_path)
        except DocumentAdapterError:
            raise
        except Exception as exc:  # any format-library failure becomes an ExtractionError
            raise ExtractionError(f"Failed to extract text from {file_path}: {exc}") from exc

        warnings: list[str] = []
        if not pages:
            warnings.append("no_pages_extracted")

        text_parts: list[str] = []
        page_offsets: list[dict[str, int]] = []
        cursor = 0
        for page_text in pages:
            start = cursor
            text_parts.append(page_text)
            cursor += len(page_text)
            page_offsets.append({"start": start, "end": cursor})
            text_parts.append("\n")
            cursor += 1

        full_text = "".join(text_parts).rstrip("\n") if text_parts else ""
        if not full_text.strip():
            warnings.append("empty_extracted_text")

        text_hash = "sha256:" + hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        return {
            "source_reference": str(file_path),
            "source_format": self.source_format,
            "extracted_at": datetime.now(UTC).isoformat(),
            "text": full_text,
            "pages": [
                {"page_number": i + 1, "start": offset["start"], "end": offset["end"]}
                for i, offset in enumerate(page_offsets)
            ],
            "text_hash": text_hash,
            "file_size_bytes": file_path.stat().st_size,
            "extraction_warnings": warnings,
        }
