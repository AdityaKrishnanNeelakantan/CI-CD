"""Route PDFs to native parsing or OCR and fail closed on weak extraction."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from synth_platform.infrastructure.documents.model import Document
from synth_platform.infrastructure.documents.pdf_loader import load_pdf
from synth_platform.errors import ExtractionQualityError


class OcrAdapter(Protocol):
    def extract(self, path: str | Path) -> Document: ...


class PdfLoadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    document: Document
    route: str
    native_coverage: float
    ocr_used: bool
    source_fingerprint: str


class PdfRouter:
    def __init__(self, ocr_adapter: OcrAdapter | None = None,
                 native_character_threshold: int = 40,
                 minimum_ocr_confidence: float = 0.65):
        self.ocr_adapter = ocr_adapter
        self.native_character_threshold = native_character_threshold
        self.minimum_ocr_confidence = minimum_ocr_confidence

    def load(self, source: bytes | str | Path, mode: str = "auto") -> PdfLoadResult:
        if mode not in {"auto", "native", "ocr"}:
            raise ValueError("mode must be auto, native, or ocr")
        temporary = None
        if isinstance(source, bytes):
            digest = hashlib.sha256(source).hexdigest()
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            handle.write(source); handle.close()
            path = Path(handle.name); temporary = path
        else:
            path = Path(source)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            native = load_pdf(path)
            pages = max(1, len(native.pages))
            coverage = native.meaningful_character_count / pages
            native_ok = coverage >= self.native_character_threshold or bool(native.tables)
            if mode == "native":
                if not native_ok:
                    raise ExtractionQualityError(
                        "native PDF extraction produced insufficient structured content")
                return PdfLoadResult(document=native, route="native",
                                     native_coverage=coverage, ocr_used=False,
                                     source_fingerprint=digest)
            if mode == "auto" and native_ok:
                return PdfLoadResult(document=native, route="native",
                                     native_coverage=coverage, ocr_used=False,
                                     source_fingerprint=digest)
            if self.ocr_adapter is None:
                raise ExtractionQualityError(
                    "OCR is required for this PDF but no OCR adapter is configured")
            document = self.ocr_adapter.extract(path)
            confidences = [line.confidence for line in document.lines]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            if document.meaningful_character_count < self.native_character_threshold:
                raise ExtractionQualityError("OCR produced insufficient content")
            if confidence < self.minimum_ocr_confidence:
                raise ExtractionQualityError(
                    f"OCR confidence {confidence:.3f} is below threshold")
            return PdfLoadResult(document=document, route="ocr",
                                 native_coverage=coverage, ocr_used=True,
                                 source_fingerprint=digest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
