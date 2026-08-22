"""OCR fallback for PDFs with no usable native text layer.

Uses pytesseract (Tesseract binary) and pdf2image (poppler) to render
pages to images and OCR them. Both are external system binaries, not
pure-Python libraries like the rest of this project's dependencies -
resolve_tesseract_cmd() and is_ocr_available() exist so callers (and
tests) can detect their absence up front and degrade to a clear
"ocr_required"/REVIEW_REQUIRED signal instead of crashing, matching the
never-crash-always-record-evidence pattern used everywhere else in
src/documents.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.errors import OCRUnavailableError

_COMMON_TESSERACT_PATHS = (
    # Windows: the official installer does not reliably add tesseract to PATH.
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    # Linux/macOS: PATH-based discovery via shutil.which() (checked first)
    # covers the common case, but these last-resort paths handle minimal
    # containers/environments where PATH is restricted or not inherited.
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)

DEFAULT_OCR_DPI = 200
LOW_CONFIDENCE_THRESHOLD = 60.0


@lru_cache(maxsize=1)
def resolve_tesseract_cmd() -> str | None:
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and Path(env_path).is_file():
        return env_path

    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    for candidate in _COMMON_TESSERACT_PATHS:
        if Path(candidate).is_file():
            return candidate

    return None


def is_ocr_available() -> bool:
    return resolve_tesseract_cmd() is not None and shutil.which("pdftoppm") is not None


def run_ocr(path: str | Path, dpi: int = DEFAULT_OCR_DPI) -> dict[str, Any]:
    """Render every page of `path` and OCR it.

    Returns per-page text plus a mean word confidence (0-100) per page.
    Only ever raises OCRUnavailableError for a missing/broken OCR
    toolchain - a genuinely blank or low-quality scan is a normal,
    successful result (empty text, low confidence), not an error.
    """
    tesseract_cmd = resolve_tesseract_cmd()
    if tesseract_cmd is None:
        raise OCRUnavailableError(
            "Tesseract OCR binary not found (set TESSERACT_CMD or install Tesseract-OCR)"
        )
    if shutil.which("pdftoppm") is None:
        raise OCRUnavailableError("Poppler (pdftoppm) not found on PATH; required by pdf2image")

    import pytesseract
    from pdf2image import convert_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFPopplerTimeoutError,
        PDFSyntaxError,
        PopplerNotInstalledError,
    )

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        images = convert_from_path(str(path), dpi=dpi)
    except (
        PDFInfoNotInstalledError,
        PopplerNotInstalledError,
        PDFPageCountError,
        PDFPopplerTimeoutError,
        PDFSyntaxError,
    ) as exc:
        raise OCRUnavailableError(f"Could not render PDF pages for OCR: {exc}") from exc

    pages_text: list[str] = []
    page_confidences: list[float] = []
    positioned_spans: list[dict[str, Any]] = []

    for page_number, image in enumerate(images, start=1):
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        img_width, img_height = image.size

        words: list[str] = []
        confidences: list[float] = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            words.append(word)
            conf = float(data["conf"][i])
            if conf < 0:
                continue
            confidences.append(conf)
            left, top, width, height = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            positioned_spans.append(
                {
                    "text": word,
                    "x0": round(left / img_width, 6),
                    "y0": round(top / img_height, 6),
                    "x1": round(min((left + width) / img_width, 1.0), 6),
                    "y1": round(min((top + height) / img_height, 1.0), 6),
                    "page_number": page_number,
                    "confidence": conf,
                }
            )

        pages_text.append(" ".join(words))
        page_confidences.append(round(sum(confidences) / len(confidences), 2) if confidences else 0.0)

    mean_confidence = round(sum(page_confidences) / len(page_confidences), 2) if page_confidences else 0.0

    return {
        "pages_text": pages_text,
        "page_confidences": page_confidences,
        "mean_confidence": mean_confidence,
        "dpi": dpi,
        # Word-level bounding boxes (fractional 0-1, top-left origin) for
        # src/documents/layout_engine.py. Never persisted as-is by any
        # caller - it's raw extracted text with coordinates attached, same
        # rule as everywhere else in src/documents.
        "positioned_spans": positioned_spans,
    }
