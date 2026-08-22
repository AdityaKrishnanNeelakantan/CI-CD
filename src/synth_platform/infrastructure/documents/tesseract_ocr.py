"""Optional local OCR adapter using Poppler + Tesseract; no network calls."""
from __future__ import annotations

from pathlib import Path

from synth_platform.infrastructure.documents.model import (
    BoundingBox, Document, Line, Page, TextBlock,
)
from synth_platform.errors import ExtractionQualityError


class TesseractOcrAdapter:
    def __init__(self, dpi: int = 200, language: str = "eng"):
        self.dpi = dpi
        self.language = language

    def extract(self, path: str | Path) -> Document:
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from pytesseract import Output
        except ImportError as exc:
            raise ExtractionQualityError(
                "OCR dependencies are unavailable (pytesseract/pdf2image)") from exc
        try:
            images = convert_from_path(str(path), dpi=self.dpi)
        except Exception as exc:
            raise ExtractionQualityError("failed to rasterize PDF for OCR") from exc
        document = Document(metadata={"ocr_engine": "tesseract"})
        for page_number, image in enumerate(images):
            data = pytesseract.image_to_data(
                image, lang=self.language, output_type=Output.DICT)
            groups: dict[tuple[int, int, int], list[tuple[int, str, float, BoundingBox]]] = {}
            for index, raw in enumerate(data.get("text", [])):
                text = str(raw).strip()
                try:
                    confidence = max(0.0, float(data["conf"][index]) / 100.0)
                except (ValueError, TypeError):
                    confidence = 0.0
                if not text or confidence <= 0:
                    continue
                left = float(data["left"][index]); top = float(data["top"][index])
                width = float(data["width"][index]); height = float(data["height"][index])
                bbox = BoundingBox(x0=left, y0=top, x1=left + width, y1=top + height)
                key = (int(data["block_num"][index]), int(data["par_num"][index]),
                       int(data["line_num"][index]))
                groups.setdefault(key, []).append((index, text, confidence, bbox))
            blocks = []
            for order, (_, words) in enumerate(sorted(groups.items())):
                words.sort(key=lambda item: item[0])
                text = " ".join(item[1] for item in words)
                confidence = sum(item[2] for item in words) / len(words)
                bbox = BoundingBox(
                    x0=min(item[3].x0 for item in words),
                    y0=min(item[3].y0 for item in words),
                    x1=max(item[3].x1 for item in words),
                    y1=max(item[3].y1 for item in words),
                )
                blocks.append(TextBlock(page=page_number, text=text, bbox=bbox,
                                        reading_order=order, confidence=confidence,
                                        provenance="ocr"))
                document.lines.append(Line(text=text, page=page_number, bbox=bbox,
                                           confidence=confidence, provenance="ocr"))
            document.pages.append(Page(number=page_number, text_blocks=blocks,
                                       image_count=1))
        return document
