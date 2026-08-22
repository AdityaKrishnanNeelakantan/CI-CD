"""Native PDF loader preserving text blocks, tables, images, and metadata."""
from __future__ import annotations

from pathlib import Path

from synth_platform.infrastructure.documents.model import (
    Bookmark, BoundingBox, Document, ExtractionWarning, Line, Page, Table, TableBlock,
    TableCell, TextBlock,
)


def load_pdf(path: str | Path, max_pages: int | None = None) -> Document:
    import pdfplumber

    doc = Document()
    with pdfplumber.open(str(path)) as pdf:
        doc.metadata = {str(k): str(v) for k, v in (pdf.metadata or {}).items()
                        if v is not None}
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for pageno, page in enumerate(pages):
            text_blocks = []
            tables = []
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            for order, word in enumerate(words):
                text = str(word.get("text", "")).strip()
                if not text:
                    continue
                bbox = BoundingBox(x0=float(word.get("x0", 0)),
                                   y0=float(word.get("top", 0)),
                                   x1=float(word.get("x1", 0)),
                                   y1=float(word.get("bottom", 0)))
                text_blocks.append(TextBlock(page=pageno, text=text, bbox=bbox,
                                             reading_order=order, provenance="native"))
            raw_text = page.extract_text() or ""
            for raw in raw_text.split("\n"):
                value = raw.strip()
                if value:
                    doc.lines.append(Line(text=value, page=pageno, provenance="native"))
            for raw_table in page.extract_tables() or []:
                rows = [[(cell or "").strip() for cell in row] for row in raw_table if row]
                if not rows:
                    continue
                cells = [TableCell(text=value, row=r, column=c)
                         for r, row in enumerate(rows) for c, value in enumerate(row)]
                table = Table(rows=rows, page=pageno, cells=cells)
                doc.tables.append(table)
                tables.append(TableBlock(page=pageno, cells=cells))
            image_count = len(page.images or [])
            if len(words) >= 20:
                left = sum(float(word.get("x0", 0)) < page.width / 2 for word in words)
                right = len(words) - left
                if left / len(words) > 0.25 and right / len(words) > 0.25:
                    doc.warnings.append(ExtractionWarning(
                        code="possible_two_column_layout",
                        message="page may require column-aware reading-order review",
                        page=pageno))
            if image_count and not raw_text.strip():
                doc.warnings.append(ExtractionWarning(
                    code="image_only_page",
                    message="page contains images but no native text layer",
                    page=pageno))
            doc.pages.append(Page(number=pageno, text_blocks=text_blocks,
                                  tables=tables, image_count=image_count))
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        def walk(items, level=0):
            for item in items or []:
                if isinstance(item, list):
                    yield from walk(item, level + 1)
                else:
                    try:
                        page_number = reader.get_destination_page_number(item)
                        yield Bookmark(title=str(getattr(item, "title", item)),
                                       page=int(page_number), level=level)
                    except Exception:
                        continue
        doc.bookmarks = list(walk(getattr(reader, "outline", [])))
    except Exception:
        pass
    return doc
