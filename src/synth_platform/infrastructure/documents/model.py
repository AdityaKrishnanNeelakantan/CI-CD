"""Format-neutral document IR with layout, provenance, and compatibility views."""
from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(_Base):
    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(_Base):
    page: int
    text: str
    bbox: BoundingBox | None = None
    reading_order: int = 0
    confidence: float = 1.0
    provenance: Literal["native", "ocr"] = "native"


class TableCell(_Base):
    text: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    bbox: BoundingBox | None = None
    confidence: float = 1.0


class TableBlock(_Base):
    page: int
    cells: list[TableCell] = Field(default_factory=list)
    title: str | None = None


class Page(_Base):
    number: int
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    image_count: int = 0


class Bookmark(_Base):
    title: str
    page: int
    level: int = 0


class ExtractionWarning(_Base):
    code: str
    message: str
    page: int | None = None


# Compatibility types retained for existing extraction adapters/tests.
@dataclass
class Line:
    text: str
    page: int = 0
    bbox: BoundingBox | None = None
    confidence: float = 1.0
    provenance: Literal["native", "ocr"] = "native"


@dataclass
class Table:
    rows: list[list[str]]
    page: int = 0
    title: str | None = None
    cells: list[TableCell] = field(default_factory=list)


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)
    bookmarks: list[Bookmark] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[ExtractionWarning] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    @property
    def meaningful_character_count(self) -> int:
        return sum(len(line.text.strip()) for line in self.lines)
