from __future__ import annotations

from dataclasses import dataclass

import pytest

from synth_platform.engine.documents.pdf.docling_engine import (
    _extract_page_texts,
    _extract_positioned_spans,
)

pytestmark = pytest.mark.unit


@dataclass
class _Size:
    width: float
    height: float


@dataclass
class _Page:
    size: _Size


@dataclass
class _BBox:
    l: float
    t: float
    r: float
    b: float

    def to_top_left_origin(self, page_height: float):
        return self


@dataclass
class _Prov:
    page_no: int
    bbox: _BBox


@dataclass
class _TextItem:
    text: str
    prov: list[_Prov]
    data: object | None = None


@dataclass
class _Cell:
    text: str
    bbox: _BBox
    start_row_offset_idx: int
    start_col_offset_idx: int


@dataclass
class _TableData:
    table_cells: list[_Cell]


@dataclass
class _TableItem:
    prov: list[_Prov]
    data: _TableData


class _Document:
    def __init__(self):
        self.pages = {1: _Page(_Size(600, 800)), 2: _Page(_Size(600, 800))}
        self._items = [
            _TextItem("Account Number: 8823471", [_Prov(1, _BBox(60, 80, 300, 100))]),
            _TableItem(
                [_Prov(2, _BBox(50, 100, 550, 300))],
                _TableData(
                    [
                        _Cell("Date", _BBox(50, 100, 150, 120), 0, 0),
                        _Cell("Amount", _BBox(200, 100, 300, 120), 0, 1),
                        _Cell("2026-08-13", _BBox(50, 140, 150, 160), 1, 0),
                        _Cell("10.00", _BBox(200, 140, 300, 160), 1, 1),
                    ]
                ),
            ),
        ]

    def iterate_items(self):
        for item in self._items:
            yield item, 0

    def export_to_text(self):
        return "fallback"


def test_docling_document_is_adapted_to_page_text():
    pages = _extract_page_texts(_Document())
    assert pages[0] == "Account Number: 8823471"
    assert "Date\tAmount" in pages[1]
    assert "2026-08-13\t10.00" in pages[1]


def test_docling_bounding_boxes_are_normalized_for_platform_layout_contract():
    spans = _extract_positioned_spans(_Document())
    assert spans
    first = next(span for span in spans if span["text"] == "Account Number: 8823471")
    assert first["page_number"] == 1
    assert first["x0"] == pytest.approx(0.1)
    assert first["x1"] == pytest.approx(0.5)
    assert first["y0"] == pytest.approx(0.1)
    assert first["y1"] == pytest.approx(0.125)
    assert any(span["text"] == "Amount" and span["page_number"] == 2 for span in spans)
