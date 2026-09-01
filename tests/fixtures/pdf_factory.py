"""Shared pytest fixtures: a temporary SQLite database built at test time.

No table or column name here is special-cased anywhere in src/ - these
fixtures exist purely to give the generic adapter something concrete to
discover, and the "no hard-coded names" pass condition is enforced by never
importing these names into src/ code.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def temp_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "fixture.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                signup_date TEXT
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );

            CREATE INDEX idx_orders_customer_id ON orders(customer_id);

            INSERT INTO customers (customer_id, name, email, signup_date) VALUES
                ('cust-1', 'Ada Lovelace', 'ada@example.com', '2024-01-01'),
                ('cust-2', 'Alan Turing', 'alan@example.com', '2024-02-15'),
                ('cust-3', 'Grace Hopper', NULL, '2024-03-20');

            INSERT INTO orders (order_id, customer_id, amount, status) VALUES
                (1, 'cust-1', 19.99, 'shipped'),
                (2, 'cust-1', 42.50, 'pending'),
                (3, 'cust-2', 7.25, 'shipped'),
                (4, 'cust-3', 100.00, 'cancelled'),
                (5, 'cust-2', 15.00, 'pending');
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def empty_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.close()
    return db_path


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Build a real PDF with known text content at deterministic page
    boundaries, so extraction correctness (including offsets) can be
    verified exactly rather than approximately.
    """
    from fpdf import FPDF

    pdf = FPDF()
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, page_text)
    pdf.output(str(path))
    return path


def make_pdf_with_metadata(
    path: Path,
    pages: list[str],
    *,
    author: str | None = None,
    title: str | None = None,
    subject: str | None = None,
    keywords: str | None = None,
) -> Path:
    """Same as make_pdf, but also sets the PDF's "Info dictionary" fields
    - the same fields real Office/PDF exporters routinely auto-populate
    with an author's real name or an internal project codename, i.e. a
    real metadata-leakage vector distinct from page content.
    """
    from fpdf import FPDF

    pdf = FPDF()
    if author is not None:
        pdf.set_author(author)
    if title is not None:
        pdf.set_title(title)
    if subject is not None:
        pdf.set_subject(subject)
    if keywords is not None:
        pdf.set_keywords(keywords)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, page_text)
    pdf.output(str(path))
    return path


def make_acroform_pdf(path: Path, fields: list[tuple[str, str, list[float]]]) -> Path:
    """Build a real, single-page fillable AcroForm PDF: each entry is
    (field_name, value, [x0, y0, x1, y1] in PDF points, bottom-left
    origin). Field values live in the form's own value objects, not the
    page content stream - page.extract_text() genuinely cannot see them
    (verified empirically), which is exactly the gap this fixture exists
    to exercise: a real fillable form's data must not be silently invisible.
    """
    import pypdf
    from pypdf.generic import (
        ArrayObject,
        BooleanObject,
        DictionaryObject,
        NameObject,
        NumberObject,
        RectangleObject,
        TextStringObject,
    )

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    field_refs = []
    for name, value, rect in fields:
        field_dict = DictionaryObject()
        field_dict.update(
            {
                NameObject("/FT"): NameObject("/Tx"),
                NameObject("/T"): TextStringObject(name),
                NameObject("/V"): TextStringObject(value),
                NameObject("/Rect"): RectangleObject(rect),
                NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/F"): NumberObject(4),
            }
        )
        field_refs.append(writer._add_object(field_dict))

    page[NameObject("/Annots")] = ArrayObject(field_refs)
    acroform = DictionaryObject()
    acroform.update(
        {
            NameObject("/Fields"): ArrayObject(field_refs),
            NameObject("/NeedAppearances"): BooleanObject(True),
        }
    )
    writer._root_object[NameObject("/AcroForm")] = acroform

    with open(path, "wb") as f:
        writer.write(f)
    return path


def make_scanned_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Build a PDF with no embedded text layer at all: every page is a
    rendered image of its lines of text, the same shape a real scanned
    document has. Uses Pillow's built-in scalable default font, not a
    system font file, so this fixture is portable across machines/OSes -
    only whether Tesseract/poppler are installed (checked via
    src/documents/ocr_engine.is_ocr_available) determines whether OCR can
    actually read it back.
    """
    from fpdf import FPDF
    from PIL import Image, ImageDraw, ImageFont

    page_size = (1000, 700)
    font = ImageFont.load_default(size=36)

    pdf = FPDF(unit="pt", format=page_size)
    for lines in pages:
        image = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(image)
        y = 40
        for line in lines:
            draw.text((40, y), line, fill="black", font=font)
            y += 50
        pdf.add_page()
        pdf.image(image, x=0, y=0, w=page_size[0], h=page_size[1])
    pdf.output(str(path))
    return path
