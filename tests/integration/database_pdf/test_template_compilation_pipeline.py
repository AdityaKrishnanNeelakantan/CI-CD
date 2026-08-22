"""Integration: run_document_profiling() -> run_template_compilation(),
using the profile's own extraction_method decision to pick the geometry
extractor, for both the native and OCR paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    load_document_template,
    run_template_compilation,
)
from tests.fixtures.pdf_factory import make_acroform_pdf, make_pdf, make_scanned_pdf

pytestmark = pytest.mark.integration


def test_native_pdf_field_and_table_regions_are_detected(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Beginning balance 69.96\n"
            "Ending balance 5340.43\n"
            "Fees 12.00"
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    profile_result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata"
    )
    assert profile_result.is_success()
    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))

    template_result = run_template_compilation(
        pdf_path, "doc1", manifest, extraction_method=profile["extraction_method"]
    )
    assert template_result.is_success()
    assert template_result.metrics["field_count"] >= 1

    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))
    region_types = {r["region_type"] for page in template["pages"] for r in page["regions"]}
    assert "field" in region_types

    serialized = str(template)
    assert "8823471" not in serialized
    assert "5340.43" not in serialized


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_scanned_pdf_template_compiles_via_ocr_geometry(tmp_path: Path):
    pdf_path = make_scanned_pdf(
        tmp_path / "scan.pdf",
        [["Account Number: 8823471", "Total Due: 401.50"]],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    profile_result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "scan1", manifest, tmp_path / "metadata"
    )
    assert profile_result.is_success()
    profile = load_document_profile(manifest.output_path(f"documents/scan1/{DOCUMENT_PROFILE_FILENAME}"))
    assert profile["extraction_method"] == "ocr"

    template_result = run_template_compilation(
        pdf_path, "scan1", manifest, extraction_method=profile["extraction_method"]
    )
    assert template_result.is_success()
    assert template_result.metrics["field_count"] >= 1

    template = load_document_template(manifest.output_path(f"documents/scan1/{DOCUMENT_TEMPLATE_FILENAME}"))
    assert template["extraction_method"] == "ocr"
    serialized = str(template)
    assert "8823471" not in serialized
    assert "401.50" not in serialized


def test_acroform_field_values_become_field_regions_with_masked_previews_not_raw_paragraph_text(
    tmp_path: Path,
):
    """Regression: found running this exact pipeline against a real
    fillable-form fixture. Two bugs compounded: (1) AcroForm field
    values are invisible to ordinary text extraction entirely (fixed by
    extract_acroform_field_spans), and (2) even once extracted, a raw
    field name like "account_number" doesn't match the field-line
    pattern's label characters (no "_" allowed) and fell through to
    "paragraph" classification - which runs the boilerplate-text
    redaction sweep and PERMANENTLY BAKED THE MASKED VALUE
    ("88***71") into the template as static text, instead of recording
    it as a field with a masked *preview* and a value_type driving fresh
    synthetic generation. Every real document's rendered twin would
    have shown the exact same masked source value forever.
    """
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf",
        [
            ("customer_name", "Ada Lovelace", [100, 700, 300, 720]),
            ("account_number", "8823471", [100, 400, 300, 420]),
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    profile_result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata"
    )
    assert profile_result.is_success()
    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))

    template_result = run_template_compilation(
        pdf_path, "doc1", manifest, extraction_method=profile["extraction_method"]
    )
    assert template_result.is_success()

    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))
    regions = template["pages"][0]["regions"]
    by_label = {r.get("label"): r for r in regions}

    assert set(by_label) == {"Customer Name", "Account Number"}
    for region in regions:
        assert region["region_type"] == "field"
        # The real value must never survive verbatim - only a masked
        # preview - and it must specifically NOT be the redact-in-place
        # form paragraph text used to fall back to.
        assert "Ada Lovelace" not in json.dumps(region)
        assert "8823471" not in json.dumps(region)
    assert by_label["Account Number"]["value_type"] == "numeric"
