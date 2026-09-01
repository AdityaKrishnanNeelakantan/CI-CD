"""End-to-end (Mode 2): document_profiling -> template_compilation ->
document_deidentification, proving the redacted PDF and its report never
leak any of the source document's real values - the leakage-test
analogue of tests/e2e/test_render_and_validation_e2e.py's Mode 3 proof,
but for de-identification, which never fabricates a replacement value at
all (unlike Mode 3's Faker-backed synthetic values).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.deidentification_service import (
    DEIDENTIFICATION_REPORT_FILENAME,
    REDACTED_PDF_FILENAME,
    load_document_deidentification_report,
    run_document_deidentification,
)
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    load_document_template,
    run_template_compilation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.e2e


def test_deidentification_pipeline_leaks_no_source_values(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    source_secrets = ["8823471", "401.50", "Grace Hopper", "grace@example.com", "2024-06-01"]

    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Contact: grace@example.com\n"
            "Issued Date: 2024-06-01"
        ],
    )

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir)
    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))

    run_template_compilation(pdf_path, "doc1", manifest, extraction_method=profile["extraction_method"])
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))

    result = run_document_deidentification(template, "doc1", manifest, template_reference=str(pdf_path))
    assert result.is_success()

    # Simulate a fresh process: re-read every artifact from disk only.
    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names == ["document_profiling", "template_compilation", "document_deidentification"]
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    report = load_document_deidentification_report(
        manifest.output_path(f"documents/doc1/{DEIDENTIFICATION_REPORT_FILENAME}")
    )
    redacted_pdf_path = manifest.output_path(f"documents/doc1/{REDACTED_PDF_FILENAME}")
    assert redacted_pdf_path.is_file()

    extracted_from_redacted = PDFDocumentAdapter().extract(redacted_pdf_path)
    serialized_report = json.dumps(report)
    serialized_template = json.dumps(template)
    for secret in source_secrets:
        assert secret not in extracted_from_redacted["text"], f"{secret!r} leaked into the redacted PDF"
        assert secret not in serialized_report, f"{secret!r} leaked into document_deidentification_report.json"
        assert secret not in serialized_template, f"{secret!r} leaked into document_template.json"

    # And unlike Mode 3, the redacted output must never contain a
    # fabricated value either - only masked previews of the original.
    assert "*" in extracted_from_redacted["text"]
