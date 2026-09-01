"""End-to-end: document_profiling -> template_compilation, fully
traceable - re-read entirely from disk afterward, same proof pattern as
tests/e2e/test_document_pipeline_e2e.py, one stage further along the PDF
track (Extract -> ... -> Layout -> Compile -> DocumentTemplateSpec).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    load_document_template,
    run_template_compilation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.e2e


def test_profile_then_template_compilation_is_fully_traceable(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Dear Dr. Grace Hopper,\n"
            "Please reply by 2024-06-01.\n"
            "Sincerely, the team."
        ],
    )

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    profile_result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir
    )
    assert profile_result.is_success()

    profile_on_disk = load_document_profile(
        manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}")
    )

    template_result = run_template_compilation(
        pdf_path, "doc1", manifest, extraction_method=profile_on_disk["extraction_method"]
    )
    assert template_result.is_success()

    manifest_on_disk_path = manifest.run_dir / "run_manifest.json"
    manifest_data = json.loads(manifest_on_disk_path.read_text(encoding="utf-8"))
    stage_names = [s["stage_name"] for s in manifest_data["stages"]]
    assert stage_names == ["document_profiling", "template_compilation"]
    assert all(s["status"] == "success" for s in manifest_data["stages"])

    template_on_disk = load_document_template(
        manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}")
    )
    region_types = {r["region_type"] for page in template_on_disk["pages"] for r in page["regions"]}
    assert "field" in region_types

    serialized = str(template_on_disk)
    assert "8823471" not in serialized
    assert "401.50" not in serialized
    assert "Grace Hopper" not in serialized
