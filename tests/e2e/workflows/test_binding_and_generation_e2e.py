"""End-to-end: document_profiling -> template_compilation -> semantic_binding
-> value_generation, fully traceable and re-read entirely from disk
afterward - the complete currently-available PDF track through Generate,
same proof pattern as the other tests/e2e/test_*_e2e.py files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.binding_service import (
    DOCUMENT_BINDING_MAP_FILENAME,
    load_document_binding_map,
    run_semantic_binding,
)
from synth_platform.engine.documents.pdf.generation_service import (
    DOCUMENT_SYNTHETIC_VALUES_FILENAME,
    load_document_synthetic_values,
    run_value_generation,
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


def test_full_pdf_track_through_generation_is_fully_traceable(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Reference Code: RC-88291\n"
            "Issued Date: 2024-06-01"
        ],
    )

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir)
    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))

    run_template_compilation(pdf_path, "doc1", manifest, extraction_method=profile["extraction_method"])
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))

    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=7)

    # Simulate a fresh process: re-read every artifact from disk only.
    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names == [
        "document_profiling",
        "template_compilation",
        "semantic_binding",
        "value_generation",
    ]
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    source_secrets = ["8823471", "401.50", "Grace Hopper", "RC-88291", "2024-06-01"]
    serialized_template = json.dumps(template)
    serialized_values = json.dumps(values)
    for secret in source_secrets:
        assert secret not in serialized_template, f"{secret!r} leaked into document_template.json"
        assert secret not in serialized_values, f"{secret!r} leaked into document_synthetic_values.json"

    fields_by_role = {f["semantic_role"]: f for f in values["fields"].values()}
    assert fields_by_role["account_number"]["value"]
    assert fields_by_role["person_name"]["value"]
    assert fields_by_role["reference_code"]["value"]
    assert fields_by_role["currency_amount"]["decimal"]
    assert fields_by_role["date"]["value"]
