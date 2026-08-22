"""Integration: run_document_profiling() -> run_template_compilation() ->
run_semantic_binding() -> run_value_generation(), the full four-stage PDF
track through Generate. Proves the generation stage never needs the
source file again once it has the template + binding map, and that
generated values never reproduce the source's real content.
"""

from __future__ import annotations

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
from synth_platform.engine.documents.pdf.service import run_document_profiling
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    load_document_template,
    run_template_compilation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.integration


def test_full_chain_generates_values_that_never_equal_the_source(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Invoice Number: INV-2024-0091"
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    profile_result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata"
    )
    assert profile_result.is_success()

    template_result = run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")
    assert template_result.is_success()
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))

    binding_result = run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    assert binding_result.is_success()
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    generation_result = run_value_generation(
        template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=42
    )
    assert generation_result.is_success()
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    fields_by_role = {f["semantic_role"]: f for f in values["fields"].values()}
    assert fields_by_role["account_number"]["value"] != "8823471"
    assert fields_by_role["person_name"]["value"] != "Grace Hopper"
    assert fields_by_role["reference_code"]["value"] != "INV-2024-0091"
    assert fields_by_role["currency_amount"]["decimal"] != "401.50"


def test_same_seed_reproduces_the_full_chain_output(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Account Number: 8823471\nTotal Due: 401.50"])

    def run_once(run_suffix: str):
        manifest = RunManifest.create(runs_dir=tmp_path / f"runs_{run_suffix}")
        run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / f"metadata_{run_suffix}")
        run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")
        template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))
        run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
        binding_map = load_document_binding_map(
            manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
        )
        run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=99)
        values = load_document_synthetic_values(
            manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
        )
        values.pop("generated_at")  # the only field allowed to differ between runs
        return values

    assert run_once("a") == run_once("b")
