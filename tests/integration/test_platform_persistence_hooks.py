from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.application.workflows.database_twin import (
    DatabaseTwinPipelineConfig,
    run_database_twin_pipeline,
)
from synth_platform.application.workflows.pdf_twin import (
    PDFDocumentAdapter,
    load_document_binding_map,
    load_document_ground_truth,
    load_document_synthetic_values,
    load_document_template,
    run_document_profiling,
    run_document_rendering,
    run_document_validation,
    run_semantic_binding,
    run_template_compilation,
    run_value_generation,
)
from synth_platform.application.workflows.schema_twin import generate_from_schema
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table
from synth_platform.infrastructure.persistence.platform_db import PlatformDB
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.integration


def test_schema_workflow_records_platform_run(tmp_path: Path) -> None:
    history = PlatformDB(tmp_path / "platform.db")
    schema = SchemaConfig(
        name="orders",
        seed=7,
        tables=[Table(name="orders", row_count=5)],
        columns={
            "orders": [
                Column(name="order_id", type="int", unique=True, min=1, max=100),
                Column(name="amount", type="float", min=1, max=50),
            ]
        },
    )

    result = generate_from_schema(
        schema,
        row_count=5,
        seed=7,
        output_dir=tmp_path / "schema_output",
        history=history,
    )

    assert result.hard_checks_passed is True
    [run] = history.list_runs(workflow_type="schema")
    assert run.validation_passed is True
    assert run.metadata["row_counts"] == {"orders": 5}


def test_database_workflow_records_platform_run(temp_sqlite_db: Path, tmp_path: Path) -> None:
    history = PlatformDB(tmp_path / "platform.db")
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    config = DatabaseTwinPipelineConfig(
        dataset_id="ds-platform-history",
        artifact_version="1.0.0",
        config_path="config/project.yaml",
        metadata_dir=tmp_path / "metadata",
        target_db_path=tmp_path / "database_b.db",
        row_counts_by_table={"customers": 3, "orders": 4},
        sample_limit=20,
        num_rows_to_generate=4,
    )

    context = run_database_twin_pipeline(adapter, manifest, config, history=history)

    assert context.artifacts["qa_report"]["hard_checks_passed"] is True
    [run] = history.list_runs(workflow_type="database")
    assert run.id == manifest.run_id
    assert run.validation_passed is True
    assert run.output_id == "database_b.db"


def test_pdf_workflow_records_platform_run(tmp_path: Path) -> None:
    history = PlatformDB(tmp_path / "platform.db")
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Reference Code: RC-88291"
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    doc_id = "doc1"

    run_document_profiling(PDFDocumentAdapter(), pdf_path, doc_id, manifest, tmp_path / "metadata")
    run_template_compilation(pdf_path, doc_id, manifest, extraction_method="native")
    template = load_document_template(manifest.output_path(f"documents/{doc_id}/document_template.json"))
    run_semantic_binding(template, doc_id, manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/{doc_id}/document_binding_map.json")
    )
    run_value_generation(template, binding_map, doc_id, manifest, binding_map_reference=str(pdf_path), seed=11)
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/{doc_id}/document_synthetic_values.json")
    )
    render_result = run_document_rendering(
        template, binding_map, values, doc_id, manifest, synthetic_values_reference=str(pdf_path)
    )
    assert render_result.is_success()
    ground_truth = load_document_ground_truth(
        manifest.output_path(f"documents/{doc_id}/document_ground_truth.json")
    )
    rendered_pdf = manifest.output_path(f"documents/{doc_id}/rendered.pdf")

    validation_result = run_document_validation(
        rendered_pdf,
        ground_truth,
        doc_id,
        manifest,
        ground_truth_reference=str(pdf_path),
        history=history,
    )

    assert validation_result.is_success()
    [run] = history.list_runs(workflow_type="pdf")
    assert run.id == f"{manifest.run_id}:{doc_id}"
    assert run.validation_passed is True
    assert run.output_id == "synthetic_twin.pdf"

