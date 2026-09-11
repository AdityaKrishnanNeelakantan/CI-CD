from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from synth_platform.application.workflows.database_twin import (
    DatabaseTwinPipelineConfig,
    run_database_twin_pipeline,
)
from synth_platform.application.workflows.interaction_twin import run_interaction_twin
from synth_platform.application.workflows.pdf_twin import run_value_generation
from synth_platform.application.workflows.schema_twin import generate_from_schema
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table
from synth_platform.infrastructure.persistence.platform_db import PlatformDB

pytestmark = pytest.mark.integration


def _settings_db(tmp_path: Path, **values) -> PlatformDB:
    db = PlatformDB(tmp_path / f"{len(list(tmp_path.glob('*.db')))}_platform.db")
    db.write_settings(
        {
            "generation_mode": "schema_driven",
            "default_record_count": 7,
            "privacy_level": "strict",
            "default_output_format": "csv",
            **values,
        }
    )
    return db


def _schema() -> SchemaConfig:
    return SchemaConfig(
        name="settings-proof",
        seed=7,
        tables=[Table(name="orders", row_count=1)],
        columns={
            "orders": [
                Column(name="order_id", type="int", unique=True, min=1, max=100),
                Column(name="amount", type="float", min=1, max=50),
            ]
        },
    )


def test_schema_twin_uses_persisted_record_count_and_export_format_until_overridden(tmp_path: Path) -> None:
    db = _settings_db(tmp_path, default_record_count=5, default_output_format="parquet")

    defaulted = generate_from_schema(
        _schema(),
        seed=7,
        output_dir=tmp_path / "schema_default",
        product_settings=db,
    )

    assert defaulted.row_counts == {"orders": 5}
    assert {path.suffix for path in defaulted.export_paths.values()} == {".parquet"}

    overridden = generate_from_schema(
        _schema(),
        row_count=3,
        seed=7,
        output_dir=tmp_path / "schema_override",
        export_format="csv",
        product_settings=db,
    )

    assert overridden.row_counts == {"orders": 3}
    assert {path.suffix for path in overridden.export_paths.values()} == {".csv"}


def test_database_twin_uses_persisted_record_count_until_per_table_override(
    temp_sqlite_db: Path, tmp_path: Path
) -> None:
    db = _settings_db(tmp_path, default_record_count=6)
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})

    manifest = RunManifest.create(runs_dir=tmp_path / "default_runs")
    default_config = DatabaseTwinPipelineConfig(
        dataset_id="settings-db-default",
        artifact_version="1.0.0",
        config_path="config/project.yaml",
        metadata_dir=tmp_path / "default_metadata",
        target_db_path=tmp_path / "default_database_b.db",
        row_counts_by_table=None,
        sample_limit=20,
        num_rows_to_generate=6,
        product_settings=db,
    )
    default_context = run_database_twin_pipeline(adapter, manifest, default_config)

    assert default_context.artifacts["qa_report"]["hard_checks_passed"] is True
    assert {
        table: entry["row_count"]
        for table, entry in default_context.artifacts["relational_report"]["tables"].items()
    } == {"customers": 6, "orders": 6}

    manifest = RunManifest.create(runs_dir=tmp_path / "override_runs")
    override_config = DatabaseTwinPipelineConfig(
        dataset_id="settings-db-override",
        artifact_version="1.0.0",
        config_path="config/project.yaml",
        metadata_dir=tmp_path / "override_metadata",
        target_db_path=tmp_path / "override_database_b.db",
        row_counts_by_table={"customers": 3, "orders": 4},
        sample_limit=20,
        num_rows_to_generate=6,
        product_settings=db,
    )
    override_context = run_database_twin_pipeline(adapter, manifest, override_config)

    assert override_context.artifacts["qa_report"]["hard_checks_passed"] is True
    assert {
        table: entry["row_count"]
        for table, entry in override_context.artifacts["relational_report"]["tables"].items()
    } == {"customers": 3, "orders": 4}


def test_pdf_twin_has_no_persisted_setting_mapping_for_value_generation() -> None:
    signature = inspect.signature(run_value_generation)

    assert "product_settings" not in signature.parameters
    assert "output_format" not in signature.parameters
    assert "privacy_level" not in signature.parameters


def test_interaction_twin_uses_privacy_default_until_per_run_override(tmp_path: Path) -> None:
    db = _settings_db(tmp_path, privacy_level="strict")
    source = "agent: Hello.\ncustomer: My name is Ada Lovelace and my card is 4199 4099 9799 8199."

    defaulted = run_interaction_twin(source, product_settings=db)

    assert defaulted.redaction_report["enabled"] is True
    assert defaulted.redaction_report["finding_count"] >= 1

    overridden = run_interaction_twin(
        source,
        remove_sensitive_information=False,
        product_settings=db,
    )

    assert overridden.redaction_report["enabled"] is False
    assert overridden.validation_report["hard_checks_passed"] is True
