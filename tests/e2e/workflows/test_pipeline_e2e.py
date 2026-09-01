"""End-to-end: the entire currently-available pipeline succeeds.

Today that pipeline is Connect -> Discover -> Profile -> Infer -> Plan
(contract approval) -> Clean -> Train (and sample) -> Export; Checkpoint 6
(Validate) onward is not built yet. This test drives it exactly the way a
real caller would - load project.yaml (with a secret resolved from an
environment variable, never hard-coded), build the adapter through the
registry, run each stage in order, then re-read everything back from disk
(not from the in-memory objects used to produce it) to prove the run is
genuinely reconstructable from its evidence alone.

As later checkpoints land (validate) this test grows by one stage at a
time rather than being replaced, so it keeps meaning "the whole
pipeline", not just "the first stages".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.discovery.database.adapters.registry import create_source_adapter
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export, run_generation_from_artifact
from synth_platform.engine.profiling.database.cleaning.service import CLEANING_REPORT_FILENAME, load_cleaning_report, run_cleaning
from synth_platform.engine.common.database.config import load_project_config
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import DISCOVERY_FILENAME, load_discovery, run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import (
    CANDIDATES_FILENAME,
    load_candidates,
    run_contract_approval,
    run_inference,
)
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.e2e

ACCEPT_ALL = "accept_all"


def write_project_yaml(tmp_path: Path) -> Path:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        """
project:
  name: "e2e-project"
  dataset_id: "e2e-dataset"

source:
  type: sqlite
  connection:
    path: "${E2E_SQLITE_DB_PATH}"
  sampling:
    default_limit: 100
    max_limit: 1000

output:
  runs_dir: "runs"
  metadata_dir: "metadata"
""",
        encoding="utf-8",
    )
    return config_path


def _run_connect_through_train(config, adapter, tmp_path: Path, decisions=None, overrides=None):
    """decisions=ACCEPT_ALL accepts every candidate as-is. `overrides`, if
    given, is layered on top of that accept-all base so a test can approve
    everything except one deliberately-overridden column, without having
    to duplicate the whole accept-all computation itself.
    """
    runs_dir = tmp_path / config.output.runs_dir
    metadata_dir = tmp_path / config.output.metadata_dir
    manifest = RunManifest.create(runs_dir=runs_dir)

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    assert discovery_result.is_success()
    discovery_on_disk = load_discovery(manifest.output_path(DISCOVERY_FILENAME))

    profiling_result = run_profiling(
        adapter, discovery_on_disk, manifest,
        discovery_reference=str(manifest.output_path(DISCOVERY_FILENAME)),
        sample_limit=config.source.sampling.default_limit,
    )
    assert profiling_result.is_success()
    profile_on_disk = load_profile(manifest.output_path(PROFILE_FILENAME))

    inference_result = run_inference(
        adapter, discovery_on_disk, profile_on_disk, manifest,
        discovery_reference=str(manifest.output_path(DISCOVERY_FILENAME)),
        profile_reference=str(manifest.output_path(PROFILE_FILENAME)),
        sample_limit=config.source.sampling.default_limit,
    )
    assert inference_result.is_success()
    candidates_on_disk = load_candidates(manifest.output_path(CANDIDATES_FILENAME))

    if decisions == ACCEPT_ALL or overrides is not None:
        decisions = {
            table: {col: cand["semantic_type"] for col, cand in cols.items()}
            for table, cols in candidates_on_disk["tables"].items()
        }
        for table, column_overrides in (overrides or {}).items():
            decisions[table].update(column_overrides)

    approval_result = run_contract_approval(
        dataset_id=config.dataset_id,
        source_fingerprint=discovery_on_disk["source_fingerprint"],
        discovery_data=discovery_on_disk,
        candidates_by_table=candidates_on_disk["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference=str(manifest.output_path(CANDIDATES_FILENAME)),
        decisions=decisions,
    )
    assert approval_result.is_success()

    contract_on_disk = load_dataset_contract(metadata_dir)
    cleaning_result = run_cleaning(
        adapter, contract_on_disk, manifest,
        contract_reference=str(metadata_dir / "dataset_contract.json"),
        sample_limit=config.source.sampling.default_limit,
    )
    assert cleaning_result.is_success()
    cleaning_report_on_disk = load_cleaning_report(manifest.output_path(CLEANING_REPORT_FILENAME))

    training_result = run_training_and_sampling(
        adapter, contract_on_disk, manifest,
        contract_reference=str(metadata_dir / "dataset_contract.json"),
        sample_limit=config.source.sampling.default_limit,
        num_rows_to_generate=20,
        seed=5,
    )
    assert training_result.is_success()
    training_report_on_disk = load_training_report(manifest.output_path("training_report.json"))

    return (
        manifest,
        discovery_on_disk,
        profile_on_disk,
        candidates_on_disk,
        cleaning_report_on_disk,
        training_report_on_disk,
    )


def test_connect_through_train_is_fully_traceable(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("E2E_SQLITE_DB_PATH", str(temp_sqlite_db))
    config_path = write_project_yaml(tmp_path)
    config = load_project_config(config_path)
    adapter = create_source_adapter(config.source.type, config.source.connection)
    assert adapter.test_connection()["healthy"] is True

    (
        manifest,
        discovery_on_disk,
        profile_on_disk,
        candidates_on_disk,
        cleaning_report_on_disk,
        training_report_on_disk,
    ) = _run_connect_through_train(config, adapter, tmp_path, decisions=ACCEPT_ALL)

    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metadata_dir = tmp_path / config.output.metadata_dir
    contract = load_dataset_contract(metadata_dir)

    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names == ["discovery", "profiling", "inference", "contract_approval", "cleaning", "training"]
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    # Full trace: training -> contract -> candidates -> profile -> discovery -> source fingerprint.
    assert contract["source_fingerprint"] == discovery_on_disk["source_fingerprint"]
    assert set(contract["tables"]) == set(discovery_on_disk["tables"]) == set(adapter.list_tables())
    for table_name, table_schema in discovery_on_disk["tables"].items():
        assert set(profile_on_disk["tables"][table_name]["columns"]) == {
            c["name"] for c in table_schema["columns"]
        }
        assert set(candidates_on_disk["tables"][table_name]) == {c["name"] for c in table_schema["columns"]}
        assert set(contract["tables"][table_name]["columns"]) == {c["name"] for c in table_schema["columns"]}

    customer_id_entry = contract["tables"]["customers"]["columns"]["customer_id"]
    assert customer_id_entry["inference_status"] == "approved"
    assert customer_id_entry["semantic_type"] == "identifier"

    # The generated data is genuinely synthetic: unique ids, none matching real ones.
    customers_training = training_report_on_disk["tables"]["customers"]
    generated = pd.read_csv(customers_training["sample_path"])
    real_df = adapter.read_sample("customers", limit=100)
    assert len(generated) == 20
    assert generated["customer_id"].is_unique
    assert set(generated["customer_id"]) & set(real_df["customer_id"]) == set()


def test_override_survives_a_full_pipeline_rerun(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A human decision recorded in one run must still hold after a second,
    completely independent run through Connect->Train with no new decision.
    """
    monkeypatch.setenv("E2E_SQLITE_DB_PATH", str(temp_sqlite_db))
    config_path = write_project_yaml(tmp_path)
    config = load_project_config(config_path)
    adapter = create_source_adapter(config.source.type, config.source.connection)

    # First run: accept every other candidate, but override customer_id to
    # "category" (disagrees with the engine's own confident "identifier"
    # proposal) - accepting the rest is what gives training enough
    # approved columns to run at all.
    _run_connect_through_train(
        config, adapter, tmp_path, overrides={"customers": {"customer_id": "category"}}
    )

    # Second run: a brand-new RunManifest/run_id, no decisions supplied at all.
    _run_connect_through_train(config, adapter, tmp_path, decisions=None)

    metadata_dir = tmp_path / config.output.metadata_dir
    contract = load_dataset_contract(metadata_dir)
    assert contract["revision"] == 2
    customer_id_entry = contract["tables"]["customers"]["columns"]["customer_id"]
    assert customer_id_entry["semantic_type"] == "category"
    assert customer_id_entry["inference_status"] == "approved"


def test_generation_succeeds_with_the_source_database_deleted(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The actual portability proof the master spec asks for: train,
    package, delete the source database entirely, then load the artifact
    and generate in what is effectively a source-free context - nothing
    after the export step touches temp_sqlite_db or any SourceAdapter.
    """
    monkeypatch.setenv("E2E_SQLITE_DB_PATH", str(temp_sqlite_db))
    config_path = write_project_yaml(tmp_path)
    config = load_project_config(config_path)
    adapter = create_source_adapter(config.source.type, config.source.connection)

    (
        manifest,
        discovery_on_disk,
        profile_on_disk,
        candidates_on_disk,
        cleaning_report_on_disk,
        training_report_on_disk,
    ) = _run_connect_through_train(config, adapter, tmp_path, decisions=ACCEPT_ALL)

    metadata_dir = tmp_path / config.output.metadata_dir
    contract_on_disk = load_dataset_contract(metadata_dir)

    export_result = run_artifact_export(
        dataset_id=config.dataset_id,
        artifact_version="1.0.0",
        discovery_data=discovery_on_disk,
        dataset_contract=contract_on_disk,
        reference_profile=profile_on_disk,
        training_report=training_report_on_disk,
        training_run_id=manifest.run_id,
        code_version=manifest.code_version,
        manifest=manifest,
        training_reference=str(manifest.output_path("training_report.json")),
    )
    assert export_result.is_success()
    assert export_result.metrics["self_test_passed"] is True
    artifact_path = export_result.output_references[0]

    # Remove the actual source database file from disk.
    os.remove(temp_sqlite_db)
    assert not temp_sqlite_db.exists()

    # A completely independent run, with no adapter and no reference to
    # the source database anywhere in this code path.
    generation_manifest = RunManifest.create(runs_dir=tmp_path / "post_delete_runs")
    loaded = load_artifact(artifact_path)
    generation_result = run_generation_from_artifact(
        artifact_path, "customers", num_rows=30, seed=11, manifest=generation_manifest,
        allow_cloudpickle_models=True,
    )

    assert generation_result.is_success()
    generated = pd.read_csv(generation_result.output_references[0])
    assert len(generated) == 30
    assert generated["customer_id"].is_unique
    assert loaded.manifest["source_fingerprint"] == discovery_on_disk["source_fingerprint"]
