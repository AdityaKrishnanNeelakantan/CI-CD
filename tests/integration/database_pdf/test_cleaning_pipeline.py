"""Integration: discovery -> profiling -> inference -> contract approval -> cleaning.

Extends the chain by one more link: cleaning consumes the approved
dataset_contract.json (not the raw candidates) and only cleans columns
that are actually marked "approved".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.profiling.database.cleaning.service import CLEANING_REPORT_FILENAME, load_cleaning_report, run_cleaning
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling

pytestmark = pytest.mark.integration


def _run_through_approval(adapter, manifest, metadata_dir, accept_all: bool):
    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())

    profiling_result = run_profiling(
        adapter, discovery_data, manifest, str(discovery_result.output_references[0]), sample_limit=100
    )
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())

    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest,
        str(discovery_result.output_references[0]), str(profiling_result.output_references[0]),
        sample_limit=100,
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())

    decisions = None
    if accept_all:
        decisions = {
            table: {col: cand["semantic_type"] for col, cand in cols.items()}
            for table, cols in candidates_data["tables"].items()
        }

    run_contract_approval(
        dataset_id="ds",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference=str(inference_result.output_references[0]),
        decisions=decisions,
    )
    return load_dataset_contract(metadata_dir)


def test_cleaning_only_touches_approved_columns(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    contract = _run_through_approval(adapter, manifest, metadata_dir, accept_all=True)

    cleaning_result = run_cleaning(
        adapter, contract, manifest, contract_reference="dataset_contract.json", sample_limit=100
    )
    assert cleaning_result.is_success()

    report = load_cleaning_report(manifest.output_path(CLEANING_REPORT_FILENAME))
    for table_name, table_contract in contract["tables"].items():
        approved_columns = {
            name for name, col in table_contract["columns"].items() if col["inference_status"] == "approved"
        }
        assert set(report["tables"][table_name]["columns"]) == approved_columns

    manifest_data = json.loads((manifest.run_dir / "run_manifest.json").read_text())
    stage_names = [s["stage_name"] for s in manifest_data["stages"]]
    assert stage_names == ["discovery", "profiling", "inference", "contract_approval", "cleaning"]


def test_cleaning_excludes_unapproved_columns_and_warns(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    # No decisions supplied - most columns stay proposed/review_required,
    # not approved, since only accepting/overriding marks them approved.
    contract = _run_through_approval(adapter, manifest, metadata_dir, accept_all=False)

    cleaning_result = run_cleaning(
        adapter, contract, manifest, contract_reference="dataset_contract.json", sample_limit=100
    )
    assert cleaning_result.is_success()

    report = load_cleaning_report(manifest.output_path(CLEANING_REPORT_FILENAME))
    for table_name in contract["tables"]:
        table_report = report["tables"][table_name]
        assert len(table_report["columns"]) == 0
        assert len(table_report["excluded_columns"]) > 0
    assert any("zero approved columns" in w for w in cleaning_result.warnings)


def test_cleaning_with_chunk_size_produces_equivalent_row_and_duplicate_counts(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest_whole = RunManifest.create(runs_dir=tmp_path / "runs_whole")
    manifest_chunked = RunManifest.create(runs_dir=tmp_path / "runs_chunked")
    metadata_dir = tmp_path / "metadata"

    contract = _run_through_approval(adapter, manifest_whole, metadata_dir, accept_all=True)

    whole_result = run_cleaning(
        adapter, contract, manifest_whole, contract_reference="dataset_contract.json", sample_limit=100
    )
    chunked_result = run_cleaning(
        adapter,
        contract,
        manifest_chunked,
        contract_reference="dataset_contract.json",
        sample_limit=100,
        chunk_size=2,
    )
    assert whole_result.is_success()
    assert chunked_result.is_success()

    whole_report = load_cleaning_report(manifest_whole.output_path(CLEANING_REPORT_FILENAME))
    chunked_report = load_cleaning_report(manifest_chunked.output_path(CLEANING_REPORT_FILENAME))

    for table_name in contract["tables"]:
        whole_table = whole_report["tables"][table_name]
        chunked_table = chunked_report["tables"][table_name]
        assert chunked_table["row_count"] == whole_table["row_count"]
        assert chunked_table["duplicate_row_count"] == whole_table["duplicate_row_count"]
        assert "chunked_cleaning_mode" in chunked_table["warnings"]
