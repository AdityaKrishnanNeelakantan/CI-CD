"""Integration: SQLite adapter -> discovery -> profiling -> inference -> contract approval.

Extends the chain by two links: semantic_candidates.json feeds off both
discovery.json and profile.json, and contract approval consumes the
candidates plus discovery to produce metadata/dataset_contract.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import CANDIDATES_FILENAME, load_candidates, run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling

pytestmark = pytest.mark.integration


def test_inference_consumes_discovery_and_profile_and_writes_candidates(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())

    profiling_result = run_profiling(
        adapter, discovery_data, manifest, str(discovery_result.output_references[0]), sample_limit=100
    )
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())

    inference_result = run_inference(
        adapter,
        discovery_data,
        profile_data,
        manifest,
        discovery_reference=str(discovery_result.output_references[0]),
        profile_reference=str(profiling_result.output_references[0]),
        sample_limit=100,
    )

    assert inference_result.is_success()
    candidates_path = manifest.output_path(CANDIDATES_FILENAME)
    candidates_data = load_candidates(candidates_path)

    # Every discovered column must get a candidate - no silent skipping.
    for table_name, table_schema in discovery_data["tables"].items():
        discovered_columns = {c["name"] for c in table_schema["columns"]}
        assert set(candidates_data["tables"][table_name]) == discovered_columns

    # customer_id: 100% unique text column named "*_id" -> identifier.
    customer_id_candidate = candidates_data["tables"]["customers"]["customer_id"]
    assert customer_id_candidate["semantic_type"] == "identifier"


def test_contract_approval_consumes_candidates_and_writes_dataset_contract(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(
        adapter, discovery_data, manifest, str(discovery_result.output_references[0]), sample_limit=100
    )
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())
    inference_result = run_inference(
        adapter,
        discovery_data,
        profile_data,
        manifest,
        str(discovery_result.output_references[0]),
        str(profiling_result.output_references[0]),
        sample_limit=100,
    )
    candidates_data = load_candidates(inference_result.output_references[0])

    approval_result = run_contract_approval(
        dataset_id="test-dataset",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference=str(inference_result.output_references[0]),
        decisions={"customers": {"customer_id": "identifier"}},
    )

    assert approval_result.is_success()
    contract = load_dataset_contract(metadata_dir)
    assert contract is not None
    assert contract["source_fingerprint"] == discovery_data["source_fingerprint"]

    customer_id_entry = contract["tables"]["customers"]["columns"]["customer_id"]
    assert customer_id_entry["inference_status"] == "approved"
    assert customer_id_entry["semantic_type"] == "identifier"
    assert customer_id_entry["sensitive"] is True

    manifest_data = json.loads((manifest.run_dir / "run_manifest.json").read_text())
    stage_names = [s["stage_name"] for s in manifest_data["stages"]]
    assert stage_names == ["discovery", "profiling", "inference", "contract_approval"]


def test_inference_with_chunk_size_produces_equivalent_candidates(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest_whole = RunManifest.create(runs_dir=tmp_path / "runs_whole")
    manifest_chunked = RunManifest.create(runs_dir=tmp_path / "runs_chunked")

    discovery_result = run_discovery(adapter, manifest_whole, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(
        adapter, discovery_data, manifest_whole, str(discovery_result.output_references[0]), sample_limit=100
    )
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())

    whole_result = run_inference(
        adapter,
        discovery_data,
        profile_data,
        manifest_whole,
        str(discovery_result.output_references[0]),
        str(profiling_result.output_references[0]),
        sample_limit=100,
    )
    chunked_result = run_inference(
        adapter,
        discovery_data,
        profile_data,
        manifest_chunked,
        str(discovery_result.output_references[0]),
        str(profiling_result.output_references[0]),
        sample_limit=100,
        chunk_size=2,
    )
    assert whole_result.is_success()
    assert chunked_result.is_success()

    whole_candidates = load_candidates(whole_result.output_references[0])
    chunked_candidates = load_candidates(chunked_result.output_references[0])

    for table_name, columns in whole_candidates["tables"].items():
        for column_name, candidate in columns.items():
            chunked_candidate = chunked_candidates["tables"][table_name][column_name]
            assert chunked_candidate["semantic_type"] == candidate["semantic_type"]
            assert chunked_candidate["status"] == candidate["status"]
