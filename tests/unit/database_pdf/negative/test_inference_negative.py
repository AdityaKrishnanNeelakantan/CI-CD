from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import (
    DatasetContractLoadError,
    build_or_update_dataset_contract,
    save_dataset_contract,
)
from synth_platform.engine.inference.database.service import CandidatesLoadError, load_candidates, run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling

pytestmark = pytest.mark.negative


DISCOVERY = {
    "tables": {
        "customers": {
            "primary_key": ["customer_id"],
            "columns": [{"name": "customer_id", "database_type": "TEXT", "nullable": False}],
        }
    }
}


def _candidates():
    return {
        "customers": {
            "customer_id": {
                "semantic_type": "identifier",
                "status": "proposed",
                "confidence": 0.9,
                "evidence": [],
                "alternatives": [],
            }
        }
    }


def test_save_refuses_to_overwrite_existing_revision_snapshot(tmp_path: Path):
    contract = build_or_update_dataset_contract("ds", "sha256:abc", DISCOVERY, _candidates())
    save_dataset_contract(contract, tmp_path)
    with pytest.raises(RuntimeError):
        save_dataset_contract(contract, tmp_path)


def test_load_dataset_contract_rejects_corrupted_json(tmp_path: Path):
    from synth_platform.engine.inference.database.contract import load_dataset_contract

    (tmp_path / "dataset_contract.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DatasetContractLoadError):
        load_dataset_contract(tmp_path)


def test_load_dataset_contract_rejects_missing_top_level_keys(tmp_path: Path):
    from synth_platform.engine.inference.database.contract import load_dataset_contract

    (tmp_path / "dataset_contract.json").write_text(
        json.dumps({"contract_version": "1.0"}), encoding="utf-8"
    )
    with pytest.raises(DatasetContractLoadError):
        load_dataset_contract(tmp_path)


def test_load_dataset_contract_rejects_column_missing_keys(tmp_path: Path):
    from synth_platform.engine.inference.database.contract import load_dataset_contract

    (tmp_path / "dataset_contract.json").write_text(
        json.dumps(
            {
                "contract_version": "1.0",
                "revision": 1,
                "dataset_id": "d",
                "source_fingerprint": "sha256:" + "a" * 64,
                "updated_at": "2026-01-01T00:00:00Z",
                "tables": {
                    "customers": {
                        "primary_key": ["customer_id"],
                        "business_rules": [],
                        "columns": {"customer_id": {"physical_type": "TEXT"}},  # missing rest
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetContractLoadError):
        load_dataset_contract(tmp_path)


def test_load_candidates_rejects_missing_file(tmp_path: Path):
    with pytest.raises(CandidatesLoadError):
        load_candidates(tmp_path / "does_not_exist.json")


def test_run_inference_fails_closed_on_write_error(temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A disk-full/permission-denied failure while writing
    semantic_candidates.json must surface as a clean failed StageResult, not
    crash run_inference() with a raw OSError.
    """
    import synth_platform.engine.inference.database.service as inference_service_module

    real_dump = inference_service_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "semantic_candidates.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(
        adapter, discovery_data, manifest, str(discovery_result.output_references[0]), sample_limit=100
    )
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())

    monkeypatch.setattr(inference_service_module.json, "dump", _raising_dump)

    result = run_inference(
        adapter,
        discovery_data,
        profile_data,
        manifest,
        discovery_reference=str(discovery_result.output_references[0]),
        profile_reference=str(profiling_result.output_references[0]),
        sample_limit=100,
    )

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("semantic_candidates.json").exists()


def test_run_contract_approval_fails_closed_on_write_error(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A disk-full/permission-denied failure while writing
    dataset_contract.json (via save_dataset_contract) must surface as a
    clean failed StageResult, not crash run_contract_approval() with a raw
    OSError.
    """
    import synth_platform.engine.inference.database.contract as contract_module

    real_dump = contract_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "dataset_contract.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

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

    monkeypatch.setattr(contract_module.json, "dump", _raising_dump)

    result = run_contract_approval(
        dataset_id="test-dataset",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference=str(inference_result.output_references[0]),
        decisions={"customers": {"customer_id": "identifier"}},
    )

    assert result.status == "failed"
    assert result.errors
    assert not (metadata_dir / "dataset_contract.json").exists()


def test_load_candidates_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "semantic_candidates.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CandidatesLoadError):
        load_candidates(path)


def test_load_candidates_rejects_candidate_missing_keys(tmp_path: Path):
    path = tmp_path / "semantic_candidates.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "tables": {"customers": {"customer_id": {"semantic_type": "identifier"}}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CandidatesLoadError):
        load_candidates(path)


def test_inference_never_overwrites_existing_output(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_data = json.loads(
        Path(run_discovery(adapter, manifest, config_path="config/project.yaml").output_references[0]).read_text()
    )
    profile_data = json.loads(
        Path(
            run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100).output_references[0]
        ).read_text()
    )

    run_inference(adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100)
    with pytest.raises(RuntimeError):
        run_inference(
            adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
        )


def test_inference_fails_closed_on_table_missing_from_source(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_data = json.loads(
        Path(run_discovery(adapter, manifest, config_path="config/project.yaml").output_references[0]).read_text()
    )
    profile_data = json.loads(
        Path(
            run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100).output_references[0]
        ).read_text()
    )
    discovery_data["tables"]["__ghost_table__"] = discovery_data["tables"]["orders"]

    result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("semantic_candidates.json").exists()
