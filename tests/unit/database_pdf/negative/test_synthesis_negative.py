from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import (
    TrainingReportLoadError,
    load_training_report,
    run_training_and_sampling,
)

pytestmark = pytest.mark.negative


def _approved_contract(temp_sqlite_db: Path, tmp_path: Path, accept_all: bool = True):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
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
        candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    return adapter, manifest, load_dataset_contract(metadata_dir)


def test_training_never_overwrites_existing_output(temp_sqlite_db: Path, tmp_path: Path):
    adapter, manifest, contract = _approved_contract(temp_sqlite_db, tmp_path)
    run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )
    with pytest.raises(RuntimeError):
        run_training_and_sampling(
            adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
        )


def test_training_fails_closed_on_table_missing_from_source(temp_sqlite_db: Path, tmp_path: Path):
    adapter, manifest, contract = _approved_contract(temp_sqlite_db, tmp_path)
    contract["tables"]["__ghost_table__"] = contract["tables"]["orders"]

    result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("training_report.json").exists()


def test_training_fails_closed_when_no_columns_are_approved(temp_sqlite_db: Path, tmp_path: Path):
    # No decisions supplied - nothing gets marked "approved", so there is
    # nothing this adapter can train on.
    adapter, manifest, contract = _approved_contract(temp_sqlite_db, tmp_path, accept_all=False)

    result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("training_report.json").exists()


def test_load_training_report_rejects_missing_file(tmp_path: Path):
    with pytest.raises(TrainingReportLoadError):
        load_training_report(tmp_path / "does_not_exist.json")


def test_load_training_report_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "training_report.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TrainingReportLoadError):
        load_training_report(path)


def test_load_training_report_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "training_report.json"
    path.write_text(json.dumps({"tables": {}}), encoding="utf-8")  # missing "trained_at"
    with pytest.raises(TrainingReportLoadError):
        load_training_report(path)


def test_load_training_report_rejects_table_missing_required_keys(tmp_path: Path):
    path = tmp_path / "training_report.json"
    path.write_text(
        json.dumps({"trained_at": "2026-01-01T00:00:00Z", "tables": {"customers": {"model_type": "x"}}}),
        encoding="utf-8",
    )
    with pytest.raises(TrainingReportLoadError):
        load_training_report(path)


def test_run_training_fails_closed_on_write_error(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A disk-full/permission-denied failure while writing training_report.json
    must surface as a clean failed StageResult, not crash run_training_and_sampling()
    with a raw OSError.
    """
    import synth_platform.engine.training.database.service as synthesis_module

    real_dump = synthesis_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "training_report.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    adapter, manifest, contract = _approved_contract(temp_sqlite_db, tmp_path)
    monkeypatch.setattr(synthesis_module.json, "dump", _raising_dump)

    result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("training_report.json").exists()
