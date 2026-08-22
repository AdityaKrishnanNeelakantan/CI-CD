"""Integration: the same full chain as tests/integration/test_artifact_pipeline.py
(discovery -> profiling -> inference -> approval -> training -> artifact
export -> load -> generate), but with model_type="safe_gaussian_copula" -
proving the pickle-free adapter works through the real pipeline, not just
in isolation, and that the resulting archive genuinely contains no pickle.
"""

from __future__ import annotations

import json
import pickletools
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export, run_generation_from_artifact
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import run_training_and_sampling

pytestmark = pytest.mark.integration


def _train_and_export(temp_sqlite_db: Path, tmp_path: Path):
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
    contract = load_dataset_contract(metadata_dir)

    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
        num_rows_to_generate=10, seed=1, model_type="safe_gaussian_copula",
    )
    training_report = json.loads(Path(training_result.output_references[0]).read_text())

    export_result = run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )
    return manifest, discovery_data, contract, training_report, export_result


def test_safe_adapter_trains_and_exports_through_the_real_pipeline(temp_sqlite_db: Path, tmp_path: Path):
    manifest, discovery_data, contract, training_report, export_result = _train_and_export(
        temp_sqlite_db, tmp_path
    )
    assert export_result.is_success()
    assert export_result.metrics["self_test_passed"] is True

    loaded = load_artifact(export_result.output_references[0])
    for table_name, table_entry in loaded.manifest["tables"].items():
        assert table_entry["model_type"] == "safe_gaussian_copula"
        assert table_entry["model_serialization_format"] == "json"
        assert table_entry["model_path"].endswith(".json")


def test_generation_from_the_json_artifact_produces_rows(temp_sqlite_db: Path, tmp_path: Path):
    manifest, discovery_data, contract, training_report, export_result = _train_and_export(
        temp_sqlite_db, tmp_path
    )
    artifact_path = export_result.output_references[0]

    generation_manifest = RunManifest.create(runs_dir=tmp_path / "gen_runs")
    generation_result = run_generation_from_artifact(
        artifact_path, "customers", 15, seed=2, manifest=generation_manifest
    )
    assert generation_result.is_success()

    generated = pd.read_csv(generation_result.output_references[0])
    assert len(generated) == 15


def test_archive_model_file_is_plain_json_with_no_pickle_opcodes(temp_sqlite_db: Path, tmp_path: Path):
    """The real point of this whole adapter: scan the actual model file
    inside a real, fully-built archive and prove it parses as plain JSON
    and does not parse as a pickle stream at all.
    """
    _manifest, _discovery, _contract, _training_report, export_result = _train_and_export(
        temp_sqlite_db, tmp_path
    )
    artifact_path = Path(export_result.output_references[0])

    with zipfile.ZipFile(artifact_path) as zf:
        model_entries = [n for n in zf.namelist() if n.startswith("models/tables/") and n.endswith("model.json")]
        assert model_entries, "expected at least one model.json in the archive"
        for entry in model_entries:
            raw_bytes = zf.read(entry)
            json.loads(raw_bytes.decode("utf-8"))  # must not raise
            with pytest.raises(Exception):
                list(pickletools.genops(raw_bytes))
