from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.training.database.artifact.builder import build_artifact
from synth_platform.engine.training.database.artifact.errors import (
    ArtifactBuildError,
    ArtifactLoadError,
    ChecksumMismatchError,
    IncompatibleArtifactError,
    PathTraversalError,
    UntrustedModelFormatError,
)
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export, run_generation_from_artifact
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import run_training_and_sampling

pytestmark = pytest.mark.negative


def _minimal_inputs(tmp_path: Path):
    model_path = tmp_path / "trained_model.pkl"
    model_path.write_bytes(b"fake-model-bytes")
    discovery_data = {"source_fingerprint": "sha256:" + "a" * 64, "tables": {}}
    dataset_contract = {"revision": 1, "tables": {}}
    reference_profile = {"tables": {}}
    training_report = {
        "tables": {
            "t": {
                "model_type": "sdv_gaussian_copula",
                "model_path": str(model_path),
                "data_fingerprint": "sha256:" + "b" * 64,
                "serialization_format": "cloudpickle",
                "fit_evidence": {"trained_columns": ["a"], "excluded_columns": [], "primary_key": "a"},
            }
        }
    }
    return discovery_data, dataset_contract, reference_profile, training_report


def _build(tmp_path: Path, name: str = "out.zip") -> Path:
    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)
    return build_artifact(
        "ds", "1.0.0", discovery_data, contract, profile, training_report,
        "run1", "abc123", tmp_path / name,
    )


def test_build_refuses_to_overwrite_existing_output(tmp_path: Path):
    zip_path = _build(tmp_path)
    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)
    with pytest.raises(ArtifactBuildError):
        build_artifact(
            "ds", "1.0.0", discovery_data, contract, profile, training_report,
            "run1", "abc123", zip_path,
        )


def test_build_fails_when_trained_model_file_missing(tmp_path: Path):
    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)
    training_report["tables"]["t"]["model_path"] = str(tmp_path / "does_not_exist.pkl")
    with pytest.raises(ArtifactBuildError):
        build_artifact(
            "ds", "1.0.0", discovery_data, contract, profile, training_report,
            "run1", "abc123", tmp_path / "out.zip",
        )


def test_build_wraps_model_copy_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil

    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", _raise)
    with pytest.raises(ArtifactBuildError):
        build_artifact(
            "ds", "1.0.0", discovery_data, contract, profile, training_report,
            "run1", "abc123", tmp_path / "out.zip",
        )


def test_build_wraps_sidecar_copy_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil

    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)
    sidecar_path = tmp_path / "trained_model.pkl.meta.json"
    sidecar_path.write_text("{}")

    real_copy2 = shutil.copy2

    def _raise_for_sidecar(src, *args, **kwargs):
        if str(src).endswith(".meta.json"):
            raise OSError("permission denied")
        return real_copy2(src, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", _raise_for_sidecar)
    with pytest.raises(ArtifactBuildError):
        build_artifact(
            "ds", "1.0.0", discovery_data, contract, profile, training_report,
            "run1", "abc123", tmp_path / "out.zip",
        )


def test_load_rejects_non_zip_file(tmp_path: Path):
    bad_path = tmp_path / "not_a_zip.zip"
    bad_path.write_bytes(b"this is not a zip file at all")
    with pytest.raises(ArtifactLoadError):
        load_artifact(bad_path)


def test_load_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ArtifactLoadError):
        load_artifact(tmp_path / "does_not_exist.zip")


def test_load_rejects_tampered_content(tmp_path: Path):
    zip_path = _build(tmp_path)

    # Tamper: rewrite the dataset_contract.json content after the archive
    # (and its checksums) were already finalised.
    with zipfile.ZipFile(zip_path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries["semantics/dataset_contract.json"] = b'{"revision": 999, "tables": {}}'

    tampered_path = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(ChecksumMismatchError):
        load_artifact(tampered_path)


def test_load_rejects_archive_missing_checksums_file(tmp_path: Path):
    zip_path = _build(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist() if name != "checksums.sha256"}
    stripped_path = tmp_path / "stripped.zip"
    with zipfile.ZipFile(stripped_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(ArtifactLoadError):
        load_artifact(stripped_path)


def test_load_rejects_path_traversal_entry(tmp_path: Path):
    malicious_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious_path, "w") as zf:
        zf.writestr("../../evil.txt", b"escaped")
        zf.writestr("manifest.json", b"{}")
        zf.writestr("checksums.sha256", b"")

    with pytest.raises(PathTraversalError):
        load_artifact(malicious_path)


def test_load_rejects_incompatible_format_version(tmp_path: Path):
    zip_path = _build(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}

    manifest = json.loads(entries["manifest.json"])
    manifest["artifact_format_version"] = "999.0"
    entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    # Recompute checksums for the modified manifest so this test isolates
    # the version-compatibility check, not the checksum check.
    import hashlib

    checksum_lines = []
    for name, data in sorted(entries.items()):
        if name == "checksums.sha256":
            continue
        checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    entries["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    incompatible_path = tmp_path / "incompatible.zip"
    with zipfile.ZipFile(incompatible_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(IncompatibleArtifactError):
        load_artifact(incompatible_path)


def test_load_rejects_malformed_json_content(tmp_path: Path):
    zip_path = _build(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries["manifest.json"] = b"{not valid json"

    # Recompute checksums so this test isolates the JSON-parsing error, not
    # the checksum check (a real-world equivalent: a builder bug that wrote
    # malformed JSON before the checksum was computed over the same bytes).
    import hashlib

    checksum_lines = []
    for name, data in sorted(entries.items()):
        if name == "checksums.sha256":
            continue
        checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    entries["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    malformed_path = tmp_path / "malformed_json.zip"
    with zipfile.ZipFile(malformed_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(ArtifactLoadError):
        load_artifact(malformed_path)


def test_load_rejects_non_object_json_content(tmp_path: Path):
    zip_path = _build(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    entries["manifest.json"] = b"[1, 2, 3]"

    import hashlib

    checksum_lines = []
    for name, data in sorted(entries.items()):
        if name == "checksums.sha256":
            continue
        checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    entries["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    non_object_path = tmp_path / "non_object_json.zip"
    with zipfile.ZipFile(non_object_path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)

    with pytest.raises(ArtifactLoadError):
        load_artifact(non_object_path)


def test_get_model_for_unknown_table_raises(tmp_path: Path):
    zip_path = _build(tmp_path)
    loaded = load_artifact(zip_path)
    with pytest.raises(ArtifactLoadError):
        loaded.get_model("__does_not_exist__")


def test_get_model_rejects_cloudpickle_by_default(tmp_path: Path):
    """A caller must explicitly opt in before a cloudpickle model (which
    can execute arbitrary code on load) is deserialized."""
    zip_path = _build(tmp_path)
    loaded = load_artifact(zip_path)
    with pytest.raises(UntrustedModelFormatError):
        loaded.get_model("t")


def test_export_never_overwrites_existing_output(temp_sqlite_db: Path, tmp_path: Path):
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
        dataset_id="ds", source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data, candidates_by_table=candidates_data["tables"],
        manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    contract = load_dataset_contract(metadata_dir)
    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )
    training_report = json.loads(Path(training_result.output_references[0]).read_text())

    run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )
    with pytest.raises(RuntimeError):
        run_artifact_export(
            "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
            manifest.run_id, manifest.code_version, manifest, "training_report.json",
        )


def test_generation_fails_closed_on_missing_artifact(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_generation_from_artifact(
        tmp_path / "does_not_exist.zip", "customers", 5, seed=1, manifest=manifest
    )
    assert result.status == "failed"
    assert result.errors


def test_generation_never_overwrites_existing_output(tmp_path: Path):
    zip_path = _build(tmp_path)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    # First call fails (table "t" has a fake, non-loadable model file), but
    # the point here is only that a second call to the same output path
    # must still refuse to silently reuse/overwrite it once it exists.
    output_path = manifest.output_path("generated/t.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("pre-existing")
    with pytest.raises(RuntimeError):
        run_generation_from_artifact(zip_path, "t", 5, seed=1, manifest=manifest)


def test_built_artifact_never_contains_source_connection_or_local_paths(
    temp_sqlite_db: Path, tmp_path: Path
):
    """The most important negative test here: scan every file in a real,
    fully-built archive for the source database's local filesystem path,
    the run directory's local path, and common credential-shaped strings.
    """
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
        dataset_id="ds", source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data, candidates_by_table=candidates_data["tables"],
        manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    contract = load_dataset_contract(metadata_dir)
    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )
    training_report = json.loads(Path(training_result.output_references[0]).read_text())
    export_result = run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )

    forbidden_strings = [
        str(temp_sqlite_db),                       # the source DB's local path
        str(temp_sqlite_db.resolve()),
        str(manifest.run_dir),                      # the training run's local directory
        str(manifest.run_dir.resolve()),
        "password",
        "sqlite:///",
    ]

    zip_path = Path(export_result.output_references[0])
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            content = zf.read(name)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                continue  # binary model file - nothing textual to scan
            for forbidden in forbidden_strings:
                assert forbidden not in text, f"forbidden string found in {name}: {forbidden!r}"


def test_exported_reference_profile_never_contains_exact_extrema_or_category_vocabulary(
    temp_sqlite_db: Path, tmp_path: Path
):
    """Regression test for a real, verified gap: src/profiling/profiler.py's
    exact minimum/maximum/category_frequencies/representative_examples
    used to be copied verbatim into reference_profile.json inside the
    exported ZIP - exact outliers and full category vocabularies (order
    status, etc.) from a real, if small, dataset. Only the safe
    companions (generation_lower_bound/upper_bound, safe_category_frequencies)
    may survive into the archive.
    """
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
        dataset_id="ds", source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data, candidates_by_table=candidates_data["tables"],
        manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    contract = load_dataset_contract(metadata_dir)
    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )
    training_report = json.loads(Path(training_result.output_references[0]).read_text())
    export_result = run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )

    # The un-sanitised source profile must actually have had the exact
    # fields (otherwise this test would trivially pass) - proves the test
    # is exercising the real gap, not a fixture that never had it.
    assert any(
        col.get("category_frequencies") is not None or col.get("maximum") is not None
        for table in profile_data["tables"].values()
        for col in table["columns"].values()
    )

    zip_path = Path(export_result.output_references[0])
    with zipfile.ZipFile(zip_path) as zf:
        reference_profile = json.loads(zf.read("validation/reference_profile.json"))

    for table in reference_profile["tables"].values():
        for column_name, column in table["columns"].items():
            assert "minimum" not in column, f"{column_name}: exact minimum leaked into exported profile"
            assert "maximum" not in column, f"{column_name}: exact maximum leaked into exported profile"
            assert "category_frequencies" not in column, (
                f"{column_name}: exact category vocabulary leaked into exported profile"
            )
            assert "representative_examples" not in column, (
                f"{column_name}: representative example values leaked into exported profile"
            )
