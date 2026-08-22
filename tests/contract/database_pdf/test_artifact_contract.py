"""Contract: every built artifact has the same top-level file structure,
and load_artifact() always returns the same LoadedArtifact shape,
regardless of how many tables were trained.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from synth_platform.engine.training.database.artifact.builder import build_artifact
from synth_platform.engine.training.database.artifact.loader import load_artifact

pytestmark = pytest.mark.contract

REQUIRED_ARCHIVE_MEMBERS = {
    "manifest.json",
    "checksums.sha256",
    "self_test.json",
    "schema/database_catalog.json",
    "semantics/dataset_contract.json",
    "validation/reference_profile.json",
}


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
                "fit_evidence": {
                    "trained_columns": ["a", "b"],
                    "excluded_columns": [],
                    "primary_key": "a",
                },
            }
        }
    }
    return discovery_data, dataset_contract, reference_profile, training_report


def test_archive_contains_required_members(tmp_path: Path):
    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)
    zip_path = build_artifact(
        "ds", "1.0.0", discovery_data, contract, profile, training_report,
        "run1", "abc123", tmp_path / "out.zip",
    )

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert REQUIRED_ARCHIVE_MEMBERS.issubset(names)
    assert "models/tables/t/model.pkl" in names


def test_loaded_artifact_has_required_attributes(tmp_path: Path):
    discovery_data, contract, profile, training_report = _minimal_inputs(tmp_path)
    zip_path = build_artifact(
        "ds", "1.0.0", discovery_data, contract, profile, training_report,
        "run1", "abc123", tmp_path / "out.zip",
    )

    loaded = load_artifact(zip_path)
    for attr in ("manifest", "database_catalog", "dataset_contract", "reference_profile", "self_test_spec"):
        assert hasattr(loaded, attr)
        assert isinstance(getattr(loaded, attr), dict)
