from __future__ import annotations

import pytest

from synth_platform.engine.training.database.artifact.manifest import ARTIFACT_FORMAT, ARTIFACT_FORMAT_VERSION, build_manifest

pytestmark = pytest.mark.unit


DISCOVERY_DATA = {"source_fingerprint": "sha256:" + "a" * 64, "tables": {}}
DATASET_CONTRACT = {"revision": 3, "tables": {}}
TRAINING_REPORT = {
    "tables": {
        "customers": {
            "model_type": "sdv_gaussian_copula",
            "model_path": "/some/training/run/models/customers.pkl",
            "data_fingerprint": "sha256:" + "b" * 64,
            "serialization_format": "cloudpickle",
            "fit_evidence": {
                "trained_columns": ["customer_id", "age"],
                "excluded_columns": [{"column": "bio", "reason": "semantic_type=free_text"}],
                "primary_key": "customer_id",
            },
        }
    }
}


def test_manifest_has_required_top_level_fields():
    manifest = build_manifest(
        "ds1", "1.0.0", DISCOVERY_DATA, DATASET_CONTRACT, TRAINING_REPORT, "run123", "abc1234"
    )
    assert manifest["artifact_format"] == ARTIFACT_FORMAT
    assert manifest["artifact_format_version"] == ARTIFACT_FORMAT_VERSION
    assert manifest["artifact_id"] == "ds1"
    assert manifest["artifact_version"] == "1.0.0"
    assert manifest["source_free"] is True
    assert manifest["contains_source_rows"] is False
    assert manifest["contains_credentials"] is False


def test_manifest_captures_full_traceability_chain():
    manifest = build_manifest(
        "ds1", "1.0.0", DISCOVERY_DATA, DATASET_CONTRACT, TRAINING_REPORT, "run123", "abc1234"
    )
    assert manifest["source_fingerprint"] == DISCOVERY_DATA["source_fingerprint"]
    assert manifest["contract_revision"] == 3
    assert manifest["training_run_id"] == "run123"
    assert manifest["code_version"] == "abc1234"


def test_manifest_table_entry_has_relative_model_path_and_evidence():
    manifest = build_manifest(
        "ds1", "1.0.0", DISCOVERY_DATA, DATASET_CONTRACT, TRAINING_REPORT, "run123", "abc1234"
    )
    table_entry = manifest["tables"]["customers"]
    assert table_entry["model_path"] == "models/tables/customers/model.pkl"
    assert table_entry["model_serialization_format"] == "cloudpickle"
    assert table_entry["trained_columns"] == ["customer_id", "age"]
    assert table_entry["primary_key"] == "customer_id"
    assert table_entry["data_fingerprint"] == TRAINING_REPORT["tables"]["customers"]["data_fingerprint"]


def test_json_model_is_labelled_with_json_serialization_format():
    training_report = {
        "tables": {
            "customers": {
                **TRAINING_REPORT["tables"]["customers"],
                "model_type": "safe_gaussian_copula",
                "model_path": "/some/training/run/models/customers.json",
                "serialization_format": "json",
            }
        }
    }
    manifest = build_manifest("ds1", "1.0.0", DISCOVERY_DATA, DATASET_CONTRACT, training_report, "run123", "abc1234")
    table_entry = manifest["tables"]["customers"]
    assert table_entry["model_path"] == "models/tables/customers/model.json"
    assert table_entry["model_serialization_format"] == "json"
