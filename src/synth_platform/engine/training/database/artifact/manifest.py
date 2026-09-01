"""Builds the manifest.json that anchors a portable artifact.

manifest.json is the first thing a loader reads (src/artifact/loader.py)
and is what makes the full traceability chain walkable from the artifact
alone: generated dataset -> generation run -> artifact version -> training
run -> approved dataset contract -> source schema fingerprint.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT = "generic-synthetic-generator"
ARTIFACT_FORMAT_VERSION = "1.0"


def build_manifest(
    dataset_id: str,
    artifact_version: str,
    discovery_data: dict[str, Any],
    dataset_contract: dict[str, Any],
    training_report: dict[str, Any],
    training_run_id: str,
    code_version: str,
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table_name, table_report in training_report["tables"].items():
        fit_evidence = table_report["fit_evidence"]
        model_extension = Path(table_report["model_path"]).suffix
        tables[table_name] = {
            "model_type": table_report["model_type"],
            "trained_columns": fit_evidence["trained_columns"],
            "excluded_columns": fit_evidence["excluded_columns"],
            "primary_key": fit_evidence["primary_key"],
            "model_path": f"models/tables/{table_name}/model{model_extension}",
            # Declared by the adapter itself (src/synthesis/base.py's
            # serialization_format) - never inferred from file_extension,
            # so a human/automated privacy reviewer can trust this field
            # without opening every model file.
            "model_serialization_format": table_report["serialization_format"],
            "data_fingerprint": table_report["data_fingerprint"],
        }

    return {
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_id": dataset_id,
        "artifact_version": artifact_version,
        "created_at": datetime.now(UTC).isoformat(),
        "source_free": True,
        "contains_source_rows": False,
        "contains_credentials": False,
        "dataset_id": dataset_id,
        "source_fingerprint": discovery_data["source_fingerprint"],
        "contract_revision": dataset_contract["revision"],
        "training_run_id": training_run_id,
        "code_version": code_version,
        "tables": tables,
    }
