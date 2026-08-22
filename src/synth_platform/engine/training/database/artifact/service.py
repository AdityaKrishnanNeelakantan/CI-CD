"""Checkpoint 5 orchestration: export a portable artifact, and (separately)
generate from one without ever touching the original source.

run_artifact_export() and run_generation_from_artifact() are deliberately
independent - the second never takes a SourceAdapter, only an artifact
path, so it can genuinely run in a fresh process with the source database
disconnected or removed entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_platform.engine.training.database.artifact.builder import build_artifact
from synth_platform.engine.training.database.artifact.errors import ArtifactError
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.self_test import run_self_test
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.training.database.base import SynthesisError

EXPORT_STAGE_NAME = "artifact_export"
GENERATE_STAGE_NAME = "generation"


def run_artifact_export(
    dataset_id: str,
    artifact_version: str,
    discovery_data: dict[str, Any],
    dataset_contract: dict[str, Any],
    reference_profile: dict[str, Any],
    training_report: dict[str, Any],
    training_run_id: str,
    code_version: str,
    manifest: RunManifest,
    training_reference: str,
) -> StageResult:
    output_path = manifest.run_dir / "artifacts" / f"generator-{dataset_id}-{artifact_version}.zip"
    if output_path.exists():
        raise RuntimeError(
            f"artifact already exists at {output_path}; a stage output must never be "
            "overwritten. Start a new run instead."
        )

    try:
        build_artifact(
            dataset_id,
            artifact_version,
            discovery_data,
            dataset_contract,
            reference_profile,
            training_report,
            training_run_id,
            code_version,
            output_path,
        )
        loaded = load_artifact(output_path)
        self_test_report = run_self_test(loaded)
    except ArtifactError as exc:
        result = StageResult(
            stage_name=EXPORT_STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[training_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    result = StageResult(
        stage_name=EXPORT_STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[training_reference],
        output_references=[str(output_path)],
        metrics={
            "table_count": len(loaded.manifest["tables"]),
            "self_test_passed": self_test_report["passed"],
        },
        warnings=[] if self_test_report["passed"] else ["self_test_failed"],
        evidence={
            "artifact_id": dataset_id,
            "artifact_version": artifact_version,
            "self_test": self_test_report,
        },
    )
    manifest.record_stage(result)
    return result


def run_generation_from_artifact(
    artifact_path: str | Path,
    table_name: str,
    num_rows: int,
    seed: int,
    manifest: RunManifest,
    allow_cloudpickle_models: bool = False,
) -> StageResult:
    """Generate rows from a portable artifact alone.

    `manifest` here is typically a brand-new run, possibly in a completely
    different process or machine than the one that trained the model -
    nothing here reads a SourceAdapter or the original source.
    """
    output_path = manifest.output_path(f"generated/{table_name}.csv")
    if output_path.exists():
        raise RuntimeError(
            f"{output_path} already exists for run {manifest.run_id}; a stage output must "
            "never be overwritten. Start a new run instead."
        )

    try:
        loaded = load_artifact(artifact_path)
        model = loaded.get_model(table_name, allow_cloudpickle_models=allow_cloudpickle_models)
        generated_df = model.sample(num_rows, seed=seed)
    except (ArtifactError, SynthesisError) as exc:
        result = StageResult(
            stage_name=GENERATE_STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[str(artifact_path)],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_df.to_csv(output_path, index=False)

    result = StageResult(
        stage_name=GENERATE_STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[str(artifact_path)],
        output_references=[str(output_path)],
        metrics={"generated_row_count": len(generated_df), "table_name": table_name},
        evidence={
            "artifact_id": loaded.manifest["artifact_id"],
            "artifact_version": loaded.manifest["artifact_version"],
        },
    )
    manifest.record_stage(result)
    return result
