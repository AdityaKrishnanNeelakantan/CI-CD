"""Normalize specialized workflow outcomes for shared progress/results screens."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synth_platform.application.dto.tool_commands import RegisterArtifactCommand
from synth_platform.application.dto.workspace import (
    ArtifactRegistration,
    ExecutionState,
    LineageRef,
    PreviewDescriptor,
    WorkflowKind,
    WorkflowResultView,
)
from synth_platform.application.services.tool_gateway import GuardedWorkspaceTools
from synth_platform.application.workflows.interaction_twin import (
    InteractionWorkflowResult,
)
from synth_platform.application.workflows.schema_twin import SchemaModeResult
from synth_platform.domain.validation.models import Status
from synth_platform.engine.common.database.core.run_manifest import RunManifest


def _artifact_format(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "file"


def _media_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".parquet": "application/vnd.apache.parquet",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".zip": "application/zip",
    }.get(path.suffix.lower(), "application/octet-stream")


def _register_paths(
    tools: GuardedWorkspaceTools,
    session_id: str,
    paths: Iterable[tuple[str, str | Path, bool]],
):
    artifacts = []
    seen: set[Path] = set()
    for label, raw_path, downloadable in paths:
        path = Path(raw_path).resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        artifacts.append(
            tools.register_artifact(
                RegisterArtifactCommand(
                    session_id=session_id,
                    registration=ArtifactRegistration(
                        path=str(path),
                        label=label,
                        media_type=_media_type(path),
                        format=_artifact_format(path),
                        downloadable=downloadable,
                    ),
                )
            )
        )
    return artifacts


def lineage_from_manifest(manifest: RunManifest) -> LineageRef:
    return LineageRef(
        run_id=manifest.run_id,
        manifest_path=str((manifest.run_dir / "run_manifest.json").resolve()),
        code_version=manifest.code_version,
        source_fingerprint=manifest.source_fingerprint,
        stage_ids=[
            str(stage.get("stage_name", "unknown")) for stage in manifest.stages
        ],
    )


def schema_result_view(
    tools: GuardedWorkspaceTools,
    session_id: str,
    result: SchemaModeResult,
    *,
    package_path: str | Path | None = None,
) -> WorkflowResultView:
    artifact_paths: list[tuple[str, str | Path, bool]] = [
        (f"{table} synthetic data", path, True)
        for table, path in result.export_paths.items()
    ]
    if package_path is not None:
        artifact_paths.append(("Synthetic dataset package", package_path, True))
    artifacts = _register_paths(tools, session_id, artifact_paths)
    highlights = result.validation_report
    hard_passed = result.hard_checks_passed
    performance = result.pipeline.performance
    metrics: dict[str, Any] = {
        "table_count": len(result.row_counts),
        "total_rows": sum(int(value) for value in result.row_counts.values()),
        "row_counts": dict(result.row_counts),
        "export_count": len(result.export_paths),
    }
    if performance is not None:
        metrics.update(
            {
                "total_time_seconds": getattr(performance, "total_time_seconds", None),
                "peak_memory_mb": getattr(performance, "peak_memory_mb", None),
                "rows_per_second": getattr(performance, "rows_per_second", None),
            }
        )
    issues = (
        highlights.get("issues")
        or highlights.get("errors")
        or highlights.get("failures")
        or []
    )
    if isinstance(issues, str):
        issues = [issues]
    return WorkflowResultView(
        session_id=session_id,
        workflow=WorkflowKind.SCHEMA,
        execution_status=ExecutionState.SUCCEEDED,
        validation_status=Status.PASS if hard_passed else Status.FAIL,
        release_verdict=None,
        blockers=[str(item) for item in issues] if not hard_passed else [],
        warnings=[str(item) for item in issues] if hard_passed else [],
        metrics=metrics,
        previews=[
            PreviewDescriptor(
                label=table,
                kind="table",
                row_count=int(result.row_counts.get(table, len(frame))),
                column_count=len(frame.columns),
                description="Synthetic table preview is held in the active UI session.",
            )
            for table, frame in result.preview_tables.items()
        ],
        artifacts=artifacts,
    )


def interaction_result_view(
    tools: GuardedWorkspaceTools,
    session_id: str,
    result: InteractionWorkflowResult,
) -> WorkflowResultView:
    artifact_paths: list[tuple[str, str | Path, bool]] = []
    if result.released:
        artifact_paths = [
            ("Sanitized source", result.sanitized_source_path, True),
            ("Interaction SSOT", result.ssot_path, True),
            ("Validation report", result.validation_path, True),
            ("Artifact manifest", result.artifact_manifest_path, True),
        ]
        if result.package_path is not None:
            artifact_paths.append(("Interaction Twin package", result.package_path, True))
    validation = result.validation_report
    stage_warnings = [
        str(warning)
        for stage in result.run_manifest.stages
        for warning in stage.get("warnings", [])
    ]
    lineage = lineage_from_manifest(result.run_manifest)
    if not result.released:
        lineage = lineage.model_copy(update={"manifest_path": None})
    return WorkflowResultView(
        session_id=session_id,
        workflow=WorkflowKind.INTERACTION,
        execution_status=ExecutionState.SUCCEEDED,
        validation_status=validation.validation.overall,
        release_verdict=validation.release.verdict,
        blockers=list(validation.release.blocking),
        warnings=stage_warnings,
        metrics={
            **validation.metrics,
            "turn_count": len(result.sanitized_transcript.turns),
            "participant_count": len(result.sanitized_transcript.participants),
            "model_requested": result.ssot.extraction.model_requested,
            "model_used": result.ssot.extraction.model_used,
        },
        previews=[
            PreviewDescriptor(
                label="Structured interaction SSOT",
                kind="json",
                description="Structured preview is held in the active UI session.",
            )
        ],
        artifacts=_register_paths(tools, session_id, artifact_paths),
        lineage=lineage,
    )


def stage_workflow_result_view(
    tools: GuardedWorkspaceTools,
    session_id: str,
    *,
    workflow: WorkflowKind,
    manifest: RunManifest,
    validation_status: Status,
    release_verdict: Status | None = None,
    blockers: Iterable[str] = (),
    warnings: Iterable[str] = (),
    metrics: dict[str, Any] | None = None,
    previews: Iterable[PreviewDescriptor] = (),
    artifacts: Iterable[tuple[str, str | Path, bool]] = (),
    execution_status: ExecutionState = ExecutionState.SUCCEEDED,
) -> WorkflowResultView:
    """Build a shared view for Database/PDF while preserving their stage APIs."""

    stage_warnings = [
        str(warning)
        for stage in manifest.stages
        for warning in stage.get("warnings", [])
    ]
    stage_errors = [
        str(error) for stage in manifest.stages for error in stage.get("errors", [])
    ]
    combined_blockers = list(dict.fromkeys([*map(str, blockers), *stage_errors]))
    combined_warnings = list(dict.fromkeys([*map(str, warnings), *stage_warnings]))
    return WorkflowResultView(
        session_id=session_id,
        workflow=workflow,
        execution_status=execution_status,
        validation_status=validation_status,
        release_verdict=release_verdict,
        blockers=combined_blockers,
        warnings=combined_warnings,
        metrics=metrics or {},
        previews=list(previews),
        artifacts=_register_paths(tools, session_id, artifacts),
        lineage=lineage_from_manifest(manifest),
    )
