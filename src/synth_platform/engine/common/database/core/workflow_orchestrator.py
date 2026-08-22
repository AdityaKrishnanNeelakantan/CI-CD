"""Workflow orchestrator: controls stage transitions, retries, failure
handling, and rollback across both the Database Twin and PDF Twin
pipelines, independent of the Streamlit UI shell.

Today the UI holds step-gating in session state; this module provides
the same sequencing as a callable service so pipelines can run
unattended, from CLI, or in CI.  Each stage is a named callable that
receives a WorkflowContext and returns a StageResult; the orchestrator
records results in the RunManifest and supports resume from the last
successful stage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult

StageCallable = Callable[["WorkflowContext"], StageResult]


class WorkflowTrack(str, Enum):  # noqa: UP042 - str(member) formatting semantics
    # differ from enum.StrEnum in ways not audited here; changing the base
    # class is a behavior-adjacent decision out of scope for a CI/CD lint
    # baseline change.
    DATABASE = "database"
    PDF = "pdf"


@dataclass
class StageDefinition:
    name: str
    handler: StageCallable
    depends_on: list[str] = field(default_factory=list)


@dataclass
class WorkflowContext:
    """Shared mutable state passed between stage handlers."""

    manifest: RunManifest
    track: WorkflowTrack
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """Drives a multi-stage pipeline with resume/retry support."""

    def __init__(self, stages: list[StageDefinition]) -> None:
        self._stages = {s.name: s for s in stages}
        self._stage_order = [s.name for s in stages]

    @property
    def stage_names(self) -> list[str]:
        return list(self._stage_order)

    def completed_stages(self, manifest: RunManifest) -> list[str]:
        return [s["stage_name"] for s in manifest.stages if s.get("status") == STATUS_SUCCESS]

    def next_stage(self, manifest: RunManifest) -> str | None:
        completed = set(self.completed_stages(manifest))
        for name in self._stage_order:
            if name not in completed:
                stage = self._stages[name]
                if all(dep in completed for dep in stage.depends_on):
                    return name
        return None

    def run(
        self,
        context: WorkflowContext,
        *,
        start_from: str | None = None,
        stop_after: str | None = None,
        max_retries: int = 0,
    ) -> list[StageResult]:
        """Execute stages sequentially, optionally resuming from a named stage."""
        results: list[StageResult] = []
        completed = set(self.completed_stages(context.manifest))
        started = start_from is None

        for name in self._stage_order:
            if not started:
                if name == start_from:
                    started = True
                else:
                    continue

            if name in completed:
                continue

            stage = self._stages[name]
            if not all(dep in completed for dep in stage.depends_on):
                break

            result = self._execute_with_retry(stage, context, max_retries)
            results.append(result)

            if result.is_success():
                completed.add(name)
            else:
                break

            if name == stop_after:
                break

        return results

    def rollback_to(self, manifest: RunManifest, target_stage: str) -> list[str]:
        """Remove stage entries after target_stage from the manifest (logical rollback).

        Does not delete output files on disk - callers must handle artifact
        cleanup separately if needed.  Returns the list of rolled-back stage names.
        """
        if target_stage not in self._stage_order:
            raise ValueError(f"unknown stage {target_stage!r}")

        target_idx = self._stage_order.index(target_stage)
        allowed = set(self._stage_order[: target_idx + 1])
        rolled_back: list[str] = []
        kept: list[dict[str, Any]] = []

        for entry in manifest.stages:
            if entry["stage_name"] in allowed:
                kept.append(entry)
            else:
                rolled_back.append(entry["stage_name"])

        manifest.stages = kept
        manifest.write()
        return rolled_back

    def _execute_with_retry(
        self, stage: StageDefinition, context: WorkflowContext, max_retries: int
    ) -> StageResult:
        last_result: StageResult | None = None
        for attempt in range(max_retries + 1):
            stages_before = len(context.manifest.stages)
            started_at = time.monotonic()
            try:
                result = stage.handler(context)
            except Exception as exc:
                result = StageResult(
                    stage_name=stage.name,
                    status=STATUS_FAILED,
                    input_references=[],
                    output_references=[],
                    errors=[str(exc)],
                    evidence={"attempt": attempt + 1},
                )
                result.evidence.setdefault("duration_seconds", round(time.monotonic() - started_at, 3))
                context.manifest.record_stage(result)
            else:
                elapsed = round(time.monotonic() - started_at, 3)
                if len(context.manifest.stages) == stages_before:
                    # Handler did not self-record (P0 guarantee): record now,
                    # with duration already attached.
                    result.evidence.setdefault("duration_seconds", elapsed)
                    context.manifest.record_stage(result)
                else:
                    # Handler already self-recorded (the common production
                    # path - see the 20+ existing service call sites): the
                    # manifest entry is an immutable snapshot taken at that
                    # earlier call (RunManifest.record_stage does
                    # `entry = result.to_dict()`), so setting
                    # `result.evidence` now would NOT reach the persisted
                    # entry. Patch the just-appended entry in place instead,
                    # guarded by a stage_name match so a pathological handler
                    # that recorded something else entirely is never touched.
                    recorded_entry = context.manifest.stages[-1]
                    if recorded_entry.get("stage_name") == stage.name:
                        entry_evidence = recorded_entry.setdefault("evidence", {})
                        if "duration_seconds" not in entry_evidence:
                            entry_evidence["duration_seconds"] = elapsed
                            context.manifest.write()
                        result.evidence.setdefault("duration_seconds", entry_evidence["duration_seconds"])

            last_result = result
            if result.is_success():
                return result

        return last_result  # type: ignore[return-value]


def build_database_workflow(
    stage_handlers: dict[str, StageCallable],
) -> WorkflowOrchestrator:
    """Build the standard Database Twin pipeline from a handler map."""
    ordered = [
        StageDefinition("discovery", stage_handlers["discovery"]),
        StageDefinition("profiling", stage_handlers["profiling"], depends_on=["discovery"]),
        StageDefinition("inference", stage_handlers["inference"], depends_on=["profiling"]),
        StageDefinition("contract_approval", stage_handlers["contract_approval"], depends_on=["inference"]),
        StageDefinition("cleaning", stage_handlers["cleaning"], depends_on=["contract_approval"]),
        StageDefinition("training", stage_handlers["training"], depends_on=["cleaning"]),
        StageDefinition("artifact_export", stage_handlers["artifact_export"], depends_on=["training"]),
        StageDefinition("relational_generation", stage_handlers["relational_generation"], depends_on=["training"]),
        StageDefinition("qa_validation", stage_handlers["qa_validation"], depends_on=["relational_generation"]),
        StageDefinition("target_write", stage_handlers["target_write"], depends_on=["qa_validation"]),
    ]
    return WorkflowOrchestrator(ordered)


def build_pdf_workflow(
    stage_handlers: dict[str, StageCallable],
) -> WorkflowOrchestrator:
    """Build the standard PDF Twin pipeline from a handler map."""
    ordered = [
        StageDefinition("document_profiling", stage_handlers["document_profiling"]),
        StageDefinition("template_compilation", stage_handlers["template_compilation"], depends_on=["document_profiling"]),
        StageDefinition("semantic_binding", stage_handlers["semantic_binding"], depends_on=["template_compilation"]),
        StageDefinition("value_generation", stage_handlers["value_generation"], depends_on=["semantic_binding"]),
        StageDefinition("render", stage_handlers["render"], depends_on=["value_generation"]),
        StageDefinition("validation", stage_handlers["validation"], depends_on=["render"]),
    ]
    return WorkflowOrchestrator(ordered)


def save_workflow_state(manifest: RunManifest, context: WorkflowContext, path: str | Path) -> None:
    """Persist workflow context artifacts for resume across process restarts."""
    state = {
        "run_id": manifest.run_id,
        "track": context.track.value,
        "config": context.config,
        "artifact_keys": sorted(context.artifacts.keys()),
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
