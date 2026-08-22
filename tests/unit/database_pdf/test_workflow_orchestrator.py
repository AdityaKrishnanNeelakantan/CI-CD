"""Unit tests for WorkflowOrchestrator: sequencing, resume, retry, rollback.

Uses simple in-memory StageDefinition handlers (not real pipeline services -
see tests/integration/test_pipeline_runner.py for the real-service wiring)
so the orchestrator's own control flow is verified in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.common.database.core.workflow_orchestrator import (
    StageDefinition,
    WorkflowContext,
    WorkflowOrchestrator,
    WorkflowTrack,
    build_database_workflow,
    build_pdf_workflow,
)


def _ok(name: str) -> StageResult:
    return StageResult(stage_name=name, status=STATUS_SUCCESS, input_references=[], output_references=[])


def _fail(name: str) -> StageResult:
    return StageResult(stage_name=name, status=STATUS_FAILED, input_references=[], output_references=[], errors=["boom"])


def test_run_executes_stages_in_dependency_order(tmp_path: Path):
    calls: list[str] = []

    def make_handler(name: str):
        def handler(ctx: WorkflowContext) -> StageResult:
            calls.append(name)
            result = _ok(name)
            ctx.manifest.record_stage(result)
            return result

        return handler

    stages = [
        StageDefinition("a", make_handler("a")),
        StageDefinition("b", make_handler("b"), depends_on=["a"]),
        StageDefinition("c", make_handler("c"), depends_on=["b"]),
    ]
    orchestrator = WorkflowOrchestrator(stages)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    results = orchestrator.run(context)

    assert calls == ["a", "b", "c"]
    assert all(r.is_success() for r in results)
    assert orchestrator.completed_stages(manifest) == ["a", "b", "c"]
    assert orchestrator.next_stage(manifest) is None


def test_run_stops_at_first_failure_and_does_not_run_later_stages(tmp_path: Path):
    calls: list[str] = []

    def handler_a(ctx: WorkflowContext) -> StageResult:
        calls.append("a")
        return _fail("a")

    def handler_b(ctx: WorkflowContext) -> StageResult:
        calls.append("b")
        return _ok("b")

    stages = [StageDefinition("a", handler_a), StageDefinition("b", handler_b, depends_on=["a"])]
    orchestrator = WorkflowOrchestrator(stages)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    results = orchestrator.run(context)

    assert calls == ["a"]
    assert len(results) == 1
    assert results[0].status == STATUS_FAILED


def test_run_resumes_from_start_from_skipping_completed_stages(tmp_path: Path):
    calls: list[str] = []

    def make_handler(name: str):
        def handler(ctx: WorkflowContext) -> StageResult:
            calls.append(name)
            return _ok(name)

        return handler

    stages = [
        StageDefinition("a", make_handler("a")),
        StageDefinition("b", make_handler("b"), depends_on=["a"]),
        StageDefinition("c", make_handler("c"), depends_on=["b"]),
    ]
    orchestrator = WorkflowOrchestrator(stages)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    manifest.record_stage(_ok("a"))
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    orchestrator.run(context, start_from="b")

    assert calls == ["b", "c"]


def test_retry_recovers_after_transient_failure(tmp_path: Path):
    attempts: list[int] = []

    def flaky_handler(ctx: WorkflowContext) -> StageResult:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient")
        return _ok("flaky")

    orchestrator = WorkflowOrchestrator([StageDefinition("flaky", flaky_handler)])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    results = orchestrator.run(context, max_retries=2)

    assert len(attempts) == 2
    assert results[0].is_success()


def test_run_records_stage_exactly_once_when_handler_forgets_to_record(tmp_path: Path):
    """A handler that returns a result without calling manifest.record_stage()
    itself (e.g. test_run_stops_at_first_failure_and_does_not_run_later_stages'
    handler_a) must still end up recorded - the orchestrator is now the
    structural guarantee, not an implicit per-handler contract."""

    def forgetful_handler(ctx: WorkflowContext) -> StageResult:
        return _ok("a")

    orchestrator = WorkflowOrchestrator([StageDefinition("a", forgetful_handler)])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    orchestrator.run(context)

    assert [s["stage_name"] for s in manifest.stages] == ["a"]


def test_run_records_failed_result_even_when_handler_does_not_record_it(tmp_path: Path):
    def forgetful_failing_handler(ctx: WorkflowContext) -> StageResult:
        return _fail("a")

    orchestrator = WorkflowOrchestrator([StageDefinition("a", forgetful_failing_handler)])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    orchestrator.run(context)

    assert [s["stage_name"] for s in manifest.stages] == ["a"]
    assert manifest.stages[0]["status"] == STATUS_FAILED


def test_run_does_not_double_record_when_handler_already_self_records(tmp_path: Path):
    """Every real service handler in src/ calls manifest.record_stage() itself
    before returning - the orchestrator's recording middleware must be a no-op
    for them, not a second duplicate entry."""

    def self_recording_handler(ctx: WorkflowContext) -> StageResult:
        result = _ok("a")
        ctx.manifest.record_stage(result)
        return result

    orchestrator = WorkflowOrchestrator([StageDefinition("a", self_recording_handler)])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    orchestrator.run(context)

    assert len(manifest.stages) == 1
    assert manifest.stages[0]["stage_name"] == "a"


def test_run_persists_duration_seconds_for_forgetful_handler(tmp_path: Path):
    def forgetful_handler(ctx: WorkflowContext) -> StageResult:
        return _ok("a")

    orchestrator = WorkflowOrchestrator([StageDefinition("a", forgetful_handler)])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    orchestrator.run(context)

    assert "duration_seconds" in manifest.stages[0]["evidence"]
    assert manifest.stages[0]["evidence"]["duration_seconds"] >= 0.0


def test_run_persists_duration_seconds_for_self_recording_handler(tmp_path: Path):
    """The common production path: a handler that calls record_stage() itself
    before returning must still end up with duration_seconds in the persisted
    manifest entry, not just on the transient in-memory StageResult."""

    def self_recording_handler(ctx: WorkflowContext) -> StageResult:
        result = _ok("a")
        ctx.manifest.record_stage(result)
        return result

    orchestrator = WorkflowOrchestrator([StageDefinition("a", self_recording_handler)])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE)

    orchestrator.run(context)

    assert len(manifest.stages) == 1
    assert "duration_seconds" in manifest.stages[0]["evidence"]
    assert manifest.stages[0]["evidence"]["duration_seconds"] >= 0.0

    # Reload from disk to prove it was actually written, not just held in
    # the in-memory manifest.stages list.
    reloaded = json.loads((manifest.run_dir / "run_manifest.json").read_text())
    assert "duration_seconds" in reloaded["stages"][0]["evidence"]


def test_rollback_to_removes_later_stage_entries_only(tmp_path: Path):
    orchestrator = WorkflowOrchestrator(
        [StageDefinition("a", lambda ctx: _ok("a")), StageDefinition("b", lambda ctx: _ok("b"), depends_on=["a"])]
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    manifest.record_stage(_ok("a"))
    manifest.record_stage(_ok("b"))

    rolled_back = orchestrator.rollback_to(manifest, "a")

    assert rolled_back == ["b"]
    assert [s["stage_name"] for s in manifest.stages] == ["a"]


def test_build_database_workflow_has_expected_stage_order():
    handlers = {name: (lambda ctx: _ok(name)) for name in [
        "discovery", "profiling", "inference", "contract_approval", "cleaning",
        "training", "artifact_export", "relational_generation", "qa_validation", "target_write",
    ]}
    orchestrator = build_database_workflow(handlers)
    assert orchestrator.stage_names == list(handlers.keys())


def test_build_pdf_workflow_has_expected_stage_order():
    handlers = {name: (lambda ctx: _ok(name)) for name in [
        "document_profiling", "template_compilation", "semantic_binding",
        "value_generation", "render", "validation",
    ]}
    orchestrator = build_pdf_workflow(handlers)
    assert orchestrator.stage_names == list(handlers.keys())
