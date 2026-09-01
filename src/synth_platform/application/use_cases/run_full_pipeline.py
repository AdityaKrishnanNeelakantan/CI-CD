"""End-to-end: train -> export -> (delete source) -> load -> generate ->
validate+evaluate -> quarantine -> promote on PASS."""
from __future__ import annotations

from pathlib import Path

from synth_platform.application.dto.commands import GenerationRequest
from synth_platform.application.dto.results import GenerationResult
from synth_platform.application.ports.artifact_store import ArtifactStore
from synth_platform.application.ports.source_connector import SourceConnector
from synth_platform.application.use_cases.export_artifact import export_artifact
from synth_platform.application.use_cases.generate_dataset import generate_dataset
from synth_platform.application.use_cases.load_artifact import load_artifact
from synth_platform.application.use_cases.train_model import train_model
from synth_platform.application.use_cases.validate_dataset import validate_dataset
from synth_platform.application.ports.release_gate import ReleaseGate
from synth_platform.domain.validation.release_gate import decide
from synth_platform.engine.validation.holdout_evaluator import (
    SecureHoldoutEvaluator, UtilityTask, holdout_split,
)


def run_full_pipeline(connector: SourceConnector, store: ArtifactStore,
                      gate: ReleaseGate,
                      artifact_path: str | Path, request: GenerationRequest,
                      approved_root: str | Path,
                      tasks: list[UtilityTask] | None = None,
                      sample_size: int = 50_000, seed: int = 42) -> GenerationResult:
    split = holdout_split(connector, seed=seed, max_rows=sample_size)
    artifact = train_model(connector, sample_size=sample_size, seed=seed)
    export_artifact(store, artifact, artifact_path)

    loaded = load_artifact(store, artifact_path)
    tables = generate_dataset(loaded, request)

    evaluator = SecureHoldoutEvaluator(split, tasks=tasks or [], seed=seed)
    report = validate_dataset(loaded, tables, evaluator.privacy_utility_checks(tables))
    decision = decide(report)  # report is already applicability-resolved; re-reads verdict

    result = gate.submit(tables, report, run_id="run")
    output_dir = None
    if result.promotable:
        gate.promote(result, approved_root)
        output_dir = str(approved_root)
    return GenerationResult(report=report, decision=decision,
                            output_dir=output_dir, staging_dir=str(result.staging_dir))
