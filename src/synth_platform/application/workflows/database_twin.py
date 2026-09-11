"""Database Twin application workflow facade.

The Streamlit page depends on this module rather than importing individual
engine packages.  The implementation remains stage-oriented underneath:
discovery -> profiling -> inference -> training -> artifact -> generation ->
validation -> target write.
"""

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.discovery.database.demo.sample_database import build_sample_database
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.training.database.planner import recommend_synthesizer
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export
from synth_platform.engine.generation.database.relational_service import (
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.generation.database.target_write_service import (
    load_target_write_report,
    run_target_write,
)
from synth_platform.engine.validation.database.qa_service import load_qa_report, run_qa_validation
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.pipeline_runner import (
    DatabaseTwinPipelineConfig,
    resolve_database_row_counts,
    run_database_twin_pipeline as _run_database_twin_pipeline,
)


def run_database_twin_pipeline(adapter, manifest: RunManifest, config: DatabaseTwinPipelineConfig, *, history=None):
    context = _run_database_twin_pipeline(adapter, manifest, config)
    if history is not None:
        try:
            qa_report = context.artifacts.get("qa_report")
            target_written = any(
                stage.get("stage_name") == "target_write" and stage.get("status") == "success"
                for stage in manifest.stages
            )
            validation_passed = bool(qa_report.get("hard_checks_passed")) if isinstance(qa_report, dict) else False
            history.record_run(
                workflow_type="database",
                project_name=config.dataset_id,
                status="completed" if target_written and validation_passed else "failed",
                validation_status="PASS" if validation_passed else "FAIL",
                validation_passed=validation_passed,
                output_id=config.target_db_path.name,
                run_id=manifest.run_id,
                metadata={
                    "dataset_id": config.dataset_id,
                    "artifact_version": config.artifact_version,
                    "target_db_path": str(config.target_db_path),
                    "row_counts_by_table": config.row_counts_by_table,
                    "manifest_run_id": manifest.run_id,
                },
            )
        except Exception:
            pass
    return context

WORKFLOW_STAGES = (
    "connect",
    "discovery",
    "profiling",
    "inference",
    "training",
    "artifact_export",
    "generation",
    "validation",
    "target_write",
)

__all__ = [
    "SQLiteSourceAdapter", "RunManifest", "DatabaseTwinPipelineConfig", "resolve_database_row_counts", "build_sample_database", "run_discovery",
    "PROFILE_FILENAME", "load_profile", "run_profiling", "load_dataset_contract",
    "run_contract_approval", "run_inference", "recommend_synthesizer",
    "get_synthesizer_adapter_class", "load_training_report", "run_training_and_sampling",
    "load_artifact", "run_artifact_export", "load_relational_generation_report",
    "run_relational_generation", "run_database_twin_pipeline", "load_target_write_report", "run_target_write",
    "load_qa_report", "run_qa_validation", "WORKFLOW_STAGES",
]
