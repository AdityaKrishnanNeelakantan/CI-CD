"""Unattended Database Twin pipeline runner.

Wires the real Checkpoint 1-8 service functions (the same functions
the Database Twin Streamlit page calls from Streamlit session-state-driven step
gating) into WorkflowOrchestrator's stage-handler contract, so the full
"Database A -> generate -> Database B" track can also run headlessly from a
CLI or CI job. The stage order and dependency graph come from
build_database_workflow() - this module only supplies the handlers.

Also attaches the canonical cross-track domain model (src.core.domain_model)
to the approved dataset contract once contract_approval succeeds, persisting
which canonical entity each table represents as domain_mapping.json - shared
vocabulary the PDF Twin track can use for parity checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.training.database.artifact.service import run_artifact_export
from synth_platform.engine.profiling.database.cleaning.service import run_cleaning
from synth_platform.engine.common.database.core.domain_model import CanonicalDomainModel, map_contract_to_domain
from synth_platform.engine.common.database.core.observability import logger
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import StageResult
from synth_platform.engine.common.database.core.workflow_orchestrator import (
    WorkflowContext,
    WorkflowTrack,
    build_database_workflow,
)
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling
from synth_platform.engine.validation.database.qa_service import QA_REPORT_FILENAME, load_qa_report, run_qa_validation
from synth_platform.engine.generation.database.relational_service import (
    RELATIONAL_REPORT_FILENAME,
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.generation.database.target_write_service import run_target_write
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling
from synth_platform.domain.product_settings import (
    ProductSettingsReader,
    read_generation_defaults,
)

DOMAIN_MAPPING_FILENAME = "domain_mapping.json"


@dataclass
class DatabaseTwinPipelineConfig:
    """Unattended-mode configuration for a full Database Twin run."""

    dataset_id: str
    artifact_version: str
    config_path: str
    metadata_dir: Path
    target_db_path: Path
    row_counts_by_table: dict[str, int] | None = None
    sample_limit: int = 1000
    num_rows_to_generate: int = 10
    seed: int = 11
    model_type: str = "safe_gaussian_copula"
    chunk_size: int | None = None
    release_mode: str | None = None
    domain_model: CanonicalDomainModel | None = None
    product_settings: ProductSettingsReader | None = None


def resolve_database_row_counts(
    dataset_contract: dict[str, Any],
    row_counts_by_table: dict[str, int] | None = None,
    *,
    product_settings: ProductSettingsReader | None = None,
) -> dict[str, int]:
    """Use explicit row-count overrides or the persisted default per table."""

    if row_counts_by_table:
        return {table: max(1, int(count)) for table, count in row_counts_by_table.items()}
    default_count = int(read_generation_defaults(product_settings)["default_record_count"])
    return {table_name: default_count for table_name in dataset_contract.get("tables", {})}


def _auto_approve_decisions(candidates_by_table: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Default unattended approval policy: accept the engine's top-ranked semantic type for every column.

    Mirrors a human reviewer accepting every suggestion as-is, including ones flagged
    REVIEW_REQUIRED; callers needing stricter gating should call run_contract_approval
    directly with their own decisions instead of this pipeline.
    """
    return {
        table_name: {col: cand["semantic_type"] for col, cand in columns.items()}
        for table_name, columns in candidates_by_table.items()
    }


def _write_domain_mapping(
    manifest: RunManifest, dataset_contract: dict[str, Any], domain_model: CanonicalDomainModel
) -> dict[str, Any]:
    mapping = map_contract_to_domain(dataset_contract, domain_model)
    unmapped = sorted(set(dataset_contract.get("tables", {})) - set(mapping))
    if unmapped:
        logger.warning(
            "tables with no canonical domain entity mapping: %s", {"run_id": manifest.run_id, "tables": unmapped}
        )
    payload = {"tables": mapping, "unmapped_tables": unmapped}
    with manifest.output_path(DOMAIN_MAPPING_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def run_database_twin_pipeline(
    adapter: SourceAdapter, manifest: RunManifest, config: DatabaseTwinPipelineConfig
) -> WorkflowContext:
    """Run the full unattended Database Twin track through WorkflowOrchestrator.

    Returns the WorkflowContext whose ``artifacts`` dict holds every intermediate
    result (discovery_data, profile_data, candidates_data, dataset_contract,
    training_report, relational_report, qa_report, domain_mapping) so callers can
    inspect the outcome without re-reading every file back from disk. Stops at the
    first failed stage, per WorkflowOrchestrator.run()'s existing contract.
    """
    context = WorkflowContext(manifest=manifest, track=WorkflowTrack.DATABASE, artifacts={})

    def discovery_handler(ctx: WorkflowContext) -> StageResult:
        result = run_discovery(adapter, ctx.manifest, config.config_path)
        if result.is_success():
            ctx.artifacts["discovery_data"] = json.loads((ctx.manifest.run_dir / "discovery.json").read_text())
        return result

    def profiling_handler(ctx: WorkflowContext) -> StageResult:
        result = run_profiling(
            adapter,
            ctx.artifacts["discovery_data"],
            ctx.manifest,
            "discovery.json",
            sample_limit=config.sample_limit,
            chunk_size=config.chunk_size,
        )
        if result.is_success():
            ctx.artifacts["profile_data"] = load_profile(ctx.manifest.output_path(PROFILE_FILENAME))
        return result

    def inference_handler(ctx: WorkflowContext) -> StageResult:
        result = run_inference(
            adapter,
            ctx.artifacts["discovery_data"],
            ctx.artifacts["profile_data"],
            ctx.manifest,
            "discovery.json",
            "profile.json",
            sample_limit=config.sample_limit,
            chunk_size=config.chunk_size,
        )
        if result.is_success():
            ctx.artifacts["candidates_data"] = json.loads(Path(result.output_references[0]).read_text())
        return result

    def contract_approval_handler(ctx: WorkflowContext) -> StageResult:
        candidates_data = ctx.artifacts["candidates_data"]
        result = run_contract_approval(
            dataset_id=config.dataset_id,
            source_fingerprint=ctx.artifacts["discovery_data"]["source_fingerprint"],
            discovery_data=ctx.artifacts["discovery_data"],
            candidates_by_table=candidates_data["tables"],
            manifest=ctx.manifest,
            metadata_dir=config.metadata_dir,
            candidates_reference="semantic_candidates.json",
            decisions=_auto_approve_decisions(candidates_data["tables"]),
        )
        if result.is_success():
            contract = load_dataset_contract(config.metadata_dir)
            ctx.artifacts["dataset_contract"] = contract
            if config.domain_model is not None:
                ctx.artifacts["domain_mapping"] = _write_domain_mapping(ctx.manifest, contract, config.domain_model)
        return result

    def cleaning_handler(ctx: WorkflowContext) -> StageResult:
        return run_cleaning(
            adapter,
            ctx.artifacts["dataset_contract"],
            ctx.manifest,
            "dataset_contract.json",
            sample_limit=config.sample_limit,
            chunk_size=config.chunk_size,
        )

    def training_handler(ctx: WorkflowContext) -> StageResult:
        result = run_training_and_sampling(
            adapter,
            ctx.artifacts["dataset_contract"],
            ctx.manifest,
            "dataset_contract.json",
            sample_limit=config.sample_limit,
            num_rows_to_generate=config.num_rows_to_generate,
            seed=config.seed,
            model_type=config.model_type,
        )
        if result.is_success():
            training_report = load_training_report(Path(result.output_references[0]))
            ctx.artifacts["training_report"] = training_report
            ctx.artifacts["adapters_by_table"] = {
                table_name: get_synthesizer_adapter_class(table_report["model_type"]).load(table_report["model_path"])
                for table_name, table_report in training_report["tables"].items()
            }
        return result

    def artifact_export_handler(ctx: WorkflowContext) -> StageResult:
        return run_artifact_export(
            dataset_id=config.dataset_id,
            artifact_version=config.artifact_version,
            discovery_data=ctx.artifacts["discovery_data"],
            dataset_contract=ctx.artifacts["dataset_contract"],
            reference_profile=ctx.artifacts["profile_data"],
            training_report=ctx.artifacts["training_report"],
            training_run_id=ctx.manifest.run_id,
            code_version=ctx.manifest.code_version,
            manifest=ctx.manifest,
            training_reference="training_report.json",
        )

    def relational_generation_handler(ctx: WorkflowContext) -> StageResult:
        row_counts = resolve_database_row_counts(
            ctx.artifacts["dataset_contract"],
            config.row_counts_by_table,
            product_settings=config.product_settings,
        )
        result = run_relational_generation(
            ctx.artifacts["dataset_contract"],
            ctx.artifacts["adapters_by_table"],
            row_counts,
            ctx.manifest,
            contract_reference="dataset_contract.json",
            seed=config.seed,
        )
        if result.is_success():
            ctx.artifacts["relational_report"] = load_relational_generation_report(
                ctx.manifest.output_path(RELATIONAL_REPORT_FILENAME)
            )
        return result

    def qa_validation_handler(ctx: WorkflowContext) -> StageResult:
        kwargs: dict[str, Any] = {}
        if config.release_mode is not None:
            kwargs["release_mode"] = config.release_mode
        result = run_qa_validation(
            ctx.artifacts["dataset_contract"],
            ctx.artifacts["relational_report"],
            ctx.manifest,
            relational_report_reference="relational_generation_report.json",
            reference_profile=ctx.artifacts.get("profile_data"),
            **kwargs,
        )
        if result.is_success():
            ctx.artifacts["qa_report"] = load_qa_report(ctx.manifest.output_path(QA_REPORT_FILENAME))
        return result

    def target_write_handler(ctx: WorkflowContext) -> StageResult:
        return run_target_write(
            config.target_db_path,
            ctx.artifacts["dataset_contract"],
            ctx.artifacts["relational_report"],
            ctx.artifacts["qa_report"],
            ctx.manifest,
            qa_report_reference="qa_report.json",
        )

    orchestrator = build_database_workflow(
        {
            "discovery": discovery_handler,
            "profiling": profiling_handler,
            "inference": inference_handler,
            "contract_approval": contract_approval_handler,
            "cleaning": cleaning_handler,
            "training": training_handler,
            "artifact_export": artifact_export_handler,
            "relational_generation": relational_generation_handler,
            "qa_validation": qa_validation_handler,
            "target_write": target_write_handler,
        }
    )
    orchestrator.run(context)
    return context
