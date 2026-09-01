"""Shared compiler/trainer for canonical RelationalDataset inputs."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.application.use_cases.compile_learning_plan import compile_learning_plan
from synth_platform.application.use_cases.materialize_connector import materialize_connector
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.artifacts.manifest import ArtifactManifest
from synth_platform.domain.constraints.compiler import compile_constraints
from synth_platform.domain.constraints.models import (
    ConstraintDefinition, ConstraintKind,
)
from synth_platform.domain.contracts.schema_adapter import database_schema_to_contract
from synth_platform.domain.contracts.dependency_adapter import dependency_profile_to_contract_edges
from synth_platform.domain.contracts.serialization import canonical_fingerprint
from synth_platform.domain.entities.assembler import entity_graph_from_contract
from synth_platform.domain.documents.models import DocumentTemplateIR
from synth_platform.domain.privacy.policies import build_policy
from synth_platform.domain.privacy.redaction import redact_profile
from synth_platform.domain.privacy.verifier import verify_source_free
from synth_platform.domain.profiling.models import SamplingRecord
from synth_platform.domain.relational.cycles import enforce_cycle_policy
from synth_platform.domain.relational.dag import build_graph_plan
from synth_platform.domain.relational.models import GraphEdge, RelationalPlan
from synth_platform.domain.runs.models import SourceKind
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.domain.schema.source import SourceConnector
from synth_platform.domain.semantics.dependencies import DependencyProfile
from synth_platform.engine.profiling.service import ProfilingService
from synth_platform.engine.inference.platform.cross_table import compile_cross_table_conditionals


def fingerprint_schema(schema: DatabaseSchema) -> str:
    """Stable schema-only digest. Never includes source URLs, credentials, or rows."""
    payload = schema.model_dump(mode="json")
    # Source counts can legitimately change without changing the schema contract.
    for table in payload.get("tables", {}).values():
        table.pop("row_count", None)
        table.pop("estimated_row_count", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def train_relational_dataset(
    dataset: RelationalDataset,
    *,
    source_kind: SourceKind,
    source_fingerprint: str,
    sample_records: dict[str, SamplingRecord],
    seed: int,
    rare_threshold: int,
    source_run_id: str | None = None,
    confirmed_rules: list[ConstraintDefinition] | None = None,
    document_template: DocumentTemplateIR | None = None,
) -> SynthArtifact:
    """Compile one isolated source dataset into one source-specific artifact."""
    dataset.validate()
    unknown_frames = set(dataset.tables) - set(dataset.schema.tables)
    if unknown_frames:
        raise ValueError(f"dataset has undeclared tables: {sorted(unknown_frames)}")

    profile, raw_sensitive = ProfilingService().run_dataset(dataset.schema, dataset.tables, sample_records)
    policy = build_policy(profile, rare_threshold)
    profile = redact_profile(profile, policy)
    verify_source_free(profile, policy, raw_sensitive)
    learning_plan = compile_learning_plan(profile)
    effective_rules = list(confirmed_rules or [])
    if document_template is not None:
        for calculation in document_template.calculations:
            lhs = calculation.expression.split("=", 1)[0].strip() if "=" in calculation.expression else None
            table = next((name for name, frame in dataset.tables.items()
                          if lhs and lhs in frame.columns), next(iter(dataset.tables), ""))
            effective_rules.append(ConstraintDefinition(
                kind=ConstraintKind.ARITHMETIC, table=table, column=lhs,
                expression=calculation.expression, tolerance=calculation.tolerance,
                source="confirmed_document_template"))
    constraints = compile_constraints(dataset.schema, profile, effective_rules)
    cross_table_conditionals = compile_cross_table_conditionals(
        dataset.schema, dataset.tables, profile)

    graph = build_graph_plan(list(dataset.schema.tables), dataset.schema.foreign_keys)
    enforce_cycle_policy(graph, allow_deferred=False)
    edges = [GraphEdge(
        parent_table=fk.parent_table,
        parent_column=fk.parent_column,
        child_table=fk.child_table,
        child_column=fk.child_column,
        confirmed=fk.confirmed,
    ) for fk in dataset.schema.foreign_keys]
    relational_plan = RelationalPlan(
        levels=graph.levels,
        edges=edges,
        self_references=graph.self_references,
        child_per_parent=profile.child_per_parent,
        cardinality_models=profile.cardinality_models,
        cross_table_conditionals=cross_table_conditionals,
    )

    run_id = source_run_id or uuid.uuid4().hex
    role = "document_relational" if source_kind == SourceKind.PDF else "relational"
    dependency_profile = DependencyProfile()
    canonical_contract = database_schema_to_contract(
        dataset.schema,
        contract_id=source_fingerprint,
        source_fingerprint=source_fingerprint,
        produced_by="train_relational_dataset",
    )
    canonical_contract.dependencies = dependency_profile_to_contract_edges(dependency_profile)
    entity_graph = entity_graph_from_contract(canonical_contract)
    contract_fingerprint = canonical_fingerprint(canonical_contract)
    manifest = ArtifactManifest(
        artifact_id="",
        created_at=datetime.now(timezone.utc).isoformat(),
        generation_order=graph.order,
        capabilities=[
            "source_free_generation", "relational_fk_integrity",
            "empirical_marginals", "learned_parent_child_cardinality",
            "compiled_constraints", "trusted_signature", "safetensors_weights",
        ] + (["synthetic_document_rendering"] if source_kind == SourceKind.PDF else []),
        limitations=[
            "no_formal_differential_privacy",
            "cross_table_conditioning_is_bounded_to_compiled_cardinality_context",
        ],
        source_kind=source_kind,
        source_run_id=run_id,
        source_schema_fingerprint=source_fingerprint,
        contract_version=canonical_contract.contract_version,
        contract_fingerprint=contract_fingerprint,
        source_fingerprint=source_fingerprint,
        artifact_role=role,
    )
    artifact = SynthArtifact(
        manifest=manifest,
        canonical_contract=canonical_contract,
        entity_graph=entity_graph,
        schema_=dataset.schema,
        relational_plan=relational_plan,
        data_profile=profile,
        dependency_profile=dependency_profile,
        learning_plan=learning_plan,
        privacy_policy=policy,
        constraints=constraints,
        document_template=document_template,
    )
    body = artifact.model_dump(exclude={"manifest"}, mode="json")
    identity = {
        "source_kind": source_kind.value,
        "source_run_id": run_id,
        "source_fingerprint": source_fingerprint,
        "seed": seed,
        "body": body,
    }
    artifact.manifest.artifact_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    return artifact


def train_model(
    connector: SourceConnector,
    sample_size: int,
    seed: int,
    rare_threshold: int = 10,
) -> SynthArtifact:
    """Compatibility wrapper for structured database connectors."""
    dataset, sampling = materialize_connector(connector, sample_size, seed)
    return train_relational_dataset(
        dataset,
        source_kind=SourceKind.DATABASE,
        source_fingerprint=fingerprint_schema(dataset.schema),
        sample_records=sampling,
        seed=seed,
        rare_threshold=rare_threshold,
    )
