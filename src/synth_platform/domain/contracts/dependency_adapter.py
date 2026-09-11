"""Adapters from dependency profiles to canonical dependency edges."""
from __future__ import annotations

from synth_platform.domain.contracts.models import DependencyEdge
from synth_platform.domain.semantics.dependencies import DependencyProfile, LearnedDependency


def dependency_profile_to_contract_edges(profile: DependencyProfile) -> list[DependencyEdge]:
    return [_dependency_to_contract_edge(dep) for dep in profile.dependencies]


def _dependency_to_contract_edge(dep: LearnedDependency) -> DependencyEdge:
    return DependencyEdge(
        dependency_id=dep.dependency_id,
        dependency_type=dep.kind.value,
        source_field_ids=[ref.field_id for ref in dep.source_fields],
        target_field_ids=[ref.field_id for ref in dep.target_fields],
        confidence=dep.confidence,
        evidence=[
            item.description or f"{item.method}:{item.score}"
            for item in dep.evidence
        ],
    )
