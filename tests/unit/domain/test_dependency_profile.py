from __future__ import annotations

import pytest

from synth_platform.domain.contracts.dependency_adapter import dependency_profile_to_contract_edges
from synth_platform.domain.semantics.dependencies import (
    DependencyEvidenceItem,
    DependencyKind,
    DependencyProfile,
    DependencyStatus,
    FieldRef,
    LearnedDependency,
)

pytestmark = pytest.mark.unit


def test_dependency_profile_filters_approved_and_table_membership():
    profile = DependencyProfile(
        dependencies=[
            LearnedDependency(
                dependency_id="dep:1",
                kind=DependencyKind.CONDITIONAL,
                source_fields=[FieldRef(table="accounts", column="account_type")],
                target_fields=[FieldRef(table="accounts", column="balance")],
                confidence=0.8,
                status=DependencyStatus.APPROVED,
            ),
            LearnedDependency(
                dependency_id="dep:2",
                kind=DependencyKind.CROSS_TABLE,
                source_fields=[FieldRef(table="customers", column="region")],
                target_fields=[FieldRef(table="accounts", column="balance")],
                confidence=0.6,
            ),
        ]
    )

    assert [dep.dependency_id for dep in profile.approved()] == ["dep:1"]
    assert {dep.dependency_id for dep in profile.for_table("accounts")} == {"dep:1", "dep:2"}
    assert {dep.dependency_id for dep in profile.for_table("customers")} == {"dep:2"}


def test_dependency_profile_adapts_to_canonical_edges():
    profile = DependencyProfile(
        dependencies=[
            LearnedDependency(
                dependency_id="dep:balance",
                kind=DependencyKind.MUTUAL_INFORMATION,
                source_fields=[FieldRef(table="accounts", column="account_type")],
                target_fields=[FieldRef(table="accounts", column="balance")],
                confidence=0.72,
                evidence=[DependencyEvidenceItem(method="mutual_information", score=0.72)],
            )
        ]
    )

    edge = dependency_profile_to_contract_edges(profile)[0]

    assert edge.dependency_id == "dep:balance"
    assert edge.dependency_type == "mutual_information"
    assert edge.source_field_ids == ["field:accounts.account_type"]
    assert edge.target_field_ids == ["field:accounts.balance"]
    assert edge.confidence == 0.72
