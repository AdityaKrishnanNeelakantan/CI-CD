"""The in-memory trained-model bundle serialized into a .synthpkg."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from synth_platform.domain.artifacts.manifest import ArtifactManifest
from synth_platform.domain.constraints.models import ConstraintSet
from synth_platform.domain.contracts.models import CanonicalContract
from synth_platform.domain.documents.models import DocumentTemplateIR
from synth_platform.domain.entities.graph import EntityGraph
from synth_platform.domain.planning.models import LearningPlan
from synth_platform.domain.privacy.models import PrivacyPolicy
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.domain.relational.models import RelationalPlan
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.domain.semantics.dependencies import DependencyProfile


class SynthArtifact(BaseModel):
    """Source-free: contains no credentials, URLs, original text, or raw rows."""
    model_config = ConfigDict(extra="forbid")

    manifest: ArtifactManifest
    canonical_contract: CanonicalContract | None = None
    entity_graph: EntityGraph | None = None
    schema_: DatabaseSchema
    relational_plan: RelationalPlan
    data_profile: DataProfile
    dependency_profile: DependencyProfile | None = None
    learning_plan: LearningPlan
    privacy_policy: PrivacyPolicy
    constraints: ConstraintSet = ConstraintSet()
    document_template: DocumentTemplateIR | None = None

    MEMBERS: ClassVar[dict[str, str]] = {
        "canonical_contract.json": "canonical_contract",
        "entity_graph.json": "entity_graph",
        "schema.json": "schema_",
        "relational_plan.json": "relational_plan",
        "semantic_profile.json": "data_profile",
        "dependency_profile.json": "dependency_profile",
        "learning_plan.json": "learning_plan",
        "privacy_policy.json": "privacy_policy",
        "constraints.json": "constraints",
    }
