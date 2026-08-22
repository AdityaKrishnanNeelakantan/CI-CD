"""The in-memory trained-model bundle serialized into a .synthpkg."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from synth_platform.domain.artifacts.manifest import ArtifactManifest
from synth_platform.domain.constraints.models import ConstraintSet
from synth_platform.domain.documents.models import DocumentTemplateIR
from synth_platform.domain.planning.models import LearningPlan
from synth_platform.domain.privacy.models import PrivacyPolicy
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.domain.relational.models import RelationalPlan
from synth_platform.domain.schema.models import DatabaseSchema


class SynthArtifact(BaseModel):
    """Source-free: contains no credentials, URLs, original text, or raw rows."""
    model_config = ConfigDict(extra="forbid")

    manifest: ArtifactManifest
    schema_: DatabaseSchema
    relational_plan: RelationalPlan
    data_profile: DataProfile
    learning_plan: LearningPlan
    privacy_policy: PrivacyPolicy
    constraints: ConstraintSet = ConstraintSet()
    document_template: DocumentTemplateIR | None = None

    MEMBERS: ClassVar[dict[str, str]] = {
        "schema.json": "schema_",
        "relational_plan.json": "relational_plan",
        "semantic_profile.json": "data_profile",
        "learning_plan.json": "learning_plan",
        "privacy_policy.json": "privacy_policy",
        "constraints.json": "constraints",
    }
