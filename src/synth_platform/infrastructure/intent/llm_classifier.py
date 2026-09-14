"""Optional LLM workflow classifier abstraction.

No provider is wired here yet. This module documents the provider boundary so
LLM routing can be added without making the API depend on a specific SDK.
"""

from __future__ import annotations

from synth_platform.application.dto.intent import WorkflowIntentRequest, WorkflowIntentResponse


SYSTEM_PROMPT = (
    "You are a workflow router for Synthetic Data Twin. Choose exactly one workflow from "
    "schema_twin, database_twin, document_twin, interaction_twin, or unknown. Use only "
    "supported platform capabilities. Do not claim unsupported functionality is available. "
    "Return strict JSON matching the provided schema."
)


class UnconfiguredLlmIntentClassifier:
    def classify(self, request: WorkflowIntentRequest) -> WorkflowIntentResponse | None:
        return None
