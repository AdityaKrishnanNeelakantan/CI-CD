"""Workflow intent routing API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from synth_platform.infrastructure.intent.llm_classifier import UnconfiguredLlmIntentClassifier
from synth_platform.infrastructure.intent.rule_based_classifier import RuleBasedWorkflowIntentClassifier
from synth_platform.application.dto.intent import WorkflowIntentRequest
from synth_platform.interfaces.api.routes.shared import ok

router = APIRouter(prefix="/api/intent", tags=["intent"])


@router.post("/workflow")
def classify_workflow_intent(body: WorkflowIntentRequest) -> dict[str, Any]:
    rules = RuleBasedWorkflowIntentClassifier()
    rule_decision = rules.classify(body)

    llm_decision = UnconfiguredLlmIntentClassifier().classify(body)
    if llm_decision is None or llm_decision.confidence < 0.55:
        return ok(rule_decision.model_dump(mode="json"))
    return ok(llm_decision.model_dump(mode="json"))
