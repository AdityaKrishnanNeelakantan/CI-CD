"""Strict domain contracts for one structured customer-interaction SSOT."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.domain.validation.models import ReleaseDecision, ValidationReport


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParticipantRole(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"
    UNKNOWN = "unknown"


class SentimentLabel(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    PENDING = "pending"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class TopicCode(StrEnum):
    ACCOUNT_ACCESS = "account_access"
    APPOINTMENT = "appointment"
    BILLING = "billing"
    CANCELLATION = "cancellation"
    DELIVERY = "delivery"
    FRAUD_SECURITY = "fraud_security"
    GENERAL_SUPPORT = "general_support"
    PAYMENT = "payment"
    REFUND = "refund"
    TECHNICAL_SUPPORT = "technical_support"


class IssueCode(StrEnum):
    ACCOUNT_LOCKED = "account_locked"
    BILLING_QUESTION = "billing_question"
    CANCELLATION_REQUEST = "cancellation_request"
    DELIVERY_DELAY = "delivery_delay"
    FRAUD_CONCERN = "fraud_concern"
    LOGIN_PROBLEM = "login_problem"
    PAYMENT_FAILURE = "payment_failure"
    REFUND_REQUEST = "refund_request"
    SERVICE_PROBLEM = "service_problem"
    UNKNOWN = "unknown"


class ActionCode(StrEnum):
    ACCOUNT_UPDATED = "account_updated"
    CREDENTIAL_RESET = "credential_reset"
    ESCALATED = "escalated"
    FOLLOW_UP_SCHEDULED = "follow_up_scheduled"
    INSTRUCTIONS_PROVIDED = "instructions_provided"
    REFUND_INITIATED = "refund_initiated"
    SECURITY_REVIEW_OPENED = "security_review_opened"
    VERIFICATION_COMPLETED = "verification_completed"


class InteractionParticipant(_StrictModel):
    participant_id: str
    role: ParticipantRole
    synthetic_alias: str


class InteractionTurnPlan(_StrictModel):
    turn_id: str
    participant_id: str
    role: ParticipantRole
    sequence: int = Field(ge=1)
    timestamp_present: bool = False
    issue_codes: list[IssueCode] = Field(default_factory=list)
    action_codes: list[ActionCode] = Field(default_factory=list)
    sentiment: SentimentLabel = SentimentLabel.NEUTRAL


class SentimentTrajectory(_StrictModel):
    initial_customer_sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    final_customer_sentiment: SentimentLabel = SentimentLabel.NEUTRAL


class ExtractionMetadata(_StrictModel):
    mode: Literal["deterministic", "model_enhanced", "deterministic_fallback"]
    model_requested: bool = False
    model_used: bool = False
    model_name: str | None = None
    fallback_reason: str | None = None


class SemanticEnhancement(_StrictModel):
    """Closed model-output surface; no model-authored free text is accepted."""

    topics: list[TopicCode] = Field(default_factory=list)
    issue_codes: list[IssueCode] = Field(default_factory=list)
    action_codes: list[ActionCode] = Field(default_factory=list)
    resolution_status: ResolutionStatus = ResolutionStatus.UNKNOWN
    initial_customer_sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    final_customer_sentiment: SentimentLabel = SentimentLabel.NEUTRAL


class InteractionSSOT(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    interaction_id: str
    source_fingerprint: str
    locale: str
    participant_count: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    participants: list[InteractionParticipant]
    turn_plan: list[InteractionTurnPlan]
    topics: list[TopicCode]
    issue_codes: list[IssueCode]
    action_codes: list[ActionCode]
    resolution_status: ResolutionStatus
    sentiment: SentimentTrajectory
    summary: str
    extraction: ExtractionMetadata


class InteractionValidationReport(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    interaction_id: str
    validation: ValidationReport
    release: ReleaseDecision
    metrics: dict[str, int | float]
    limitations: list[str] = Field(default_factory=list)


class ArtifactRecord(_StrictModel):
    sha256: str
    size_bytes: int = Field(ge=0)


class InteractionArtifactManifest(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    package_type: Literal["customer_interaction_twin"] = "customer_interaction_twin"
    interaction_id: str
    run_id: str
    code_version: str
    created_at: str
    source_fingerprint: str
    source_media_type: Literal["text/plain", "text/log"]
    seed: int
    locale: str
    model_requested: bool
    model_used: bool
    model_name: str | None = None
    deterministic_fallback: bool
    validation_verdict: str
    artifacts: dict[str, ArtifactRecord]
