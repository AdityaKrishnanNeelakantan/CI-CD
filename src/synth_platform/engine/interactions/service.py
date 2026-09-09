"""Deterministic parsing, sanitization, SSOT construction, and validation.

Raw transcript text is accepted in memory and is never written by this module.
The implementation is deliberately rule-based and US/English-centric. Optional
model enhancement is orchestrated by the application workflow after this module
has produced a sanitized transcript.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TypeVar

from synth_platform.domain.interactions.models import (
    ActionCode,
    ExtractionMetadata,
    InteractionParticipant,
    InteractionSSOT,
    InteractionTurnPlan,
    InteractionValidationReport,
    IssueCode,
    ParticipantRole,
    ResolutionStatus,
    SemanticEnhancement,
    SentimentLabel,
    SentimentTrajectory,
    TopicCode,
)
from synth_platform.domain.validation.models import (
    CheckResult,
    Status,
    ValidationReport,
)
from synth_platform.domain.validation.release_gate import decide
from synth_platform.engine.documents.pdf.entities import extract_entities
from synth_platform.engine.documents.pdf.pii import detect_pii
from synth_platform.engine.interactions.patterns import (
    INTERACTION_PATTERNS,
    SENSITIVE_FIELD_LABELS,
)

_MAX_TRANSCRIPT_CHARACTERS = 1_000_000
_SPEAKER_LINE = re.compile(
    r"^(?:\[(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\]\s*)?"
    r"(?P<speaker>[^:\n]{1,64}):\s*(?P<text>.+)$"
)
_WORD = re.compile(r"[a-z][a-z0-9']{1,31}")
_GENERIC_SPEAKERS = {
    "agent",
    "advisor",
    "caller",
    "client",
    "customer",
    "employee",
    "member",
    "participant",
    "representative",
    "rep",
    "support",
    "unknown",
    "user",
}
_AGENT_MARKERS = {"agent", "advisor", "employee", "representative", "rep", "support"}
_CUSTOMER_MARKERS = {"caller", "client", "customer", "member", "user"}
_POSITIVE_WORDS = {
    "appreciate",
    "excellent",
    "fixed",
    "glad",
    "good",
    "great",
    "happy",
    "helpful",
    "resolved",
    "thanks",
    "thank",
    "working",
}
_NEGATIVE_WORDS = {
    "angry",
    "broken",
    "complaint",
    "failed",
    "frustrated",
    "issue",
    "late",
    "locked",
    "missing",
    "problem",
    "stolen",
    "unable",
    "wrong",
}
_TOPIC_TERMS: dict[TopicCode, set[str]] = {
    TopicCode.ACCOUNT_ACCESS: {"account", "login", "locked", "password", "access"},
    TopicCode.APPOINTMENT: {"appointment", "schedule", "booking"},
    TopicCode.BILLING: {"bill", "billing", "charge", "invoice"},
    TopicCode.CANCELLATION: {"cancel", "cancellation", "close"},
    TopicCode.DELIVERY: {"delivery", "shipment", "package", "tracking"},
    TopicCode.FRAUD_SECURITY: {"fraud", "stolen", "security", "unauthorized"},
    TopicCode.PAYMENT: {"payment", "card", "pay", "transaction"},
    TopicCode.REFUND: {"refund", "reimbursement", "reimburse"},
    TopicCode.TECHNICAL_SUPPORT: {"error", "technical", "device", "application", "app"},
}
_ISSUE_TERMS: dict[IssueCode, set[str]] = {
    IssueCode.ACCOUNT_LOCKED: {"locked", "lockout"},
    IssueCode.BILLING_QUESTION: {"bill", "billing", "charge", "invoice"},
    IssueCode.CANCELLATION_REQUEST: {"cancel", "cancellation", "close"},
    IssueCode.DELIVERY_DELAY: {"late", "delay", "delivery", "missing"},
    IssueCode.FRAUD_CONCERN: {"fraud", "stolen", "unauthorized"},
    IssueCode.LOGIN_PROBLEM: {"login", "password", "signin", "access"},
    IssueCode.PAYMENT_FAILURE: {"payment", "declined", "transaction", "failed"},
    IssueCode.REFUND_REQUEST: {"refund", "reimbursement", "reimburse"},
    IssueCode.SERVICE_PROBLEM: {"problem", "issue", "broken", "error"},
}
_ACTION_PHRASES: dict[ActionCode, tuple[str, ...]] = {
    ActionCode.ACCOUNT_UPDATED: (
        "updated the account",
        "account updated",
        "changed the account",
    ),
    ActionCode.CREDENTIAL_RESET: (
        "reset the password",
        "password reset",
        "reset your pin",
    ),
    ActionCode.ESCALATED: ("escalated", "escalate", "supervisor"),
    ActionCode.FOLLOW_UP_SCHEDULED: ("follow up", "follow-up", "call you back"),
    ActionCode.INSTRUCTIONS_PROVIDED: ("steps to", "instructions", "please try"),
    ActionCode.REFUND_INITIATED: (
        "refund issued",
        "refund initiated",
        "processed the refund",
    ),
    ActionCode.SECURITY_REVIEW_OPENED: (
        "security review",
        "fraud review",
        "investigation opened",
    ),
    ActionCode.VERIFICATION_COMPLETED: (
        "verified",
        "verification complete",
        "identity confirmed",
    ),
}
_RESOLVED_PHRASES = (
    "resolved",
    "fixed",
    "completed",
    "working now",
    "refund issued",
    "successfully reset",
)
_PENDING_PHRASES = ("follow up", "escalated", "pending", "under review", "call back")
_T = TypeVar("_T")


@dataclass(frozen=True)
class ParsedTurn:
    raw_speaker: str
    raw_text: str
    timestamp: str | None


@dataclass(frozen=True)
class SanitizedTurn:
    turn_id: str
    participant_id: str
    role: ParticipantRole
    sequence: int
    timestamp_present: bool
    text: str
    raw_text: str


@dataclass(frozen=True)
class SanitizedTranscript:
    interaction_id: str
    source_fingerprint: str
    participants: tuple[InteractionParticipant, ...]
    turns: tuple[SanitizedTurn, ...]
    sensitive_values: tuple[str, ...]
    raw_speaker_labels: tuple[str, ...]
    finding_counts: dict[str, int]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _SensitiveSpan:
    kind: str
    start: int
    end: int


def _normalise_source(source_text: str) -> str:
    text = source_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    if "\x00" in text:
        raise ValueError("Transcript must be plain UTF-8 text, not binary data.")
    if len(text) > _MAX_TRANSCRIPT_CHARACTERS:
        raise ValueError("Transcript exceeds the 1,000,000-character safety limit.")
    if not text.strip():
        raise ValueError("Transcript is empty.")
    return text


def _parse_turns(source_text: str) -> list[ParsedTurn]:
    parsed: list[ParsedTurn] = []
    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SPEAKER_LINE.match(line)
        if match:
            speaker = match.group("speaker").strip()
            if (
                speaker.casefold() not in SENSITIVE_FIELD_LABELS
                and len(speaker.split()) <= 8
                and not speaker.lower().startswith(("http", "www"))
            ):
                parsed.append(
                    ParsedTurn(
                        raw_speaker=speaker,
                        raw_text=match.group("text").strip(),
                        timestamp=match.group("timestamp"),
                    )
                )
                continue
        if parsed:
            previous = parsed[-1]
            parsed[-1] = replace(previous, raw_text=f"{previous.raw_text} {line}")
        else:
            parsed.append(ParsedTurn("unknown", line, None))
    if not parsed:
        raise ValueError("Transcript contains no readable turns.")
    return parsed


def _role_for_speaker(label: str) -> ParticipantRole:
    words = set(_WORD.findall(label.lower()))
    if words & _AGENT_MARKERS:
        return ParticipantRole.AGENT
    if words & _CUSTOMER_MARKERS:
        return ParticipantRole.CUSTOMER
    return ParticipantRole.UNKNOWN


def _collect_sensitive_spans(text: str) -> list[_SensitiveSpan]:
    candidates: list[_SensitiveSpan] = []
    for finding in detect_pii(text):
        candidates.append(
            _SensitiveSpan(
                str(finding["type"]), int(finding["start"]), int(finding["end"])
            )
        )
    for kind, pattern in INTERACTION_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span(1)
            candidates.append(_SensitiveSpan(kind, start, end))
    for finding in extract_entities(text):
        candidates.append(
            _SensitiveSpan(
                str(finding["type"]).lower(),
                int(finding["start"]),
                int(finding["end"]),
            )
        )

    accepted: list[_SensitiveSpan] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.start, -(item.end - item.start))
    ):
        if any(
            candidate.start < item.end and candidate.end > item.start
            for item in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: item.start)


def _sanitize_text(
    text: str,
    token_by_value: dict[tuple[str, str], str],
    counters: dict[str, int],
) -> tuple[str, list[str], dict[str, int]]:
    spans = _collect_sensitive_spans(text)
    sensitive_values: list[str] = []
    counts: dict[str, int] = {}
    output: list[str] = []
    cursor = 0
    for span in spans:
        value = text[span.start : span.end]
        key = (span.kind, value.casefold())
        token = token_by_value.get(key)
        if token is None:
            counters[span.kind] = counters.get(span.kind, 0) + 1
            token = f"[{span.kind.upper()}_{counters[span.kind]:03d}]"
            token_by_value[key] = token
        output.append(text[cursor : span.start])
        output.append(token)
        cursor = span.end
        sensitive_values.append(value)
        counts[span.kind] = counts.get(span.kind, 0) + 1
    output.append(text[cursor:])
    return "".join(output), sensitive_values, counts


def parse_and_sanitize_transcript(source_text: str) -> SanitizedTranscript:
    """Parse and sanitize one transcript without persisting its raw content."""

    normalized = _normalise_source(source_text)
    parsed = _parse_turns(normalized)
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    interaction_id = f"interaction_{fingerprint[:12]}"

    participant_by_label: dict[str, InteractionParticipant] = {}
    token_by_value: dict[tuple[str, str], str] = {}
    token_counters: dict[str, int] = {}
    all_sensitive: list[str] = []
    finding_counts: dict[str, int] = {}
    sanitized_turns: list[SanitizedTurn] = []

    for sequence, turn in enumerate(parsed, start=1):
        speaker_key = turn.raw_speaker.casefold().strip()
        participant = participant_by_label.get(speaker_key)
        if participant is None:
            role = _role_for_speaker(turn.raw_speaker)
            participant_id = f"participant_{len(participant_by_label) + 1:03d}"
            role_index = (
                sum(1 for item in participant_by_label.values() if item.role == role)
                + 1
            )
            participant = InteractionParticipant(
                participant_id=participant_id,
                role=role,
                synthetic_alias=f"Synthetic {role.value.title()} {role_index}",
            )
            participant_by_label[speaker_key] = participant

        sanitized, sensitive_values, counts = _sanitize_text(
            turn.raw_text,
            token_by_value,
            token_counters,
        )
        all_sensitive.extend(sensitive_values)
        for kind, count in counts.items():
            finding_counts[kind] = finding_counts.get(kind, 0) + count
        sanitized_turns.append(
            SanitizedTurn(
                turn_id=f"turn_{sequence:04d}",
                participant_id=participant.participant_id,
                role=participant.role,
                sequence=sequence,
                timestamp_present=turn.timestamp is not None,
                text=sanitized,
                raw_text=turn.raw_text,
            )
        )

    unknown_ratio = sum(
        turn.role == ParticipantRole.UNKNOWN for turn in sanitized_turns
    ) / len(sanitized_turns)
    warnings = [
        "PII and entity detection is deterministic and US/English-centric; unsupported formats may require review."
    ]
    if unknown_ratio > 0.5:
        warnings.append("More than half of turns use unknown participant roles.")

    return SanitizedTranscript(
        interaction_id=interaction_id,
        source_fingerprint=fingerprint,
        participants=tuple(participant_by_label.values()),
        turns=tuple(sanitized_turns),
        sensitive_values=tuple(dict.fromkeys(all_sensitive)),
        raw_speaker_labels=tuple(
            label
            for label in (turn.raw_speaker for turn in parsed)
            if label.casefold() not in _GENERIC_SPEAKERS
        ),
        finding_counts=finding_counts,
        warnings=tuple(warnings),
    )


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _codes_for_words(words: set[str], mapping: dict[_T, set[str]]) -> list[_T]:
    return [code for code, terms in mapping.items() if words & terms]


def _actions_for_text(text: str) -> list[ActionCode]:
    lowered = text.lower()
    return [
        code
        for code, phrases in _ACTION_PHRASES.items()
        if any(item in lowered for item in phrases)
    ]


def _sentiment(text: str) -> SentimentLabel:
    words = _words(text)
    score = len(words & _POSITIVE_WORDS) - len(words & _NEGATIVE_WORDS)
    if score > 0:
        return SentimentLabel.POSITIVE
    if score < 0:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def deterministic_semantics(transcript: SanitizedTranscript) -> SemanticEnhancement:
    combined = " ".join(turn.text for turn in transcript.turns)
    words = _words(combined)
    topics = _codes_for_words(words, _TOPIC_TERMS) or [TopicCode.GENERAL_SUPPORT]
    issues = _codes_for_words(words, _ISSUE_TERMS) or [IssueCode.UNKNOWN]
    actions = _actions_for_text(combined)
    lowered = combined.lower()
    if any(phrase in lowered for phrase in _RESOLVED_PHRASES):
        resolution = ResolutionStatus.RESOLVED
    elif any(phrase in lowered for phrase in _PENDING_PHRASES):
        resolution = ResolutionStatus.PENDING
    elif issues != [IssueCode.UNKNOWN]:
        resolution = ResolutionStatus.UNRESOLVED
    else:
        resolution = ResolutionStatus.UNKNOWN

    customer_turns = [
        turn for turn in transcript.turns if turn.role == ParticipantRole.CUSTOMER
    ]
    first_customer = (
        customer_turns[0].text if customer_turns else transcript.turns[0].text
    )
    final_customer = (
        customer_turns[-1].text if customer_turns else transcript.turns[-1].text
    )
    return SemanticEnhancement(
        topics=topics,
        issue_codes=issues,
        action_codes=actions,
        resolution_status=resolution,
        initial_customer_sentiment=_sentiment(first_customer),
        final_customer_sentiment=_sentiment(final_customer),
    )


def _deduplicate(values: Iterable[_T]) -> list[_T]:
    return list(dict.fromkeys(values))


def _summary(semantic: SemanticEnhancement) -> str:
    topics = ", ".join(code.value.replace("_", " ") for code in semantic.topics)
    actions = ", ".join(code.value.replace("_", " ") for code in semantic.action_codes)
    action_clause = actions if actions else "no explicit completed action"
    return (
        f"Synthetic interaction record covering {topics}. "
        f"Outcome: {semantic.resolution_status.value}. Actions: {action_clause}."
    )


def build_interaction_ssot(
    transcript: SanitizedTranscript,
    *,
    locale: str = "en_US",
    enhancement: SemanticEnhancement | None = None,
    extraction: ExtractionMetadata | None = None,
) -> InteractionSSOT:
    semantic = enhancement or deterministic_semantics(transcript)
    semantic = semantic.model_copy(
        update={
            "topics": _deduplicate(semantic.topics) or [TopicCode.GENERAL_SUPPORT],
            "issue_codes": _deduplicate(semantic.issue_codes) or [IssueCode.UNKNOWN],
            "action_codes": _deduplicate(semantic.action_codes),
        }
    )

    turn_plan: list[InteractionTurnPlan] = []
    for turn in transcript.turns:
        turn_words = _words(turn.text)
        turn_plan.append(
            InteractionTurnPlan(
                turn_id=turn.turn_id,
                participant_id=turn.participant_id,
                role=turn.role,
                sequence=turn.sequence,
                timestamp_present=turn.timestamp_present,
                issue_codes=_codes_for_words(turn_words, _ISSUE_TERMS),
                action_codes=_actions_for_text(turn.text),
                sentiment=_sentiment(turn.text),
            )
        )

    return InteractionSSOT(
        interaction_id=transcript.interaction_id,
        source_fingerprint=transcript.source_fingerprint,
        locale=locale,
        participant_count=len(transcript.participants),
        turn_count=len(transcript.turns),
        participants=list(transcript.participants),
        turn_plan=turn_plan,
        topics=semantic.topics,
        issue_codes=semantic.issue_codes,
        action_codes=semantic.action_codes,
        resolution_status=semantic.resolution_status,
        sentiment=SentimentTrajectory(
            initial_customer_sentiment=semantic.initial_customer_sentiment,
            final_customer_sentiment=semantic.final_customer_sentiment,
        ),
        summary=_summary(semantic),
        extraction=extraction
        or ExtractionMetadata(
            mode="deterministic",
            model_requested=False,
            model_used=False,
        ),
    )


def render_sanitized_source(transcript: SanitizedTranscript) -> str:
    return "\n".join(
        f"[{turn.turn_id}] {turn.participant_id}/{turn.role.value}: {turn.text}"
        for turn in transcript.turns
    )


def _normalise_for_replay(value: str) -> str:
    return " ".join(_WORD.findall(value.lower()))


def _source_ngrams(source: str, size: int = 8) -> set[tuple[str, ...]]:
    words = _WORD.findall(source.lower())
    return {
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    }


def _status_check(
    name: str,
    dimension: str,
    passed: bool,
    metric: float,
    threshold: float,
    detail: str,
    *,
    critical: bool = True,
) -> CheckResult:
    return CheckResult(
        name=name,
        dimension=dimension,
        status=Status.PASS if passed else Status.FAIL,
        ran=True,
        critical=critical,
        required=critical,
        metric=float(metric),
        threshold=float(threshold),
        detail=detail,
    )


def _source_facts_match(
    transcript: SanitizedTranscript,
    ssot: InteractionSSOT,
) -> bool:
    participant_ids = [participant.participant_id for participant in ssot.participants]
    if len(participant_ids) != len(set(participant_ids)):
        return False
    if not (
        ssot.participant_count == len(ssot.participants) == len(transcript.participants)
        and ssot.turn_count == len(ssot.turn_plan) == len(transcript.turns)
        and ssot.participants == list(transcript.participants)
    ):
        return False
    return all(
        planned.turn_id == source.turn_id
        and planned.participant_id == source.participant_id
        and planned.role == source.role
        and planned.sequence == source.sequence
        and planned.timestamp_present == source.timestamp_present
        for planned, source in zip(ssot.turn_plan, transcript.turns, strict=True)
    )


def _contains_source_value(persisted_text: str, value: str) -> bool:
    folded = value.casefold()
    if not folded:
        return False
    if folded[0].isalnum() and folded[-1].isalnum():
        return (
            re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", persisted_text) is not None
        )
    return folded in persisted_text


def validate_interaction_ssot(
    transcript: SanitizedTranscript,
    ssot: InteractionSSOT,
) -> InteractionValidationReport:
    sanitized_source = render_sanitized_source(transcript)
    ssot_json = ssot.model_dump_json()
    persisted_text = f"{sanitized_source}\n{ssot_json}".casefold()

    residual_pii = detect_pii(sanitized_source)
    residual_entities = extract_entities(sanitized_source)
    residual_interaction = sum(
        1
        for _kind, pattern in INTERACTION_PATTERNS
        for _ in pattern.finditer(sanitized_source)
    )
    residual_count = len(residual_pii) + len(residual_entities) + residual_interaction
    sensitive_leaks = sum(
        1
        for value in transcript.sensitive_values
        if _contains_source_value(persisted_text, value)
    )
    speaker_leaks = sum(
        1
        for value in transcript.raw_speaker_labels
        if _contains_source_value(persisted_text, value)
    )
    replay_count = sum(
        1
        for turn in transcript.turns
        if len(_normalise_for_replay(turn.raw_text).split()) >= 5
        and _normalise_for_replay(turn.raw_text) in _normalise_for_replay(ssot_json)
    )
    source_ngrams = _source_ngrams(" ".join(turn.raw_text for turn in transcript.turns))
    ssot_ngrams = _source_ngrams(ssot.summary)
    overlap_count = len(source_ngrams & ssot_ngrams)
    overlap_ratio = overlap_count / max(1, len(source_ngrams))

    facts_valid = _source_facts_match(transcript, ssot)
    unknown_ratio = sum(
        turn.role == ParticipantRole.UNKNOWN for turn in transcript.turns
    ) / len(transcript.turns)

    checks = [
        _status_check(
            "ssot_schema_valid",
            "structural",
            True,
            1,
            1,
            "SSOT is a strict InteractionSSOT instance.",
        ),
        _status_check(
            "residual_pii_absent",
            "privacy",
            residual_count == 0,
            residual_count,
            0,
            "Sanitized source was rescanned for PII, entities, identifiers, and secrets.",
        ),
        _status_check(
            "source_sensitive_span_absent",
            "privacy",
            sensitive_leaks == 0,
            sensitive_leaks,
            0,
            "Detected source values must not appear in persisted artifacts.",
        ),
        _status_check(
            "raw_speaker_label_absent",
            "privacy",
            speaker_leaks == 0,
            speaker_leaks,
            0,
            "Non-generic source speaker labels must not appear in persisted artifacts.",
        ),
        _status_check(
            "exact_turn_replay_absent",
            "privacy",
            replay_count == 0,
            replay_count,
            0,
            "Five-word-or-longer source turns must not be reproduced in the SSOT.",
        ),
        _status_check(
            "long_source_overlap_absent",
            "privacy",
            overlap_count == 0,
            overlap_ratio,
            0,
            "SSOT summary must not share an eight-word sequence with source turns.",
        ),
        _status_check(
            "participant_reference_integrity",
            "structural",
            facts_valid,
            1 if facts_valid else 0,
            1,
            "Participant references and source-derived counts must remain consistent.",
        ),
    ]
    if unknown_ratio > 0:
        checks.append(
            CheckResult(
                name="unknown_participant_ratio",
                dimension="content",
                status=Status.WARN,
                ran=True,
                critical=False,
                required=False,
                metric=unknown_ratio,
                threshold=0.5,
                detail="Unknown roles are retained rather than guessed.",
            )
        )

    validation = ValidationReport(checks=checks)
    release = decide(validation)
    metrics: dict[str, int | float] = {
        "turn_count": len(transcript.turns),
        "participant_count": len(transcript.participants),
        "detected_sensitive_value_count": sum(transcript.finding_counts.values()),
        "residual_pii_count": residual_count,
        "source_sensitive_span_leak_count": sensitive_leaks,
        "raw_speaker_leak_count": speaker_leaks,
        "exact_turn_replay_count": replay_count,
        "source_8gram_overlap_count": overlap_count,
        "source_8gram_overlap_ratio": overlap_ratio,
        "unknown_participant_ratio": unknown_ratio,
    }
    return InteractionValidationReport(
        interaction_id=ssot.interaction_id,
        validation=validation,
        release=release,
        metrics=metrics,
        limitations=list(transcript.warnings),
    )
