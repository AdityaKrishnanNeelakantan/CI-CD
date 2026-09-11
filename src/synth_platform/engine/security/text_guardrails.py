"""Deterministic, local guardrails for text sent to and returned by models.

The checks are deliberately inspectable and conservative. They are not claimed
as a comprehensive safety classifier. Reports contain only categories, counts,
and fixed policy messages; prompt fragments and matched sensitive values are
never included.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from synth_platform.domain.guardrails.models import (
    GuardedOutput,
    GuardedPrompt,
    GuardrailAction,
    GuardrailFinding,
    GuardrailPolicy,
    GuardrailReport,
    GuardrailStage,
)
from synth_platform.engine.documents.pdf.pii import detect_pii


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    category: str


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "secret",
        re.compile(
            r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password|passwd)"
            r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+\-=]{8,})"
        ),
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_.\-]{12,})"),
    ),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
)

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
    re.compile(r"(?i)\b(?:reveal|show|print|repeat|leak)\s+(?:the\s+)?(?:system|developer)\s+prompt\b"),
    re.compile(r"(?i)\b(?:act|behave)\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|developer)\b"),
    re.compile(r"(?i)<\|(?:system|assistant|developer|tool)\|>"),
    re.compile(r"(?i)\b(?:call|invoke|execute)\s+(?:a\s+)?tool\s+(?:to|for)\b"),
    re.compile(r"(?i)\b(?:bypass|disable|override)\s+(?:the\s+)?(?:guardrails?|safety|policy)\b"),
)

_CONTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "dangerous_instructions",
        re.compile(
            r"(?i)\b(?:how\s+to|steps?\s+to|instructions?\s+for)\b.{0,80}"
            r"\b(?:build|make|create|deploy)\b.{0,40}"
            r"\b(?:bomb|explosive|ransomware|malware|poison)\b"
        ),
    ),
    (
        "credential_abuse",
        re.compile(
            r"(?i)\b(?:steal|harvest|phish|exfiltrate)\b.{0,60}"
            r"\b(?:passwords?|credentials?|tokens?|accounts?)\b"
        ),
    ),
    (
        "self_harm_instructions",
        re.compile(
            r"(?i)\b(?:best|effective|painless|instructions?|steps?)\b.{0,60}"
            r"\b(?:suicide|kill myself|self[- ]harm)\b"
        ),
    ),
    (
        "child_sexual_abuse",
        re.compile(
            r"(?i)\b(?:sexual|explicit|nude|pornographic)\b.{0,40}"
            r"\b(?:child|minor|underage)\b"
        ),
    ),
)

_CHECKS = (
    "sensitive_data",
    "prompt_injection",
    "scope_validation",
    "content_safety",
)


def _action(findings: list[GuardrailFinding]) -> GuardrailAction:
    if any(item.action == GuardrailAction.BLOCK for item in findings):
        return GuardrailAction.BLOCK
    if findings:
        return GuardrailAction.WARN
    return GuardrailAction.ALLOW


def _finding(
    code: str,
    category: str,
    action: GuardrailAction,
    count: int,
    message: str,
) -> GuardrailFinding:
    return GuardrailFinding(
        code=code,
        category=category,
        action=action,
        count=count,
        message=message,
    )


def _sensitive_spans(text: str) -> list[_Span]:
    candidates = [
        _Span(int(item["start"]), int(item["end"]), str(item["type"]))
        for item in detect_pii(text)
    ]
    for category, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex:
                start, end = match.span(match.lastindex)
            else:
                start, end = match.span()
            candidates.append(_Span(start, end, category))

    accepted: list[_Span] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, -(item.end - item.start))):
        if any(
            candidate.start < existing.end and candidate.end > existing.start
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: item.start)


def mask_sensitive_text(text: str) -> tuple[str, Counter[str]]:
    spans = _sensitive_spans(text)
    if not spans:
        return text, Counter()
    counts: Counter[str] = Counter()
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(text[cursor : span.start])
        normalized = re.sub(r"[^A-Z0-9]+", "_", span.category.upper()).strip("_")
        parts.append(f"[REDACTED_{normalized or 'VALUE'}]")
        counts[span.category] += 1
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts), counts


def _pattern_counts(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for category, pattern in patterns:
        counts[category] += len(tuple(pattern.finditer(text)))
    return +counts


class DeterministicModelGuardrails:
    """Run local, category-only checks around a ChatModel invocation."""

    def inspect_input(
        self,
        system: str,
        user: str,
        policy: GuardrailPolicy,
    ) -> GuardedPrompt:
        findings: list[GuardrailFinding] = []
        if len(system) + len(user) > policy.max_input_characters:
            findings.append(
                _finding(
                    "input_too_large",
                    "scope_validation",
                    GuardrailAction.BLOCK,
                    1,
                    "Model input exceeds the configured policy limit.",
                )
            )

        missing_markers = [
            marker for marker in policy.required_input_markers if marker not in user
        ]
        if missing_markers:
            findings.append(
                _finding(
                    "scope_marker_missing",
                    "scope_validation",
                    GuardrailAction.BLOCK,
                    len(missing_markers),
                    "Model input is outside the configured workflow prompt shape.",
                )
            )

        injection_count = sum(
            len(tuple(pattern.finditer(user))) for pattern in _INJECTION_PATTERNS
        )
        if injection_count and policy.block_prompt_injection:
            findings.append(
                _finding(
                    "prompt_injection_detected",
                    "prompt_injection",
                    GuardrailAction.BLOCK,
                    injection_count,
                    "Untrusted input contains instruction-override indicators.",
                )
            )

        if policy.enforce_content_safety:
            for category, count in _pattern_counts(user, _CONTENT_PATTERNS).items():
                findings.append(
                    _finding(
                        f"unsafe_{category}",
                        "content_safety",
                        GuardrailAction.BLOCK,
                        count,
                        "Input was blocked by the deterministic content policy.",
                    )
                )

        masked_system, system_counts = mask_sensitive_text(system)
        masked_user, user_counts = mask_sensitive_text(user)
        sensitive_counts = system_counts + user_counts
        if sensitive_counts:
            findings.append(
                _finding(
                    "sensitive_input_masked",
                    "sensitive_data",
                    GuardrailAction.WARN,
                    sum(sensitive_counts.values()),
                    "Sensitive input values were masked before model use.",
                )
            )
        if not policy.mask_sensitive_input:
            masked_system, masked_user = system, user

        report = GuardrailReport(
            policy_name=policy.name,
            policy_version=policy.version,
            stage=GuardrailStage.INPUT,
            action=_action(findings),
            findings=tuple(findings),
            masked_value_count=(
                sum(sensitive_counts.values()) if policy.mask_sensitive_input else 0
            ),
            checks=_CHECKS,
        )
        return GuardedPrompt(system=masked_system, user=masked_user, report=report)

    def inspect_output(
        self,
        output: str,
        policy: GuardrailPolicy,
        *,
        json_only: bool,
    ) -> GuardedOutput:
        findings: list[GuardrailFinding] = []
        guarded_output = output
        if len(output) > policy.max_output_characters:
            findings.append(
                _finding(
                    "output_too_large",
                    "scope_validation",
                    GuardrailAction.BLOCK,
                    1,
                    "Model output exceeds the configured policy limit.",
                )
            )

        parsed: object | None = None
        if json_only or policy.require_json_object:
            try:
                parsed = json.loads(output)
            except (json.JSONDecodeError, TypeError):
                findings.append(
                    _finding(
                        "invalid_json_output",
                        "schema_validation",
                        GuardrailAction.BLOCK,
                        1,
                        "Model output did not satisfy the required JSON contract.",
                    )
                )
            if (
                parsed is not None
                and policy.require_json_object
                and not isinstance(parsed, dict)
            ):
                findings.append(
                    _finding(
                        "non_object_json_output",
                        "schema_validation",
                        GuardrailAction.BLOCK,
                        1,
                        "Model output must be one JSON object.",
                    )
                )
            if isinstance(parsed, dict) and policy.allowed_json_keys:
                unknown = set(parsed) - set(policy.allowed_json_keys)
                if unknown:
                    findings.append(
                        _finding(
                            "unknown_json_keys",
                            "schema_validation",
                            GuardrailAction.BLOCK,
                            len(unknown),
                            "Model output contained fields outside the workflow contract.",
                        )
                    )

        if policy.enforce_content_safety:
            for category, count in _pattern_counts(output, _CONTENT_PATTERNS).items():
                findings.append(
                    _finding(
                        f"unsafe_output_{category}",
                        "content_safety",
                        GuardrailAction.BLOCK,
                        count,
                        "Model output was blocked by the deterministic content policy.",
                    )
                )

        masked_output, sensitive_counts = mask_sensitive_text(output)
        if sensitive_counts:
            sensitive_action = (
                GuardrailAction.BLOCK
                if json_only and policy.block_sensitive_json_output
                else GuardrailAction.WARN
            )
            findings.append(
                _finding(
                    "sensitive_output_detected",
                    "sensitive_data",
                    sensitive_action,
                    sum(sensitive_counts.values()),
                    "Sensitive model output was rejected or masked by policy.",
                )
            )
            if not json_only:
                guarded_output = masked_output

        report = GuardrailReport(
            policy_name=policy.name,
            policy_version=policy.version,
            stage=GuardrailStage.OUTPUT,
            action=_action(findings),
            findings=tuple(findings),
            masked_value_count=(0 if json_only else sum(sensitive_counts.values())),
            checks=(
                "schema_validation",
                "sensitive_data",
                "content_safety",
            ),
        )
        return GuardedOutput(text=guarded_output, report=report)
