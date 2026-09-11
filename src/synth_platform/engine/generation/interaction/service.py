"""Deterministic synthesis for customer-service transcript twins."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.pii import detect_pii


_TURN_PATTERN = re.compile(
    r"(?is)\b(internal|external|agent|customer|user|representative|rep)\s*:\s*(.*?)"
    r"(?=(?:[\"']?\s*,\s*[\r\n\s]*[\"']?\s*"
    r"(?:internal|external|agent|customer|user|representative|rep)\s*:)|\Z)"
)
_LINE_PATTERN = re.compile(
    r"^\s*(?:\[(?P<timestamp>[^\]]+)\]\s*)?"
    r"(?P<speaker>[A-Za-z][A-Za-z0-9 _.-]{0,40})\s*:\s*(?P<text>.+?)\s*$"
)
_CARD_FRAGMENT_PATTERN = re.compile(r"\b\d{4}\b")
_NAME_CONTEXT_PATTERN = re.compile(
    r"\b(?:my name is|name is|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
)


@dataclass(frozen=True)
class TranscriptTurn:
    index: int
    speaker: str
    text: str
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "text": self.text,
        }


@dataclass(frozen=True)
class Transcript:
    turns: list[TranscriptTurn]

    @property
    def speaker_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for turn in self.turns:
            counts[turn.speaker] = counts.get(turn.speaker, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_count": len(self.turns),
            "speaker_counts": self.speaker_counts,
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True)
class InteractionGenerationConfig:
    interaction_type: str = "Customer Support"
    output_format: str = "Structured JSON + Synthetic Logs"
    remove_sensitive_information: bool = True
    seed: int = 42


@dataclass(frozen=True)
class InteractionTwinResult:
    source: Transcript
    synthetic: Transcript
    validation_report: dict[str, Any]
    redaction_report: dict[str, Any]
    output_paths: dict[str, Path] = field(default_factory=dict)
    package_bytes: bytes | None = None

    @property
    def hard_checks_passed(self) -> bool:
        return self.validation_report.get("hard_checks_passed") is True


def parse_transcript(text: str) -> Transcript:
    """Parse common transcript exports into normalized speaker turns."""

    turns: list[TranscriptTurn] = []
    for match in _TURN_PATTERN.finditer(text):
        speaker = _normalize_speaker(match.group(1))
        content = _clean_turn_text(match.group(2))
        if content:
            turns.append(TranscriptTurn(index=len(turns), speaker=speaker, text=content))

    if not turns:
        for line in text.splitlines():
            line_match = _LINE_PATTERN.match(line)
            if line_match is None:
                continue
            turns.append(
                TranscriptTurn(
                    index=len(turns),
                    speaker=_normalize_speaker(line_match.group("speaker")),
                    timestamp=line_match.group("timestamp"),
                    text=_clean_turn_text(line_match.group("text")),
                )
            )

    if not turns:
        raise ValueError("No transcript turns were found. Upload a .txt or .log file with speaker-labeled turns.")
    return Transcript(turns=turns)


def redact_sensitive_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Redact PII/PCI spans using existing rule-based PII detection plus transcript PCI fragments."""

    findings = detect_pii(text)
    findings.extend(_detect_card_fragments(text))
    findings.extend(_detect_contextual_names(text))
    findings = _merge_findings(findings)
    if not findings:
        return text, []

    parts: list[str] = []
    cursor = 0
    for finding in findings:
        start = int(finding["start"])
        end = int(finding["end"])
        parts.append(text[cursor:start])
        parts.append(f"[REDACTED_{str(finding['type']).upper()}]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), findings


def generate_interaction_twin(
    transcript_text: str,
    config: InteractionGenerationConfig | None = None,
    *,
    output_dir: str | Path | None = None,
) -> InteractionTwinResult:
    config = config or InteractionGenerationConfig()
    source = parse_transcript(transcript_text)
    source_for_generation, redaction_report = _redacted_source(source, config.remove_sensitive_information)
    synthetic = _synthesize(source_for_generation, config)
    validation_report = validate_interaction_twin(source, synthetic, redaction_report, config)

    result = InteractionTwinResult(
        source=source,
        synthetic=synthetic,
        validation_report=validation_report,
        redaction_report=redaction_report,
    )
    if output_dir is not None:
        paths = write_interaction_outputs(result, output_dir, config)
        package = package_interaction_outputs(paths)
        result = InteractionTwinResult(
            source=source,
            synthetic=synthetic,
            validation_report=validation_report,
            redaction_report=redaction_report,
            output_paths=paths,
            package_bytes=package,
        )
    return result


def validate_interaction_twin(
    source: Transcript,
    synthetic: Transcript,
    redaction_report: dict[str, Any],
    config: InteractionGenerationConfig,
) -> dict[str, Any]:
    synthetic_text = "\n".join(turn.text for turn in synthetic.turns)
    source_texts = {turn.text.strip() for turn in source.turns if len(turn.text.strip()) >= 12}
    copied_turns = [turn.text for turn in synthetic.turns if turn.text.strip() in source_texts]
    residual_findings = detect_pii(synthetic_text) if config.remove_sensitive_information else []
    checks = [
        {
            "name": "turns_present",
            "passed": bool(source.turns and synthetic.turns),
            "required": True,
            "detail": f"{len(synthetic.turns)} synthetic turns generated",
        },
        {
            "name": "speaker_structure_preserved",
            "passed": [t.speaker for t in source.turns] == [t.speaker for t in synthetic.turns],
            "required": True,
            "detail": "speaker sequence matches source transcript",
        },
        {
            "name": "source_text_not_copied",
            "passed": not copied_turns,
            "required": True,
            "detail": f"{len(copied_turns)} copied source turns detected",
        },
        {
            "name": "sensitive_output_redacted",
            "passed": not residual_findings,
            "required": bool(config.remove_sensitive_information),
            "detail": f"{len(residual_findings)} PII/PCI findings remain in synthetic output",
        },
    ]
    passed = all(check["passed"] for check in checks if check["required"])
    return {
        "workflow": "interaction_twin",
        "status": "PASS" if passed else "FAIL",
        "hard_checks_passed": passed,
        "passed": passed,
        "export_ready": passed,
        "checks": checks,
        "metrics": {
            "source_turn_count": len(source.turns),
            "synthetic_turn_count": len(synthetic.turns),
            "redaction_findings": redaction_report.get("finding_count", 0),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }


def write_interaction_outputs(
    result: InteractionTwinResult,
    output_dir: str | Path,
    config: InteractionGenerationConfig,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "synthetic_interaction.json"
    log_path = out / "synthetic_interaction.log"
    validation_path = out / "validation_report.json"
    redaction_path = out / "redaction_report.json"

    json_path.write_text(
        json.dumps(
            {
                "interaction_type": config.interaction_type,
                "output_format": config.output_format,
                "transcript": result.synthetic.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log_path.write_text(_to_log(result.synthetic), encoding="utf-8")
    validation_path.write_text(json.dumps(result.validation_report, indent=2, sort_keys=True), encoding="utf-8")
    redaction_path.write_text(json.dumps(result.redaction_report, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "json": json_path,
        "log": log_path,
        "validation_report": validation_path,
        "redaction_report": redaction_path,
    }


def package_interaction_outputs(paths: dict[str, Path]) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths.values():
            zf.write(path, arcname=path.name)
    return buffer.getvalue()


def _redacted_source(source: Transcript, enabled: bool) -> tuple[Transcript, dict[str, Any]]:
    if not enabled:
        return source, {"enabled": False, "finding_count": 0, "findings_by_type": {}}
    turns: list[TranscriptTurn] = []
    findings_by_type: dict[str, int] = {}
    finding_count = 0
    for turn in source.turns:
        redacted, findings = redact_sensitive_text(turn.text)
        for finding in findings:
            pii_type = str(finding["type"])
            findings_by_type[pii_type] = findings_by_type.get(pii_type, 0) + 1
        finding_count += len(findings)
        turns.append(
            TranscriptTurn(index=turn.index, speaker=turn.speaker, timestamp=turn.timestamp, text=redacted)
        )
    return Transcript(turns), {
        "enabled": True,
        "finding_count": finding_count,
        "findings_by_type": findings_by_type,
    }


def _synthesize(source: Transcript, config: InteractionGenerationConfig) -> Transcript:
    agent_templates = [
        "Thank you for contacting support. I can help review this request.",
        "I understand the concern and will check the available account details.",
        "I am going to verify the request and keep the explanation concise.",
        "The synthetic record shows this was resolved through the standard support path.",
        "Thanks for your patience. I have documented the outcome for follow-up.",
    ]
    customer_templates = [
        "Hi, I need help with a recent service question.",
        "That makes sense. I can provide the requested detail.",
        "Please check what happened and let me know the next step.",
        "I appreciate the clarification.",
        "Great, I will follow that guidance. Thank you.",
    ]
    turns: list[TranscriptTurn] = []
    for turn in source.turns:
        templates = agent_templates if turn.speaker == "agent" else customer_templates
        text = templates[turn.index % len(templates)]
        if config.interaction_type and turn.index == 0 and turn.speaker == "agent":
            text = f"Thank you for contacting {config.interaction_type.lower()} support. I can help with this request."
        turns.append(TranscriptTurn(index=turn.index, speaker=turn.speaker, timestamp=turn.timestamp, text=text))
    return Transcript(turns)


def _detect_card_fragments(text: str) -> list[dict[str, Any]]:
    matches = list(_CARD_FRAGMENT_PATTERN.finditer(text))
    return [
        {
            "type": "credit_card_fragment",
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.65,
            "redaction_preview": "****",
        }
        for match in matches
    ]


def _detect_contextual_names(text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "person_name",
            "start": match.start(1),
            "end": match.end(1),
            "confidence": 0.7,
            "redaction_preview": "[name]",
        }
        for match in _NAME_CONTEXT_PATTERN.finditer(text)
    ]


def _merge_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(findings, key=lambda item: (int(item["start"]), -(int(item["end"]) - int(item["start"]))))
    merged: list[dict[str, Any]] = []
    for finding in ordered:
        start = int(finding["start"])
        end = int(finding["end"])
        if any(start < int(existing["end"]) and end > int(existing["start"]) for existing in merged):
            continue
        merged.append(finding)
    return merged


def _normalize_speaker(value: str) -> str:
    raw = value.strip().lower()
    if raw in {"internal", "agent", "representative", "rep"}:
        return "agent"
    return "customer"


def _clean_turn_text(value: str) -> str:
    text = value.strip().strip(",").strip()
    text = re.sub(r"^[\"']+|[\"']+$", "", text).strip()
    text = text.replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _to_log(transcript: Transcript) -> str:
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in transcript.turns) + "\n"


def new_run_id() -> str:
    return f"interaction-{uuid.uuid4().hex}"
