"""Deterministic workflow intent classification."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from synth_platform.application.dto.intent import (
    DetectedWorkflowInput,
    WorkflowIntentAlternative,
    WorkflowIntentAttachment,
    WorkflowIntentPrefill,
    WorkflowIntentRequest,
    WorkflowIntentResponse,
)

ROUTES = {
    "schema_twin": "/schema",
    "database_twin": "/database",
    "document_twin": "/document",
    "interaction_twin": "/interaction",
    "unknown": "/",
}

EXTENSION_SIGNALS = {
    ".sqlite": ("database_twin", 0.72, "SQLite database upload"),
    ".sqlite3": ("database_twin", 0.72, "SQLite database upload"),
    ".db": ("database_twin", 0.72, "SQLite database upload"),
    ".pdf": ("document_twin", 0.72, "PDF document upload"),
    ".txt": ("interaction_twin", 0.54, "TXT transcript/log upload"),
    ".log": ("interaction_twin", 0.64, "LOG transcript upload"),
    ".sql": ("schema_twin", 0.68, "SQL schema upload"),
    ".json": ("schema_twin", 0.58, "JSON schema-like upload"),
    ".yaml": ("schema_twin", 0.58, "YAML schema-like upload"),
    ".yml": ("schema_twin", 0.58, "YAML schema-like upload"),
    ".xlsx": ("schema_twin", 0.42, "Excel-style schema upload"),
    ".xls": ("schema_twin", 0.36, "Excel-style schema upload"),
}

KEYWORDS = {
    "schema_twin": (
        "schema",
        "columns",
        "fields",
        "table definition",
        "records",
        "synthetic rows",
        "fake rows",
        "fake records",
        "synthetic records",
        "sql",
        "json schema",
        "tabular",
    ),
    "database_twin": (
        "database",
        "sqlite",
        "db",
        "relational",
        "source database",
        "tables",
    ),
    "document_twin": (
        "pdf",
        "document",
        "report",
        "preserve layout",
        "same structure",
        "medical report",
    ),
    "interaction_twin": (
        "transcript",
        "chat",
        "conversation",
        "support log",
        "customer interaction",
        "messages",
        "support conversations",
        "customer chats",
        "call center",
    ),
}


class RuleBasedWorkflowIntentClassifier:
    def classify(self, request: WorkflowIntentRequest) -> WorkflowIntentResponse:
        message = request.message.strip()
        text = _normalize(message)
        attachment = request.attachments[0] if request.attachments else None
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)
        warnings: list[str] = []

        detected_source = "prompt"
        detected_file_type: str | None = None
        detected_kind = "natural_language" if message else "unknown"

        if attachment:
            ext = _extension_for(attachment)
            detected_source = "attachment"
            detected_file_type = ext.lstrip(".") if ext else None
            detected_kind = _kind_for_extension(ext)
            if ext in EXTENSION_SIGNALS:
                workflow, score, reason = EXTENSION_SIGNALS[ext]
                scores[workflow] += score
                reasons[workflow].append(reason)
            elif ext == ".csv":
                _score_csv(text, scores, reasons, warnings)
            elif ext:
                warnings.append(f"{ext} files are not currently supported by the available workflows.")

            preview_kind = _sniff_preview(attachment.content_preview or "")
            if preview_kind:
                workflow, reason = preview_kind
                scores[workflow] += 0.18
                reasons[workflow].append(reason)

        for workflow, words in KEYWORDS.items():
            hits = [word for word in words if word in text]
            if hits:
                scores[workflow] += min(0.72, 0.46 + 0.08 * len(hits))
                reasons[workflow].append(f"Prompt mentions {', '.join(hits[:3])}")

        record_count = _extract_record_count(text)
        if record_count:
            scores["schema_twin"] += 0.62
            reasons["schema_twin"].append("Prompt includes a target record count")

        if not scores:
            return WorkflowIntentResponse(
                workflow_type="unknown",
                confidence=0.0,
                reason="No supported workflow signal was detected.",
                suggested_route="/",
                can_auto_start=False,
                next_action="choose_workflow",
                detected_input=DetectedWorkflowInput(kind=detected_kind, file_type=detected_file_type, source=detected_source),
                alternatives=[],
                warnings=warnings,
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_workflow, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        conflict = second_score and top_score - second_score < 0.18
        confidence = max(0.0, min(0.98, top_score if not conflict else top_score - 0.18))
        next_action = "configure_required"

        if conflict or confidence < 0.55:
            next_action = "choose_workflow"
        elif attachment is None and top_workflow in {"database_twin", "document_twin", "interaction_twin"}:
            next_action = "upload_required"
        elif _is_unsupported_csv_database(attachment, text, top_workflow):
            next_action = "unsupported"
            confidence = min(confidence, 0.62)
            warnings.append("CSV-to-Database Twin routing is not supported yet. Database Twin currently supports SQLite uploads only.")
        elif attachment is not None:
            next_action = "ready_to_generate" if request.allow_auto_start and confidence >= 0.82 else "configure_required"

        alternatives = [
            WorkflowIntentAlternative(
                workflow_type=workflow, confidence=max(0.0, min(0.98, score)), reason="; ".join(reasons[workflow][:2])
            )
            for workflow, score in ranked[1:4]
        ]
        if conflict and ranked:
            alternatives.insert(
                0,
                WorkflowIntentAlternative(
                    workflow_type=top_workflow,
                    confidence=max(0.0, min(0.98, top_score)),
                    reason="; ".join(reasons[top_workflow][:2]) or "Likely workflow",
                ),
            )

        return WorkflowIntentResponse(
            workflow_type=top_workflow,
            confidence=round(confidence, 2),
            reason=_reason_for(top_workflow, reasons),
            suggested_route=ROUTES[top_workflow],
            can_auto_start=bool(request.allow_auto_start and next_action == "ready_to_generate"),
            next_action=next_action,
            detected_input=DetectedWorkflowInput(kind=detected_kind, file_type=detected_file_type, source=detected_source),
            prefill=WorkflowIntentPrefill(
                record_count=record_count,
                privacy_level=_privacy_level(text),
                output_format=_output_format(text),
                interaction_type=_interaction_type(text),
                document_type=_document_type(text),
                other=_supported_options(text),
            ),
            alternatives=alternatives,
            warnings=warnings,
        )


def _extension_for(attachment: WorkflowIntentAttachment) -> str:
    ext = (attachment.extension or Path(attachment.filename or "").suffix).strip().lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _kind_for_extension(ext: str) -> str:
    if ext in {".db", ".sqlite", ".sqlite3"}:
        return "database"
    if ext == ".pdf":
        return "document"
    if ext in {".txt", ".log"}:
        return "conversation"
    if ext in {".sql", ".json", ".yaml", ".yml", ".csv", ".xlsx", ".xls"}:
        return "schema"
    return "unknown"


def _score_csv(text: str, scores: dict[str, float], reasons: dict[str, list[str]], warnings: list[str]) -> None:
    if any(word in text for word in ("schema", "columns", "fields", "table definition")):
        scores["schema_twin"] += 0.58
        reasons["schema_twin"].append("CSV plus schema-oriented prompt")
    elif any(word in text for word in ("database", "source database", "twin from data")):
        scores["database_twin"] += 0.44
        reasons["database_twin"].append("CSV plus database-oriented prompt")
        warnings.append("Database Twin currently supports SQLite uploads only; CSV database sources are not available yet.")
    else:
        scores["schema_twin"] += 0.36
        scores["database_twin"] += 0.34
        reasons["schema_twin"].append("CSV can describe tabular schema")
        reasons["database_twin"].append("CSV can also represent source rows")


def _sniff_preview(preview: str) -> tuple[str, str] | None:
    sample = preview[:4096].strip().lower()
    if not sample:
        return None
    if sample.startswith("%pdf"):
        return ("document_twin", "Preview has a PDF header")
    if "create table" in sample or "foreign key" in sample:
        return ("schema_twin", "Preview contains SQL schema statements")
    if sample.startswith("{") and any(key in sample for key in ('"tables"', '"columns"', '"fields"', '"schema"')):
        return ("schema_twin", "Preview looks like structured schema JSON")
    if any(token in sample for token in ("agent:", "customer:", "user:", "assistant:", "message:")):
        return ("interaction_twin", "Preview looks like a conversation transcript")
    return None


def _extract_record_count(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,7})\b(?=[\w\s-]{0,80}\b(?:records|rows|data|items|examples)\b)", text)
    if not match:
        return None
    value = int(match.group(1).replace(",", ""))
    return max(1, min(value, 1_000_000))


def _output_format(text: str) -> str | None:
    if "parquet" in text:
        return "parquet"
    if "csv" in text:
        return "csv"
    if "json" in text:
        return "json"
    if "synthetic log" in text or "logs" in text:
        return "logs"
    return None


def _privacy_level(text: str) -> str | None:
    if any(term in text for term in ("strict privacy", "high privacy", "maximum privacy", "hipaa", "de-identify", "deidentify", "anonymize")):
        return "strict"
    if any(term in text for term in ("standard privacy", "default privacy")):
        return "standard"
    return None


def _interaction_type(text: str) -> str | None:
    if "call center" in text or "phone" in text:
        return "call_center"
    if "email" in text:
        return "email_thread"
    if "chat" in text or "support" in text or "conversation" in text:
        return "support_chat"
    return None


def _document_type(text: str) -> str | None:
    if "medical" in text:
        return "medical_report"
    if "report" in text:
        return "report"
    if "pdf" in text or "document" in text:
        return "document"
    return None


def _supported_options(text: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    locale = _locale(text)
    seed = _seed(text)
    sample_limit = _sample_limit(text)
    extraction_method = _extraction_method(text)
    if locale:
        options["locale"] = locale
    if seed is not None:
        options["seed"] = seed
    if sample_limit is not None:
        options["sample_limit"] = sample_limit
    if extraction_method:
        options["extraction_method"] = extraction_method
    if any(term in text for term in ("llm text", "use llm", "llm enabled", "richer text", "realistic text")):
        options["llm_text_enabled"] = True
    if any(term in text for term in ("no llm", "without llm", "disable llm")):
        options["llm_text_enabled"] = False
    disables_redaction = any(term in text for term in ("keep sensitive", "do not remove sensitive", "without redaction", "no redaction"))
    enables_redaction = any(term in text for term in ("remove sensitive", "redact", "anonymize", "de-identify", "deidentify"))
    if disables_redaction:
        options["remove_sensitive_information"] = False
    elif enables_redaction:
        options["remove_sensitive_information"] = True
    return options


def _locale(text: str) -> str | None:
    aliases = {
        "india": "en_IN",
        "indian": "en_IN",
        "uk": "en_GB",
        "british": "en_GB",
        "united kingdom": "en_GB",
        "germany": "de_DE",
        "german": "de_DE",
        "france": "fr_FR",
        "french": "fr_FR",
        "us": "en_US",
        "usa": "en_US",
        "american": "en_US",
    }
    match = re.search(r"\b(en_US|en_GB|en_IN|de_DE|fr_FR)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    for phrase, locale in aliases.items():
        if phrase in text:
            return locale
    return None


def _seed(text: str) -> int | None:
    match = re.search(r"\bseed(?:\s+is|\s*=|:)?\s*(\d{1,9})\b", text)
    if not match:
        return None
    return max(0, min(int(match.group(1)), 999_999_999))


def _sample_limit(text: str) -> int | None:
    match = re.search(r"\bsample(?:\s+limit|\s+size)?(?:\s+is|\s*=|:)?\s*(\d{1,7})\b", text)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 1_000_000))


def _extraction_method(text: str) -> str | None:
    if "docling" in text:
        return "docling"
    if "ocr" in text:
        return "ocr"
    if "native pdf" in text or "native extraction" in text:
        return "native"
    return None


def _is_unsupported_csv_database(attachment: WorkflowIntentAttachment | None, text: str, workflow: str) -> bool:
    return bool(attachment and _extension_for(attachment) == ".csv" and workflow == "database_twin" and "database" in text)


def _reason_for(workflow: str, reasons: dict[str, list[str]]) -> str:
    values = reasons.get(workflow) or []
    return "; ".join(values[:3]) if values else "Selected by deterministic workflow rules."
