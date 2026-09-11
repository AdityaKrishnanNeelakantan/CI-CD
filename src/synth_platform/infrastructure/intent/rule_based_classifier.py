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
            scores["schema_twin"] += 0.12
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
                output_format=_output_format(text),
                interaction_type=_interaction_type(text),
                document_type=_document_type(text),
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
    match = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,7})\b(?=[\w\s-]{0,80}\b(?:records|rows)\b)", text)
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


def _is_unsupported_csv_database(attachment: WorkflowIntentAttachment | None, text: str, workflow: str) -> bool:
    return bool(attachment and _extension_for(attachment) == ".csv" and workflow == "database_twin" and "database" in text)


def _reason_for(workflow: str, reasons: dict[str, list[str]]) -> str:
    values = reasons.get(workflow) or []
    return "; ".join(values[:3]) if values else "Selected by deterministic workflow rules."
