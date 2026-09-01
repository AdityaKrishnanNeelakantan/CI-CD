"""Determine whether a column is eligible for LLM text generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import pandas as pd

from synth_platform.engine.generation.schema.pii_columns import resolve_column_semantic
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig

ALLOWED_TEXT_ROLES = frozenset(
    {
        "long_text",
        "notes",
        "comments",
        "comment",
        "review",
        "description",
        "summary",
        "feedback",
        "support_message",
        "ticket_message",
        "case_note",
        "narrative",
        "service_case_note",
        "support_case_note",
        "transaction_note",
    }
)

BLOCKED_COLUMN_TYPES = frozenset(
    {
        "int",
        "float",
        "decimal",
        "money",
        "currency",
        "boolean",
        "date",
        "datetime",
        "time",
        "categorical",
        "foreign_key",
        "email",
        "phone",
        "url",
        "address",
        "uuid",
        "ssn",
        "aadhaar",
        "routing_number",
        "account_number",
        "bank_account",
        "iban",
        "ifsc_code",
        "swift_bic",
        "credit_card",
    }
)

BLOCKED_NAME_PARTS = frozenset(
    {
        "name",
        "email",
        "phone",
        "mobile",
        "address",
        "ssn",
        "aadhaar",
        "account",
        "routing",
        "iban",
        "ifsc",
        "card",
        "password",
        "token",
        "secret",
        "username",
        "user_id",
        "customer_id",
        "account_id",
        "transaction_id",
        "order_id",
        "pk",
        "uuid",
        "guid",
    }
)

SHORT_STATUS_PATTERN = re.compile(r"^(status|state|stage|phase|type|code|flag|level|priority)$", re.I)


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: List[str] = field(default_factory=list)
    text_role: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "text_role": self.text_role,
        }


def _infer_text_role(column: Column) -> Optional[str]:
    params = column.distribution_params or {}
    explicit = str(params.get("text_type") or params.get("text_role") or "").lower().strip()
    if explicit in ALLOWED_TEXT_ROLES:
        return explicit
    name = column.name.lower()
    for role in ALLOWED_TEXT_ROLES:
        if role in name or name.endswith(f"_{role}") or name == role:
            return role
    narrative_hints = ("note", "comment", "description", "summary", "feedback", "message", "review", "narrative")
    if any(h in name for h in narrative_hints):
        return "narrative"
    if params.get("llm_text") or params.get("llm_enabled"):
        return explicit or "narrative"
    return None


def assess_column_eligibility(
    column: Column,
    *,
    table_name: str = "",
    protected_columns: Optional[Sequence[str]] = None,
    profile_column: Any = None,
) -> EligibilityResult:
    """Return whether a column may use LLM text generation."""
    reasons: List[str] = []
    protected = {c.lower() for c in (protected_columns or [])}

    if column.name.lower() in protected:
        reasons.append("protected column")
    if column.type != "text":
        reasons.append(f"column type '{column.type}' is not free text")
    if column.unique:
        reasons.append("unique/identity column")
    if column.type in BLOCKED_COLUMN_TYPES:
        reasons.append(f"blocked type '{column.type}'")

    semantic = resolve_column_semantic(column.name)
    if semantic:
        reasons.append(f"PII semantic '{semantic}'")

    lowered = column.name.lower()
    for part in BLOCKED_NAME_PARTS:
        if part == lowered or part in lowered.split("_"):
            if part not in {"level", "priority"}:  # priority alone blocked below
                reasons.append(f"blocked name pattern '{part}'")
                break

    if SHORT_STATUS_PATTERN.match(column.name):
        reasons.append("short structured status field")

    params = column.distribution_params or {}
    choices = params.get("choices") or []
    if choices and len(choices) <= 12 and column.name.lower() in {"status", "state", "stage", "type", "category"}:
        reasons.append("low-cardinality categorical status field")

    if profile_column is not None:
        kind = str(getattr(profile_column, "kind", "") or "").lower()
        if kind in {"numeric", "categorical", "boolean", "date", "id_like"}:
            reasons.append(f"source profile kind '{kind}'")

    text_role = _infer_text_role(column)
    if column.type == "text" and text_role is None and not (params.get("llm_text") or params.get("llm_enabled")):
        reasons.append("no recognized text-heavy role")

    eligible = not reasons and text_role is not None
    return EligibilityResult(eligible=eligible, reasons=reasons, text_role=text_role)


def list_eligible_columns(
    schema: SchemaConfig,
    *,
    protected_columns: Optional[Sequence[str]] = None,
) -> List[dict]:
    """List all columns with eligibility assessment."""
    rows: List[dict] = []
    for table in schema.tables:
        for column in schema.get_columns(table.name):
            result = assess_column_eligibility(
                column,
                table_name=table.name,
                protected_columns=protected_columns,
            )
            rows.append(
                {
                    "table": table.name,
                    "column": column.name,
                    "eligible": result.eligible,
                    "text_role": result.text_role,
                    "reasons": result.reasons,
                }
            )
    return rows
