"""Bank statement document archetype.

Encapsulates field labels, generator priors, and validation rules
specific to bank/financial statement documents.  Registered as the first
concrete archetype; other document types fall to generic_shape_fallback
until additional archetypes are added.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.documents.pdf.archetypes.base import DocumentArchetype

_BANK_STATEMENT_KEYWORDS = frozenset({
    "account number", "statement period", "opening balance", "closing balance",
    "beginning balance", "ending balance", "service charge", "transaction",
    "deposits", "withdrawals", "available balance", "account summary",
})


class BankStatementArchetype(DocumentArchetype):
    archetype_id = "bank_statement"
    document_types = frozenset({"bank_statement", "report"})
    display_name = "Bank Statement"

    def expected_field_labels(self) -> list[str]:
        return [
            "Account Number",
            "Statement Date",
            "Opening Balance",
            "Closing Balance",
            "Available Balance",
            "Statement Period",
        ]

    def generator_priors(self) -> dict[str, Any]:
        return {
            "account_number": {"pattern": "##########", "min_length": 8, "max_length": 12},
            "currency_amount": {"min_value": 0.0, "max_value": 999_999.99, "decimal_places": 2},
            "date": {"format": "MM/DD/YYYY"},
            "transaction_description": {"max_length": 80},
        }

    def validation_rules(self) -> list[dict[str, Any]]:
        return [
            {"rule_id": "balance_consistency", "expression": "closing_balance >= 0"},
            {"rule_id": "date_order", "expression": "statement_end_date >= statement_start_date"},
        ]

    def match_score(self, text: str, classification: dict[str, Any]) -> float:
        lowered = text.lower()
        keyword_hits = sum(1 for kw in _BANK_STATEMENT_KEYWORDS if kw in lowered)
        keyword_score = min(keyword_hits / 4.0, 1.0)
        base_score = super().match_score(text, classification)
        return max(base_score, keyword_score * 0.8)
