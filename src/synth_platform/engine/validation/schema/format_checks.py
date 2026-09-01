"""Schema-typed format validation for generated data.

Checks run against explicit column types from ``SchemaConfig`` when available.
Name-based inference is only a fallback for unschema'd CSV validation paths.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from synth_platform.engine.generation.schema.smart_values import is_aba_routing_number, is_luhn_valid, is_verhoeff_valid

TYPED_FORMAT_COLUMN_TYPES = frozenset(
    {
        "routing_number",
        "account_number",
        "bank_account",
        "ssn",
        "aadhaar",
        "credit_card",
        "ifsc_code",
        "swift_bic",
        "iban",
    }
)


def _ssn_invalid_mask(text: pd.Series) -> pd.Series:
    digits = text.str.replace(r"\D", "", regex=True)
    area = pd.to_numeric(digits.str[:3], errors="coerce")
    return (
        (digits.str.len() != 9)
        | digits.str.startswith("000")
        | digits.str.startswith("666")
        | area.ge(900).fillna(True)
        | (digits.str[3:5] == "00")
        | (digits.str[5:] == "0000")
    )


def invalid_typed_mask(
    column_name: str,
    values: pd.Series,
    *,
    column_type: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return a boolean mask for invalid typed-format values, or None if not applicable."""
    col_type = (column_type or "").lower().strip()
    name = column_name.lower()
    text = values.dropna().astype(str)
    if text.empty:
        return None

    if col_type in {"", "text", "categorical", "foreign_key"}:
        col_type = ""

    if col_type == "routing_number" or (not col_type and ("routing" in name or "aba" in name)):
        invalid = ~text.map(is_aba_routing_number)
    elif col_type == "aadhaar" or (not col_type and ("aadhaar" in name or "aadhar" in name)):
        invalid = ~text.map(is_verhoeff_valid)
    elif col_type == "ssn" or (not col_type and (name == "ssn" or "social_security" in name)):
        invalid = _ssn_invalid_mask(text)
    elif col_type == "credit_card" or (
        not col_type and ("credit_card" in name or name.endswith("card_number") or name == "card_number")
    ):
        invalid = ~text.map(is_luhn_valid)
    elif col_type == "ifsc_code" or (not col_type and "ifsc" in name):
        invalid = ~text.str.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", na=False)
    elif col_type == "swift_bic" or (not col_type and ("swift" in name or "bic" in name)):
        invalid = ~text.str.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$", na=False)
    elif col_type == "iban" or (not col_type and "iban" in name):
        invalid = ~text.str.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}$", na=False)
    elif col_type in {"account_number", "bank_account"} or (
        not col_type and "account" in name and ("bank" in name or "acct" in name or "number" in name)
    ):
        digits = text.str.replace(r"\D", "", regex=True)
        invalid = (digits.str.len() < 6) | (digits.str.len() > 18)
    else:
        return None

    return invalid.reindex(values.index, fill_value=False)


def schema_column_type_map(schema_config: Any) -> Dict[tuple[str, str], str]:
    """Build ``(table, column) -> type`` lookup from a schema config."""
    mapping: Dict[tuple[str, str], str] = {}
    if schema_config is None:
        return mapping
    for table in getattr(schema_config, "tables", []):
        for column in schema_config.get_columns(table.name):
            mapping[(table.name, column.name)] = str(column.type)
    return mapping


def _banking_invalid_mask(column_name: str, values: pd.Series) -> Optional[pd.Series]:
    """Backward-compatible alias for name-inferred format checks."""
    return invalid_typed_mask(column_name, values)
