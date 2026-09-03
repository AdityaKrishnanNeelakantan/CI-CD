"""
Semantic column inference for automatic type detection.

This module detects column semantics from names and applies
the correct data generators, even if the LLM misses it.

Bare ``text`` columns otherwise default to ``text_type="sentence"``
(lorem ipsum). Schema Mode JSON uploads often declare only
``type: "text"`` with empty params — this module is the safety net.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from synth_platform.engine.inference.schema.schema import Column, Relationship, SchemaConfig, Table


# Semantic patterns: regex -> (type, distribution_params)
SEMANTIC_PATTERNS: List[Tuple[str, str, Dict[str, Any]]] = [
    # Email patterns
    (r"^email$|^e_?mail$|^user_?email$|^customer_?email$", "text", {"text_type": "email"}),

    # Name patterns
    (r"^name$|^full_?name$|^user_?name$|^customer_?name$|^display_?name$", "text", {"text_type": "name"}),
    (r"^first_?name$", "text", {"text_type": "first_name"}),
    (r"^last_?name$|^surname$|^family_?name$", "text", {"text_type": "last_name"}),
    (r"^manager_?name$|^owner_?name$|^contact_?name$", "text", {"text_type": "name"}),

    # Phone patterns
    (r"^phone$|^phone_?number$|^mobile$|^cell$|^telephone$", "text", {"text_type": "phone"}),

    # Locale-specific identity documents
    (r"^national_?id$|^ssn$|^cpf$|^aadhaar$|^aadhar$|^nid$|^tax_?id$", "text", {"text_type": "national_id"}),

    # Address / geo patterns
    (r"^address$|^street$|^full_?address$|^billing_?address$|^shipping_?address$", "text", {"text_type": "address"}),
    (r"^city$|_city$", "text", {"text_type": "city"}),
    (r"^state$|^province$|_state$|^region$", "text", {"text_type": "state"}),
    (r"^country$|_country$", "text", {"text_type": "country"}),
    (r"^zip$|^zip_?code$|^postal$|^postal_?code$|^postcode$", "text", {"text_type": "postal_code"}),

    # Company / merchant
    (r"^company$|^company_?name$|^organization$|^org_?name$|^employer$", "text", {"text_type": "company"}),
    (r"^merchant$|^merchant_?name$", "text", {"text_type": "company"}),
    (r"^branch_?name$|^store_?name$|^office_?name$|^location_?name$", "text", {"text_type": "company"}),

    # URL patterns
    (r"^url$|^website$|^web_?url$|^link$|^profile_?url$", "text", {"text_type": "url"}),

    # Banking / cards
    (r"^credit_?score$", "int", {"distribution": "uniform", "min": 300, "max": 850}),
    (
        r"^card_?type$",
        "categorical",
        {"choices": ["credit", "debit", "prepaid", "charge"]},
    ),
    (
        r"^account_?type$|^account_?status$",
        "categorical",
        {"choices": ["checking", "savings", "credit", "loan", "closed"]},
    ),
    (
        r"^loan_?type$|^loan_?status$",
        "categorical",
        {"choices": ["personal", "auto", "mortgage", "student", "active", "paid_off"]},
    ),

    # Price/Money patterns (must be positive)
    (r"^interest_?rate$|^apr$|^apy$", "float", {"distribution": "uniform", "min": 1.0, "max": 24.99, "decimals": 2}),
    (r"^price$|^cost$|^amount$|_amount$|^fee$|_fee$|^total$|_total$|^subtotal$|^tax$|^balance$|_balance$|^principal$|_usd$", "float", {"distribution": "uniform", "min": 0, "max": 1000, "decimals": 2}),
    (r"^mrr$|^arr$|^revenue$|^income$|^salary$|^wage$", "float", {"distribution": "uniform", "min": 0, "max": 100000, "decimals": 2}),

    # Age patterns
    (r"^age$|^user_?age$|^customer_?age$", "int", {"distribution": "uniform", "min": 18, "max": 80}),

    # Count patterns (non-negative integers)
    (r"^count$|^quantity$|^qty$|^num_|^number_of_|_count$", "int", {"distribution": "poisson", "lambda": 5, "min": 0}),

    # Percentage patterns
    (r"^percent|percentage$|_pct$|_percent$|^rate$", "float", {"distribution": "uniform", "min": 0, "max": 100, "decimals": 1}),

    # Duration patterns
    (r"^duration$|^duration_?minutes$|^duration_?hours$|^length$|^time_?spent$", "int", {"distribution": "uniform", "min": 1, "max": 120}),

    # Weight/Height patterns
    (r"^weight$|^weight_?kg$", "float", {"distribution": "normal", "mean": 70, "std": 15, "min": 30, "max": 200}),
    (r"^height$|^height_?cm$", "float", {"distribution": "normal", "mean": 170, "std": 10, "min": 140, "max": 220}),

    # Rating patterns (exclude credit_score — handled above)
    (r"^rating$|^stars$|^review_?score$", "float", {"distribution": "uniform", "min": 1, "max": 5, "decimals": 1}),

    # Boolean patterns
    (r"^is_|^has_|^can_|^should_|^active$|^enabled$|^verified$|^confirmed$", "boolean", {"probability": 0.5}),

    # Status patterns
    (r"^status$|^state$|^order_?status$|^subscription_?status$", "categorical", {"choices": ["active", "inactive", "pending", "cancelled"]}),

    # Date patterns (already handled by type, but ensure proper params)
    (r"^date$|^created_?at$|^updated_?at$|^start_?date$|^end_?date$|_date$|_at$", "date", {"start": "2020-01-01", "end": "2024-12-31"}),
]


def _is_identifier_name(column_name: str) -> bool:
    name = column_name.lower()
    return name in {"id", "uuid", "guid"} or name.endswith("_id") or name.endswith("_uuid")


def _table_name_candidates(stem: str) -> List[str]:
    """Map ``customer`` from ``customer_id`` to likely parent table names."""
    stem = stem.lower().strip("_")
    if not stem:
        return []
    candidates = [stem, f"{stem}s", f"{stem}es"]
    if stem.endswith("y") and len(stem) > 1:
        candidates.append(stem[:-1] + "ies")
    if stem.endswith("s"):
        candidates.append(stem[:-1])
    # de-dupe preserving order
    seen: set[str] = set()
    out: List[str] = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def infer_relationships_from_column_names(
    tables: Sequence[Table],
    columns: Dict[str, List[Column]],
) -> List[Relationship]:
    """Infer parent→child FKs from ``*_id`` columns when relationships are absent."""
    table_names = {t.name for t in tables}
    existing: List[Relationship] = []
    seen: set[tuple[str, str, str, str]] = set()

    for child_table, cols in columns.items():
        pk_names = {c.name for c in cols if c.unique}
        for col in cols:
            if not col.name.endswith("_id"):
                continue
            # Don't treat the table's own primary key as an FK to itself.
            if col.name in pk_names or col.name in {f"{child_table}_id", "id"}:
                if col.unique:
                    continue
            stem = col.name[:-3]
            resolved: Optional[Tuple[str, str]] = None
            for parent_table in _table_name_candidates(stem):
                if parent_table not in table_names or parent_table == child_table:
                    continue
                parent_cols = {c.name for c in columns.get(parent_table, [])}
                if col.name in parent_cols:
                    resolved = (parent_table, col.name)
                    break
                preferred = f"{parent_table}_id"
                if preferred in parent_cols:
                    resolved = (parent_table, preferred)
                    break
                if "id" in parent_cols:
                    resolved = (parent_table, "id")
                    break
            if resolved is None:
                continue
            parent_table, parent_key = resolved
            key = (parent_table, parent_key, child_table, col.name)
            if key in seen:
                continue
            seen.add(key)
            existing.append(
                Relationship(
                    parent_table=parent_table,
                    parent_key=parent_key,
                    child_table=child_table,
                    child_key=col.name,
                )
            )
    return existing


def promote_foreign_keys(
    columns: Dict[str, List[Column]],
    relationships: Sequence[Relationship],
) -> Dict[str, List[Column]]:
    """Mark relationship child keys as ``foreign_key`` so they sample parent PKs."""
    child_keys = {(r.child_table, r.child_key): r for r in relationships}
    if not child_keys:
        return columns

    fixed: Dict[str, List[Column]] = {}
    for table_name, cols in columns.items():
        updated: List[Column] = []
        for col in cols:
            rel = child_keys.get((table_name, col.name))
            if rel is None or col.type == "foreign_key":
                updated.append(col)
                continue
            updated.append(
                Column(
                    name=col.name,
                    type="foreign_key",
                    nullable=col.nullable,
                    unique=False,
                    distribution_params={
                        **dict(col.distribution_params or {}),
                        "references": f"{rel.parent_table}.{rel.parent_key}",
                    },
                    description=col.description,
                )
            )
        fixed[table_name] = updated
    return fixed


class SemanticInference:
    """
    Automatically infer and fix column semantics based on naming patterns.

    This acts as a safety net - if the LLM generates incorrect column types
    or parameters, semantic inference can fix them based on column names.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize semantic inference.

        Args:
            strict_mode: If True, always override LLM; if False, only fix obvious errors
        """
        self.strict_mode = strict_mode
        self.patterns = [(re.compile(p, re.IGNORECASE), t, params)
                         for p, t, params in SEMANTIC_PATTERNS]

    def infer_column(self, column_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Infer column type and parameters from name.

        Args:
            column_name: Name of the column

        Returns:
            Tuple of (type, distribution_params) or None if no match
        """
        for pattern, col_type, params in self.patterns:
            if pattern.search(column_name):
                return (col_type, params.copy())
        return None

    def fix_column(self, column: Column, table_name: str = "") -> Column:
        """
        Fix a column's type/params based on semantic inference.

        Args:
            column: Column to potentially fix
            table_name: Name of the table (for context)

        Returns:
            Fixed column (or original if no fix needed)
        """
        # Never rewrite declared foreign keys.
        if column.type == "foreign_key":
            return column

        # Unique / primary-looking identifiers must not fall through to lorem sentences.
        if column.type == "text" and _is_identifier_name(column.name):
            current_text_type = column.distribution_params.get("text_type", "sentence")
            if current_text_type in {"sentence", "word"} or "text_type" not in column.distribution_params:
                return Column(
                    name=column.name,
                    type="text",
                    distribution_params={**dict(column.distribution_params or {}), "text_type": "uuid"},
                    nullable=column.nullable,
                    unique=column.unique,
                    description=column.description,
                )

        inferred = self.infer_column(column.name)

        if inferred is None:
            return column

        inferred_type, inferred_params = inferred

        # Determine if we should apply the fix
        should_fix = False

        if self.strict_mode:
            # Always use inferred semantics
            should_fix = True
        else:
            # Only fix if current type seems wrong
            # Case 1: Column named "email"/"city"/… but still default sentence text
            if column.type == "text":
                current_text_type = column.distribution_params.get("text_type", "sentence")
                if current_text_type == "sentence" or "text_type" not in column.distribution_params:
                    # Default sentence generation - probably wrong for semantic names
                    should_fix = True

            # Case 2: Numeric column that could be negative but shouldn't be
            if column.type in ["int", "float"]:
                lowered = column.name.lower()
                if any(token in lowered for token in ("price", "age", "score", "amount", "balance", "rate", "apr", "apy")):
                    if "min" not in column.distribution_params:
                        should_fix = True

            # Case 3: Typed as generic text/int but name implies categorical/score
            if inferred_type != column.type and not column.distribution_params:
                should_fix = True

        if should_fix:
            # Merge inferred params with existing (inferred takes precedence)
            merged_params = {**column.distribution_params, **inferred_params}
            return Column(
                name=column.name,
                type=inferred_type,
                distribution_params=merged_params,
                nullable=column.nullable,
                unique=column.unique,
                description=column.description,
            )

        return column

    def fix_schema_columns(self, columns: Dict[str, List[Column]]) -> Dict[str, List[Column]]:
        """
        Fix all columns in a schema using semantic inference.

        Args:
            columns: Dict mapping table names to column lists

        Returns:
            Fixed columns dict
        """
        fixed = {}
        for table_name, cols in columns.items():
            fixed[table_name] = [self.fix_column(c, table_name) for c in cols]
        return fixed


def apply_semantic_inference(columns: Dict[str, List[Column]], strict: bool = False) -> Dict[str, List[Column]]:
    """
    Apply semantic inference to fix column definitions.

    Args:
        columns: Schema columns to fix
        strict: If True, always apply semantic rules

    Returns:
        Fixed columns
    """
    inference = SemanticInference(strict_mode=strict)
    return inference.fix_schema_columns(columns)


def enrich_schema_semantics(schema: SchemaConfig, *, strict: bool = False) -> SchemaConfig:
    """Infer FKs + column semantics so bare text schemas do not emit lorem ipsum.

    Order:
      1. Infer missing relationships from ``*_id`` naming
      2. Promote child keys to ``foreign_key``
      3. Apply name-based text/numeric semantic fixes
    """
    prepared = schema.model_copy(deep=True)
    relationships = list(prepared.relationships or [])
    if not relationships:
        relationships = infer_relationships_from_column_names(prepared.tables, prepared.columns)
        prepared.relationships = relationships

    prepared.columns = promote_foreign_keys(prepared.columns, prepared.relationships)
    prepared.columns = apply_semantic_inference(prepared.columns, strict=strict)
    return prepared
