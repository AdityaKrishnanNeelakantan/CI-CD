"""Generic duplicate detection and repair helpers for generated synthetic tables.

The guard is intentionally schema-driven, not banking-specific.  It derives
protected columns from schema metadata (unique columns, relationship keys,
identifier-like names, and sensitive column types) and only repairs safe
business columns.  Hard duplicate failures are still caught again by
``mvp.validation`` before export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROTECTED_TYPES: Set[str] = {
    "foreign_key",
    "uuid",
    "email",
    "phone",
    "url",
    "address",
    "bank_account",
    "account_number",
    "routing_number",
    "ssn",
    "aadhaar",
    "iban",
    "swift_bic",
    "ifsc_code",
    "credit_card",
}

PROTECTED_NAME_PARTS: Set[str] = {
    "id",
    "uuid",
    "guid",
    "ssn",
    "aadhaar",
    "routing",
    "account_number",
    "accountnumber",
    "bank_account",
    "iban",
    "swift",
    "ifsc",
    "credit_card",
    "card_number",
    "email",
    "phone",
    "mobile",
    "address",
}

VOLATILE_NAME_PARTS: Set[str] = {
    "created_at",
    "updated_at",
    "posted_at",
    "timestamp",
    "datetime",
    "date",
    "time",
}


@dataclass
class DuplicateRepairAction:
    table: str
    column: Optional[str]
    category: str
    message: str
    rows_repaired: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateRepairReport:
    actions: List[DuplicateRepairAction] = field(default_factory=list)

    @property
    def repaired_rows(self) -> int:
        return int(sum(a.rows_repaired for a in self.actions))

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repaired_rows": self.repaired_rows,
            "actions": [
                {
                    "table": a.table,
                    "column": a.column,
                    "category": a.category,
                    "message": a.message,
                    "rows_repaired": a.rows_repaired,
                    "details": a.details,
                }
                for a in self.actions
            ],
        }


def enforce_duplicate_policy(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
    *,
    seed: Optional[int] = None,
    locked_columns: Optional[Mapping[str, Iterable[str]]] = None,
    repair: bool = True,
    max_rounds: int = 3,
) -> Tuple[Dict[str, pd.DataFrame], DuplicateRepairReport]:
    """Repair avoidable duplicates across generated tables.

    Parameters
    ----------
    tables:
        Generated tables.
    schema:
        ``SchemaConfig``-like object with ``get_columns`` and ``relationships``.
    seed:
        Deterministic seed for repeatable repair choices.
    locked_columns:
        Columns that should not be mutated, for example columns referenced by
        user distribution rules.  Keys are table names, values are column names.
    repair:
        If False, only returns copies and an empty report.
    max_rounds:
        Number of exact/fingerprint repair passes.

    Notes
    -----
    Relationship keys are never changed here.  If a key column duplicates when it
    should be unique, validation will block export rather than silently breaking
    referential integrity.
    """
    output = {name: df.copy() for name, df in tables.items()}
    report = DuplicateRepairReport()

    if not repair:
        return output, report

    locked = {
        table: {str(c) for c in cols}
        for table, cols in (locked_columns or {}).items()
    }
    rng = np.random.default_rng(int(seed if seed is not None else 0))

    for table_name, df in output.items():
        if df.empty:
            continue

        # 1) Repair duplicate values in unique non-relationship columns.
        for col in _schema_columns(schema, table_name):
            col_name = getattr(col, "name", None)
            if not col_name or col_name not in df.columns:
                continue
            if not bool(getattr(col, "unique", False)):
                continue
            if col_name in locked.get(table_name, set()):
                continue
            if _is_relationship_key(schema, table_name, col_name):
                continue

            dup_mask = df[col_name].duplicated(keep="first") & df[col_name].notna()
            if int(dup_mask.sum()) <= 0:
                continue

            repaired = _repair_unique_column(df, col_name, col, dup_mask, rng)
            if repaired:
                report.actions.append(
                    DuplicateRepairAction(
                        table=table_name,
                        column=col_name,
                        category="unique_column_repair",
                        message=f"Repaired {repaired} duplicate values in unique column '{table_name}.{col_name}'.",
                        rows_repaired=repaired,
                    )
                )

        # 2) Repair exact full-row duplicates and business-fingerprint duplicates.
        # Multiple passes are useful when a repair creates a new collision by bad luck.
        for _ in range(max(1, int(max_rounds))):
            repaired_this_round = 0
            safe_cols = _business_columns(schema, table_name, df, locked.get(table_name, set()))
            if not safe_cols:
                break

            exact_dup_mask = df.duplicated(keep="first")
            exact_count = int(exact_dup_mask.sum())
            if exact_count:
                repaired = _mutate_duplicate_rows(df, schema, table_name, exact_dup_mask, safe_cols, rng)
                repaired_this_round += repaired
                if repaired:
                    report.actions.append(
                        DuplicateRepairAction(
                            table=table_name,
                            column=None,
                            category="exact_row_repair",
                            message=f"Repaired {repaired} exact duplicate rows in '{table_name}'.",
                            rows_repaired=repaired,
                            details={"business_columns_used": safe_cols},
                        )
                    )

            # Business fingerprint detects rows that only differ by technical IDs.
            # Require at least two business columns so we don't overreact to valid
            # repeated single attributes like two people sharing one name.
            if len(safe_cols) >= 2:
                fp_dup_mask = df.duplicated(subset=safe_cols, keep="first")
                fp_count = int(fp_dup_mask.sum())
                if fp_count:
                    repaired = _mutate_duplicate_rows(df, schema, table_name, fp_dup_mask, safe_cols, rng)
                    repaired_this_round += repaired
                    if repaired:
                        report.actions.append(
                            DuplicateRepairAction(
                                table=table_name,
                                column=None,
                                category="business_fingerprint_repair",
                                message=(
                                    f"Repaired {repaired} duplicate business fingerprints in '{table_name}' "
                                    "without changing keys or protected identifiers."
                                ),
                                rows_repaired=repaired,
                                details={"fingerprint_columns": safe_cols},
                            )
                        )

            if repaired_this_round == 0:
                break

    return output, report


def duplicate_validation_records(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
) -> List[Dict[str, Any]]:
    """Return hard duplicate issues for validation.py."""
    records: List[Dict[str, Any]] = []

    for table_name, df in tables.items():
        if df.empty:
            continue

        exact_count = int(df.duplicated(keep="first").sum())
        if exact_count:
            records.append(
                {
                    "table": table_name,
                    "column": None,
                    "message": f"Contains {exact_count} exact duplicate rows",
                    "affected_rows": exact_count,
                    "sample_values": [],
                }
            )

        for col in _schema_columns(schema, table_name):
            col_name = getattr(col, "name", None)
            if not col_name or col_name not in df.columns:
                continue
            # Child foreign keys are allowed to repeat.  Only declared unique
            # columns and protected non-relationship identifiers are hard failures.
            is_unique = bool(getattr(col, "unique", False))
            is_protected_identifier = (
                _is_protected_column(schema, table_name, col_name, col)
                and not _is_relationship_key(schema, table_name, col_name)
            )
            if is_unique or is_protected_identifier:
                dup_mask = df[col_name].duplicated(keep="first") & df[col_name].notna()
                dup_count = int(dup_mask.sum())
                if dup_count:
                    records.append(
                        {
                            "table": table_name,
                            "column": col_name,
                            "message": f"Contains {dup_count} duplicate values in unique/protected column '{col_name}'",
                            "affected_rows": dup_count,
                            "sample_values": df.loc[dup_mask, col_name].head(5).astype(str).tolist(),
                        }
                    )

    return records


def duplicate_quality_records(
    tables: Mapping[str, pd.DataFrame],
    schema: Any,
    *,
    min_rows: int = 50,
    top_value_ratio_warning: float = 0.15,
    fingerprint_ratio_warning: float = 0.10,
) -> List[Dict[str, Any]]:
    """Return advisory duplicate-density records for quality.py.

    These are warnings, not hard failures.  Repeated names/categories can be valid
    in real data, but high concentration should be visible to the user.
    """
    records: List[Dict[str, Any]] = []

    for table_name, df in tables.items():
        if len(df) < min_rows:
            continue

        locked: Set[str] = set()
        business_cols = _business_columns(schema, table_name, df, locked)

        for col_name in business_cols:
            spec = _schema_column(schema, table_name, col_name)
            col_type = str(getattr(spec, "type", "") or "").lower()

            # Repeated categories like status/account_type/risk_tier are expected.
            # Column-level duplicate-density warnings are useful mainly for
            # free-text/name-like columns where excessive repetition looks fake.
            if col_type not in {"text", "address", "email", "phone", "url"}:
                continue

            series = df[col_name].dropna()
            if series.empty:
                continue
            counts = series.astype(str).value_counts(dropna=True)
            if counts.empty:
                continue
            top_count = int(counts.iloc[0])
            top_ratio = float(top_count / len(df))
            if top_ratio >= top_value_ratio_warning:
                records.append(
                    {
                        "table": table_name,
                        "column": col_name,
                        "category": "duplicate_density",
                        "message": (
                            f"High repetition in '{table_name}.{col_name}': top value appears "
                            f"in {top_ratio:.1%} of rows."
                        ),
                        "details": {
                            "top_value": str(counts.index[0]),
                            "top_count": top_count,
                            "top_ratio": round(top_ratio, 4),
                        },
                    }
                )

        if len(business_cols) >= 2:
            fp_dup_count = int(df.duplicated(subset=business_cols, keep="first").sum())
            fp_ratio = fp_dup_count / len(df)
            if fp_ratio >= fingerprint_ratio_warning:
                records.append(
                    {
                        "table": table_name,
                        "column": None,
                        "category": "duplicate_fingerprint_density",
                        "message": (
                            f"{fp_dup_count} rows in '{table_name}' repeat the same business fingerprint "
                            f"({fp_ratio:.1%} of rows)."
                        ),
                        "details": {
                            "duplicate_rows": fp_dup_count,
                            "duplicate_ratio": round(fp_ratio, 4),
                            "fingerprint_columns": business_cols,
                        },
                    }
                )

    return records


def locked_columns_from_distribution_rules(rules: Sequence[Any]) -> Dict[str, Set[str]]:
    """Build a locked-column map from distribution rules."""
    locked: Dict[str, Set[str]] = {}
    for rule in rules or []:
        table = getattr(rule, "table", None)
        column = getattr(rule, "column", None)
        if table and column:
            locked.setdefault(str(table), set()).add(str(column))
    return locked


def _schema_columns(schema: Any, table_name: str) -> List[Any]:
    if schema is None:
        return []
    if hasattr(schema, "get_columns"):
        return list(schema.get_columns(table_name) or [])
    return list(getattr(schema, "columns", {}).get(table_name, []) or [])


def _schema_column(schema: Any, table_name: str, column_name: str) -> Optional[Any]:
    for col in _schema_columns(schema, table_name):
        if getattr(col, "name", None) == column_name:
            return col
    return None


def _relationship_key_names(schema: Any, table_name: str) -> Set[str]:
    keys: Set[str] = set()
    for rel in getattr(schema, "relationships", []) or []:
        if getattr(rel, "parent_table", None) == table_name:
            keys.add(str(getattr(rel, "parent_key", "")))
        if getattr(rel, "child_table", None) == table_name:
            keys.add(str(getattr(rel, "child_key", "")))
    return {k for k in keys if k}


def _is_relationship_key(schema: Any, table_name: str, column_name: str) -> bool:
    return column_name in _relationship_key_names(schema, table_name)


def _is_protected_column(schema: Any, table_name: str, column_name: str, column_spec: Optional[Any]) -> bool:
    name = str(column_name).lower()
    compact = name.replace("_", "")
    col_type = str(getattr(column_spec, "type", "") or "").lower()

    if bool(getattr(column_spec, "unique", False)):
        return True
    if col_type in PROTECTED_TYPES:
        return True
    if _is_relationship_key(schema, table_name, column_name):
        return True
    if name == "id" or name.endswith("_id") or (name.endswith("id") and len(name) <= 12):
        return True

    for part in PROTECTED_NAME_PARTS:
        part_compact = part.replace("_", "")
        if name == part or compact == part_compact or part in name:
            return True

    return False


def _is_volatile_column(column_name: str, column_spec: Optional[Any]) -> bool:
    name = str(column_name).lower()
    col_type = str(getattr(column_spec, "type", "") or "").lower()
    if col_type in {"date", "time", "datetime"}:
        return True
    return any(part in name for part in VOLATILE_NAME_PARTS)


def _business_columns(schema: Any, table_name: str, df: pd.DataFrame, locked: Set[str]) -> List[str]:
    cols: List[str] = []
    for column_name in df.columns:
        if column_name in locked:
            continue
        spec = _schema_column(schema, table_name, column_name)
        if _is_protected_column(schema, table_name, column_name, spec):
            continue
        if _is_volatile_column(column_name, spec):
            continue
        if df[column_name].isna().all():
            continue
        cols.append(column_name)
    return cols


def _repair_unique_column(df: pd.DataFrame, col_name: str, column_spec: Any, dup_mask: pd.Series, rng: np.random.Generator) -> int:
    used = {str(v) for v in df.loc[~dup_mask, col_name].dropna().tolist()}
    indices = df.index[dup_mask].tolist()
    col_type = str(getattr(column_spec, "type", "") or "").lower()
    series = df[col_name]
    repaired = 0

    for offset, idx in enumerate(indices, start=1):
        new_value = _new_unique_value(series, col_type, used, offset, rng)
        df.at[idx, col_name] = new_value
        used.add(str(new_value))
        repaired += 1
    return repaired


def _new_unique_value(series: pd.Series, col_type: str, used: Set[str], offset: int, rng: np.random.Generator) -> Any:
    if col_type == "uuid":
        # Deterministic enough for synthetic repair, without importing uuid state.
        while True:
            candidate = f"synthetic-{int(rng.integers(10**12, 10**13 - 1))}"
            if candidate not in used:
                return candidate

    if pd.api.types.is_integer_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        base = int(numeric.max()) if numeric.notna().any() else 0
        candidate = base + offset
        while str(candidate) in used:
            candidate += 1
        return candidate

    if pd.api.types.is_float_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        base = float(numeric.max()) if numeric.notna().any() else 0.0
        candidate = base + (offset / 1000.0)
        while str(candidate) in used:
            candidate += 0.001
        return candidate

    base_text = "synthetic_unique"
    candidate = f"{base_text}_{offset}"
    while candidate in used:
        offset += 1
        candidate = f"{base_text}_{offset}"
    return candidate


def _mutate_duplicate_rows(
    df: pd.DataFrame,
    schema: Any,
    table_name: str,
    duplicate_mask: pd.Series,
    safe_cols: Sequence[str],
    rng: np.random.Generator,
) -> int:
    indices = df.index[duplicate_mask].tolist()
    if not indices:
        return 0

    repaired = 0
    preferred_cols = _rank_mutation_columns(df, schema, table_name, safe_cols)
    if not preferred_cols:
        return 0

    for n, idx in enumerate(indices, start=1):
        col_name = preferred_cols[(n - 1) % len(preferred_cols)]
        spec = _schema_column(schema, table_name, col_name)
        old_value = df.at[idx, col_name]
        df.at[idx, col_name] = _mutated_value(old_value, df[col_name], spec, n, rng)
        repaired += 1

    return repaired


def _rank_mutation_columns(df: pd.DataFrame, schema: Any, table_name: str, safe_cols: Sequence[str]) -> List[str]:
    # Prefer categorical/numeric columns because mutating names or free text can look artificial.
    ranked: List[Tuple[int, str]] = []
    for col_name in safe_cols:
        spec = _schema_column(schema, table_name, col_name)
        col_type = str(getattr(spec, "type", "") or "").lower()
        if col_type == "categorical":
            priority = 0
        elif pd.api.types.is_numeric_dtype(df[col_name]) or col_type in {"int", "float", "decimal", "money"}:
            priority = 1
        elif pd.api.types.is_bool_dtype(df[col_name]) or col_type == "boolean":
            priority = 2
        else:
            priority = 3
        ranked.append((priority, col_name))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [col for _, col in ranked]


def _mutated_value(value: Any, series: pd.Series, column_spec: Optional[Any], n: int, rng: np.random.Generator) -> Any:
    col_type = str(getattr(column_spec, "type", "") or "").lower()
    params = dict(getattr(column_spec, "distribution_params", {}) or {})

    if col_type == "categorical":
        choices = [c for c in params.get("choices", []) if str(c) != str(value)]
        if choices:
            return choices[(n - 1) % len(choices)]

    if pd.api.types.is_bool_dtype(series) or isinstance(value, (bool, np.bool_)):
        return not bool(value)

    if pd.api.types.is_integer_dtype(series) or isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        try:
            return int(value) + n
        except Exception:
            return n

    if pd.api.types.is_float_dtype(series) or isinstance(value, (float, np.floating)):
        try:
            return round(float(value) + (n / 100.0), 6)
        except Exception:
            return float(n) / 100.0

    text = "" if pd.isna(value) else str(value)
    suffix = f" variant {n}"
    max_len = int(params.get("max_length", 0) or 0)
    candidate = f"{text}{suffix}" if text else f"synthetic variant {n}"
    if max_len > 0:
        candidate = candidate[:max_len]
    return candidate


__all__ = [
    "DuplicateRepairAction",
    "DuplicateRepairReport",
    "duplicate_quality_records",
    "duplicate_validation_records",
    "enforce_duplicate_policy",
    "locked_columns_from_distribution_rules",
]
