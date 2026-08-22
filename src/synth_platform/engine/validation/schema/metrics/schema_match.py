"""Schema match metric — compare expected vs actual schema fields."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from synth_platform.engine.inference.schema.schema import SchemaConfig


def _field_key(table: str, column: str, aspect: str) -> str:
    return f"{table}.{column}:{aspect}"


def compare_schema_to_tables(
    schema: SchemaConfig,
    tables: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Compare declared schema against generated table columns and dtypes."""
    expected: List[str] = []
    matched: List[str] = []
    mismatches: List[Dict[str, Any]] = []

    for table in schema.tables:
        table_name = table.name
        df = tables.get(table_name)
        declared_cols = schema.get_columns(table_name)
        if df is None:
            for col in declared_cols:
                expected.append(_field_key(table_name, col.name, "present"))
                mismatches.append({"table": table_name, "column": col.name, "issue": "table missing"})
            continue

        actual_cols = set(df.columns)
        for col in declared_cols:
            expected.append(_field_key(table_name, col.name, "present"))
            if col.name not in actual_cols:
                mismatches.append({"table": table_name, "column": col.name, "issue": "column missing"})
                continue
            matched.append(_field_key(table_name, col.name, "present"))

            expected.append(_field_key(table_name, col.name, "nullable"))
            series = df[col.name]
            has_nulls = bool(series.isna().any())
            if col.nullable or not has_nulls:
                matched.append(_field_key(table_name, col.name, "nullable"))
            else:
                mismatches.append(
                    {"table": table_name, "column": col.name, "issue": "unexpected nulls in non-nullable column"}
                )

            if col.type == "categorical":
                choices = (col.distribution_params or {}).get("choices") or []
                if choices:
                    expected.append(_field_key(table_name, col.name, "categories"))
                    observed = set(series.dropna().astype(str).unique())
                    allowed = {str(c) for c in choices}
                    if observed.issubset(allowed):
                        matched.append(_field_key(table_name, col.name, "categories"))
                    else:
                        extra = sorted(observed - allowed)[:5]
                        mismatches.append(
                            {
                                "table": table_name,
                                "column": col.name,
                                "issue": "values outside allowed categories",
                                "extra_values": extra,
                            }
                        )

    total = max(len(expected), 1)
    score = len(matched) / total
    return {
        "metric": "schema_match",
        "score": round(score, 6),
        "target": 1.0,
        "passed": score >= 1.0,
        "matched_fields": len(matched),
        "total_expected_fields": len(expected),
        "mismatches": mismatches,
    }


def compare_source_profile_schema(
    source_columns: Sequence[str],
    synthetic_columns: Sequence[str],
) -> Dict[str, Any]:
    """Compare source and synthetic column sets for source-driven mode."""
    source_set = list(dict.fromkeys(source_columns))
    synthetic_set = set(synthetic_columns)
    matched = [name for name in source_set if name in synthetic_set]
    missing = [name for name in source_set if name not in synthetic_set]
    extra = [name for name in synthetic_columns if name not in set(source_set)]
    total = max(len(source_set), 1)
    score = len(matched) / total
    return {
        "metric": "schema_match",
        "score": round(score, 6),
        "target": 1.0,
        "passed": score >= 1.0 and not missing,
        "matched_fields": len(matched),
        "total_expected_fields": len(source_set),
        "missing_columns": missing,
        "extra_columns": extra,
    }
