"""Type adherence metric — share of non-null values matching declared types."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from synth_platform.engine.validation.schema.format_checks import invalid_typed_mask, schema_column_type_map
from synth_platform.engine.inference.schema.schema import SchemaConfig


def compute_type_adherence(
    tables: Dict[str, pd.DataFrame],
    schema: SchemaConfig,
) -> Dict[str, Any]:
    """Return aggregate type-adherence score across all typed columns."""
    type_map = schema_column_type_map(schema)
    per_column: List[Dict[str, Any]] = []
    matching = 0
    non_null_total = 0

    for table_name, df in tables.items():
        for column_name, declared_type in type_map.get(table_name, {}).items():
            if column_name not in df.columns:
                continue
            series = df[column_name]
            non_null = series.dropna()
            if non_null.empty:
                continue
            invalid = invalid_typed_mask(non_null, declared_type)
            valid_count = int((~invalid).sum())
            count = int(len(non_null))
            non_null_total += count
            matching += valid_count
            per_column.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "declared_type": declared_type,
                    "non_null_values": count,
                    "matching_values": valid_count,
                    "adherence": round(valid_count / count, 6) if count else 1.0,
                }
            )

    score = matching / non_null_total if non_null_total else 1.0
    target = 0.999
    return {
        "metric": "type_adherence",
        "score": round(score, 6),
        "target": target,
        "passed": score >= target,
        "matching_values": matching,
        "non_null_values": non_null_total,
        "columns": per_column,
    }
