"""Human-readable schema summaries for pipeline results."""

from __future__ import annotations

from typing import Any, Optional

from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table


def source_profile_schema_summary(
    profile: Any,
    *,
    synthetic_rows: int,
    model_type: Optional[str] = None,
) -> str:
    from synth_platform.engine.inference.schema.source_driven import ColumnKind

    schema_columns: list[Column] = []
    for name, col_profile in profile.columns.items():
        kind = str(col_profile.kind)
        if kind == ColumnKind.NUMERIC.value:
            col_type = "float"
        elif kind == ColumnKind.CATEGORICAL.value:
            col_type = "categorical"
        elif kind == ColumnKind.BOOLEAN.value:
            col_type = "boolean"
        else:
            col_type = "text"
        schema_columns.append(Column(name=name, type=col_type))

    schema = SchemaConfig(
        name=f"{profile.table_name} (source-driven)",
        domain="source_driven",
        tables=[Table(name=profile.table_name, row_count=synthetic_rows)],
        columns={profile.table_name: schema_columns},
    )
    header = [
        f"Source file rows: {profile.row_count:,}",
        f"Preview synthetic rows: {synthetic_rows:,}",
        f"SDV model: {model_type or profile.recommended_model}",
        "",
    ]
    return "\n".join(header) + schema.summary()
