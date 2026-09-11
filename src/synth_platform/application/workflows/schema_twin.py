"""Schema Mode application facade.

Thin orchestration around the existing schema-driven pipeline so presentation
layers (Streamlit Schema Twin) never own generation, validation, or export
algorithms.

All generation delegates to ``run_schema_pipeline``.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

from synth_platform.application.orchestration.schema.config import PipelineConfig
from synth_platform.application.orchestration.schema.export import make_zip_from_paths
from synth_platform.application.orchestration.schema.result import PipelineResult
from synth_platform.application.orchestration.schema.schema_driven import run_schema_pipeline
from synth_platform.domain.product_settings import (
    ProductSettingsReader,
    read_generation_defaults,
)
from synth_platform.engine.inference.schema.schema import Column, RealismConfig, Relationship, SchemaConfig, Table
from synth_platform.engine.inference.schema.schema_columns import align_unique_int_ranges
from synth_platform.engine.inference.schema.planning import GenerationPlanner
from synth_platform.engine.inference.schema.yaml_schema import load_yaml_schema

SchemaPayload = Union[Mapping[str, Any], bytes, str, Path, SchemaConfig]

WORKFLOW_STAGES = (
    "intent",
    "schema_input",
    "schema_review",
    "generation_configuration",
    "generation",
    "preview",
    "validation",
    "export",
)


@dataclass(frozen=True)
class SchemaSummary:
    """UI-facing schema overview — no generation side effects."""

    name: str
    table_count: int
    column_count: int
    relationship_count: int
    tables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SchemaModeResult:
    """Presentation-ready outcome of a Schema Mode run."""

    schema: SchemaConfig
    pipeline: PipelineResult

    @property
    def preview_tables(self) -> Dict[str, pd.DataFrame]:
        return self.pipeline.preview_tables

    @property
    def row_counts(self) -> Dict[str, int]:
        return self.pipeline.row_counts

    @property
    def validation_report(self) -> Dict[str, Any]:
        report = self.pipeline.validation_report
        return report if isinstance(report, Mapping) else {}

    @property
    def export_paths(self) -> Dict[str, Path]:
        return self.pipeline.export_paths

    @property
    def hard_checks_passed(self) -> bool:
        report = self.validation_report
        if not isinstance(report, Mapping) or not report:
            return False
        if "hard_checks_passed" in report:
            return report["hard_checks_passed"] is True
        if "passed" in report:
            return report["passed"] is True
        status = report.get("status")
        if isinstance(status, str):
            return status.lower() in {"pass", "passed", "ok", "success"}
        return False


def _normalize_column_type(value: Any) -> str:
    raw = str(value or "text").strip().lower()
    aliases = {
        "integer": "int",
        "number": "float",
        "double": "float",
        "string": "text",
        "str": "text",
        "object": "text",
        "bool": "boolean",
        "category": "categorical",
        "enum": "categorical",
        "fk": "foreign_key",
    }
    return aliases.get(raw, raw)


def _column_from_flexible_spec(name: str, spec: Any) -> Column:
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        raise ValueError(f"Column spec for {name!r} must be a string or object.")

    params = dict(spec.get("distribution_params") or spec.get("params") or spec.get("generation") or {})
    col_type = _normalize_column_type(spec.get("type") or spec.get("data_type") or spec.get("datatype"))

    for key in ("min", "max", "mean", "std", "decimals", "probability", "start", "end", "text_type"):
        if key in spec and key not in params:
            params[key] = spec[key]
    if "choices" in spec and "choices" not in params:
        params["choices"] = spec["choices"]
    if "probabilities" in spec and "probabilities" not in params:
        params["probabilities"] = spec["probabilities"]

    return Column(
        name=str(spec.get("name") or name),
        type=col_type,  # type: ignore[arg-type]
        unique=bool(spec.get("unique", False)),
        nullable=bool(spec.get("nullable", False)),
        description=spec.get("description"),
        distribution_params=params,
    )


def _relationship_from_flexible_spec(raw: Any) -> Relationship:
    if isinstance(raw, str):
        arrow = "->" if "->" in raw else "→"
        lhs, rhs = [part.strip() for part in raw.split(arrow, 1)]
        parent_table, parent_key = lhs.rsplit(".", 1)
        child_table, child_key = rhs.rsplit(".", 1)
        return Relationship(
            parent_table=parent_table,
            parent_key=parent_key,
            child_table=child_table,
            child_key=child_key,
        )
    if not isinstance(raw, Mapping):
        raise ValueError("Relationship must be a string or object.")
    if "parent" in raw and "child" in raw:
        parent_table, parent_key = str(raw["parent"]).rsplit(".", 1)
        child_table, child_key = str(raw["child"]).rsplit(".", 1)
        return Relationship(
            parent_table=parent_table,
            parent_key=parent_key,
            child_table=child_table,
            child_key=child_key,
        )
    return Relationship(
        parent_table=str(raw.get("parent_table")),
        parent_key=str(raw.get("parent_key") or raw.get("parent_column")),
        child_table=str(raw.get("child_table")),
        child_key=str(raw.get("child_key") or raw.get("child_column")),
    )


def schema_from_flexible_json(payload: Mapping[str, Any], *, seed: Optional[int] = None) -> SchemaConfig:
    """Accept common table-oriented JSON shapes as SchemaConfig."""
    raw_tables = payload.get("tables")
    if raw_tables is None:
        raise ValueError("JSON schema must contain a 'tables' section.")

    tables: List[Table] = []
    columns: Dict[str, List[Column]] = {}

    if isinstance(raw_tables, Mapping):
        table_items = []
        for table_name, table_spec in raw_tables.items():
            table_spec = table_spec or {}
            if not isinstance(table_spec, Mapping):
                raise ValueError(f"Table spec for {table_name!r} must be an object.")
            table_items.append((str(table_name), table_spec))
    elif isinstance(raw_tables, list):
        table_items = []
        for table_spec in raw_tables:
            if not isinstance(table_spec, Mapping):
                raise ValueError("Each table entry must be an object.")
            table_name = str(table_spec.get("name") or table_spec.get("table"))
            if not table_name or table_name == "None":
                raise ValueError("Each table entry must include a name.")
            table_items.append((table_name, table_spec))
    else:
        raise ValueError("'tables' must be either an object or a list.")

    for table_name, table_spec in table_items:
        row_count = int(table_spec.get("row_count") or table_spec.get("rows") or table_spec.get("count") or 100)
        tables.append(
            Table(name=table_name, row_count=max(1, row_count), description=table_spec.get("description"))
        )

        raw_columns = table_spec.get("columns")
        if raw_columns is None and isinstance(payload.get("columns"), Mapping):
            raw_columns = payload["columns"].get(table_name)
        if isinstance(raw_columns, Mapping):
            columns[table_name] = [
                _column_from_flexible_spec(col_name, col_spec) for col_name, col_spec in raw_columns.items()
            ]
        elif isinstance(raw_columns, list):
            columns[table_name] = [
                _column_from_flexible_spec(str(col.get("name")), col)
                for col in raw_columns
                if isinstance(col, Mapping)
            ]
        else:
            raise ValueError(f"Table {table_name!r} must include columns.")

    relationships = [
        _relationship_from_flexible_spec(rel)
        for rel in (payload.get("relationships") or payload.get("foreign_keys") or [])
    ]

    return SchemaConfig(
        name=str(payload.get("name") or payload.get("dataset") or "Uploaded Dataset"),
        description=payload.get("description"),
        domain=payload.get("domain"),
        seed=seed if seed is not None else payload.get("seed"),
        tables=tables,
        columns=columns,
        relationships=relationships,
    )


def load_schema_bytes(
    uploaded_bytes: bytes,
    filename: str,
    *,
    seed: Optional[int] = None,
) -> SchemaConfig:
    """Load SchemaConfig from uploaded YAML/YML/JSON/SQL-DDL bytes."""
    from synth_platform.engine.inference.schema.ddl import from_ddl

    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        payload = json.loads(uploaded_bytes.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("The schema must contain a JSON object at its root.")
        try:
            schema = SchemaConfig.model_validate(payload)
        except Exception:
            schema = schema_from_flexible_json(payload, seed=seed)
        if seed is not None:
            schema.seed = seed
        return schema

    if suffix == ".sql":
        # Generic SQL DDL ingest — reuses the schema DDL parser (CREATE TABLE).
        text = uploaded_bytes.decode("utf-8-sig")
        schema = from_ddl(text, infer_fks=True, default_rows=100)
        updates: Dict[str, Any] = {}
        if seed is not None:
            updates["seed"] = seed
        if not schema.name or schema.name == "from_ddl":
            updates["name"] = Path(filename).stem or "sql_schema"
        return schema.model_copy(update=updates) if updates else schema

    if suffix not in {".yaml", ".yml"}:
        raise ValueError("Upload a .yaml, .yml, .json, or .sql schema file.")

    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = Path(tmp.name)
    try:
        schema = load_yaml_schema(tmp_path, seed=seed)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    if seed is not None:
        schema.seed = seed
    return schema


def load_schema(payload: SchemaPayload, *, filename: str = "schema.json", seed: Optional[int] = None) -> SchemaConfig:
    """Normalize dict/bytes/path/SchemaConfig inputs into SchemaConfig."""
    if isinstance(payload, SchemaConfig):
        schema = payload.model_copy(deep=True)
        if seed is not None:
            schema.seed = seed
        return schema
    if isinstance(payload, Path):
        return load_schema_bytes(payload.read_bytes(), payload.name, seed=seed)
    if isinstance(payload, str) and Path(payload).exists():
        path = Path(payload)
        return load_schema_bytes(path.read_bytes(), path.name, seed=seed)
    if isinstance(payload, (bytes, bytearray)):
        return load_schema_bytes(bytes(payload), filename, seed=seed)
    if isinstance(payload, Mapping):
        try:
            schema = SchemaConfig.model_validate(payload)
        except Exception:
            schema = schema_from_flexible_json(payload, seed=seed)
        if seed is not None:
            schema.seed = seed
        return schema
    raise TypeError(f"Unsupported schema payload type: {type(payload)!r}")


def _friendly_table_description(table_name: str, existing: Any = None) -> str:
    if existing and str(existing).strip():
        return str(existing).strip()
    name = (table_name or "").strip().lower().replace("-", "_")
    presets = {
        "customers": "People or organizations you do business with",
        "customer": "People or organizations you do business with",
        "patients": "Patient identity and demographics",
        "patient": "Patient identity and demographics",
        "accounts": "Accounts owned by customers",
        "account": "Accounts owned by customers",
        "branches": "Branch / location reference data",
        "branch": "Branch / location reference data",
        "cards": "Payment cards linked to accounts",
        "card": "Payment cards linked to accounts",
        "merchants": "Merchant / payee reference data",
        "merchant": "Merchant / payee reference data",
        "loans": "Loan products linked to customers",
        "loan": "Loan products linked to customers",
        "transactions": "Money-movement events on accounts",
        "transaction": "Money-movement events on accounts",
        "orders": "Orders placed by customers",
        "order": "Orders placed by customers",
        "providers": "Care providers / organizations",
        "encounters": "Clinical visits / encounters",
        "medications": "Current or historical medications",
        "diagnoses": "Diagnoses linked to encounters",
    }
    if name in presets:
        return presets[name]
    label = table_name.replace("_", " ").strip().title() or "Table"
    return f"{label} records"


def summarize_schema(schema: SchemaConfig) -> SchemaSummary:
    """Best-effort table/column summary for UI review (with PK/FK/description)."""
    rels = list(schema.relationships or [])
    fks_by_table: dict[str, list[str]] = {}
    for rel in rels:
        child = getattr(rel, "child_table", None) or ""
        child_key = getattr(rel, "child_key", None) or ""
        parent = getattr(rel, "parent_table", None) or ""
        parent_key = getattr(rel, "parent_key", None) or ""
        if not child:
            continue
        fks_by_table.setdefault(child, []).append(f"{child_key} → {parent}.{parent_key}")

    table_rows: List[Dict[str, Any]] = []
    column_count = 0
    for table in schema.tables:
        cols = schema.columns.get(table.name, []) or []
        column_count += len(cols)
        pk_cols = [c.name for c in cols if getattr(c, "unique", False)]
        fk_cols = [c.name for c in cols if getattr(c, "type", None) == "foreign_key"]
        fk_links = fks_by_table.get(table.name) or [f"{name} (FK)" for name in fk_cols]
        table_rows.append(
            {
                "table": table.name,
                "rows": int(table.row_count or 0),
                "columns": len(cols),
                "primary_key": ", ".join(pk_cols) or "-",
                "foreign_keys": "; ".join(fk_links) if fk_links else "-",
                "description": _friendly_table_description(table.name, getattr(table, "description", None)),
            }
        )
    return SchemaSummary(
        name=schema.name,
        table_count=len(schema.tables),
        column_count=column_count,
        relationship_count=len(rels),
        tables=table_rows,
    )


def schema_column_details(schema: SchemaConfig) -> List[Dict[str, Any]]:
    """Flat column list for a readable schema-details table."""
    rows: List[Dict[str, Any]] = []
    fk_targets = {
        (rel.child_table, rel.child_key): f"{rel.parent_table}.{rel.parent_key}"
        for rel in (schema.relationships or [])
    }
    for table in schema.tables:
        for col in schema.columns.get(table.name, []) or []:
            role = "PK" if col.unique else ("FK" if col.type == "foreign_key" else "")
            rows.append(
                {
                    "table": table.name,
                    "column": col.name,
                    "type": col.type,
                    "role": role or "-",
                    "nullable": "yes" if col.nullable else "no",
                    "references": fk_targets.get((table.name, col.name), "-"),
                    "description": (col.description or "").strip() or "-",
                }
            )
    return rows


def list_llm_text_columns(schema: SchemaConfig) -> List[Dict[str, str]]:
    """Text-heavy columns eligible for optional LLM generation."""
    from synth_platform.engine.generation.text.eligibility import list_eligible_columns

    return [
        {
            "table": str(row["table"]),
            "column": str(row["column"]),
            "role": str(row.get("text_role") or "text"),
        }
        for row in list_eligible_columns(schema)
        if row.get("eligible")
    ]


def apply_llm_text_flags(schema: SchemaConfig, *, enabled: bool) -> SchemaConfig:
    """Mark eligible text-heavy columns so the simulator routes them through LLM+fallback.

    The UI toggle alone is not enough: DataSimulator only enters the LLM path when
    a column has ``llm_enabled`` / ``llm_text`` in distribution_params (or when the
    simulator sees global enable + eligibility). This makes the flag explicit on
    the schema before pipeline run.
    """
    from synth_platform.engine.generation.text.eligibility import list_eligible_columns

    prepared = schema.model_copy(deep=True)
    eligible = {
        (str(row["table"]), str(row["column"])): str(row.get("text_role") or "narrative")
        for row in list_eligible_columns(prepared)
        if row.get("eligible")
    }
    for table in prepared.tables:
        cols = prepared.columns.get(table.name) or []
        updated: List[Column] = []
        for col in cols:
            key = (table.name, col.name)
            if not (enabled and key in eligible):
                updated.append(col)
                continue
            params = dict(col.distribution_params or {})
            params["llm_enabled"] = True
            params["llm_text"] = True
            params.setdefault("text_type", eligible[key])
            params.setdefault("text_role", eligible[key])
            updated.append(col.model_copy(update={"distribution_params": params}))
        prepared.columns[table.name] = updated
    return prepared


def prepare_schema_for_generation(
    schema: SchemaConfig,
    *,
    row_count: int,
    seed: int,
    locale: Optional[str] = None,
    llm_text_enabled: bool = False,
) -> SchemaConfig:
    """Apply Schema Twin settings and materialize FK-aware table cardinalities."""
    from synth_platform.engine.inference.schema.semantic import enrich_schema_semantics

    prepared = enrich_schema_semantics(schema.model_copy(deep=True))
    if not prepared.domain and _looks_like_banking_schema(prepared):
        prepared.domain = "fintech"
    prepared.seed = int(seed)
    rows = max(1, int(row_count))
    for table in prepared.tables:
        table.row_count = rows
        if not (table.description or "").strip():
            table.description = _friendly_table_description(table.name)
    if locale:
        if prepared.realism is None:
            prepared.realism = RealismConfig()
        prepared.realism.locale = locale

    # Schema inference already provides a relationship graph. Reuse the engine's
    # established planner to turn the global setting into a base count for roots
    # and proportional counts for children. Materialize the plan before entering
    # the pipeline so validation/export expectations use the same counts.
    if prepared.relationships:
        if prepared.realism is None:
            prepared.realism = RealismConfig()
        prepared.realism.row_planning = "heuristic"
        prepared.realism.row_planning_base_rows = rows
        plan = GenerationPlanner(prepared).build()
        for table in prepared.tables:
            table.row_count = max(1, int(plan.row_count_for(table.name, rows)))
        prepared.realism.row_planning = "off"

    for table in prepared.tables:
        align_unique_int_ranges(prepared.columns.get(table.name, []), int(table.row_count))
    if llm_text_enabled:
        prepared = apply_llm_text_flags(prepared, enabled=True)
    return prepared


def _looks_like_banking_schema(schema: SchemaConfig) -> bool:
    """Recognize the concrete banking constellation used by Schema Twin."""
    table_names = {table.name.lower() for table in schema.tables}

    def contains(concept: str) -> bool:
        return concept in table_names or f"{concept}s" in table_names

    return (
        contains("account")
        and contains("transaction")
        and any(contains(concept) for concept in ("customer", "branch", "card", "loan", "merchant"))
    )


def generate_from_schema(
    schema: SchemaConfig,
    *,
    row_count: int | None = None,
    seed: int = 42,
    locale: str = "en_US",
    output_dir: Union[str, Path] | None = None,
    export_format: str | None = None,
    preview_rows: int | None = None,
    llm_text_enabled: bool = False,
    max_llm_rows: int = 50,
    history: Any | None = None,
    product_settings: ProductSettingsReader | None = None,
) -> SchemaModeResult:
    """Run the canonical schema-driven pipeline for Schema Mode."""
    defaults = read_generation_defaults(product_settings)
    resolved_row_count = int(row_count) if row_count is not None else int(defaults["default_record_count"])
    resolved_export_format = str(export_format or defaults["default_output_format"]).lower()
    if resolved_export_format not in {"csv", "parquet"}:
        resolved_export_format = "csv"
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="schema_twin_output_"))
    resolved_preview_rows = int(preview_rows) if preview_rows is not None else min(100, resolved_row_count)
    prepared = prepare_schema_for_generation(
        schema,
        row_count=resolved_row_count,
        seed=seed,
        locale=locale,
        llm_text_enabled=bool(llm_text_enabled),
    )
    total_rows = sum(max(1, int(table.row_count or 1)) for table in prepared.tables)
    config = PipelineConfig(
        generation_mode="schema_driven",
        seed=int(seed),
        preview_rows=max(1, min(int(resolved_preview_rows), int(resolved_row_count))),
        full_rows=total_rows,
        preview_only=False,
        export_format=resolved_export_format,
        output_dir=resolved_output_dir,
        write_reports=True,
        chunk_size=min(10_000, max(1_000, int(resolved_row_count))),
        llm_text_enabled=bool(llm_text_enabled),
        llm_full_enabled=bool(llm_text_enabled),
        max_llm_rows=max(1, int(max_llm_rows)),
    )
    pipeline = run_schema_pipeline(prepared, config)
    result = SchemaModeResult(schema=prepared, pipeline=pipeline)
    if history is not None:
        try:
            history.record_run(
                workflow_type="schema",
                project_name=prepared.name or None,
                status="completed" if result.hard_checks_passed else "failed",
                validation_status="PASS" if result.hard_checks_passed else "FAIL",
                validation_passed=result.hard_checks_passed,
                output_id=str(config.output_dir),
                metadata={
                    "generation_mode": pipeline.generation_mode,
                    "row_counts": pipeline.row_counts,
                    "export_format": resolved_export_format,
                    "artifact_dir": str(config.output_dir),
                },
            )
        except Exception:
            pass
    return result


def package_download(result: SchemaModeResult) -> bytes:
    """Package exported tables and validation report into a ZIP."""
    report = result.validation_report or result.pipeline.to_dict()
    return make_zip_from_paths(
        result.export_paths,
        report_payload=report,
        report_name="validation_report.json",
    )


def validation_highlights(result: SchemaModeResult) -> Dict[str, Any]:
    """Flatten a few user-facing validation signals for the UI."""
    report = result.validation_report
    return {
        "hard_checks_passed": result.hard_checks_passed,
        "row_counts": dict(result.row_counts),
        "tables": list(result.preview_tables.keys()),
        "export_count": len(result.export_paths),
        "status": report.get("status") or report.get("overall_status") or (
            "passed" if result.hard_checks_passed else "failed"
        ),
        "issues": report.get("issues") or report.get("errors") or report.get("failures") or [],
    }


__all__ = [
    "SchemaModeResult",
    "SchemaSummary",
    "apply_llm_text_flags",
    "generate_from_schema",
    "list_llm_text_columns",
    "load_schema",
    "load_schema_bytes",
    "package_download",
    "prepare_schema_for_generation",
    "schema_column_details",
    "schema_from_flexible_json",
    "summarize_schema",
    "validation_highlights",
]
