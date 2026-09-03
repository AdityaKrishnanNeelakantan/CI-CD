"""Manual benchmark runner for the four Synthetic Data Twin modes.

The runner exercises the same workflow facades used by the app and writes a
small JSON report with timing, output counts, and NVIDIA NeMo SDK status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synth_platform.application.workflows.schema_twin import generate_from_schema, load_schema_bytes
from synth_platform.application.workflows.schema_prompt_draft import (
    draft_schema_from_prompt as shared_draft_schema_from_prompt,
)
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.discovery.database.demo.sample_database import build_sample_database
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.documents.pdf.binding_service import (
    DOCUMENT_BINDING_MAP_FILENAME,
    load_document_binding_map,
    run_semantic_binding,
)
from synth_platform.engine.documents.pdf.generation_service import (
    DOCUMENT_SYNTHETIC_VALUES_FILENAME,
    load_document_synthetic_values,
    run_value_generation,
)
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.render_service import (
    GROUND_TRUTH_FILENAME,
    RENDERED_PDF_FILENAME,
    load_document_ground_truth,
    run_document_rendering,
)
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from synth_platform.engine.documents.pdf.template_service import DOCUMENT_TEMPLATE_FILENAME, load_document_template, run_template_compilation
from synth_platform.engine.documents.pdf.validation_service import VALIDATION_REPORT_FILENAME, load_document_validation_report, run_document_validation
from synth_platform.engine.generation.backends import GeneratorFactory, UnsupportedGenerationBackend
from synth_platform.engine.generation.data_designer_provider import (
    build_data_designer_model_config,
    build_data_designer_provider,
    load_data_designer_sdk,
)
from synth_platform.engine.generation.slm_runtime import (
    data_designer_skip_health_check,
    preflight_slm_endpoint,
    require_data_designer_api_key,
    resolve_platform_slm_runtime,
)
from synth_platform.engine.generation.database.relational_service import run_relational_generation
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling
from synth_platform.engine.transcripts import build_transcript_contract, validate_transcript_non_replay
from synth_platform.infrastructure.integrations.nvidia_nemo import inspect_nvidia_nemo_environment


@contextmanager
def timer() -> Any:
    start = time.perf_counter()
    box: dict[str, float] = {}
    try:
        yield box
    finally:
        box["seconds"] = round(time.perf_counter() - start, 6)


def _status(result: Any) -> str:
    if hasattr(result, "is_success"):
        return "success" if result.is_success() else "failed"
    return "success"


def _run_stage(name: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    with timer() as elapsed:
        result = fn()
    return result, {"name": name, "seconds": elapsed["seconds"], "status": _status(result)}


def _write_pdf(path: Path, text: str) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(str(path))


def _preview_to_frame(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset
    if hasattr(dataset, "dataset"):
        return _preview_to_frame(dataset.dataset)
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    if hasattr(dataset, "to_dataframe"):
        return dataset.to_dataframe()
    if isinstance(dataset, list):
        return pd.DataFrame(dataset)
    raise RuntimeError("Preview returned an unsupported dataset shape.")


def _schema_draft_output_format() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["name", "tables"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "domain": {"type": "string"},
            "tables": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "columns"],
                    "properties": {
                        "name": {"type": "string"},
                        "row_count": {"type": "integer", "minimum": 1},
                        "description": {"type": "string"},
                        "columns": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["name", "type"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "int",
                                            "float",
                                            "date",
                                            "time",
                                            "datetime",
                                            "categorical",
                                            "foreign_key",
                                            "text",
                                            "boolean",
                                            "uuid",
                                            "email",
                                            "phone",
                                            "url",
                                            "address",
                                            "currency",
                                            "money",
                                            "decimal",
                                            "bank_account",
                                            "account_number",
                                            "routing_number",
                                            "ssn",
                                            "aadhaar",
                                            "iban",
                                            "swift_bic",
                                            "ifsc_code",
                                            "credit_card",
                                        ],
                                    },
                                    "unique": {"type": "boolean"},
                                    "nullable": {"type": "boolean"},
                                    "description": {"type": "string"},
                                    "params": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "required": ["parent_table", "parent_key", "child_table", "child_key"],
                            "properties": {
                                "parent_table": {"type": "string"},
                                "parent_key": {"type": "string"},
                                "child_table": {"type": "string"},
                                "child_key": {"type": "string"},
                            },
                        },
                    ]
                },
            },
        },
    }


def _coerce_schema_draft_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        payload = json.loads(_extract_json_object_text(value))
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"Schema draft returned an unsupported structured value: {value!r}")


def _extract_json_object_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _schema_prompt_table_hints(prompt: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "banking",
        "create",
        "data",
        "include",
        "normalized",
        "relationship",
        "relationships",
        "schema",
        "synthetic",
        "tables",
        "the",
        "to",
        "with",
    }
    hints: list[str] = []
    for match in re.finditer(r"\b[a-z][a-z0-9_]{2,}\b", prompt.lower()):
        token = match.group(0)
        if token in stopwords or token.endswith("_id"):
            continue
        if token.endswith("s") or "_" in token:
            hints.append(token)
    return sorted(dict.fromkeys(hints))


def _singular_table_name(table_name: str) -> str:
    if table_name.endswith("ies"):
        return table_name[:-3] + "y"
    return table_name.rstrip("s") or table_name


def _plural_table_name(value: str) -> str:
    token = value.strip().lower()
    if token.endswith("s"):
        return token
    if token.endswith("y"):
        return token[:-1] + "ies"
    return token + "s"


def _primary_key_for_table(table_name: str) -> str:
    singular = _singular_table_name(table_name)
    return f"{singular}_id"


def _default_columns_for_table(table_name: str) -> list[dict[str, Any]]:
    pk = _primary_key_for_table(table_name)
    singular = _singular_table_name(table_name)
    return [
        {"name": pk, "type": "uuid", "unique": True, "nullable": False},
        {"name": f"{singular}_name", "type": "text", "nullable": False},
        {"name": "status", "type": "categorical", "nullable": False, "params": {"choices": ["active", "inactive"]}},
        {"name": "created_at", "type": "datetime", "nullable": False},
    ]


def _table_columns(table: dict[str, Any]) -> set[str]:
    columns = table.get("columns") or []
    if isinstance(columns, list):
        return {str(column.get("name")) for column in columns if isinstance(column, dict)}
    if isinstance(columns, dict):
        return {str(name) for name in columns}
    return set()


def _ensure_column(table: dict[str, Any], column: dict[str, Any]) -> bool:
    columns = table.setdefault("columns", [])
    if not isinstance(columns, list):
        return False
    name = str(column.get("name"))
    if name in _table_columns(table):
        return False
    columns.append(column)
    return True


def _relationship_hints_from_prompt(prompt: str, table_hints: list[str]) -> list[tuple[str, str]]:
    known = set(table_hints)
    hints: list[tuple[str, str]] = []
    for parent, child in re.findall(r"\b([a-z][a-z0-9_]*)\s+to\s+([a-z][a-z0-9_]*)\b", prompt.lower()):
        parent_table = parent if parent in known else _plural_table_name(parent)
        child_table = child if child in known else _plural_table_name(child)
        if parent_table in known and child_table in known and parent_table != child_table:
            hints.append((parent_table, child_table))
    return list(dict.fromkeys(hints))


def _repair_schema_draft_payload(
    payload: dict[str, Any],
    table_hints: list[str],
    relationship_hints: list[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = json.loads(json.dumps(payload))
    raw_tables = repaired.get("tables")
    if not isinstance(raw_tables, list):
        return repaired, []
    repairs: list[dict[str, Any]] = []
    normalized_tables: list[dict[str, Any]] = []
    extra_relationship_hints = list(relationship_hints)
    for item in raw_tables:
        if isinstance(item, dict):
            normalized_tables.append(item)
            continue
        if isinstance(item, list) and len(item) == 2:
            parent_table = item[0] if str(item[0]) in table_hints else _plural_table_name(str(item[0]))
            child_table = item[1] if str(item[1]) in table_hints else _plural_table_name(str(item[1]))
            if parent_table in table_hints and child_table in table_hints:
                extra_relationship_hints.append((parent_table, child_table))
                repairs.append(
                    {
                        "action": "moved_table_list_item_to_relationship_hint",
                        "relationship": f"{parent_table}->{child_table}",
                    }
                )
            continue
        repairs.append({"action": "dropped_invalid_table_entry", "value": str(item)[:120]})
    repaired["tables"] = normalized_tables
    relationship_hints = list(dict.fromkeys(extra_relationship_hints))
    raw_tables = normalized_tables
    by_name = {str(table.get("name")): table for table in raw_tables if isinstance(table, dict)}
    for table_name in table_hints:
        if table_name not in by_name:
            table = {
                "name": table_name,
                "row_count": 100,
                "description": f"{table_name.replace('_', ' ').title()} records inferred from the user prompt.",
                "columns": _default_columns_for_table(table_name),
            }
            raw_tables.append(table)
            by_name[table_name] = table
            repairs.append({"action": "added_missing_prompt_table", "table": table_name})

    for parent_table, child_table in relationship_hints:
        if parent_table not in by_name or child_table not in by_name:
            continue
        child_key = _primary_key_for_table(parent_table)
        if _ensure_column(by_name[child_table], {"name": child_key, "type": "foreign_key", "nullable": False}):
            repairs.append({"action": "added_child_foreign_key_column", "table": child_table, "column": child_key})

    relationships = repaired.setdefault("relationships", [])
    if not isinstance(relationships, list):
        relationships = []
        repaired["relationships"] = relationships
        repairs.append({"action": "replaced_invalid_relationships"})
    existing = {
        (
            str(rel.get("parent_table")),
            str(rel.get("parent_key")),
            str(rel.get("child_table")),
            str(rel.get("child_key")),
        )
        for rel in relationships
        if isinstance(rel, dict)
    }
    candidates = [
        (parent_table, _primary_key_for_table(parent_table), child_table, _primary_key_for_table(parent_table))
        for parent_table, child_table in relationship_hints
    ]
    for parent_table, parent_key, child_table, child_key in candidates:
        if parent_table not in by_name or child_table not in by_name:
            continue
        if parent_key not in _table_columns(by_name[parent_table]) or child_key not in _table_columns(by_name[child_table]):
            continue
        key = (parent_table, parent_key, child_table, child_key)
        if key in existing:
            continue
        relationships.append(
            {
                "parent_table": parent_table,
                "parent_key": parent_key,
                "child_table": child_table,
                "child_key": child_key,
            }
        )
        repairs.append({"action": "added_inferred_relationship", "relationship": ".".join(key)})
    return repaired, repairs


def _scaffold_schema_from_prompt_hints(prompt: str, table_hints: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = {
        "name": "ai_drafted_schema",
        "description": prompt.strip()[:500],
        "domain": "synthetic",
        "tables": [],
        "relationships": [],
    }
    repaired, repairs = _repair_schema_draft_payload(
        payload,
        table_hints,
        _relationship_hints_from_prompt(prompt, table_hints),
    )
    repairs.insert(0, {"action": "used_prompt_scaffold_after_invalid_ai_json"})
    return repaired, repairs


def _schema_draft_cache_enabled() -> bool:
    return os.getenv("SP_SCHEMA_DRAFT_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _schema_prompt_cache_path(prompt: str, model: str) -> Path:
    key = hashlib.sha256(f"{model}|{prompt.strip()}".encode("utf-8")).hexdigest()[:16]
    cache_dir = ROOT / "build" / "schema_draft_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _load_schema_draft_cache(prompt: str, model: str) -> tuple[Any, dict[str, Any], list[dict[str, Any]]] | None:
    if not _schema_draft_cache_enabled():
        return None
    cache_path = _schema_prompt_cache_path(prompt, model)
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    schema_payload = payload.get("schema") if isinstance(payload, dict) else None
    if not isinstance(schema_payload, dict):
        return None
    schema = load_schema_bytes(json.dumps(schema_payload).encode("utf-8"), "ai_schema.json")
    repairs = payload.get("repairs", []) if isinstance(payload.get("repairs"), list) else []
    repairs = [{"action": "loaded_schema_draft_cache", "path": str(cache_path)}] + repairs
    return schema, schema_payload, repairs


def _write_schema_draft_cache(prompt: str, model: str, payload: dict[str, Any], repairs: list[dict[str, Any]]) -> None:
    if not _schema_draft_cache_enabled():
        return
    cache_path = _schema_prompt_cache_path(prompt, model)
    cache_path.write_text(
        json.dumps({"schema": payload, "repairs": repairs}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def draft_schema_from_prompt(prompt: str, *, seed: int) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    runtime = resolve_platform_slm_runtime()
    cached = _load_schema_draft_cache(prompt, runtime.model_id)
    if cached is not None:
        schema, payload, repairs = cached
        if seed is not None:
            schema.seed = seed
        return schema, payload, repairs

    try:
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Prompt-to-schema drafting requires the standalone data-designer package.") from exc

    require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    model_alias = "schema-draft-generator"
    table_hints = _schema_prompt_table_hints(prompt)
    relationship_hints = _relationship_hints_from_prompt(prompt, table_hints)
    seed_frame = pd.DataFrame(
        [
            {
                "request_id": f"schema-draft-{seed}",
                "user_request": prompt.strip(),
                "required_tables_json": json.dumps(table_hints),
                "required_relationships_json": json.dumps(relationship_hints),
            }
        ]
    )
    model_config = build_data_designer_model_config(
        dd,
        model_alias,
        workflow="schema-draft",
        runtime=runtime,
        temperature=0.1,
        top_p=0.8,
        max_tokens=768,
        skip_health_check=data_designer_skip_health_check(),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="schema_json",
            model_alias=model_alias,
            system_prompt="Return only valid JSON. No Markdown. No explanation.",
            prompt=(
                "User request:\n"
                "{{ user_request }}\n\n"
                "Required table names inferred from the user request:\n"
                "{{ required_tables_json }}\n\n"
                "Required parent-to-child relationship table pairs inferred from the user request:\n"
                "{{ required_relationships_json }}\n\n"
                "Draft a concise relational synthetic-data schema JSON object with this root shape:\n"
                "{\"name\":\"...\",\"description\":\"...\",\"domain\":\"...\",\"tables\":[...],\"relationships\":[...]}\n"
                "Each table must have name, row_count, columns. Each column must have name, type, unique, nullable, "
                "description, and optional params. Include every required table exactly once. For every relationship, "
                "use this shape only: {\"parent_table\":\"customers\",\"parent_key\":\"customer_id\","
                "\"child_table\":\"accounts\",\"child_key\":\"customer_id\"}."
            ),
        )
    )
    frame = _preview_to_frame(
        DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)]).preview(builder, num_records=1)
    )
    if frame.empty or "schema_json" not in frame:
        raise RuntimeError(f"Schema draft did not return a `schema_json` column. Raw frame: {frame}")
    try:
        payload = _coerce_schema_draft_value(frame.loc[0, "schema_json"])
        payload, repairs = _repair_schema_draft_payload(payload, table_hints, relationship_hints)
    except json.JSONDecodeError:
        payload, repairs = _scaffold_schema_from_prompt_hints(prompt, table_hints)
    schema = load_schema_bytes(json.dumps(payload).encode("utf-8"), "ai_schema.json", seed=seed)
    table_names = {table.name for table in schema.tables}
    missing_hints = [name for name in table_hints if name not in table_names]
    if missing_hints:
        raise RuntimeError(
            "Schema draft missed table names inferred from the user prompt: "
            f"{missing_hints}. Drafted tables: {sorted(table_names)}"
        )
    _write_schema_draft_cache(prompt, runtime.model_id, payload, repairs)
    return schema, payload, repairs


def _generation_error_result(
    *,
    stages: list[dict[str, Any]],
    stage_name: str,
    backend_name: str,
    error: Exception,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stages.append(
        {
            "name": stage_name,
            "seconds": 0.0,
            "status": "error",
            "backend": backend_name,
            "error": str(error),
        }
    )
    result = {
        "status": "error",
        "stages": stages,
        "generation_backend": backend_name,
        "error": str(error),
    }
    error_metadata = getattr(error, "metadata", None)
    if error_metadata:
        result["backend_metadata"] = error_metadata
    raw_exception = getattr(error, "raw_exception", None)
    if raw_exception:
        result["raw_data_designer_exception"] = raw_exception
    raw_value = getattr(error, "raw_value", None)
    if raw_value is not None:
        result["raw_data_designer_value"] = raw_value
    result.update(extra or {})
    return result


def _generation_not_implemented_result(
    *,
    stages: list[dict[str, Any]],
    stage_name: str,
    backend_name: str,
    error: UnsupportedGenerationBackend,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stages.append(
        {
            "name": stage_name,
            "seconds": 0.0,
            "status": "not_implemented",
            "backend": backend_name,
            "reason": str(error),
        }
    )
    result = {
        "status": "not_implemented",
        "stages": stages,
        "generation_backend": backend_name,
        "reason": str(error),
    }
    result.update(extra or {})
    return result


def run_schema_mode(
    out_dir: Path,
    rows: int,
    seed: int,
    generation_backend: str = "current",
    data_dir: Path | None = None,
    schema_prompt: str | None = None,
) -> dict[str, Any]:
    stage_rows: list[dict[str, Any]] = []
    schema_input: str
    drafted_schema: dict[str, Any] | None = None
    draft_repairs: list[dict[str, Any]] = []
    if schema_prompt and schema_prompt.strip():
        schema, stage = _run_stage(
            "draft_schema_from_prompt",
            lambda: shared_draft_schema_from_prompt(schema_prompt, seed=seed),
        )
        stage_rows.append(stage)
        schema, drafted_schema, draft_repairs = schema.schema, schema.payload, schema.repairs
        schema_input = "prompt"
    else:
        schema_path = (data_dir / "schema.sql") if data_dir is not None else None
        if schema_path is None or not schema_path.exists():
            schema_path = ROOT / "tests" / "fixtures" / "schema" / "schema_twin_minimal.sql"
        ddl = schema_path.read_bytes()
        schema = load_schema_bytes(ddl, schema_path.name, seed=seed)
        schema_input = str(schema_path)
    if schema_prompt and schema_prompt.strip():
        table_rows = {table.name: max(1, int(table.row_count or rows)) for table in schema.tables}
    else:
        table_rows = {table.name: max(1, rows // max(1, len(schema.tables))) for table in schema.tables}
    backend = GeneratorFactory.create(generation_backend)
    if backend.name != "current":
        try:
            result, stage = _run_stage(
                "generate_validate_export",
                lambda: backend.generate_schema(
                    schema,
                    row_count=rows,
                    table_row_counts=table_rows,
                    seed=seed,
                    output_dir=out_dir / "exports",
                    export_format="csv",
                ),
            )
            stage_rows.append(stage)
            return {
                "status": "success" if bool(result.hard_checks_passed) else "review",
                "stages": stage_rows,
                "generation_backend": backend.name,
                "schema_input": schema_input,
                "schema_prompt": schema_prompt or "",
                "drafted_schema": drafted_schema,
                "schema_draft_repairs": draft_repairs,
                "schema_row_generation_mode": os.getenv("SP_SCHEMA_ROW_GENERATION_MODE", "deterministic"),
                "table_names": [table.name for table in schema.tables],
                "row_counts": dict(result.row_counts),
                "export_paths": [str(path) for path in result.export_paths.values()],
                "hard_checks_passed": bool(result.hard_checks_passed),
                "backend_metadata": getattr(result, "metadata", {}),
            }
        except UnsupportedGenerationBackend as exc:
            return _generation_not_implemented_result(
                stages=stage_rows,
                stage_name="generate_validate_export",
                backend_name=backend.name,
                error=exc,
                extra={
                    "schema_input": schema_input,
                    "schema_prompt": schema_prompt or "",
                    "drafted_schema": drafted_schema,
                    "schema_draft_repairs": draft_repairs,
                    "schema_row_generation_mode": os.getenv("SP_SCHEMA_ROW_GENERATION_MODE", "deterministic"),
                    "table_names": [table.name for table in schema.tables],
                },
            )
        except Exception as exc:
            return _generation_error_result(
                stages=stage_rows,
                stage_name="generate_validate_export",
                backend_name=backend.name,
                error=exc,
                extra={
                    "schema_input": schema_input,
                    "schema_prompt": schema_prompt or "",
                    "drafted_schema": drafted_schema,
                    "schema_draft_repairs": draft_repairs,
                    "schema_row_generation_mode": os.getenv("SP_SCHEMA_ROW_GENERATION_MODE", "deterministic"),
                    "table_names": [table.name for table in schema.tables],
                },
            )
    result, stage = _run_stage(
        "generate_validate_export",
        lambda: generate_from_schema(
            schema,
            row_count=rows,
            table_row_counts=table_rows,
            seed=seed,
            output_dir=out_dir / "exports",
            export_format="csv",
        ),
    )
    stage_rows.append(stage)
    return {
        "status": "success",
        "stages": stage_rows,
        "generation_backend": backend.name,
        "schema_input": schema_input,
        "schema_prompt": schema_prompt or "",
        "drafted_schema": drafted_schema,
        "schema_draft_repairs": draft_repairs,
        "schema_row_generation_mode": os.getenv("SP_SCHEMA_ROW_GENERATION_MODE", "deterministic"),
        "table_names": [table.name for table in schema.tables],
        "row_counts": dict(result.row_counts),
        "export_paths": [str(path) for path in result.export_paths.values()],
        "hard_checks_passed": bool(result.hard_checks_passed),
    }


def run_database_mode(out_dir: Path, rows: int, seed: int, model_type: str, generation_backend: str = "current") -> dict[str, Any]:
    db_path = out_dir / "source.db"
    build_sample_database(db_path, seed=seed, customer_count=max(20, rows))
    adapter = SQLiteSourceAdapter({"path": str(db_path)})
    manifest = RunManifest.create(runs_dir=out_dir / "runs")
    metadata_dir = out_dir / "metadata"
    stages: list[dict[str, Any]] = []

    discovery_result, stage = _run_stage("discovery", lambda: run_discovery(adapter, manifest, config_path="manual"))
    stages.append(stage)
    discovery = json.loads(Path(discovery_result.output_references[0]).read_text(encoding="utf-8"))

    profiling_result, stage = _run_stage("profiling", lambda: run_profiling(adapter, discovery, manifest, "discovery.json", sample_limit=rows))
    stages.append(stage)
    profile = json.loads(Path(profiling_result.output_references[0]).read_text(encoding="utf-8"))

    inference_result, stage = _run_stage(
        "inference",
        lambda: run_inference(adapter, discovery, profile, manifest, "discovery.json", "profile.json", sample_limit=rows),
    )
    stages.append(stage)
    candidates = json.loads(Path(inference_result.output_references[0]).read_text(encoding="utf-8"))
    decisions = {
        table: {column: candidate["semantic_type"] for column, candidate in columns.items()}
        for table, columns in candidates["tables"].items()
    }

    _approval, stage = _run_stage(
        "contract_approval",
        lambda: run_contract_approval(
            dataset_id="manual-benchmark",
            source_fingerprint=discovery["source_fingerprint"],
            discovery_data=discovery,
            candidates_by_table=candidates["tables"],
            manifest=manifest,
            metadata_dir=metadata_dir,
            candidates_reference="semantic_candidates.json",
            decisions=decisions,
        ),
    )
    stages.append(stage)
    contract = load_dataset_contract(metadata_dir)
    backend = GeneratorFactory.create(generation_backend)
    if backend.name != "current":
        try:
            table_count = max(1, len(contract["tables"]))
            row_plan = {table_name: max(5, rows // table_count) for table_name in contract["tables"]}
            result, stage = _run_stage(
                "data_designer_generation",
                lambda: backend.generate_database(
                    contract,
                    adapter=adapter,
                    manifest=manifest,
                    rows=rows,
                    seed=seed,
                    model_type=model_type,
                    row_plan=row_plan,
                    contract_reference="dataset_contract.json",
                ),
            )
            stages.append(stage)
            return {
                "status": "success",
                "stages": stages,
                "generation_backend": backend.name,
                "tables": sorted(contract["tables"]),
                "generation_metrics": getattr(result, "metrics", {}),
            }
        except UnsupportedGenerationBackend as exc:
            return _generation_not_implemented_result(
                stages=stages,
                stage_name="data_designer_generation",
                backend_name=backend.name,
                error=exc,
                extra={"tables": sorted(contract["tables"]) if contract else []},
            )
        except Exception as exc:
            return _generation_error_result(
                stages=stages,
                stage_name="data_designer_generation",
                backend_name=backend.name,
                error=exc,
                extra={"tables": sorted(contract["tables"]) if contract else []},
            )

    training_result, stage = _run_stage(
        "training_and_sampling",
        lambda: run_training_and_sampling(
            adapter,
            contract,
            manifest,
            "dataset_contract.json",
            sample_limit=rows,
            num_rows_to_generate=max(5, rows // 4),
            seed=seed,
            model_type=model_type,
        ),
    )
    stages.append(stage)
    training_report = load_training_report(Path(training_result.output_references[0]))

    adapters = {}
    row_plan = {}
    for table_name, table_report in training_report["tables"].items():
        adapters[table_name] = get_synthesizer_adapter_class(table_report["model_type"]).load(table_report["model_path"])
        row_plan[table_name] = max(5, rows // max(1, len(training_report["tables"])))
    generation_result, stage = _run_stage(
        "relational_generation",
        lambda: run_relational_generation(contract, adapters, row_plan, manifest, contract_reference="dataset_contract.json", seed=seed),
    )
    stages.append(stage)
    return {
        "status": "success",
        "stages": stages,
        "generation_backend": backend.name,
        "model_type": model_type,
        "tables": sorted(training_report["tables"]),
        "generation_metrics": getattr(generation_result, "metrics", {}),
    }


def run_pdf_mode(out_dir: Path, seed: int, pdf_input: Path | None = None, generation_backend: str = "current") -> dict[str, Any]:
    pdf_path = pdf_input or out_dir / "statement.pdf"
    if pdf_input is None:
        _write_pdf(
            pdf_path,
            "Account Number: 8823471\nTotal Due: 401.50\nCustomer Name: Grace Hopper\n"
            "Reference Code: RC-88291\nIssued Date: 2024-06-01\nBeginning balance 69.96\nEnding balance 5340.43",
        )
    manifest = RunManifest.create(runs_dir=out_dir / "runs")
    metadata_dir = out_dir / "metadata"
    doc_id = "manual_pdf"
    stages: list[dict[str, Any]] = []

    _result, stage = _run_stage("document_profiling", lambda: run_document_profiling(PDFDocumentAdapter(), pdf_path, doc_id, manifest, metadata_dir))
    stages.append(stage)
    profile = load_document_profile(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}"))

    _result, stage = _run_stage("template_compilation", lambda: run_template_compilation(pdf_path, doc_id, manifest, extraction_method=profile["extraction_method"]))
    stages.append(stage)
    template = load_document_template(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_TEMPLATE_FILENAME}"))

    _result, stage = _run_stage("semantic_binding", lambda: run_semantic_binding(template, doc_id, manifest, template_reference=str(pdf_path)))
    stages.append(stage)
    binding_map = load_document_binding_map(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_BINDING_MAP_FILENAME}"))
    backend = GeneratorFactory.create(generation_backend)
    if backend.name != "current":
        try:
            values, stage = _run_stage(
                "value_generation",
                lambda: backend.generate_pdf(
                    {
                        "template": template,
                        "binding_map": binding_map,
                        "doc_id": doc_id,
                        "manifest": manifest,
                        "binding_map_reference": str(pdf_path),
                    },
                    seed=seed,
                ),
            )
            stages.append(stage)
        except UnsupportedGenerationBackend as exc:
            return _generation_not_implemented_result(
                stages=stages,
                stage_name="value_generation",
                backend_name=backend.name,
                error=exc,
                extra={
                    "input_pdf": str(pdf_path),
                    "extraction_method": profile["extraction_method"],
                },
            )
        except Exception as exc:
            return _generation_error_result(
                stages=stages,
                stage_name="value_generation",
                backend_name=backend.name,
                error=exc,
                extra={
                    "input_pdf": str(pdf_path),
                    "extraction_method": profile["extraction_method"],
                },
            )
    else:
        _result, stage = _run_stage("value_generation", lambda: run_value_generation(template, binding_map, doc_id, manifest, binding_map_reference=str(pdf_path), seed=seed))
        stages.append(stage)
        values = load_document_synthetic_values(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}"))

    _result, stage = _run_stage("rendering", lambda: run_document_rendering(template, binding_map, values, doc_id, manifest, synthetic_values_reference=str(pdf_path)))
    stages.append(stage)
    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/{doc_id}/{GROUND_TRUTH_FILENAME}"))
    rendered_pdf = manifest.output_path(f"documents/{doc_id}/{RENDERED_PDF_FILENAME}")

    _result, stage = _run_stage("validation", lambda: run_document_validation(rendered_pdf, ground_truth, doc_id, manifest, ground_truth_reference=str(pdf_path)))
    stages.append(stage)
    validation = load_document_validation_report(manifest.output_path(f"documents/{doc_id}/{VALIDATION_REPORT_FILENAME}"))
    return {
        "status": "success",
        "stages": stages,
        "generation_backend": backend.name,
        "extraction_method": profile["extraction_method"],
        "input_pdf": str(pdf_path),
        "rendered_pdf": str(rendered_pdf),
        "hard_checks_passed": validation.get("hard_checks_passed"),
    }


def run_transcript_mode(
    out_dir: Path,
    seed: int,
    *,
    generation_backend: str,
    transcript_input: Path | None,
    nvidia_enabled: bool,
    curator_base_url: str,
    curator_api_key: str,
    curator_model: str,
    guardrails_config_path: str,
) -> dict[str, Any]:
    source_name = "manual_transcript.txt"
    if transcript_input is not None:
        text = transcript_input.read_text(encoding="utf-8")
        source_name = transcript_input.name
    else:
        text = (
            "Customer: My name is Grace Hopper and my email is grace@example.com. "
            "I need help with claim CLM-12345678 after my vehicle was rear-ended.\n"
            "Agent: I can help. Can you confirm the policy number POL-ABC-1234 and phone 555-123-4567?\n"
            "Customer: Yes, please send the claim status update."
        )
    options = {
        "enabled": nvidia_enabled,
        "curator_base_url": curator_base_url,
        "curator_api_key": curator_api_key,
        "curator_model": curator_model,
        "guardrails_config_path": guardrails_config_path,
    }
    stages: list[dict[str, Any]] = []
    contract, stage = _run_stage("build_contract", lambda: build_transcript_contract(text, source_name=source_name, nvidia_options=options))
    stages.append(stage)
    output_mode = os.getenv("SP_TRANSCRIPT_TWIN_OUTPUT_MODE", "ssot").strip().lower()
    if output_mode != "conversation":
        ssot = _structured_ssot_from_contract(contract)
        validation, stage = _run_stage("validate_transcript_ssot", lambda: _validate_transcript_ssot(contract, ssot))
        stages.append(stage)
        output_path = out_dir / "transcript_twin_ssot.json"
        output_path.write_text(json.dumps(ssot, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "status": "success" if validation["passed"] else "review",
            "stages": stages,
            "input_transcript": str(transcript_input) if transcript_input else source_name,
            "generation_backend": generation_backend,
            "output_mode": "ssot",
            "output_path": str(output_path),
            "model": os.getenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm"),
            "model_alias": "transcript-ssot-builder",
            "nvidia_nemo_curator": contract.privacy_policy.get("nvidia_nemo_curator"),
            "nvidia_nemo_guardrails": "not_run",
            "validation_passed": validation["passed"],
            "validation": validation,
        }
    backend = GeneratorFactory.create(generation_backend)
    try:
        synthetic, stage = _run_stage("generate_interaction", lambda: backend.generate_transcript(contract, turn_count=8, seed=seed))
        stages.append(stage)
    except Exception as exc:
        stages.append(
            {
                "name": "generate_interaction",
                "seconds": 0.0,
                "status": "error",
                "backend": backend.name,
                "error": str(exc),
            }
        )
        return {
            "status": "error",
            "stages": stages,
            "input_transcript": str(transcript_input) if transcript_input else source_name,
            "generation_backend": backend.name,
            "error": str(exc),
            "nvidia_nemo_curator": contract.privacy_policy.get("nvidia_nemo_curator"),
            "nvidia_nemo_guardrails": "not_run",
            "validation_passed": False,
        }
    validation, stage = _run_stage("validate_interaction", lambda: validate_transcript_non_replay(contract, synthetic, nvidia_options=options))
    stages.append(stage)
    (out_dir / "synthetic_transcript.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    (out_dir / "transcript_validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {
        "status": "success" if bool(validation.get("passed")) else "review",
        "stages": stages,
        "input_transcript": str(transcript_input) if transcript_input else source_name,
        "generation_backend": backend.name,
        "output_mode": "conversation",
        "turn_count": len(synthetic),
        "nvidia_nemo_curator": contract.privacy_policy.get("nvidia_nemo_curator"),
        "nvidia_nemo_guardrails": validation.get("nvidia_nemo_guardrails"),
        "validation_passed": validation.get("passed"),
        "validation": validation,
    }


def _structured_ssot_from_contract(contract: Any) -> dict[str, Any]:
    metadata = contract.entities[0].metadata if contract and contract.entities else {}
    ssot = metadata.get("structured_ssot")
    return dict(ssot) if isinstance(ssot, dict) else {"status": "missing"}


def _validate_transcript_ssot(contract: Any, ssot: dict[str, Any]) -> dict[str, Any]:
    required = {
        "metadata",
        "entities",
        "support_context",
        "resolved_issues",
        "actions_taken",
        "sentiment_analysis",
        "privacy_validation",
    }
    missing = sorted(required - set(ssot))
    serialized = json.dumps(ssot, sort_keys=True, default=str).lower()
    forbidden_speakers = any(label in serialized for label in ("speaker_1", "speaker_2"))
    raw_hashes_present = "source_turn_hashes" in serialized or "source_ngram_hashes" in serialized
    raw_source_text_used = False
    if contract and contract.entities:
        source_hashes = set(contract.entities[0].metadata.get("source_turn_hashes") or [])
        raw_source_text_used = any(source_hash and source_hash in serialized for source_hash in source_hashes)
    empty_sections = _empty_transcript_ssot_sections(ssot)
    passed = (
        ssot.get("status") == "generated"
        and not missing
        and not empty_sections
        and not forbidden_speakers
        and not raw_hashes_present
        and not raw_source_text_used
    )
    return {
        "passed": passed,
        "status": "passed" if passed else "review",
        "missing_required_sections": missing,
        "empty_required_sections": empty_sections,
        "forbidden_speaker_labels_present": forbidden_speakers,
        "raw_hashes_present": raw_hashes_present,
        "raw_source_text_used": raw_source_text_used,
    }


def _empty_transcript_ssot_sections(ssot: dict[str, Any]) -> list[str]:
    empty: list[str] = []
    for key in ("metadata", "entities", "support_context", "sentiment_analysis", "privacy_validation"):
        value = ssot.get(key)
        if not isinstance(value, dict) or not any(str(item).strip() for item in value.values()):
            empty.append(key)
    for key in ("resolved_issues", "actions_taken"):
        value = ssot.get(key)
        if not isinstance(value, list) or not any(isinstance(item, dict) and item for item in value):
            empty.append(key)
    return empty


def _summarize_mode(result: dict[str, Any]) -> dict[str, Any]:
    seconds = round(sum(float(stage["seconds"]) for stage in result.get("stages", [])), 6)
    return {"status": result.get("status"), "seconds": seconds, "stage_count": len(result.get("stages", []))}


def _run_mode(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        with timer() as elapsed:
            result = fn()
        result["total_seconds"] = elapsed["seconds"]
        result["summary"] = _summarize_mode(result)
        return result
    except Exception as exc:
        return {
            "status": "error",
            "summary": {"status": "error", "seconds": 0.0, "stage_count": 0},
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _write_markdown_report(report: dict[str, Any], report_path: Path) -> None:
    comparison = report.get("comparison", {})
    modes = report.get("modes", {})
    lines = [
        "# NeMo SDK Workflow Benchmark",
        "",
        "## Inputs",
        "",
        f'- Data folder: `{report["parameters"].get("data_dir")}`',
        f'- NVIDIA enabled: `{report["parameters"].get("nvidia_enabled")}`',
        f'- Generation backend: `{report["parameters"].get("generation_backend")}`',
        f'- Python: `{str(report.get("python", "")).split()[0]}`',
        "",
        "## Workflow Summary",
        "",
        "| Workflow | Status | Runtime Seconds | Stage Count | Key Validation |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for workflow in ["schema", "database", "pdf", "transcript"]:
        row = comparison.get(workflow, {})
        mode = modes.get(workflow, {})
        validation = ""
        if workflow == "schema":
            validation = f'hard_checks_passed={mode.get("hard_checks_passed")}'
        elif workflow == "database":
            metrics = mode.get("generation_metrics", {})
            validation = f'fk_validity={metrics.get("overall_fk_validity")}, rows={metrics.get("total_rows")}'
        elif workflow == "pdf":
            validation = f'hard_checks_passed={mode.get("hard_checks_passed")}'
        elif workflow == "transcript":
            validation = f'validation_passed={mode.get("validation_passed")}'
        lines.append(
            f'| {workflow} | {row.get("status", "")} | {row.get("seconds", "")} | '
            f'{row.get("stage_count", "")} | {validation} |'
        )

    lines.extend(
        [
            "",
            "## NeMo SDK Status",
            "",
            "| Role | Package | Installed | Python Supported | Version |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for role, status in report.get("nvidia_nemo_environment", {}).items():
        lines.append(
            f'| {role} | {status.get("package")} | {status.get("installed")} | '
            f'{status.get("python_supported")} | {status.get("version") or ""} |'
        )

    transcript = modes.get("transcript", {})
    curator_status = transcript.get("nvidia_nemo_curator") or {}
    guardrails_status = transcript.get("nvidia_nemo_guardrails") or {}
    if not isinstance(curator_status, dict):
        curator_status = {"status": curator_status}
    if not isinstance(guardrails_status, dict):
        guardrails_status = {"status": guardrails_status}
    lines.extend(
        [
            "",
            "## Data-Backed Evidence",
            "",
            f'- Schema input: `{modes.get("schema", {}).get("schema_input", "")}`',
            f'- Schema prompt: `{modes.get("schema", {}).get("schema_prompt", "")}`',
            f'- Schema row generation mode: `{modes.get("schema", {}).get("schema_row_generation_mode", "")}`',
            f'- Schema tables: `{", ".join(modes.get("schema", {}).get("table_names", []) or [])}`',
            f'- Schema row counts: `{json.dumps(modes.get("schema", {}).get("row_counts", {}), sort_keys=True)}`',
            f'- Schema draft repairs: `{len(modes.get("schema", {}).get("schema_draft_repairs", []) or [])}`',
            f'- Schema quality repairs: `{len(((modes.get("schema", {}).get("backend_metadata") or {}).get("data_designer") or {}).get("quality_repairs", []) or [])}`',
            f'- PDF input: `{modes.get("pdf", {}).get("input_pdf", "")}`',
            f'- Transcript input: `{transcript.get("input_transcript", "")}`',
            f'- NeMo Curator status: `{curator_status.get("status", "")}`',
            f'- NeMo Guardrails status: `{guardrails_status.get("status", "")}`',
            "",
            "## Decision",
            "",
            "Keep the platform workflow contracts and evaluate NeMo only as an optional generation, curation, or guardrails backend.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Schema, Database, PDF, and Transcript Twin modes.")
    parser.add_argument("--out-dir", default=str(ROOT / "build" / "manual_nemo_benchmark"))
    parser.add_argument("--rows", type=int, default=60, help="Approximate row scale for schema/database modes.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-type", default="safe_gaussian_copula", help="Database Twin synthesizer model type.")
    parser.add_argument("--only", nargs="*", choices=["schema", "database", "pdf", "transcript"], default=None)
    parser.add_argument("--generation-backend", choices=["current", "nemo"], default="current", help="Generator backend for stages with a backend abstraction.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Folder containing local evaluation inputs.")
    parser.add_argument("--schema-prompt", default="", help="Natural-language request used to draft a schema before schema generation.")
    parser.add_argument("--schema-prompt-file", type=Path, default=None, help="File containing a natural-language schema draft request.")
    parser.add_argument("--pdf-input", type=Path, default=None, help="PDF input for PDF Twin mode.")
    parser.add_argument("--transcript-input", type=Path, default=None, help="Transcript input for Customer Interaction Twin mode.")
    parser.add_argument("--nvidia-enabled", action="store_true", help="Enable NeMo Curator/Guardrails hooks where wired.")
    parser.add_argument("--curator-base-url", default="")
    parser.add_argument("--curator-api-key", default="")
    parser.add_argument("--curator-model", default="meta/llama-3.1-70b-instruct")
    parser.add_argument("--guardrails-config-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or ["schema", "database", "pdf", "transcript"])
    schema_prompt = args.schema_prompt or ""
    if args.schema_prompt_file:
        schema_prompt = args.schema_prompt_file.read_text(encoding="utf-8").strip()
    pdf_input = args.pdf_input or args.data_dir / "sample_clinical_study_records.pdf"
    transcript_input = args.transcript_input or args.data_dir / "Transcript1-HP.txt"
    if not pdf_input.exists():
        pdf_input = None
    if not transcript_input.exists():
        transcript_input = None
    env = inspect_nvidia_nemo_environment()
    report: dict[str, Any] = {
        "parameters": vars(args),
        "python": sys.version,
        "nvidia_nemo_environment": {
            status.role: {
                "package": status.package,
                "import_name": status.import_name,
                "installed": status.installed,
                "version": status.version,
                "python_supported": status.python_supported,
            }
            for status in env.packages
        },
        "modes": {},
    }
    mode_fns: dict[str, Callable[[], dict[str, Any]]] = {
        "schema": lambda: run_schema_mode(
            out_dir / "schema",
            args.rows,
            args.seed,
            generation_backend=args.generation_backend,
            data_dir=args.data_dir,
            schema_prompt=schema_prompt,
        ),
        "database": lambda: run_database_mode(out_dir / "database", args.rows, args.seed, args.model_type, generation_backend=args.generation_backend),
        "pdf": lambda: run_pdf_mode(out_dir / "pdf", args.seed, pdf_input=pdf_input, generation_backend=args.generation_backend),
        "transcript": lambda: run_transcript_mode(
            out_dir / "transcript",
            args.seed,
            generation_backend=args.generation_backend,
            transcript_input=transcript_input,
            nvidia_enabled=args.nvidia_enabled,
            curator_base_url=args.curator_base_url,
            curator_api_key=args.curator_api_key,
            curator_model=args.curator_model,
            guardrails_config_path=args.guardrails_config_path,
        ),
    }
    for mode_name, fn in mode_fns.items():
        if mode_name in selected:
            (out_dir / mode_name).mkdir(parents=True, exist_ok=True)
            print(f"Running {mode_name} mode...")
            report["modes"][mode_name] = _run_mode(mode_name, fn)

    comparison = {
        mode_name: mode_report.get("summary", {})
        for mode_name, mode_report in report["modes"].items()
    }
    report["comparison"] = comparison
    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    markdown_path = out_dir / "benchmark_summary.md"
    _write_markdown_report(report, markdown_path)
    print(json.dumps(comparison, indent=2))
    print(f"Report: {report_path}")
    print(f"Summary: {markdown_path}")
    return 1 if any(row.get("status") == "error" for row in report["modes"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
