"""Prompt-to-schema drafting through the configured local SLM provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synth_platform.application.workflows.schema_twin import load_schema_bytes
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

CACHE_VERSION = "schema_prompt_draft_v2"


@dataclass(frozen=True)
class SchemaPromptDraftResult:
    """Validated schema draft and metadata produced from a user prompt."""

    schema: Any
    payload: dict[str, Any]
    schema_json: str
    repairs: list[dict[str, Any]]
    cache_hit: bool
    model: str
    provider: str
    endpoint: str


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


def _coerce_schema_draft_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        payload = json.loads(_extract_json_object_text(value))
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"Schema draft returned an unsupported structured value: {value!r}")


def _schema_prompt_table_hints(prompt: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "build",
        "create",
        "data",
        "dataset",
        "field",
        "fields",
        "foreign",
        "generate",
        "include",
        "keep",
        "key",
        "keys",
        "normalized",
        "operational",
        "per",
        "primary",
        "realistic",
        "relationship",
        "relationships",
        "row",
        "rows",
        "schema",
        "synthetic",
        "tables",
        "the",
        "to",
        "with",
    }
    hints = _schema_prompt_table_list_hints(prompt, stopwords)
    if hints:
        return hints

    hints: list[str] = []
    for match in re.finditer(r"\b[a-z][a-z0-9_]{2,}\b", prompt.lower()):
        token = match.group(0)
        if token in stopwords or token.endswith("_id"):
            continue
        if token.endswith("s") or "_" in token:
            hints.append(token)
    return _dedupe_table_hints(hints)


def _schema_prompt_table_list_hints(prompt: str, stopwords: set[str]) -> list[str]:
    normalized = prompt.lower().replace("-", " ")
    matches = re.findall(
        r"\b(?:with|tables?|entities?)\b\s+(.*?)(?:\.\s*(?:include|add|use|make|keep|generate)\b|\binclude\b|\bkeep\b|\bgenerate\b|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    hints: list[str] = []
    for body in matches:
        for phrase in re.split(r",|\band\b", body):
            table = _normalize_table_phrase(phrase, stopwords)
            if table:
                hints.append(table)
    return _dedupe_table_hints(hints)


def _normalize_table_phrase(phrase: str, stopwords: set[str]) -> str | None:
    words = [
        word
        for word in re.findall(r"[a-z][a-z0-9_]*", phrase.lower())
        if word not in stopwords and not word.endswith("_id")
    ]
    if not words:
        return None
    table = "_".join(words)
    if table.endswith("_table"):
        table = table[:-6]
    return table or None


def _row_count_hint_from_prompt(prompt: str) -> int | None:
    matches = re.findall(r"\b(\d{1,6})\s+rows?\s+per\s+table\b", prompt.lower())
    if matches:
        return max(1, int(matches[-1]))
    matches = re.findall(r"\bgenerate\s+(\d{1,6})\s+rows?\b", prompt.lower())
    if matches:
        return max(1, int(matches[-1]))
    return None


def _dedupe_table_hints(hints: list[str]) -> list[str]:
    deduped = list(dict.fromkeys(hints))
    compound_parts = {
        part
        for hint in deduped
        if "_" in hint
        for part in hint.split("_")
    }
    return [hint for hint in deduped if not (hint in compound_parts and any(other != hint for other in deduped))]


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
    return f"{_singular_table_name(table_name)}_id"


def _default_columns_for_table(table_name: str) -> list[dict[str, Any]]:
    singular = _singular_table_name(table_name)
    return [
        {"name": _primary_key_for_table(table_name), "type": "uuid", "unique": True, "nullable": False},
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
    if not hints:
        hints.extend(_infer_relationship_hints_from_tables(table_hints))
    return list(dict.fromkeys(hints))


def _infer_relationship_hints_from_tables(table_hints: list[str]) -> list[tuple[str, str]]:
    hints: list[tuple[str, str]] = []
    known = set(table_hints)
    for parent_table in table_hints:
        parent_singular = _singular_table_name(parent_table)
        for child_table in table_hints:
            if parent_table == child_table:
                continue
            child_parts = set(child_table.split("_"))
            if parent_singular in child_parts or parent_table in child_parts:
                hints.append((parent_table, child_table))

    anchor_names = {
        "appointment",
        "booking",
        "invoice",
        "order",
        "reservation",
        "ticket",
        "transaction",
    }
    anchors = [table for table in table_hints if _singular_table_name(table) in anchor_names]
    entities_before_anchor = {
        "account",
        "author",
        "course",
        "customer",
        "member",
        "patient",
        "product",
        "provider",
        "student",
        "teacher",
        "user",
        "vendor",
    }
    dependent_names = {
        "invoice",
        "item",
        "line_item",
        "payment",
        "refund_transaction",
        "return_request",
        "shipment",
    }
    for anchor in anchors:
        anchor_index = table_hints.index(anchor)
        for table in table_hints[:anchor_index]:
            if _singular_table_name(table) in entities_before_anchor:
                hints.append((table, anchor))
        for table in table_hints:
            singular = _singular_table_name(table)
            if table != anchor and singular in dependent_names:
                hints.append((anchor, table))
    if "order_items" in known:
        if "orders" in known:
            hints.append(("orders", "order_items"))
        if "products" in known:
            hints.append(("products", "order_items"))
    if "orders" in known and "customers" in known:
        hints.append(("customers", "orders"))
    if "return_requests" in known:
        if "orders" in known:
            hints.append(("orders", "return_requests"))
        if "products" in known:
            hints.append(("products", "return_requests"))
        if "customers" in known:
            hints.append(("customers", "return_requests"))
        if "support_agents" in known:
            hints.append(("support_agents", "return_requests"))
    if "refund_transactions" in known and "return_requests" in known:
        hints.append(("return_requests", "refund_transactions"))
    return list(dict.fromkeys(hints))


def _repair_schema_draft_payload(
    payload: dict[str, Any],
    table_hints: list[str],
    relationship_hints: list[tuple[str, str]],
    *,
    default_row_count: int | None = None,
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
    by_name = {str(table.get("name")): table for table in normalized_tables if isinstance(table, dict)}

    for table_name in table_hints:
        if table_name not in by_name:
            table = {
                "name": table_name,
                "row_count": int(default_row_count or 100),
                "description": f"{table_name.replace('_', ' ').title()} records inferred from the user prompt.",
                "columns": _default_columns_for_table(table_name),
            }
            normalized_tables.append(table)
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
    for parent_table, child_table in relationship_hints:
        candidate = (parent_table, _primary_key_for_table(parent_table), child_table, _primary_key_for_table(parent_table))
        if parent_table not in by_name or child_table not in by_name:
            continue
        if candidate[1] not in _table_columns(by_name[parent_table]) or candidate[3] not in _table_columns(by_name[child_table]):
            continue
        if candidate in existing:
            continue
        relationships.append(
            {
                "parent_table": candidate[0],
                "parent_key": candidate[1],
                "child_table": candidate[2],
                "child_key": candidate[3],
            }
        )
        repairs.append({"action": "added_inferred_relationship", "relationship": ".".join(candidate)})
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
        default_row_count=_row_count_hint_from_prompt(prompt),
    )
    repairs.insert(0, {"action": "used_prompt_scaffold_after_invalid_ai_json"})
    return repaired, repairs


def _schema_draft_cache_enabled() -> bool:
    return os.getenv("SP_SCHEMA_DRAFT_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _schema_prompt_cache_path(prompt: str, model: str) -> Path:
    key = hashlib.sha256(f"{CACHE_VERSION}|{model}|{prompt.strip()}".encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(os.getenv("SP_SCHEMA_DRAFT_CACHE_DIR", Path.cwd() / "build" / "schema_draft_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _load_schema_draft_cache(prompt: str, model: str, seed: int | None) -> SchemaPromptDraftResult | None:
    if not _schema_draft_cache_enabled():
        return None
    cache_path = _schema_prompt_cache_path(prompt, model)
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    schema_payload = payload.get("schema") if isinstance(payload, dict) else None
    if not isinstance(schema_payload, dict):
        return None
    schema = load_schema_bytes(json.dumps(schema_payload).encode("utf-8"), "ai_schema.json", seed=seed)
    repairs = payload.get("repairs", []) if isinstance(payload.get("repairs"), list) else []
    repairs = [{"action": "loaded_schema_draft_cache", "path": str(cache_path)}] + repairs
    runtime = resolve_platform_slm_runtime()
    return SchemaPromptDraftResult(
        schema=schema,
        payload=schema_payload,
        schema_json=json.dumps(schema_payload, indent=2, sort_keys=True),
        repairs=repairs,
        cache_hit=True,
        model=runtime.model_id,
        provider=runtime.provider,
        endpoint=runtime.endpoint,
    )


def _write_schema_draft_cache(prompt: str, model: str, payload: dict[str, Any], repairs: list[dict[str, Any]]) -> None:
    if not _schema_draft_cache_enabled():
        return
    cache_path = _schema_prompt_cache_path(prompt, model)
    cache_path.write_text(
        json.dumps({"schema": payload, "repairs": repairs}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def draft_schema_from_prompt(prompt: str, *, seed: int | None = None) -> SchemaPromptDraftResult:
    """Draft, repair, cache, and validate a schema from a plain-English user prompt."""

    runtime = resolve_platform_slm_runtime()
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("A schema prompt is required.")

    cached = _load_schema_draft_cache(prompt, runtime.model_id, seed)
    if cached is not None:
        return cached

    try:
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Prompt-to-schema drafting requires the standalone data-designer package.") from exc

    require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    table_hints = _schema_prompt_table_hints(prompt)
    relationship_hints = _relationship_hints_from_prompt(prompt, table_hints)
    row_count_hint = _row_count_hint_from_prompt(prompt)
    request_seed = 0 if seed is None else int(seed)
    seed_frame = pd.DataFrame(
        [
            {
                "request_id": f"schema-draft-{request_seed}",
                "user_request": prompt,
                "required_tables_json": json.dumps(table_hints),
                "required_relationships_json": json.dumps(relationship_hints),
            }
        ]
    )
    model_alias = "schema-draft-generator"
    model_config = build_data_designer_model_config(
        dd,
        model_alias,
        workflow="schema-draft",
        runtime=runtime,
        temperature=0.1,
        top_p=0.8,
        max_tokens=int(os.getenv("SP_SCHEMA_DRAFT_MAX_TOKENS", "768")),
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
        payload, repairs = _repair_schema_draft_payload(
            payload,
            table_hints,
            relationship_hints,
            default_row_count=row_count_hint,
        )
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
    return SchemaPromptDraftResult(
        schema=schema,
        payload=payload,
        schema_json=json.dumps(payload, indent=2, sort_keys=True),
        repairs=repairs,
        cache_hit=False,
        model=runtime.model_id,
        provider=runtime.provider,
        endpoint=runtime.endpoint,
    )
