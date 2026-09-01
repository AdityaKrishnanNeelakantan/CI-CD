"""Generation backend boundary for current and optional NeMo generators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import util
import json
import os
from pathlib import Path
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from synth_platform.domain.contracts.models import CanonicalContract
from synth_platform.engine.common.database.core.stage_result import STATUS_SUCCESS, StageResult
from synth_platform.engine.transcripts.service import generate_synthetic_transcript


class UnsupportedGenerationBackend(NotImplementedError):
    """Raised when a backend is selected for a workflow it does not support yet."""


class NemoSchemaGenerationError(RuntimeError):
    """Raised when Schema NeMo generation fails with diagnostic metadata."""

    def __init__(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        raw_exception: str | None = None,
        raw_value: str | None = None,
    ) -> None:
        super().__init__(message)
        self.metadata = metadata or {}
        self.raw_exception = raw_exception
        self.raw_value = raw_value


class GenerationBackend(ABC):
    """Interface for generator implementations behind stable workflow contracts."""

    name: str

    def generate_schema(self, contract: Any, **kwargs: Any) -> Any:
        raise UnsupportedGenerationBackend("NeMo schema generation is not implemented yet.")

    def generate_database(self, contract: Any, **kwargs: Any) -> Any:
        raise UnsupportedGenerationBackend("NeMo database generation is not implemented yet.")

    def generate_pdf(self, contract: Any, **kwargs: Any) -> Any:
        raise UnsupportedGenerationBackend("NeMo PDF generation is not implemented yet.")

    @abstractmethod
    def generate_transcript(
        self,
        contract: CanonicalContract,
        *,
        turn_count: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a synthetic customer interaction from a transcript contract."""


@dataclass(frozen=True)
class BackendSchemaResult:
    preview_tables: dict[str, pd.DataFrame]
    row_counts: dict[str, int]
    validation_report: dict[str, Any]
    export_paths: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hard_checks_passed(self) -> bool:
        return bool(self.validation_report.get("hard_checks_passed"))


@dataclass(frozen=True)
class BackendDatabaseResult:
    tables: dict[str, pd.DataFrame]
    metrics: dict[str, Any]
    sample_paths: dict[str, Path]
    report_path: Path


@dataclass(frozen=True)
class DataDesignerRuntime:
    model_id: str
    provider: str
    endpoint: str
    api_key_env: str
    model_family: str


@dataclass(frozen=True)
class CurrentBackend(GenerationBackend):
    """Existing platform generator implementation."""

    name: str = "current"

    def generate_transcript(
        self,
        contract: CanonicalContract,
        *,
        turn_count: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        return generate_synthetic_transcript(contract, turn_count=turn_count, seed=seed)


@dataclass(frozen=True)
class NemoBackend(GenerationBackend):
    """Optional NVIDIA/Data Designer generator boundary.

    The implementation refuses to masquerade as NeMo generation when the SDK is
    absent. This gives benchmarks a real signal: either NeMo generated output,
    or the benchmark records that no replacement decision can be made.
    """

    name: str = "nemo"

    def generate_database(self, contract: Any, **kwargs: Any) -> BackendDatabaseResult:
        manifest = kwargs["manifest"]
        rows = int(kwargs.get("rows") or 100)
        seed = int(kwargs.get("seed") or 0)
        row_plan = dict(kwargs.get("row_plan") or _database_row_plan(contract, rows))
        if _mock_data_designer_enabled():
            tables = _mock_database_tables(contract, row_plan=row_plan, seed=seed)
        else:
            if not _data_designer_available():
                raise RuntimeError(
                    "NeMo/Data Designer database generation is not available. "
                    "Install data-designer in a Python <3.14 environment."
                )
            tables = _generate_database_with_data_designer(contract, row_plan=row_plan, seed=seed)
        return _write_database_generation_result(
            contract,
            tables,
            manifest=manifest,
            contract_reference=str(kwargs.get("contract_reference") or "dataset_contract.json"),
            seed=seed,
        )

    def generate_schema(self, contract: Any, **kwargs: Any) -> BackendSchemaResult:
        row_count = int(kwargs.get("row_count") or 100)
        table_row_counts = dict(kwargs.get("table_row_counts") or {})
        seed = int(kwargs.get("seed") or 0)
        output_dir = Path(kwargs["output_dir"])
        export_format = str(kwargs.get("export_format") or "csv")
        if export_format != "csv":
            raise ValueError("NeMo schema backend currently supports CSV export only.")
        if _mock_data_designer_enabled():
            tables = _mock_schema_tables(contract, row_count=row_count, table_row_counts=table_row_counts, seed=seed)
            metadata = {
                "backend": self.name,
                "data_designer": {"mode": "mock", "model_alias": None, "provider": None, "model": None},
            }
        else:
            if not _data_designer_available():
                raise RuntimeError(
                    "NeMo/Data Designer schema generation is not available. "
                    "Install data-designer in a Python <3.14 environment."
                )
            tables, metadata = _generate_schema_with_data_designer(
                contract,
                row_count=row_count,
                table_row_counts=table_row_counts,
                seed=seed,
            )
        validation_report = _validate_schema_tables(contract, tables)
        validation_report["generation_backend"] = self.name
        validation_report["data_designer"] = metadata.get("data_designer", {})
        if not validation_report.get("hard_checks_passed"):
            raise NemoSchemaGenerationError(
                "Schema Data Designer output failed validation before export.",
                metadata=metadata,
                raw_exception=json.dumps(validation_report, default=str),
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        export_paths: dict[str, Path] = {}
        for table_name, frame in tables.items():
            path = output_dir / f"{table_name}.csv"
            frame.to_csv(path, index=False)
            export_paths[table_name] = path
        validation_path = output_dir / "validation_report.json"
        with validation_path.open("w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, sort_keys=True, default=str)
        return BackendSchemaResult(
            preview_tables={name: frame.head(100) for name, frame in tables.items()},
            row_counts={name: len(frame) for name, frame in tables.items()},
            validation_report=validation_report,
            export_paths=export_paths,
            metadata=metadata,
        )

    def generate_pdf(self, contract: Any, **kwargs: Any) -> dict[str, Any]:
        template = contract["template"]
        binding_map = contract["binding_map"]
        doc_id = str(contract["doc_id"])
        manifest = contract["manifest"]
        binding_map_reference = str(contract.get("binding_map_reference") or kwargs.get("binding_map_reference") or "")
        seed = kwargs.get("seed")
        if _mock_data_designer_enabled():
            generated = _mock_data_designer_pdf_values(template, binding_map, seed=seed)
        else:
            if not _data_designer_available():
                raise RuntimeError(
                    "NeMo/Data Designer PDF value generation is not available. "
                    "Install data-designer in a Python <3.14 environment."
                )
            generated = _generate_pdf_values_with_data_designer(template, binding_map, seed=seed)
        runtime = _digital_twin_data_designer_runtime()
        generated.setdefault("model_family", runtime.model_family)
        generated.setdefault("model", runtime.model_id)
        generated.setdefault("model_provider", runtime.provider)
        document_synthetic_values = {
            "doc_id": doc_id,
            "source_reference": binding_map_reference,
            "generated_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "generation_backend": "nemo",
            "model_family": generated.get("model_family"),
            "model": generated.get("model"),
            "model_provider": generated.get("model_provider"),
            "fields": generated["fields"],
            "tables": generated["tables"],
            "inline_spans": generated["inline_spans"],
        }
        output_path = manifest.output_path(f"documents/{doc_id}/document_synthetic_values.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(document_synthetic_values, f, indent=2, sort_keys=True)
        table_cell_count = sum(len(row) for rows in generated["tables"].values() for row in rows)
        manifest.record_stage(
            StageResult(
                stage_name="value_generation",
                status=STATUS_SUCCESS,
                input_references=[binding_map_reference],
                output_references=[str(output_path)],
                metrics={
                    "field_count": len(generated["fields"]),
                    "table_count": len(generated["tables"]),
                    "table_cell_count": table_cell_count,
                    "seed": seed,
                    "generation_backend": "nemo",
                },
                evidence={"generated_at": document_synthetic_values["generated_at"]},
            )
        )
        return document_synthetic_values

    def generate_transcript(
        self,
        contract: CanonicalContract,
        *,
        turn_count: int | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        if _mock_data_designer_enabled():
            return _mock_data_designer_transcript(contract, turn_count=turn_count, seed=seed)
        if not _data_designer_available():
            raise RuntimeError(
                "NeMo/Data Designer transcript generation is not available. "
                "Install data-designer or nemo-microservices[data-designer] in a Python <3.14 environment."
            )
        return _generate_transcript_with_data_designer(
            contract,
            turn_count=turn_count,
            seed=seed,
        )


class GeneratorFactory:
    """Create generation backends by stable backend id."""

    @staticmethod
    def create(backend: str | None) -> GenerationBackend:
        normalized = (backend or "current").strip().lower()
        if normalized in {"current", "platform", "mine"}:
            return CurrentBackend()
        if normalized in {"nemo", "nvidia", "data-designer", "datadesigner"}:
            return NemoBackend()
        raise ValueError(f"Unknown generation backend: {backend}")


def _data_designer_available() -> bool:
    return (
        util.find_spec("data_designer") is not None
        or util.find_spec("nemo_microservices") is not None
    )


def _mock_data_designer_enabled() -> bool:
    return os.getenv("SP_NEMO_DATA_DESIGNER_MOCK", "").strip().lower() in {"1", "true", "yes"}


def _data_designer_user_prompt() -> str:
    return (
        os.getenv("SP_NEMO_DATA_DESIGNER_USER_PROMPT")
        or "Generate realistic privacy-safe synthetic data. Preserve schema, relationships, formats, and validation constraints."
    ).strip()


def _data_designer_api_key_env() -> str:
    return "SP_NEMO_DATA_DESIGNER_API_KEY"


def _data_designer_api_key() -> str | None:
    return os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY")


def _data_designer_endpoint() -> str:
    return os.getenv("SP_NEMO_DATA_DESIGNER_ENDPOINT", "http://localhost:8000/v1")


def _is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _data_designer_model_provider_api_key(runtime: DataDesignerRuntime) -> str | None:
    key = os.getenv(runtime.api_key_env) or _data_designer_api_key()
    if key:
        if not os.getenv(runtime.api_key_env):
            os.environ[runtime.api_key_env] = key
        return runtime.api_key_env
    if _is_local_endpoint(runtime.endpoint):
        return None
    return runtime.api_key_env


def _data_designer_chat_extra_body(runtime: DataDesignerRuntime) -> dict[str, Any] | None:
    return None


def _digital_twin_data_designer_runtime() -> DataDesignerRuntime:
    """Resolve the open-source SLM used by PDF/TXT twin generation."""
    return DataDesignerRuntime(
        model_id=os.getenv(
            "SP_DIGITAL_TWIN_SLM_MODEL",
            os.getenv("SP_NEMO_DATA_DESIGNER_MODEL", "local/slm"),
        ),
        provider=os.getenv(
            "SP_DIGITAL_TWIN_SLM_PROVIDER",
            os.getenv("SP_NEMO_DATA_DESIGNER_PROVIDER", "internal"),
        ),
        endpoint=os.getenv("SP_DIGITAL_TWIN_SLM_ENDPOINT", _data_designer_endpoint()),
        api_key_env=os.getenv("SP_DIGITAL_TWIN_SLM_API_KEY_ENV", _data_designer_api_key_env()),
        model_family=os.getenv("SP_DIGITAL_TWIN_MODEL_FAMILY", "open_source_slm"),
    )


def _require_data_designer_api_key(runtime: DataDesignerRuntime) -> None:
    key = os.getenv(runtime.api_key_env) or _data_designer_api_key()
    if key:
        if not os.getenv(runtime.api_key_env):
            os.environ[runtime.api_key_env] = key
        return
    if _is_local_endpoint(runtime.endpoint):
        return
    env_names = list(dict.fromkeys([runtime.api_key_env, "SP_NEMO_DATA_DESIGNER_API_KEY"]))
    raise RuntimeError(
        "Set one of these environment variables before running this workflow: "
        f"{', '.join(env_names)}."
    )


def _data_designer_skip_health_check() -> bool:
    return os.getenv("SP_NEMO_DATA_DESIGNER_SKIP_HEALTH_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _generate_transcript_with_data_designer(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Generate transcript turns through NVIDIA NeMo Data Designer."""
    if util.find_spec("data_designer") is not None:
        return _generate_transcript_with_standalone_data_designer(
            contract,
            turn_count=turn_count,
            seed=seed,
        )
    return _generate_transcript_with_nemo_microservices(
        contract,
        turn_count=turn_count,
        seed=seed,
    )


def _generate_transcript_with_standalone_data_designer(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "transcript-generator"
    target_turns = int(turn_count or _contract_turn_count(contract) or 8)
    seed_frame = pd.DataFrame(
        [
            {
                "source_contract_json": _transcript_data_designer_seed(contract, target_turns),
                "target_turn_count": target_turns,
                "user_prompt": _data_designer_user_prompt(),
            }
        ]
    )
    prompt = _transcript_data_designer_prompt()

    model_config = dd.ModelConfig(
        alias=model_alias,
        model=runtime.model_id,
        provider=runtime.provider,
        skip_health_check=skip_health_check,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=float(os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.65")),
            top_p=float(os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.95")),
            max_tokens=int(os.getenv("SP_NEMO_DATA_DESIGNER_MAX_TOKENS", "2048")),
            extra_body=_data_designer_chat_extra_body(runtime),
        ),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="synthetic_transcript_json",
            prompt=prompt,
            system_prompt=(
                "Generate only valid JSON. Do not include Markdown, comments, source identifiers, "
                "or raw source transcript text."
            ),
            model_alias=model_alias,
            output_format=_transcript_json_schema(),
        )
    )

    data_designer = DataDesigner(
        model_providers=[
            dd.ModelProvider(
                name=runtime.provider,
                endpoint=runtime.endpoint,
                provider_type="openai",
                api_key=_data_designer_model_provider_api_key(runtime),
            )
        ]
    )
    preview = data_designer.preview(builder, num_records=1)
    dataset = getattr(preview, "dataset", preview)
    value = _extract_data_designer_value(dataset, "synthetic_transcript_json")
    rows = _parse_transcript_json(value)
    return _normalize_transcript_rows(rows, target_turns=target_turns)


def _generate_transcript_with_nemo_microservices(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    try:
        from nemo_microservices.data_designer.essentials import (
            DataDesignerConfigBuilder,
            InferenceParameters,
            LLMTextColumnConfig,
            ModelConfig,
            NeMoDataDesignerClient,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Installed Data Designer package does not expose the expected NeMo Microservices SDK API."
        ) from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    base_url = os.getenv("SP_NEMO_DATA_DESIGNER_BASE_URL", "https://ai.api.nvidia.com/v1/nemo/dd")
    model_alias = "transcript-generator"
    target_turns = int(turn_count or _contract_turn_count(contract) or 8)
    prompt = _transcript_data_designer_inline_prompt(contract, target_turns)

    model_config_kwargs: dict[str, Any] = {
        "alias": model_alias,
        "model": runtime.model_id,
        "provider": runtime.provider,
        "inference_parameters": InferenceParameters(
            temperature=float(os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.65")),
            top_p=float(os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.95")),
            max_tokens=int(os.getenv("SP_NEMO_DATA_DESIGNER_MAX_TOKENS", "2048")),
        ),
    }

    builder = DataDesignerConfigBuilder(
        [
            ModelConfig(**model_config_kwargs),
        ]
    )
    builder.add_column(
        LLMTextColumnConfig(
            name="synthetic_transcript_json",
            prompt=prompt,
            system_prompt=(
                "Generate only valid JSON. Do not include Markdown, comments, source identifiers, "
                "or raw source transcript text."
            ),
            model_alias=model_alias,
        )
    )
    api_key = os.getenv(runtime.api_key_env) or _data_designer_api_key() or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = NeMoDataDesignerClient(base_url=base_url, default_headers=headers)
    preview = client.preview(builder, num_records=1)
    dataset = getattr(preview, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("NeMo Data Designer returned no transcript records.")
    value = _extract_data_designer_value(dataset, "synthetic_transcript_json")
    rows = _parse_transcript_json(str(value))
    return _normalize_transcript_rows(rows, target_turns=target_turns)


def _extract_data_designer_value(dataset: Any, column: str) -> Any:
    if hasattr(dataset, "iloc"):
        if len(dataset) == 0:
            raise RuntimeError("NeMo Data Designer returned no transcript records.")
        return dataset.iloc[0][column]
    if isinstance(dataset, list):
        if not dataset:
            raise RuntimeError("NeMo Data Designer returned no transcript records.")
        return dataset[0][column]
    if isinstance(dataset, dict):
        value = dataset.get(column)
        if isinstance(value, list):
            return value[0]
        return value
    raise RuntimeError("Unsupported NeMo Data Designer preview response shape.")


def _transcript_data_designer_seed(contract: CanonicalContract, turn_count: int) -> str:
    entity = contract.entities[0] if contract.entities else None
    metadata = dict(entity.metadata if entity else {})
    context = dict(metadata.get("synthetic_context") or {})
    speakers = metadata.get("speakers") or ["speaker_1", "speaker_2"]
    topic_terms = metadata.get("topic_terms") or []
    return json.dumps(
        {
            "turn_count": turn_count,
            "source_speaker_count": len(speakers),
            "topic_terms": topic_terms,
            "synthetic_context": context,
        },
        sort_keys=True,
        default=str,
    )


def _transcript_data_designer_prompt() -> str:
    return (
        "Create a synthetic customer support transcript from sanitized contract metadata only.\n"
        "User generation instructions: {{ user_prompt }}\n"
        "Target turn count: {{ target_turn_count }}\n"
        "Sanitized contract metadata JSON: {{ source_contract_json }}\n"
        "Return a JSON object with key turns. Each turn must have: turn, speaker, timestamp, text.\n"
        "Use Customer and Agent speaker labels. Keep the conversation realistic, concise, "
        "privacy-safe, and consistent with the synthetic context. Do not copy source text."
    )


def _transcript_data_designer_inline_prompt(contract: CanonicalContract, turn_count: int) -> str:
    return (
        "Create a synthetic customer support transcript from sanitized contract metadata only.\n"
        f"Sanitized contract metadata JSON: {_transcript_data_designer_seed(contract, turn_count)}\n"
        "Return a JSON object with key turns. Each turn must have: turn, speaker, timestamp, text.\n"
        "Use Customer and Agent speaker labels. Keep the conversation realistic, concise, "
        "privacy-safe, and consistent with the synthetic context. Do not copy source text."
    )


def _transcript_json_schema() -> dict[str, Any]:
    turn = {
        "type": "object",
        "properties": {
            "turn": {"type": "integer"},
            "speaker": {"type": "string"},
            "timestamp": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["turn", "speaker", "text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "items": turn,
            }
        },
        "required": ["turns"],
        "additionalProperties": False,
    }


def _parse_transcript_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, list):
        payload = value
    else:
        text = str(value).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("NeMo Data Designer did not return valid transcript JSON.") from exc
    if isinstance(payload, dict):
        payload = payload.get("turns") or payload.get("transcript") or payload.get("messages")
    if not isinstance(payload, list):
        raise RuntimeError("NeMo Data Designer transcript output must be a JSON array.")
    return [row for row in payload if isinstance(row, dict)]


def _normalize_transcript_rows(rows: list[dict[str, Any]], *, target_turns: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:target_turns], start=1):
        text = str(row.get("text") or row.get("content") or row.get("message") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "turn": int(row.get("turn") or row.get("index") or idx),
                "speaker": str(row.get("speaker") or row.get("role") or ("Customer" if idx % 2 else "Agent")),
                "timestamp": str(row.get("timestamp") or ""),
                "text": text,
            }
        )
    if not normalized:
        raise RuntimeError("NeMo Data Designer transcript output contained no usable turns.")
    return normalized


def _mock_data_designer_transcript(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    total = int(turn_count or _contract_turn_count(contract) or 6)
    entity = contract.entities[0] if contract.entities else None
    metadata = entity.metadata if entity else {}
    context = dict(metadata.get("synthetic_context") or {})
    topic_terms = [str(term) for term in metadata.get("topic_terms") or []]
    topic = " ".join(topic_terms[:2]) or "support request"
    customer_name = context.get("customer_name") or f"Customer {int(seed or 0) % 1000:03d}"
    reference = context.get("claim_id") or context.get("policy_id") or context.get("tracking_id") or "case reference"
    contact = " or ".join(str(value) for value in (context.get("email"), context.get("phone")) if value)
    dob = context.get("date_of_birth")
    templates = [
        f"Hi, I need help with the synthetic {topic}.",
        f"I can help, {customer_name}. I have opened {reference}.",
        f"Please use {contact} for follow up." if contact else "Can you confirm the next step for this request?",
        "The next step is recorded in the synthetic case notes.",
        f"Thanks. My synthetic date of birth is {dob}." if dob else "Thanks, that gives me what I need.",
        "You are welcome. The update is ready for review.",
    ]
    return [
        {
            "turn": idx + 1,
            "speaker": "Customer" if idx % 2 == 0 else "Agent",
            "timestamp": "",
            "text": templates[idx % len(templates)],
        }
        for idx in range(max(1, total))
    ]


def _contract_turn_count(contract: CanonicalContract) -> int:
    if not contract.entities:
        return 0
    return int(contract.entities[0].metadata.get("turn_count") or 0)


def _mock_schema_tables(
    schema: Any,
    *,
    row_count: int,
    table_row_counts: dict[str, int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    return _deterministic_schema_tables(
        schema,
        row_count=row_count,
        table_row_counts=table_row_counts,
        seed=seed,
        text_prefix="nemo_mock",
    )


def _database_row_plan(contract: dict[str, Any], rows: int) -> dict[str, int]:
    tables = contract.get("tables", {})
    per_table = max(5, rows // max(1, len(tables)))
    return {table_name: per_table for table_name in tables}


def _mock_database_tables(
    contract: dict[str, Any],
    *,
    row_plan: dict[str, int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    graph = _database_schema_graph(contract)
    for table_name in graph["generation_order"]:
        table_contract = contract["tables"][table_name]
        count = int(row_plan.get(table_name) or 5)
        rows = [
            _database_deterministic_row(
                table_name,
                table_contract,
                row_index=idx,
                seed=seed,
                generated_tables=tables,
                prefix="nemo_mock",
            )
            for idx in range(count)
        ]
        tables[table_name] = pd.DataFrame(rows)
    _enforce_database_foreign_keys(contract, tables, seed=seed)
    return tables


def _generate_database_with_data_designer(
    contract: dict[str, Any],
    *,
    row_plan: dict[str, int],
    seed: int,
) -> dict[str, pd.DataFrame]:
    if util.find_spec("data_designer") is None:
        raise RuntimeError("Database Data Designer generation requires the standalone data-designer package.")
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    model_id = os.getenv("SP_NEMO_DATA_DESIGNER_MODEL", "local/slm")
    model_provider = os.getenv("SP_NEMO_DATA_DESIGNER_PROVIDER", "internal")
    provider_endpoint = _data_designer_endpoint()
    runtime = DataDesignerRuntime(
        model_id=model_id,
        provider=model_provider,
        endpoint=provider_endpoint,
        api_key_env=_data_designer_api_key_env(),
        model_family="open_source_slm",
    )
    _require_data_designer_api_key(runtime)
    model_alias = "database-row-generator"
    data_designer = DataDesigner(
        model_providers=[
            dd.ModelProvider(
                name=model_provider,
                endpoint=provider_endpoint,
                provider_type="openai",
                api_key=_data_designer_model_provider_api_key(runtime),
            )
        ]
    )
    graph = _database_schema_graph(contract)
    tables: dict[str, pd.DataFrame] = {}
    for table_name in graph["generation_order"]:
        count = int(row_plan.get(table_name) or 5)
        seed_frame = pd.DataFrame(
        [
            {
                "table_name": table_name,
                "row_count": count,
                "column_contract_json": _database_column_contract_json(contract, table_name),
                "foreign_keys_json": json.dumps(contract["tables"][table_name].get("foreign_keys", []), default=str),
                "parent_rows_json": _database_parent_rows_json(contract, table_name, generated_tables=tables),
                "user_prompt": _data_designer_user_prompt(),
            }
        ]
    )
        prompt = _database_data_designer_prompt()
        model_config = dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=model_provider,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=float(os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.35")),
                top_p=float(os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.9")),
                max_tokens=int(os.getenv("SP_NEMO_DATA_DESIGNER_MAX_TOKENS", "4096")),
                extra_body=_data_designer_chat_extra_body(runtime),
            ),
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="rows_json",
                prompt=prompt,
                system_prompt="Return only valid JSON. No Markdown. No explanation. Do not emit source PII.",
                model_alias=model_alias,
            )
        )
        preview = data_designer.preview(builder, num_records=1)
        dataset = getattr(preview, "dataset", preview)
        value = _extract_data_designer_value(dataset, "rows_json")
        rows = _parse_rows_json(str(value))
        tables[table_name] = _coerce_database_frame(
            contract,
            table_name,
            pd.DataFrame(rows),
            count=count,
            seed=seed,
            generated_tables=tables,
        )
    _enforce_database_foreign_keys(contract, tables, seed=seed)
    return tables


def _database_schema_graph(contract: dict[str, Any]) -> dict[str, Any]:
    from synth_platform.engine.generation.database.schema_graph import build_schema_graph

    return build_schema_graph(contract)


def _database_column_contract_json(contract: dict[str, Any], table_name: str) -> str:
    table = contract["tables"][table_name]
    columns = [
        {
            "name": name,
            "physical_type": details.get("physical_type"),
            "semantic_type": details.get("semantic_type"),
            "nullable": bool(details.get("nullable")),
            "sensitive": bool(details.get("sensitive")),
            "primary_key": name in table.get("primary_key", []),
            "foreign_key": any(fk.get("column") == name for fk in table.get("foreign_keys", [])),
        }
        for name, details in table.get("columns", {}).items()
    ]
    return json.dumps(columns, default=str)


def _database_parent_rows_json(
    contract: dict[str, Any],
    table_name: str,
    *,
    generated_tables: dict[str, pd.DataFrame],
) -> str:
    table = contract["tables"][table_name]
    parent_samples: dict[str, Any] = {}
    for fk in table.get("foreign_keys", []):
        parent_name = fk.get("references_table")
        if parent_name in generated_tables:
            parent_samples[parent_name] = generated_tables[parent_name].head(25).to_dict(orient="records")
    return json.dumps(parent_samples, default=str)


def _database_data_designer_prompt() -> str:
    return (
        "Generate {{ row_count }} privacy-safe synthetic database rows for table {{ table_name }}.\n"
        "User generation instructions: {{ user_prompt }}\n"
        "Columns: {{ column_contract_json }}\n"
        "Table foreign keys: {{ foreign_keys_json }}\n"
        "Previously generated parent rows for referential integrity: {{ parent_rows_json }}\n"
        "Rules: preserve exact column names, primary keys must be unique, non-nullable columns must be populated, "
        "foreign-key columns must use values from the parent rows when parent rows are provided, and sensitive "
        "values must be new synthetic values rather than copied from any source data.\n"
        "Return a JSON array of row objects only."
    )


def _coerce_database_frame(
    contract: dict[str, Any],
    table_name: str,
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
    generated_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    table = contract["tables"][table_name]
    columns = list(table.get("columns", {}).keys())
    if len(frame) < count:
        filler = pd.DataFrame(
            [
                _database_deterministic_row(
                    table_name,
                    table,
                    row_index=idx,
                    seed=seed,
                    generated_tables=generated_tables,
                    prefix="nemo_fill",
                )
                for idx in range(count)
            ]
        )
        frame = pd.concat([frame, filler], ignore_index=True).head(count)
    frame = frame.head(count).copy()
    for idx in range(len(frame)):
        for column_name, column in table.get("columns", {}).items():
            if column_name not in frame.columns or pd.isna(frame.at[idx, column_name]) or str(frame.at[idx, column_name]).strip() == "":
                if column_name not in frame.columns:
                    frame[column_name] = None
                frame.at[idx, column_name] = _database_cell_value(
                    table_name,
                    column_name,
                    column,
                    row_index=idx,
                    seed=seed,
                    prefix="nemo_fill",
                )
    for pk in table.get("primary_key", []):
        if pk in frame.columns:
            frame[pk] = [
                _database_primary_key_value(table_name, pk, table["columns"].get(pk, {}), idx, prefix="nemo")
                for idx in range(len(frame))
            ]
    return frame[columns]


def _database_deterministic_row(
    table_name: str,
    table: dict[str, Any],
    *,
    row_index: int,
    seed: int,
    generated_tables: dict[str, pd.DataFrame],
    prefix: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column_name, column in table.get("columns", {}).items():
        fk = next((item for item in table.get("foreign_keys", []) if item.get("column") == column_name), None)
        if fk and fk.get("references_table") in generated_tables:
            parent = generated_tables[fk["references_table"]]
            parent_key = fk["references_column"]
            if parent_key in parent.columns and len(parent) > 0:
                row[column_name] = parent.iloc[row_index % len(parent)][parent_key]
                continue
        if column_name in table.get("primary_key", []):
            row[column_name] = _database_primary_key_value(table_name, column_name, column, row_index, prefix=prefix)
        else:
            row[column_name] = _database_cell_value(
                table_name,
                column_name,
                column,
                row_index=row_index,
                seed=seed,
                prefix=prefix,
            )
    return row


def _database_primary_key_value(
    table_name: str,
    column_name: str,
    column: dict[str, Any],
    row_index: int,
    *,
    prefix: str,
) -> Any:
    physical_type = str(column.get("physical_type") or "").lower()
    if any(token in physical_type for token in ("int", "number", "numeric")):
        table_offset = 100000 + (sum(ord(ch) for ch in table_name) % 900) * 1000
        return table_offset + row_index + 1
    return f"{prefix}_{table_name}_{column_name}_{row_index + 1}"


def _database_cell_value(
    table_name: str,
    column_name: str,
    column: dict[str, Any],
    *,
    row_index: int,
    seed: int,
    prefix: str,
) -> Any:
    semantic = str(column.get("semantic_type") or "").lower()
    physical = str(column.get("physical_type") or "").lower()
    base = row_index + seed + 1
    if semantic == "email" or "email" in column_name.lower():
        return f"{prefix}_{table_name}_{row_index + 1}@example.com"
    if semantic == "person_name" or column_name.lower() in {"name", "full_name", "customer_name"}:
        return f"{prefix.title()} Person {row_index + 1}"
    if semantic == "category" or any(token in column_name.lower() for token in ("status", "type", "category", "gender")):
        choices = ["standard", "premium", "review", "active"]
        return choices[base % len(choices)]
    if semantic in {"datetime", "date"} or "date" in column_name.lower():
        return f"2024-01-{(base % 28) + 1:02d}"
    if semantic == "numerical" or any(token in physical for token in ("real", "float", "double", "decimal", "numeric")):
        return round(25.0 + (base % 250) * 1.37, 2)
    if any(token in physical for token in ("int", "number")):
        return int(base)
    return f"{prefix}_{table_name}_{column_name}_{row_index + 1}"


def _enforce_database_foreign_keys(contract: dict[str, Any], tables: dict[str, pd.DataFrame], *, seed: int) -> None:
    graph = _database_schema_graph(contract)
    for edge_index, edge in enumerate(graph["edges"]):
        child = tables.get(edge["child"])
        parent = tables.get(edge["parent"])
        if child is None or parent is None:
            continue
        parent_key = edge["parent_key"]
        child_key = edge["child_key"]
        if parent_key not in parent.columns or child_key not in child.columns or parent.empty:
            continue
        parent_values = parent[parent_key].dropna().tolist()
        if not parent_values:
            continue
        child[child_key] = [
            parent_values[(idx + seed + edge_index) % len(parent_values)]
            for idx in range(len(child))
        ]


def _write_database_generation_result(
    contract: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    *,
    manifest: Any,
    contract_reference: str,
    seed: int,
) -> BackendDatabaseResult:
    from synth_platform.engine.generation.database.relational_generator import compute_fk_validity

    graph = _database_schema_graph(contract)
    samples_dir = manifest.run_dir / "generated_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_paths: dict[str, Path] = {}
    for table_name, frame in tables.items():
        table_path = samples_dir / f"{table_name}.csv"
        frame.to_csv(table_path, index=False)
        sample_paths[table_name] = table_path

    fk_validity = compute_fk_validity(tables, graph)
    report_path = manifest.output_path("relational_generation_report.json")
    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_backend": "nemo",
        "generation_order": graph["generation_order"],
        "fk_validity": fk_validity,
        "tables": {
            table_name: {"row_count": len(frame), "path": str(sample_paths[table_name])}
            for table_name, frame in tables.items()
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, sort_keys=True)
    metrics = {
        "table_count": len(tables),
        "total_rows": sum(len(frame) for frame in tables.values()),
        "overall_fk_validity": fk_validity["overall_fk_validity"],
        "generation_backend": "nemo",
        "seed": seed,
    }
    manifest.record_stage(
        StageResult(
            stage_name="relational_generation",
            status=STATUS_SUCCESS,
            input_references=[contract_reference],
            output_references=[str(report_path)] + [str(path) for path in sample_paths.values()],
            metrics=metrics,
            evidence={"generated_at": report_data["generated_at"]},
        )
    )
    return BackendDatabaseResult(
        tables=tables,
        metrics=metrics,
        sample_paths=sample_paths,
        report_path=report_path,
    )


def _generate_schema_with_data_designer(
    schema: Any,
    *,
    row_count: int,
    table_row_counts: dict[str, int],
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if util.find_spec("data_designer") is None:
        raise RuntimeError("Schema Data Designer generation requires the standalone data-designer package.")
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    model_id = os.getenv("SP_NEMO_DATA_DESIGNER_MODEL", "local/slm")
    model_provider = os.getenv("SP_NEMO_DATA_DESIGNER_PROVIDER", "internal")
    provider_endpoint = _data_designer_endpoint()
    runtime = DataDesignerRuntime(
        model_id=model_id,
        provider=model_provider,
        endpoint=provider_endpoint,
        api_key_env=_data_designer_api_key_env(),
        model_family="open_source_slm",
    )
    _require_data_designer_api_key(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "schema-row-generator"
    metadata: dict[str, Any] = {
        "backend": "nemo",
        "data_designer": {
            "backend_name": "data-designer",
            "model": model_id,
            "provider": model_provider,
            "provider_endpoint": provider_endpoint,
            "model_alias": model_alias,
            "skip_health_check": skip_health_check,
            "row_count_repairs": [],
        },
    }
    data_designer = DataDesigner(
        model_providers=[
            dd.ModelProvider(
                name=model_provider,
                endpoint=provider_endpoint,
                provider_type="openai",
                api_key=_data_designer_model_provider_api_key(runtime),
            )
        ]
    )
    tables: dict[str, pd.DataFrame] = {}
    for table in schema.tables:
        table_name = table.name
        count = int(table_row_counts.get(table_name) or row_count)
        seed_frame = pd.DataFrame(
            [
                {
                    "table_name": table_name,
                    "row_count": count,
                    "column_contract_json": _schema_column_contract_json(schema, table_name),
                    "relationships_json": _schema_relationships_json(schema, table_name),
                    "parent_rows_json": json.dumps(
                        {name: frame.head(20).to_dict(orient="records") for name, frame in tables.items()},
                        default=str,
                    ),
                    "user_prompt": _data_designer_user_prompt(),
                }
            ]
        )
        prompt = _schema_data_designer_prompt()
        model_config = dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=model_provider,
            skip_health_check=skip_health_check,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=float(os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.35")),
                top_p=float(os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.9")),
                max_tokens=int(os.getenv("SP_NEMO_DATA_DESIGNER_MAX_TOKENS", "4096")),
                extra_body=_data_designer_chat_extra_body(runtime),
            ),
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="rows_json",
                prompt=prompt,
                system_prompt=(
                    "Return strict JSON only. The response must be a JSON array of row objects. "
                    "Do not include Markdown, prose, comments, code fences, or wrapper objects."
                ),
                model_alias=model_alias,
            )
        )
        raw_value: str | None = None
        try:
            preview = data_designer.preview(builder, num_records=1)
            dataset = getattr(preview, "dataset", preview)
            value = _extract_data_designer_value(dataset, "rows_json")
            raw_value = str(value)
            rows = _parse_schema_rows_json(raw_value, table_name=table_name, expected_count=count, metadata=metadata)
        except NemoSchemaGenerationError:
            raise
        except Exception as exc:
            raise NemoSchemaGenerationError(
                f"Schema Data Designer generation failed for table {table_name!r}.",
                metadata=metadata,
                raw_exception=repr(exc),
                raw_value=raw_value,
            ) from exc
        frame = pd.DataFrame(rows)
        tables[table_name] = _coerce_schema_frame(schema, table_name, frame, count=count, metadata=metadata)
    return tables, metadata


def _deterministic_schema_tables(
    schema: Any,
    *,
    row_count: int,
    table_row_counts: dict[str, int],
    seed: int,
    text_prefix: str,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    relationships = list(getattr(schema, "relationships", []) or [])
    for table in schema.tables:
        table_name = table.name
        count = int(table_row_counts.get(table_name) or row_count)
        rows: list[dict[str, Any]] = []
        cols = list(schema.columns.get(table_name, []) or [])
        for idx in range(count):
            row: dict[str, Any] = {}
            for col in cols:
                rel = next((r for r in relationships if r.child_table == table_name and r.child_key == col.name), None)
                if rel and rel.parent_table in tables and rel.parent_key in tables[rel.parent_table]:
                    parent_values = tables[rel.parent_table][rel.parent_key].tolist()
                    row[col.name] = parent_values[idx % len(parent_values)]
                elif getattr(col, "unique", False):
                    row[col.name] = idx + 1
                else:
                    row[col.name] = _schema_cell_value(col, idx=idx, seed=seed, prefix=text_prefix)
            rows.append(row)
        tables[table_name] = pd.DataFrame(rows)
    return tables


def _schema_cell_value(col: Any, *, idx: int, seed: int, prefix: str) -> Any:
    col_type = getattr(col, "type", "text")
    params = dict(getattr(col, "distribution_params", {}) or {})
    if col_type in {"int", "foreign_key"}:
        return idx + 1
    if col_type in {"float", "decimal", "money", "currency"}:
        return round(float(params.get("min", 10.0)) + ((idx + seed) % 100) * 1.25, 2)
    if col_type in {"date", "datetime"}:
        return f"2024-01-{(idx % 28) + 1:02d}"
    if col_type == "email":
        return f"{prefix}_{idx + 1}@example.com"
    if col_type == "boolean":
        return (idx + seed) % 2 == 0
    if col_type == "categorical":
        choices = list(params.get("choices") or ["new", "active", "closed"])
        return choices[idx % len(choices)]
    return f"{prefix}_{col.name}_{idx + 1}"


def _schema_column_contract_json(schema: Any, table_name: str) -> str:
    columns = [
        {
            "name": col.name,
            "type": getattr(col, "type", "text"),
            "nullable": bool(getattr(col, "nullable", False)),
            "unique": bool(getattr(col, "unique", False)),
            "params": dict(getattr(col, "distribution_params", {}) or {}),
        }
        for col in (schema.columns.get(table_name, []) or [])
    ]
    return json.dumps(columns, default=str)


def _schema_relationships_json(schema: Any, table_name: str) -> str:
    relationships = [
        rel.model_dump(mode="json") if hasattr(rel, "model_dump") else dict(rel)
        for rel in (getattr(schema, "relationships", []) or [])
        if rel.child_table == table_name or rel.parent_table == table_name
    ]
    return json.dumps(relationships, default=str)


def _schema_data_designer_prompt() -> str:
    return (
        "Generate {{ row_count }} synthetic rows for table {{ table_name }}.\n"
        "User generation instructions: {{ user_prompt }}\n"
        "Columns: {{ column_contract_json }}\n"
        "Relationships: {{ relationships_json }}\n"
        "Previously generated parent table samples: {{ parent_rows_json }}\n"
        "Preserve primary-key uniqueness, foreign-key validity, non-null constraints, and column names.\n"
        "Return strict JSON only: a JSON array of exactly {{ row_count }} row objects. "
        "Before returning, count the array items. If row_count is 30, return exactly 30 objects, not 29 and not 31. "
        "Every object must contain every requested column exactly once. "
        "Do not include Markdown, prose, comments, code fences, or wrapper objects."
    )


def _parse_schema_rows_json(
    value: str,
    *,
    table_name: str,
    expected_count: int,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    text = value.strip()
    if not text:
        raise NemoSchemaGenerationError(
            f"Schema Data Designer returned empty rows_json for table {table_name!r}.",
            metadata=metadata,
            raw_value=value,
        )
    if text.startswith("```") or text.endswith("```"):
        raise NemoSchemaGenerationError(
            f"Schema Data Designer returned Markdown/code fences for table {table_name!r}; strict JSON is required.",
            metadata=metadata,
            raw_value=value,
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NemoSchemaGenerationError(
            f"Schema Data Designer did not return valid rows_json for table {table_name!r}.",
            metadata=metadata,
            raw_exception=repr(exc),
            raw_value=value,
        ) from exc
    if not isinstance(payload, list):
        raise NemoSchemaGenerationError(
            f"Schema Data Designer rows_json for table {table_name!r} must be a JSON array.",
            metadata=metadata,
            raw_value=value,
        )
    rows = [row for row in payload if isinstance(row, dict)]
    if len(rows) != len(payload):
        raise NemoSchemaGenerationError(
            f"Schema Data Designer rows_json for table {table_name!r} contains non-object rows.",
            metadata=metadata,
            raw_value=value,
        )
    if not rows:
        raise NemoSchemaGenerationError(
            f"Schema Data Designer returned {len(rows)} row(s) for table {table_name!r}; expected {expected_count}.",
            metadata=metadata,
            raw_value=value,
        )
    if len(rows) < int(expected_count):
        raise NemoSchemaGenerationError(
            f"Schema Data Designer returned only {len(rows)} row(s) for table {table_name!r}; expected {expected_count}.",
            metadata=metadata,
            raw_value=value,
        )
    if len(rows) > int(expected_count):
        if metadata is not None:
            repairs = metadata.setdefault("data_designer", {}).setdefault("row_count_repairs", [])
            repairs.append(
                {
                    "table": table_name,
                    "requested_rows": int(expected_count),
                    "returned_rows": len(rows),
                    "action": "trimmed_extra_rows",
                }
            )
        return rows[: int(expected_count)]
    return rows


def _parse_rows_json(value: str) -> list[dict[str, Any]]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("NeMo Data Designer did not return valid rows JSON.") from exc
    if isinstance(payload, dict):
        payload = payload.get("rows") or payload.get("data") or payload.get("records")
    if not isinstance(payload, list):
        raise RuntimeError("NeMo Data Designer schema output must be a JSON array.")
    return [row for row in payload if isinstance(row, dict)]


def _coerce_schema_frame(
    schema: Any,
    table_name: str,
    frame: pd.DataFrame,
    *,
    count: int,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = list(schema.columns.get(table_name, []) or [])
    if len(frame) != count:
        raise NemoSchemaGenerationError(
            f"Schema Data Designer returned {len(frame)} row(s) for table {table_name!r}; expected {count}.",
            metadata=metadata,
        )
    frame = frame.head(count).copy()
    for col in columns:
        if col.name not in frame.columns:
            raise NemoSchemaGenerationError(
                f"Schema Data Designer output for table {table_name!r} is missing required column {col.name!r}.",
                metadata=metadata,
            )
        if getattr(col, "unique", False):
            duplicated = frame[col.name].duplicated().any()
            if duplicated:
                raise NemoSchemaGenerationError(
                    f"Schema Data Designer output for table {table_name!r} has duplicate values in unique column {col.name!r}.",
                    metadata=metadata,
                )
    return frame[[col.name for col in columns]]


def _validate_schema_tables(schema: Any, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    issues: list[str] = []
    for table in schema.tables:
        frame = tables.get(table.name)
        if frame is None:
            issues.append(f"{table.name}: missing table")
            continue
        for col in schema.columns.get(table.name, []) or []:
            if col.name not in frame.columns:
                issues.append(f"{table.name}.{col.name}: missing column")
                continue
            if not getattr(col, "nullable", False) and frame[col.name].isna().any():
                issues.append(f"{table.name}.{col.name}: null values in non-null column")
            if getattr(col, "unique", False) and frame[col.name].duplicated().any():
                issues.append(f"{table.name}.{col.name}: duplicate values in unique column")
    for rel in getattr(schema, "relationships", []) or []:
        parent = tables.get(rel.parent_table)
        child = tables.get(rel.child_table)
        if parent is None or child is None:
            continue
        if rel.parent_key not in parent.columns or rel.child_key not in child.columns:
            issues.append(f"{rel.child_table}.{rel.child_key}: FK columns missing")
            continue
        missing = set(child[rel.child_key].dropna()) - set(parent[rel.parent_key].dropna())
        if missing:
            issues.append(f"{rel.child_table}.{rel.child_key}: {len(missing)} FK values missing from parent")
    return {
        "hard_checks_passed": not issues,
        "status": "passed" if not issues else "failed",
        "issues": issues,
    }


def _generate_pdf_values_with_data_designer(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    *,
    seed: int | None,
) -> dict[str, Any]:
    plan = _pdf_generation_plan(template, binding_map)
    if not plan["bindings"]:
        runtime = _digital_twin_data_designer_runtime()
        return {
            "fields": {},
            "tables": {},
            "inline_spans": {},
            "model_family": runtime.model_family,
            "model": runtime.model_id,
            "model_provider": runtime.provider,
        }
    if util.find_spec("data_designer") is None:
        raise RuntimeError("PDF Data Designer generation requires the standalone data-designer package.")
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "pdf-value-generator"
    prompt = _pdf_data_designer_prompt()
    model_config = dd.ModelConfig(
        alias=model_alias,
        model=runtime.model_id,
        provider=runtime.provider,
        skip_health_check=skip_health_check,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=float(os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.45")),
            top_p=float(os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.9")),
            max_tokens=int(os.getenv("SP_NEMO_DATA_DESIGNER_MAX_TOKENS", "4096")),
            extra_body=_data_designer_chat_extra_body(runtime),
        ),
    )
    data_designer = DataDesigner(
        model_providers=[
            dd.ModelProvider(
                name=runtime.provider,
                endpoint=runtime.endpoint,
                provider_type="openai",
                api_key=_data_designer_model_provider_api_key(runtime),
            )
        ]
    )
    merged_sdk_values: dict[str, Any] = {"values": {}}
    for chunk_index, plan_chunk in enumerate(_pdf_generation_plan_chunks(plan), start=1):
        seed_frame = pd.DataFrame(
            [
                {
                    "binding_plan_json": json.dumps(plan_chunk, default=str),
                    "user_prompt": _data_designer_user_prompt(),
                }
            ]
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="pdf_values_json",
                prompt=prompt,
                system_prompt="Return only valid JSON. No Markdown. No explanation.",
                model_alias=model_alias,
                output_format=_pdf_values_json_schema(plan_chunk),
            )
        )
        try:
            preview = data_designer.preview(builder, num_records=1)
        except Exception as exc:
            chunk_size = len(plan_chunk.get("bindings") or [])
            raise RuntimeError(
                "NeMo Data Designer PDF value generation failed "
                f"for binding batch {chunk_index} containing {chunk_size} bindings."
            ) from exc
        dataset = getattr(preview, "dataset", preview)
        value = _extract_data_designer_value(dataset, "pdf_values_json")
        sdk_values = _parse_pdf_values_json(value)
        chunk_values = _pdf_values_by_binding_id(sdk_values)
        if chunk_values is None:
            raise RuntimeError(
                "NeMo Data Designer PDF value generation must return a `values` object "
                f"for binding batch {chunk_index}."
            )
        merged_sdk_values["values"].update(chunk_values)
    sdk_values = merged_sdk_values
    values = _coerce_pdf_values_from_sdk(template, binding_map, sdk_values)
    values["model_family"] = runtime.model_family
    values["model"] = runtime.model_id
    values["model_provider"] = runtime.provider
    return values


def _regions_by_id(template: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(region.get("region_id")): region
        for page in template.get("pages", [])
        for region in page.get("regions", [])
        if region.get("region_id")
    }


def _pdf_generation_plan(template: dict[str, Any], binding_map: dict[str, Any]) -> dict[str, Any]:
    regions_by_id = _regions_by_id(template)
    bindings: list[dict[str, Any]] = []
    for binding in binding_map.get("bindings", []):
        binding_type = binding.get("binding_type")
        region_id = str(binding.get("region_id") or "")
        region = regions_by_id.get(region_id, {})
        if binding_type == "field":
            bindings.append(
                {
                    "binding_id": region_id,
                    "binding_type": "field",
                    "region_id": region_id,
                    "label": binding.get("label"),
                    "semantic_role": binding.get("semantic_role"),
                    "generator_strategy": binding.get("generator_strategy"),
                    "shape_pattern": region.get("shape_pattern"),
                    "value_type": region.get("value_type"),
                }
            )
        elif binding_type == "table":
            header_rows = int(region.get("header_row_count") or 0)
            all_rows = list(region.get("rows") or [])
            for data_row_index, row in enumerate(all_rows[header_rows:]):
                for column in binding.get("columns", []):
                    column_index = column.get("column_index")
                    if not isinstance(column_index, int):
                        continue
                    source_cell = row[column_index] if column_index < len(row) else {}
                    bindings.append(
                        {
                            "binding_id": _pdf_table_cell_binding_id(region_id, data_row_index, column_index),
                            "binding_type": "table_cell",
                            "region_id": region_id,
                            "row_index": data_row_index,
                            "column_index": column_index,
                            "label": column.get("label"),
                            "semantic_role": column.get("semantic_role"),
                            "generator_strategy": column.get("generator_strategy"),
                            "shape_pattern": source_cell.get("shape_pattern") if isinstance(source_cell, dict) else None,
                        }
                    )
        elif binding_type == "inline_spans":
            source_spans = list(region.get("inline_variable_spans") or [])
            for idx, span in enumerate(binding.get("spans", [])):
                bindings.append(
                    {
                        "binding_id": _pdf_inline_span_binding_id(region_id, idx),
                        "binding_type": "inline_span",
                        "region_id": region_id,
                        "span_index": idx,
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "semantic_role": span.get("semantic_role"),
                        "generator_strategy": span.get("generator_strategy"),
                        "shape_pattern": (
                            source_spans[idx].get("shape_pattern")
                            if idx < len(source_spans) and isinstance(source_spans[idx], dict)
                            else None
                        ),
                    }
                )
    return {"bindings": bindings}


def _pdf_generation_plan_chunks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    bindings = list(plan.get("bindings") or [])
    if not bindings:
        return [{"bindings": []}]
    chunk_size = max(1, int(os.getenv("SP_PDF_DATA_DESIGNER_BINDINGS_PER_BATCH", "40")))
    return [{"bindings": bindings[index : index + chunk_size]} for index in range(0, len(bindings), chunk_size)]


def _pdf_table_cell_binding_id(region_id: str, row_index: int, column_index: int) -> str:
    return f"{region_id}__r{row_index}__c{column_index}"


def _pdf_inline_span_binding_id(region_id: str, span_index: int) -> str:
    return f"{region_id}__s{span_index}"


def _pdf_data_designer_prompt() -> str:
    return (
        "Generate privacy-safe synthetic replacement values for a PDF template.\n"
        "User generation instructions: {{ user_prompt }}\n"
        "Binding plan: {{ binding_plan_json }}\n"
        "Return a compact JSON object with exactly one key: values.\n"
        "values must be an object whose keys are binding_id values from the binding plan "
        "and whose values are synthetic replacement strings.\n"
        "Do not return arrays. Do not return nested table rows. Do not return region metadata. "
        "Only include binding_id keys provided in the input. "
        "Preserve meaning implied by labels and semantic roles. Do not include real PII."
    )


def _pdf_values_json_schema(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    binding_ids = [
        str(binding.get("binding_id"))
        for binding in (plan or {}).get("bindings", [])
        if binding.get("binding_id")
    ]
    values_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    if binding_ids:
        values_schema = {
            "type": "object",
            "properties": {binding_id: {"type": "string"} for binding_id in binding_ids},
            "required": binding_ids,
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "values": values_schema,
        },
        "required": ["values"],
        "additionalProperties": False,
    }


def _parse_pdf_values_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise RuntimeError("NeMo Data Designer did not return valid PDF values JSON.") from exc
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as nested_exc:
            raise RuntimeError("NeMo Data Designer did not return valid PDF values JSON.") from nested_exc
    if not isinstance(payload, dict):
        raise RuntimeError("NeMo Data Designer PDF output must be a JSON object.")
    return payload


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_pdf_values_from_sdk(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    sdk_values: dict[str, Any],
) -> dict[str, Any]:
    regions_by_id = _regions_by_id(template)
    fields: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    inline_spans: dict[str, Any] = {}
    sdk_values_by_binding = _pdf_values_by_binding_id(sdk_values)
    has_binding_value_map = sdk_values_by_binding is not None
    sdk_fields = _pdf_field_values_by_region(sdk_values.get("fields") or {})
    sdk_tables = _pdf_table_values_by_region(sdk_values.get("tables") or {})
    sdk_inline_spans = _pdf_inline_values_by_region(sdk_values.get("inline_spans") or {})

    for binding in binding_map.get("bindings", []):
        binding_type = binding.get("binding_type")
        region_id = str(binding.get("region_id") or "")
        region = regions_by_id.get(region_id, {})
        if binding_type == "field":
            fields[region_id] = {
                "label": binding.get("label"),
                "semantic_role": binding.get("semantic_role"),
                "value": _sdk_value_for(
                    sdk_values_by_binding if has_binding_value_map else sdk_fields,
                    region_id,
                    "values" if has_binding_value_map else "fields",
                ),
            }
        elif binding_type == "inline_spans":
            spans = sdk_inline_spans.get(region_id, [])
            if not has_binding_value_map and not isinstance(spans, list):
                raise RuntimeError(f"SDK output missing inline spans for region `{region_id}`.")
            expected_spans = list(binding.get("spans") or [])
            inline_spans[region_id] = [
                {
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "semantic_role": span.get("semantic_role"),
                    "value": _sdk_value_for(
                        sdk_values_by_binding,
                        _pdf_inline_span_binding_id(region_id, idx),
                        "values",
                    )
                    if has_binding_value_map
                    else _sdk_list_value(spans, idx, f"inline_spans.{region_id}[{idx}]"),
                }
                for idx, span in enumerate(expected_spans)
            ]
        elif binding_type == "table":
            header_row_count = int(region.get("header_row_count") or 0)
            source_rows = list(region.get("rows") or [])
            sdk_rows = sdk_tables.get(region_id)
            if not has_binding_value_map and not isinstance(sdk_rows, list):
                raise RuntimeError(f"SDK output missing table rows for region `{region_id}`.")
            rows_out = []
            for row_index, row in enumerate(source_rows):
                if row_index < header_row_count:
                    rows_out.append([{"semantic_role": None, "value": cell.get("text", "")} for cell in row])
                    continue
                data_row_index = row_index - header_row_count
                row_out = []
                for col_idx, column in enumerate(binding.get("columns", [])):
                    row_out.append(
                        {
                            "semantic_role": column.get("semantic_role"),
                            "value": _sdk_value_for(
                                sdk_values_by_binding,
                                _pdf_table_cell_binding_id(region_id, data_row_index, col_idx),
                                "values",
                            )
                            if has_binding_value_map
                            else _sdk_table_cell_value(
                                sdk_rows,
                                data_row_index,
                                col_idx,
                                f"tables.{region_id}[{data_row_index}][{col_idx}]",
                            ),
                        }
                    )
                rows_out.append(row_out)
            tables[region_id] = rows_out
    return {"fields": fields, "tables": tables, "inline_spans": inline_spans}


def _pdf_values_by_binding_id(values: dict[str, Any]) -> dict[str, Any] | None:
    if "values" not in values:
        return None
    raw_values = values.get("values")
    if not isinstance(raw_values, dict):
        raise RuntimeError("SDK output `values` must be an object keyed by binding_id.")
    return {str(key): value for key, value in raw_values.items()}


def _pdf_field_values_by_region(values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        return values
    if not isinstance(values, list):
        return {}
    out: dict[str, Any] = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        region_id = str(row.get("region_id") or "")
        if region_id:
            out[region_id] = row
    return out


def _pdf_table_values_by_region(values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        return values
    if not isinstance(values, list):
        return {}
    out: dict[str, Any] = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        region_id = str(row.get("region_id") or "")
        if region_id:
            out[region_id] = row.get("cells") if "cells" in row else row.get("rows")
    return out


def _pdf_inline_values_by_region(values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        return values
    if not isinstance(values, list):
        return {}
    out: dict[str, Any] = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        region_id = str(row.get("region_id") or "")
        if region_id:
            out[region_id] = row.get("spans")
    return out


def _sdk_value_for(values: dict[str, Any], region_id: str, scope: str) -> Any:
    value = values.get(region_id)
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        raise RuntimeError(f"SDK output missing value for `{scope}.{region_id}`.")
    return value


def _sdk_list_value(values: list[Any], index: int, scope: str) -> Any:
    if index >= len(values):
        raise RuntimeError(f"SDK output missing value for `{scope}`.")
    value = values[index]
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        raise RuntimeError(f"SDK output missing value for `{scope}`.")
    return value


def _sdk_table_cell_value(values: Any, data_row_index: int, column_index: int, scope: str) -> Any:
    if isinstance(values, list):
        if values and all(isinstance(row, list) for row in values):
            return _sdk_list_value(values[data_row_index] if data_row_index < len(values) else [], column_index, scope)
        for cell in values:
            if not isinstance(cell, dict):
                continue
            if int(cell.get("row_index", -1)) == data_row_index and int(cell.get("column_index", -1)) == column_index:
                value = cell.get("value")
                if value is not None:
                    return value
    raise RuntimeError(f"SDK output missing value for `{scope}`.")


def _mock_data_designer_pdf_values(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    *,
    seed: int | None,
) -> dict[str, Any]:
    rng_seed = int(seed or 0)
    plan = _pdf_generation_plan(template, binding_map)
    sdk_values = {
        "values": {
            row["binding_id"]: f"sdk_mock_{rng_seed}_{idx}_{row.get('semantic_role') or row.get('binding_type') or 'value'}"
            for idx, row in enumerate(plan["bindings"])
        },
    }
    return _coerce_pdf_values_from_sdk(template, binding_map, sdk_values)
