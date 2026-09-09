"""Generation backend boundary for current and optional NeMo generators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import hashlib
from importlib import util
import json
import os
from pathlib import Path
import re
from datetime import UTC, datetime
from typing import Any
import uuid

import pandas as pd

from synth_platform.domain.contracts.models import CanonicalContract
from synth_platform.engine.common.database.core.stage_result import STATUS_SUCCESS, StageResult
from synth_platform.engine.transcripts import ssot_designer as transcript_ssot_designer
from synth_platform.engine.transcripts.service import generate_synthetic_transcript
from synth_platform.engine.generation.slm_runtime import (
    PlatformSLMRuntime,
    data_designer_max_parallel_requests,
    data_designer_max_tokens,
    data_designer_skip_health_check,
    data_designer_timeout,
    is_local_endpoint,
    preflight_slm_endpoint,
    require_data_designer_api_key,
    resolve_platform_slm_runtime,
)
from synth_platform.engine.generation.data_designer_provider import (
    build_data_designer_model_config,
    build_data_designer_provider,
    load_data_designer_sdk,
)


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
class BackendTranscriptTwinResult:
    turns: list[dict[str, Any]]
    structured_ssot: dict[str, Any]
    metadata: dict[str, Any]


DataDesignerRuntime = PlatformSLMRuntime


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
        schema_row_mode = os.getenv("SP_SCHEMA_ROW_GENERATION_MODE", "deterministic").strip().lower()
        if export_format != "csv":
            raise ValueError("NeMo schema backend currently supports CSV export only.")
        if schema_row_mode in {"deterministic", "fast", "platform"}:
            tables = _deterministic_schema_tables(
                contract,
                row_count=row_count,
                table_row_counts=table_row_counts,
                seed=seed,
                text_prefix="nemo_schema",
            )
            metadata = {
                "backend": self.name,
                "data_designer": {
                    "mode": "schema_draft_plus_deterministic_rows",
                    "model_alias": "schema-draft-generator",
                    "row_generation": "deterministic_platform_generator",
                },
            }
        elif _mock_data_designer_enabled():
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
    return resolve_platform_slm_runtime().api_key_env


def _data_designer_api_key() -> str | None:
    return os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY")


def _data_designer_endpoint() -> str:
    return resolve_platform_slm_runtime().endpoint


def _is_local_endpoint(endpoint: str) -> bool:
    return is_local_endpoint(endpoint)


def _data_designer_chat_extra_body(runtime: DataDesignerRuntime) -> dict[str, Any] | None:
    return None


def _digital_twin_data_designer_runtime() -> DataDesignerRuntime:
    """Resolve the open-source SLM used by PDF/TXT twin generation."""
    return resolve_platform_slm_runtime()


def _require_data_designer_api_key(runtime: DataDesignerRuntime) -> None:
    require_data_designer_api_key(runtime)


def _data_designer_skip_health_check() -> bool:
    return data_designer_skip_health_check()


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
    mode = os.getenv("SP_TRANSCRIPT_GENERATION_MODE", "structured").strip().lower()
    if mode in {"row", "per_turn", "per-turn"}:
        return _generate_transcript_rows_with_standalone_data_designer(
            contract,
            turn_count=turn_count,
            seed=seed,
        )
    return _generate_transcript_structured_with_standalone_data_designer(
        contract,
        turn_count=turn_count,
        seed=seed,
    )


def _generate_transcript_structured_with_standalone_data_designer(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    try:
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "transcript-generator"
    target_turns = int(turn_count or _contract_turn_count(contract) or 8)
    turn_seed_frame = _transcript_turn_seed_frame(contract, target_turns=target_turns, seed=seed)
    seed_frame = pd.DataFrame(
        [
            {
                "source_contract_json": _transcript_structured_generation_seed(turn_seed_frame, target_turns),
                "target_turn_count": target_turns,
                "user_prompt": _data_designer_user_prompt(),
            }
        ]
    )

    model_config = build_data_designer_model_config(
        dd,
        model_alias,
        workflow="transcript-conversation-structured",
        runtime=runtime,
        temperature=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TOP_P", "0.9")),
        timeout_env="SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT",
        parallel_env="SP_TRANSCRIPT_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
        extra_body=_data_designer_chat_extra_body(runtime),
        skip_health_check=skip_health_check,
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="synthetic_transcript_json",
            prompt=_transcript_data_designer_prompt(),
            system_prompt=(
                "Return only the requested JSON object. Do not include Markdown, comments, source identifiers, "
                "or raw source transcript text."
            ),
            model_alias=model_alias,
            output_format=_transcript_json_schema(target_turns),
        )
    )

    data_designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
    preview = data_designer.preview(builder, num_records=1)
    dataset = getattr(preview, "dataset", preview)
    value = _extract_data_designer_value(dataset, "synthetic_transcript_json")
    rows = _parse_transcript_json(value)
    return _transcript_rows_from_structured_data_designer(rows, turn_seed_frame)


def generate_transcript_twin_with_data_designer(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> BackendTranscriptTwinResult:
    """Generate structured SSOT and synthetic transcript turns in one SDK preview."""
    output_mode = os.getenv("SP_TRANSCRIPT_TWIN_OUTPUT_MODE", "ssot").strip().lower()
    if output_mode == "ssot":
        ssot = transcript_ssot_designer.build_transcript_ssot_from_contract_evidence(contract, enabled=True)
        if ssot.get("status") != "generated":
            raise RuntimeError(f"Transcript SSOT generation failed: {ssot.get('reason') or ssot.get('status')}")
        return BackendTranscriptTwinResult(
            turns=generate_synthetic_transcript(contract, turn_count=turn_count, seed=seed),
            structured_ssot=ssot,
            metadata={
                "mode": "ssot_only_sdk_plus_platform_turns",
                "model_alias": "transcript-ssot-builder",
                "sdk_preview_calls": 1,
            },
        )
    if _mock_data_designer_enabled():
        rows = _mock_data_designer_transcript(contract, turn_count=turn_count, seed=seed)
        ssot = transcript_ssot_designer.build_transcript_ssot_from_contract_evidence(contract, enabled=True)
        return BackendTranscriptTwinResult(
            turns=rows,
            structured_ssot=ssot,
            metadata={"mode": "mock", "sdk_preview_calls": 0},
        )
    if util.find_spec("data_designer") is None:
        raise RuntimeError(
            "Combined transcript twin generation requires the standalone Data Designer SDK. "
            "Install data-designer in a Python <3.14 environment."
        )
    return _generate_transcript_twin_with_standalone_data_designer(
        contract,
        turn_count=turn_count,
        seed=seed,
    )


def _generate_transcript_twin_with_standalone_data_designer(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> BackendTranscriptTwinResult:
    try:
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "transcript-twin-generator"
    target_turns = int(turn_count or _contract_turn_count(contract) or 8)
    turn_seed_frame = _transcript_turn_seed_frame(contract, target_turns=target_turns, seed=seed)
    seed_frame = pd.DataFrame(
        [
            {
                "source_contract_json": _transcript_structured_generation_seed(turn_seed_frame, target_turns),
                "target_turn_count": target_turns,
                "user_prompt": _data_designer_user_prompt(),
            }
        ]
    )
    model_config = build_data_designer_model_config(
        dd,
        model_alias,
        workflow="transcript-twin-structured",
        runtime=runtime,
        temperature=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TOP_P", "0.9")),
        timeout_env="SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT",
        parallel_env="SP_TRANSCRIPT_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
        extra_body=_data_designer_chat_extra_body(runtime),
        skip_health_check=skip_health_check,
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="synthetic_twin_json",
            prompt=_transcript_twin_data_designer_prompt(),
            system_prompt=(
                "Return only the requested JSON object. Do not include Markdown, comments, source identifiers, "
                "or raw source transcript text."
            ),
            model_alias=model_alias,
            output_format=_transcript_twin_json_schema(target_turns),
        )
    )
    data_designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
    preview = data_designer.preview(builder, num_records=1)
    dataset = getattr(preview, "dataset", preview)
    payload = _coerce_data_designer_mapping(_extract_data_designer_value(dataset, "synthetic_twin_json"))
    turns = _transcript_rows_from_structured_data_designer(_parse_transcript_json(payload.get("turns")), turn_seed_frame)
    ssot = transcript_ssot_designer._enrich_ssot_payload(
        _coerce_data_designer_mapping(payload.get("structured_ssot")),
        transcript_ssot_designer._contract_evidence_turns(_contract_metadata_dict(contract)),
        source_name=str(contract.contract_id or "transcript"),
    )
    ssot["status"] = "generated"
    validation = transcript_ssot_designer._validate_ssot_value(ssot)
    if not transcript_ssot_designer._validation_passed(validation):
        raise RuntimeError(
            "NeMo Data Designer transcript twin SSOT failed SDK validation: "
            f"{transcript_ssot_designer._validation_error(validation)}"
        )
    return BackendTranscriptTwinResult(
        turns=turns,
        structured_ssot=ssot,
        metadata={
            "mode": "transcript_twin_single_structured_preview",
            "model_alias": model_alias,
            "provider": runtime.provider,
            "model": runtime.model_id,
            "sdk_preview_calls": 1,
        },
    )


def _generate_transcript_rows_with_standalone_data_designer(
    contract: CanonicalContract,
    *,
    turn_count: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    try:
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "transcript-generator"
    target_turns = int(turn_count or _contract_turn_count(contract) or 8)
    seed_frame = _transcript_turn_seed_frame(contract, target_turns=target_turns, seed=seed)

    model_config = build_data_designer_model_config(
        dd,
        model_alias,
        workflow="transcript-conversation-row",
        runtime=runtime,
        temperature=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TOP_P", "0.9")),
        timeout_env="SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT",
        parallel_env="SP_TRANSCRIPT_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
        extra_body=_data_designer_chat_extra_body(runtime),
        skip_health_check=skip_health_check,
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="synthetic_message",
            prompt=_transcript_turn_prompt(),
            system_prompt=(
                "Return only the final customer-interaction utterance text. "
                "Do not include speaker labels, speaker IDs, placeholders, Markdown, JSON, phone numbers, emails, "
                "account numbers, source identifiers, or raw source text."
            ),
            model_alias=model_alias,
        )
    )
    builder.add_column(
        dd.ValidationColumnConfig(
            name="synthetic_message_validation",
            target_columns=["synthetic_message"],
            validator_type=dd.ValidatorType.LOCAL_CALLABLE,
            validator_params=dd.LocalCallableValidatorParams(
                validation_function=_validate_data_designer_transcript_messages
            ),
            batch_size=25,
        )
    )

    data_designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
    preview = data_designer.preview(builder, num_records=target_turns)
    dataset = getattr(preview, "dataset", preview)
    rows = _transcript_rows_from_data_designer(dataset, seed_frame)
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
    base_url = os.getenv("SP_NEMO_DATA_DESIGNER_BASE_URL", runtime.endpoint)
    model_alias = "transcript-generator"
    target_turns = int(turn_count or _contract_turn_count(contract) or 8)
    prompt = _transcript_data_designer_inline_prompt(contract, target_turns)

    model_config_kwargs: dict[str, Any] = {
        "alias": model_alias,
        "model": runtime.model_id,
        "provider": runtime.provider,
        "inference_parameters": InferenceParameters(
            max_parallel_requests=data_designer_max_parallel_requests(
                workflow_env="SP_TRANSCRIPT_DATA_DESIGNER_MAX_PARALLEL_REQUESTS"
            ),
            timeout=data_designer_timeout(workflow_env="SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT"),
            temperature=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("SP_TRANSCRIPT_DATA_DESIGNER_TOP_P", "0.9")),
            max_tokens=data_designer_max_tokens(512, workflow_env="SP_TRANSCRIPT_DATA_DESIGNER_MAX_TOKENS"),
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


def _transcript_structured_generation_seed(turn_seed_frame: pd.DataFrame, target_turns: int) -> str:
    prompt_columns = [
        "turn",
        "speaker",
        "speaker_role",
        "conversation_phase",
        "issue_type",
        "issue_summary",
        "conversation_topic",
        "topic_terms",
        "domain",
        "source_locale",
        "target_locale",
        "target_persona",
        "message_goal",
        "reference",
    ]
    available_columns = [column for column in prompt_columns if column in turn_seed_frame.columns]
    return json.dumps(
        {
            "target_turn_count": target_turns,
            "turn_plan": turn_seed_frame[available_columns].to_dict(orient="records"),
        },
        sort_keys=True,
        default=str,
    )


def _transcript_twin_data_designer_prompt() -> str:
    return (
        "Create one complete synthetic customer interaction twin from sanitized contract metadata only.\n"
        "User generation instructions: {{ user_prompt }}\n"
        "Target turn count: {{ target_turn_count }}\n"
        "Sanitized contract metadata JSON: {{ source_contract_json }}\n"
        "Return a JSON object with exactly two keys: structured_ssot and turns.\n"
        "structured_ssot must follow the requested structured twin schema.\n"
        "turns must contain exactly the requested number of synthetic messages with turn, speaker, timestamp, and text.\n"
        "Use the turn plan speaker and speaker_role values. Preserve the support flow from the issue summary, "
        "message goals, and grounding terms. Do not copy source text. Do not include phone numbers, emails, "
        "account numbers, markdown, labels inside message text, or raw source transcript text."
    )


def _transcript_twin_json_schema(target_turns: int | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "structured_ssot": transcript_ssot_designer._ssot_schema(),
            "turns": _transcript_json_schema(target_turns)["properties"]["turns"],
        },
        "required": ["structured_ssot", "turns"],
        "additionalProperties": False,
    }


def _coerce_data_designer_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise RuntimeError("NeMo Data Designer structured output must be a JSON object.")
    return parsed


def _contract_metadata_dict(contract: CanonicalContract) -> dict[str, Any]:
    entity = contract.entities[0] if contract.entities else None
    return dict(entity.metadata if entity else {})


def _transcript_rows_from_structured_data_designer(
    rows: list[dict[str, Any]],
    seed_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    seed_records = seed_frame.to_dict(orient="records")
    if len(rows) != len(seed_records):
        raise RuntimeError(
            "NeMo Data Designer transcript batch failed SDK validation: "
            f"generated {len(rows)} turns for {len(seed_records)} requested turns"
        )
    aligned_rows: list[dict[str, Any]] = []
    for idx, seed_record in enumerate(seed_records):
        source_row = rows[idx]
        text = _clean_transcript_message(
            source_row.get("text") or source_row.get("content") or source_row.get("message")
        )
        validation = _validate_data_designer_transcript_message(
            {
                **seed_record,
                "synthetic_message": text,
            }
        )
        if not validation["is_valid"]:
            raise RuntimeError(
                "NeMo Data Designer transcript message failed SDK validation: "
                f"{validation['error_messages']}"
            )
        aligned_rows.append(
            {
                "turn": int(seed_record.get("turn") or idx + 1),
                "speaker": str(seed_record.get("speaker") or ("Customer" if idx % 2 == 0 else "Agent")),
                "timestamp": str(source_row.get("timestamp") or ""),
                "text": text,
            }
        )
    _validate_transcript_row_batch_quality(aligned_rows, expected_rows=len(seed_records))
    return aligned_rows


def _transcript_turn_seed_frame(
    contract: CanonicalContract,
    *,
    target_turns: int,
    seed: int | None,
) -> pd.DataFrame:
    entity = contract.entities[0] if contract.entities else None
    metadata = dict(entity.metadata if entity else {})
    context = dict(metadata.get("synthetic_context") or {})
    support_context = _transcript_support_context(metadata)
    topic_terms = _transcript_generation_topic_terms(support_context, metadata)
    source_turn_hashes_json = json.dumps([str(value) for value in metadata.get("source_turn_hashes") or []])
    source_ngram_hashes_json = json.dumps([str(value) for value in metadata.get("source_ngram_hashes") or []])
    topic = _transcript_conversation_topic(topic_terms, support_context=support_context)
    reference = "the synthetic case"
    if context.get("claim_id") or context.get("tracking_id"):
        reference = "the mock case reference"
    elif context.get("policy_id"):
        reference = "the mock policy reference"
    elif context.get("vehicle_id"):
        reference = "the mock item reference"
    rows: list[dict[str, Any]] = []
    speaker_sequence = _transcript_speaker_sequence(metadata, max(1, target_turns))
    turn_plan = [row for row in metadata.get("turn_plan") or [] if isinstance(row, dict)]
    speaker_roles = {str(key): str(value) for key, value in dict(metadata.get("speaker_roles") or {}).items()}
    speaker_occurrences: dict[str, int] = {}
    for idx in range(max(1, target_turns)):
        speaker = speaker_sequence[idx]
        occurrence = speaker_occurrences.get(speaker, 0)
        speaker_occurrences[speaker] = occurrence + 1
        intent = (
            str(turn_plan[idx % len(turn_plan)].get("intent") or "")
            if turn_plan
            else _transcript_turn_goal(idx, target_turns)
        )
        if re.fullmatch(r"speaker_\d+", speaker.strip().lower()):
            role = "customer" if idx % 2 == 0 else "agent"
        else:
            role = (
                str(turn_plan[idx % len(turn_plan)].get("role") or "")
                if turn_plan
                else speaker_roles.get(speaker, "")
            )
            role = role or speaker_roles.get(speaker) or ("customer" if idx % 2 == 0 else "agent")
        display_speaker = _friendly_transcript_speaker(speaker, role, idx)
        if display_speaker in {"Customer", "Agent"}:
            role = display_speaker.lower()
        rows.append(
            {
                "turn": idx + 1,
                "speaker": display_speaker,
                "source_speaker_key": speaker,
                "speaker_role": role,
                "speaker_occurrence": occurrence + 1,
                "conversation_phase": _transcript_conversation_phase(idx, target_turns),
                "topic_terms": ", ".join(topic_terms[:6]) or "customer support",
                "conversation_topic": topic,
                "issue_type": _label_from_token(str(support_context.get("issue_type") or "customer_support_request")),
                "issue_summary": str(support_context.get("issue_summary") or "").strip(),
                "domain": context.get("domain") or "customer_support",
                "source_locale": context.get("source_locale") or "en_US",
                "target_locale": str(context.get("target_locale") or context.get("source_locale") or "en_US"),
                "target_persona": str(context.get("target_persona") or _transcript_persona_for_role(role)),
                "source_turn_hashes_json": source_turn_hashes_json,
                "source_ngram_hashes_json": source_ngram_hashes_json,
                "customer_name": "the mock customer" if context.get("customer_name") else "",
                "reference": reference,
                "message_goal": intent,
                "seed": int(seed or 0),
            }
        )
    return pd.DataFrame(rows)


def _transcript_support_context(metadata: dict[str, Any]) -> dict[str, Any]:
    ssot = metadata.get("structured_ssot")
    if not isinstance(ssot, dict):
        return {}
    support_context = ssot.get("support_context")
    return dict(support_context) if isinstance(support_context, dict) else {}


def _transcript_generation_topic_terms(support_context: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    terms = _split_topic_terms(support_context.get("topic_terms"))
    if not terms:
        terms = [str(term).strip() for term in metadata.get("topic_terms") or [] if str(term).strip()]
    if not terms:
        issue_type = _label_from_token(str(support_context.get("issue_type") or "customer_support_request"))
        terms = [issue_type]
    return terms


def _split_topic_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,;\n]+", text) if item.strip()]


def _label_from_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ")).strip()


def _transcript_speaker_sequence(metadata: dict[str, Any], target_turns: int) -> list[str]:
    source_sequence = [str(value) for value in metadata.get("speaker_sequence") or [] if str(value).strip()]
    if source_sequence:
        labels = set(source_sequence)
        if labels and labels <= {"Customer", "Agent"}:
            first = source_sequence[0] if source_sequence[0] in {"Customer", "Agent"} else "Customer"
            second = "Agent" if first == "Customer" else "Customer"
            return [first if idx % 2 == 0 else second for idx in range(max(1, target_turns))]
        return [source_sequence[idx % len(source_sequence)] for idx in range(max(1, target_turns))]
    return ["Customer" if idx % 2 == 0 else "Agent" for idx in range(max(1, target_turns))]


def _friendly_transcript_speaker(speaker: str, role: str, index: int) -> str:
    normalized = str(speaker or "").strip()
    lower = normalized.lower()
    if lower in {"customer", "caller", "client", "member", "patient"}:
        return "Customer"
    if lower in {"agent, representative", "agent", "support", "specialist", "supervisor"}:
        return "Agent" if lower != "supervisor" else "Supervisor"
    role_lower = str(role or "").strip().lower()
    if role_lower == "customer":
        return "Customer"
    if role_lower == "agent":
        return "Agent"
    return f"Participant {index + 1}"


def _transcript_conversation_topic(topic_terms: list[str], *, support_context: dict[str, Any] | None = None) -> str:
    support_context = support_context or {}
    summary = str(support_context.get("issue_summary") or "").strip()
    if summary:
        return summary
    issue_type = _label_from_token(str(support_context.get("issue_type") or ""))
    if issue_type:
        return issue_type
    useful = [str(term).replace("_", " ").strip().lower() for term in topic_terms if str(term).strip()]
    return ", ".join(useful[:4]) or "customer support request"


def _transcript_persona_for_role(role: str) -> str:
    role = str(role or "").strip().lower()
    if role == "agent":
        return "professional support representative"
    if role == "customer":
        return "customer requesting help"
    return "support conversation participant"


def _transcript_conversation_phase(index: int, target_turns: int) -> str:
    if index == 0:
        return "opening"
    if index >= max(0, target_turns - 2):
        return "closing"
    if index < max(2, target_turns // 3):
        return "discovery"
    return "resolution"


def _transcript_turn_goal(index: int, target_turns: int) -> str:
    if index == 0:
        return "customer opens the issue"
    if index == 1:
        return "agent acknowledges and starts helping"
    if index >= target_turns - 2:
        return "resolve or confirm next step"
    return "progress the support conversation"


def _transcript_turn_prompt() -> str:
    return (
        "Create one synthetic transcript message.\n"
        "Turn: {{ turn }}\n"
        "Speaker: {{ speaker }}\n"
        "Speaker role: {{ speaker_role }}\n"
        "Speaker occurrence: {{ speaker_occurrence }}\n"
        "Conversation phase: {{ conversation_phase }}\n"
        "Issue type: {{ issue_type }}\n"
        "Issue summary: {{ issue_summary }}\n"
        "Conversation topic: {{ conversation_topic }}\n"
        "Grounding terms: {{ topic_terms }}\n"
        "Domain: {{ domain }}\n"
        "Source locale: {{ source_locale }}\n"
        "Target locale: {{ target_locale }}\n"
        "Target persona: {{ target_persona }}\n"
        "Synthetic customer name, if any: {{ customer_name }}\n"
        "Synthetic reference: {{ reference }}\n"
        "Message goal: {{ message_goal }}\n"
        "Write one natural, concise message for this speaker and turn role. "
        "Use the target locale exactly; if source locale and target locale match, do not translate. "
        "Preserve the support flow described by the structured SSOT issue summary and grounding terms. "
        "If speaker role is customer, write as the person requesting help. "
        "If speaker role is agent, write as the support representative helping the customer. "
        "Do not copy source text. "
        "Do not include phone numbers, emails, account numbers, markdown, labels, or JSON."
    )


def _transcript_rows_from_data_designer(dataset: Any, seed_frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = _data_designer_dataset_to_frame(dataset, "transcript")
    if frame.empty:
        raise RuntimeError("NeMo Data Designer returned no transcript records.")
    if "synthetic_message" not in frame.columns:
        raise RuntimeError("NeMo Data Designer transcript output missing `synthetic_message` column.")

    rows: list[dict[str, Any]] = []
    seed_records = seed_frame.to_dict(orient="records")
    for idx, record in enumerate(frame.to_dict(orient="records")):
        seed_record = seed_records[idx] if idx < len(seed_records) else {}
        validation = record.get("synthetic_message_validation")
        if not _data_designer_validation_passed(validation):
            raise RuntimeError(
                "NeMo Data Designer transcript message failed SDK validation: "
                f"{_data_designer_validation_error(validation)}"
            )
        text = _clean_transcript_message(record.get("synthetic_message"))
        if not text:
            continue
        rows.append(
            {
                "turn": int(record.get("turn") or seed_record.get("turn") or idx + 1),
                "speaker": str(record.get("speaker") or seed_record.get("speaker") or ("Customer" if idx % 2 == 0 else "Agent")),
                "timestamp": "",
                "text": text,
            }
        )
    _validate_transcript_row_batch_quality(rows, expected_rows=len(seed_records))
    return rows


def _validate_transcript_row_batch_quality(rows: list[dict[str, Any]], *, expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise RuntimeError(
            "NeMo Data Designer transcript batch failed SDK validation: "
            f"generated {len(rows)} usable records for {expected_rows} requested records"
        )
    used_texts: list[str] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if _is_repetitive_transcript_text(text, used_texts):
            raise RuntimeError("NeMo Data Designer transcript batch failed SDK validation: repeated_message")
        used_texts.append(text)


def _is_source_replay_transcript_text(text: str, seed_record: dict[str, Any]) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return True
    if _transcript_text_hash(normalized) in _hashes_from_seed_record(seed_record, "source_turn_hashes_json"):
        return True
    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", normalized.lower())
    if len(words) < 5:
        return False
    source_ngram_hashes = _hashes_from_seed_record(seed_record, "source_ngram_hashes_json")
    return any(
        _transcript_text_hash(" ".join(words[idx : idx + 5])) in source_ngram_hashes
        for idx in range(0, len(words) - 4)
    )


def _hashes_from_seed_record(seed_record: dict[str, Any], key: str) -> set[str]:
    value = seed_record.get(key)
    if isinstance(value, set):
        return {str(item) for item in value}
    if isinstance(value, list):
        return {str(item) for item in value}
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in parsed} if isinstance(parsed, list) else set()


def _transcript_text_hash(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def _is_repetitive_transcript_text(text: str, used_texts: list[str]) -> bool:
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    for existing in used_texts:
        other = re.sub(r"\s+", " ", existing.strip().lower())
        if normalized == other:
            return True
        if SequenceMatcher(None, normalized, other).ratio() >= 0.85:
            return True
    return False


def _text_conflicts_with_transcript_role(text: str, *, speaker: str, role: str) -> bool:
    lower = text.lower()
    speaker_lower = speaker.lower()
    is_agent = role == "agent" or speaker_lower == "agent"
    is_customer = role == "customer" or speaker_lower == "customer"

    support_action_start = re.match(
        r"^\s*(?:i\s+(?:can|will|am|have)|i'll|let\s+me|we\s+can|give\s+me)\b",
        lower,
    )
    first_person_problem = re.search(
        r"\b(?:my\s+\w+|i(?:'m| am| have|'ve)\s+(?:having|seeing|getting|unable|trying|concerned|worried|charged|missing|locked|stuck))\b",
        lower,
    )
    asks_for_help = re.search(
        r"\b(?:can|could|please|help|check|investigate|explain|confirm|why|issue|problem)\b",
        lower,
    )
    if is_customer and support_action_start:
        return True
    if is_agent and first_person_problem and asks_for_help and not support_action_start:
        return True
    if is_agent and lower.startswith("agent,"):
        return True
    return False


def _data_designer_dataset_to_frame(dataset: Any, workflow: str) -> pd.DataFrame:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    if isinstance(dataset, pd.DataFrame):
        return dataset
    if isinstance(dataset, list):
        return pd.DataFrame(dataset)
    if isinstance(dataset, dict):
        return pd.DataFrame(dataset)
    raise RuntimeError(f"Unsupported NeMo Data Designer {workflow} preview response shape.")


def _clean_transcript_message(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = text.splitlines()[0].strip() if "\n" in text else text
    return text.strip("\"' ,")


def _validate_data_designer_transcript_messages(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [_validate_data_designer_transcript_message(row) for row in frame.to_dict(orient="records")]
    )


def _validate_data_designer_transcript_message(value: Any) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {"synthetic_message": value}
    text = str(row.get("synthetic_message") or "").strip()
    reasons: list[str] = []
    if not text or len(text.split()) < 3:
        reasons.append("message_too_short")
    lower = text.lower()
    blocked_fragments = (
        "speaker:",
        "speaker_",
        "company name",
        "[company",
        "{company",
        "conversation phase",
        "message goal",
        "speaking as",
        "as a customer",
        "as an agent",
    )
    if any(fragment in lower for fragment in blocked_fragments):
        reasons.append("contains_generation_scaffold")
    if re.search(r"\b(?:speaker|participant)\s+\w+\s+says\b", text, flags=re.IGNORECASE):
        reasons.append("contains_speaker_narration")
    if re.fullmatch(r"(?:speaker|participant)\s*[:#-]?\s*[\w -]+\.?", text, flags=re.IGNORECASE):
        reasons.append("label_only_message")
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
        reasons.append("contains_email")
    if re.search(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", text):
        reasons.append("contains_phone_or_identifier")
    speaker = str(row.get("speaker") or "")
    role = str(row.get("speaker_role") or "").lower()
    if _text_conflicts_with_transcript_role(text, speaker=speaker, role=role):
        reasons.append("speaker_role_conflict")
    if _is_source_replay_transcript_text(text, row):
        reasons.append("source_replay")
    return {
        "is_valid": not reasons,
        "error_messages": ";".join(reasons),
    }


def _data_designer_validation_passed(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("is_valid"))
    return False


def _data_designer_validation_error(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("error_messages") or "invalid_message")
    return "missing_validation_result"


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


def _transcript_json_schema(target_turns: int | None = None) -> dict[str, Any]:
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
    turns_schema: dict[str, Any] = {
        "type": "array",
        "items": turn,
    }
    if target_turns is not None:
        turns_schema["minItems"] = int(target_turns)
        turns_schema["maxItems"] = int(target_turns)
    return {
        "type": "object",
        "properties": {
            "turns": turns_schema,
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
    rows = generate_synthetic_transcript(
        contract,
        turn_count=int(turn_count or _contract_turn_count(contract) or 6),
        seed=seed,
    )
    return [
        {
            **row,
            "speaker": _friendly_transcript_speaker(
                str(row.get("speaker") or ""),
                "customer" if idx % 2 == 0 else "agent",
                idx,
            ),
        }
        for idx, row in enumerate(rows)
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
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = resolve_platform_slm_runtime()
    _require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    model_alias = "database-row-generator"
    data_designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
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
        model_config = build_data_designer_model_config(
            dd,
            model_alias,
            workflow="database-row-generation",
            runtime=runtime,
            temperature=float(os.getenv("SP_DATABASE_DATA_DESIGNER_TEMPERATURE", os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.35"))),
            top_p=float(os.getenv("SP_DATABASE_DATA_DESIGNER_TOP_P", os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.9"))),
            timeout_env="SP_DATABASE_DATA_DESIGNER_TIMEOUT",
            parallel_env="SP_DATABASE_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
            extra_body=_data_designer_chat_extra_body(runtime),
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
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = resolve_platform_slm_runtime()
    _require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "schema-row-generator"
    metadata: dict[str, Any] = {
        "backend": "nemo",
        "data_designer": {
            "backend_name": "data-designer",
            "model": runtime.model_id,
            "provider": runtime.provider,
            "provider_endpoint": runtime.endpoint,
            "model_alias": model_alias,
            "skip_health_check": skip_health_check,
            "row_count_repairs": [],
        },
    }
    data_designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
    schema_tables_by_name = {table.name: table for table in schema.tables}
    tables: dict[str, pd.DataFrame] = {}
    for table_name in _schema_generation_order(schema):
        table = schema_tables_by_name[table_name]
        count = int(table_row_counts.get(table_name) or row_count)
        seed_frame = pd.DataFrame(
            [
                {
                    "table_name": table_name,
                    "row_count": count,
                    "column_contract_json": _schema_column_contract_json(schema, table_name),
                    "relationships_json": _schema_relationships_json(schema, table_name),
                    "parent_rows_json": _schema_parent_key_context(schema, table_name, tables),
                    "user_prompt": _data_designer_user_prompt(),
                }
            ]
        )
        prompt = _schema_data_designer_prompt()
        model_config = build_data_designer_model_config(
            dd,
            model_alias,
            workflow="schema-row-generation",
            runtime=runtime,
            temperature=float(os.getenv("SP_SCHEMA_DATA_DESIGNER_TEMPERATURE", os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.35"))),
            top_p=float(os.getenv("SP_SCHEMA_DATA_DESIGNER_TOP_P", os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.9"))),
            timeout_env="SP_SCHEMA_DATA_DESIGNER_TIMEOUT",
            parallel_env="SP_SCHEMA_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
            extra_body=_data_designer_chat_extra_body(runtime),
            skip_health_check=skip_health_check,
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
            rows = _parse_schema_rows_json(
                raw_value,
                table_name=table_name,
                expected_count=count,
                column_names=[col.name for col in schema.columns.get(table_name, []) or []],
                metadata=metadata,
            )
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
        tables[table_name] = _coerce_schema_frame(
            schema,
            table_name,
            frame,
            count=count,
            parent_tables=tables,
            metadata=metadata,
        )
    return tables, metadata


def _schema_parent_key_context(schema: Any, table_name: str, parent_tables: dict[str, pd.DataFrame]) -> str:
    context: dict[str, Any] = {}
    for rel in getattr(schema, "relationships", []) or []:
        if rel.child_table != table_name:
            continue
        parent_frame = parent_tables.get(rel.parent_table)
        if parent_frame is None or rel.parent_key not in parent_frame.columns:
            continue
        context[f"{rel.parent_table}.{rel.parent_key}"] = (
            parent_frame[rel.parent_key].dropna().astype(str).head(10).tolist()
        )
    return json.dumps(context, default=str)


def _schema_generation_order(schema: Any) -> list[str]:
    """Return schema tables with FK parents before children."""
    from collections import defaultdict, deque

    table_names = [table.name for table in schema.tables]
    known = set(table_names)
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree = {name: 0 for name in table_names}
    for rel in getattr(schema, "relationships", []) or []:
        parent = getattr(rel, "parent_table", None)
        child = getattr(rel, "child_table", None)
        if parent not in known or child not in known or parent == child:
            continue
        graph[parent].append(child)
        in_degree[child] += 1

    ready = deque(name for name in table_names if in_degree[name] == 0)
    ordered: list[str] = []
    while ready:
        table_name = ready.popleft()
        ordered.append(table_name)
        for child in graph[table_name]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    if len(ordered) != len(table_names):
        return table_names
    return ordered


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
    schema_tables_by_name = {table.name: table for table in schema.tables}
    for table_name in _schema_generation_order(schema):
        table = schema_tables_by_name[table_name]
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
                elif str(getattr(col, "type", "text")) == "uuid":
                    row[col.name] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{text_prefix}.{table_name}.{col.name}.{seed}.{idx}"))
                elif getattr(col, "unique", False):
                    row[col.name] = idx + 1
                else:
                    row[col.name] = _schema_cell_value(col, table_name=table_name, idx=idx, seed=seed, prefix=text_prefix)
            rows.append(row)
        tables[table_name] = pd.DataFrame(rows)
    return tables


def _schema_cell_value(col: Any, *, table_name: str, idx: int, seed: int, prefix: str) -> Any:
    col_type = getattr(col, "type", "text")
    col_name = str(getattr(col, "name", "value") or "value")
    normalized_name = col_name.lower()
    params = dict(getattr(col, "distribution_params", {}) or {})
    if col_type in {"int", "foreign_key"}:
        return idx + 1
    if col_type in {"float", "decimal", "money", "currency"}:
        return round(float(params.get("min", 10.0)) + ((idx + seed) % 100) * 1.25, 2)
    if col_type in {"date", "datetime"}:
        return f"2024-01-{(idx % 28) + 1:02d}"
    if col_type == "email":
        return f"{_schema_entity_slug(table_name)}{idx + 1:03d}@example.com"
    if col_type == "phone":
        return f"+1-555-01{idx % 100:02d}"
    if col_type == "url":
        return f"https://example.com/{_schema_entity_slug(table_name)}/{idx + 1:03d}"
    if col_type == "address":
        streets = ["100 Main St", "250 Market Ave", "42 Center Rd", "808 North Pkwy"]
        return streets[(idx + seed) % len(streets)]
    if col_type == "boolean":
        return (idx + seed) % 2 == 0
    if col_type == "categorical":
        choices = list(params.get("choices") or ["new", "active", "closed"])
        probabilities = _schema_category_probabilities(params, len(choices))
        if probabilities is None:
            return choices[idx % len(choices)]
        return choices[_schema_weighted_choice_index(probabilities, idx=idx, seed=seed)]
    if normalized_name in {"name", "full_name"} or normalized_name.endswith("_name"):
        return f"{_schema_entity_label(table_name)} {idx + 1:03d}"
    if "description" in normalized_name or "notes" in normalized_name:
        return f"{_schema_entity_label(table_name)} synthetic note {idx + 1:03d}"
    if normalized_name in {"city", "town"}:
        return ["Phoenix", "Austin", "Denver", "Raleigh"][(idx + seed) % 4]
    if normalized_name in {"state", "region"}:
        return ["AZ", "TX", "CO", "NC"][(idx + seed) % 4]
    if normalized_name in {"specialty", "department"}:
        return ["Primary Care", "Scheduling", "Billing", "Operations"][(idx + seed) % 4]
    if normalized_name in {"code", "reference", "reference_number"} or normalized_name.endswith("_code"):
        return f"{_schema_entity_slug(table_name).upper()}-{idx + 1:04d}"
    return f"{_schema_entity_label(table_name)} {col_name.replace('_', ' ')} {idx + 1:03d}"


def _schema_entity_label(table_name: str) -> str:
    value = str(table_name or "record").replace("_", " ").strip()
    if value.endswith("ies"):
        value = value[:-3] + "y"
    elif value.endswith("s"):
        value = value[:-1]
    return value.title() or "Record"


def _schema_entity_slug(table_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _schema_entity_label(table_name).lower()) or "record"


def _schema_category_probabilities(params: dict[str, Any], choice_count: int) -> list[float] | None:
    weights = params.get("weights") or params.get("probabilities")
    if not isinstance(weights, list) or len(weights) != choice_count:
        return None
    total = sum(float(weight) for weight in weights)
    if total <= 0:
        return None
    return [float(weight) / total for weight in weights]


def _schema_weighted_choice_index(probabilities: list[float], *, idx: int, seed: int) -> int:
    slot = ((idx * 37 + seed * 17) % 1000) / 1000.0
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if slot <= cumulative:
            return index
    return len(probabilities) - 1


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
        "Return strict JSON only: a JSON array of exactly {{ row_count }} row objects, not arrays. "
        "Before returning, count the array items. If row_count is 30, return exactly 30 objects, not 29 and not 31. "
        "Every object must contain every requested column exactly once. "
        "Use realistic values for the column name and type. Credit scores must be between 300 and 850. "
        "Money and transaction amounts must be non-negative. Percent/rate fields must be positive realistic decimals. "
        "Foreign-key values must come from the provided parent table samples when available. "
        "Do not include Markdown, prose, comments, code fences, or wrapper objects."
    )


def _parse_schema_rows_json(
    value: str,
    *,
    table_name: str,
    expected_count: int,
    column_names: list[str] | None = None,
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
    if len(rows) != len(payload) and column_names and all(isinstance(row, list) for row in payload):
        rows = [
            {
                name: row[index] if index < len(row) else None
                for index, name in enumerate(column_names)
            }
            for row in payload
        ]
        if metadata is not None:
            repairs = metadata.setdefault("data_designer", {}).setdefault("row_count_repairs", [])
            repairs.append(
                {
                    "table": table_name,
                    "requested_rows": int(expected_count),
                    "returned_rows": len(rows),
                    "action": "mapped_array_rows_to_objects",
                }
            )
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
    parent_tables: dict[str, pd.DataFrame] | None = None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    columns = list(schema.columns.get(table_name, []) or [])
    if len(frame) != count:
        raise NemoSchemaGenerationError(
            f"Schema Data Designer returned {len(frame)} row(s) for table {table_name!r}; expected {count}.",
            metadata=metadata,
        )
    frame = frame.head(count).copy()
    parent_tables = parent_tables or {}
    quality_repairs = metadata.setdefault("data_designer", {}).setdefault("quality_repairs", []) if metadata is not None else []
    for col in columns:
        if col.name not in frame.columns:
            raise NemoSchemaGenerationError(
                f"Schema Data Designer output for table {table_name!r} is missing required column {col.name!r}.",
                metadata=metadata,
            )
        if not getattr(col, "nullable", False) and frame[col.name].isna().any():
            frame[col.name] = [
                value if pd.notna(value) and str(value).strip() else _schema_cell_value(col, idx=idx, seed=0, prefix=table_name)
                for idx, value in enumerate(frame[col.name].tolist())
            ]
            quality_repairs.append({"table": table_name, "column": col.name, "action": "filled_non_null_values"})
        if getattr(col, "unique", False):
            frame[col.name] = _repair_unique_column(frame[col.name], table_name=table_name, column_name=col.name)
            quality_repairs.append({"table": table_name, "column": col.name, "action": "enforced_unique_values"})
        rel = next(
            (
                relationship
                for relationship in (getattr(schema, "relationships", []) or [])
                if relationship.child_table == table_name and relationship.child_key == col.name
            ),
            None,
        )
        if rel is not None and rel.parent_table in parent_tables and rel.parent_key in parent_tables[rel.parent_table]:
            parent_values = parent_tables[rel.parent_table][rel.parent_key].dropna().tolist()
            if parent_values:
                frame[col.name] = [parent_values[idx % len(parent_values)] for idx in range(len(frame))]
                quality_repairs.append(
                    {
                        "table": table_name,
                        "column": col.name,
                        "action": "aligned_foreign_keys_to_parent_rows",
                        "parent": f"{rel.parent_table}.{rel.parent_key}",
                    }
                )
        frame[col.name] = _repair_schema_quality_values(frame[col.name], col, table_name=table_name)
    return frame[[col.name for col in columns]]


def _repair_unique_column(series: pd.Series, *, table_name: str, column_name: str) -> list[Any]:
    values: list[Any] = []
    seen: set[str] = set()
    for idx, value in enumerate(series.tolist()):
        text = str(value).strip() if pd.notna(value) else ""
        if not text or text in seen:
            text = f"{table_name}_{column_name}_{idx + 1}"
        while text in seen:
            text = f"{table_name}_{column_name}_{idx + 1}_{len(seen)}"
        seen.add(text)
        values.append(text)
    return values


def _repair_schema_quality_values(series: pd.Series, col: Any, *, table_name: str) -> pd.Series:
    name = str(getattr(col, "name", "")).lower()
    col_type = str(getattr(col, "type", "text"))
    if col_type == "uuid":
        repaired = []
        seen: set[str] = set()
        for idx, value in enumerate(series.tolist()):
            text = str(value).strip() if pd.notna(value) else ""
            try:
                repaired_uuid = str(uuid.UUID(text))
            except (TypeError, ValueError, AttributeError):
                repaired_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{table_name}.{name}.{idx}.{text}"))
            while repaired_uuid in seen:
                repaired_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{table_name}.{name}.{idx}.{len(seen)}"))
            seen.add(repaired_uuid)
            repaired.append(repaired_uuid)
        return pd.Series(repaired, index=series.index)
    if col_type in {"int", "float", "decimal", "money", "currency"}:
        numeric = pd.to_numeric(series, errors="coerce")
        if "credit_score" in name or name == "credit":
            return numeric.fillna(650).clip(lower=300, upper=850).round().astype(int)
        if any(token in name for token in ("amount", "balance", "total", "price", "usd")):
            return numeric.fillna(0).clip(lower=0)
        if any(token in name for token in ("interest", "rate", "percent")):
            return numeric.fillna(5.0).clip(lower=0.1, upper=36.0)
        return numeric.fillna(0 if col_type == "int" else 0.0)
    if "email" in name:
        repaired = []
        for idx, value in enumerate(series.tolist()):
            text = str(value).strip().lower() if pd.notna(value) else ""
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", text):
                text = f"{table_name}_{idx + 1}@example.com"
            repaired.append(text)
        return pd.Series(repaired, index=series.index)
    return series


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
            if str(getattr(col, "type", "text")) == "uuid":
                invalid_uuid_count = sum(
                    1
                    for value in frame[col.name].dropna().tolist()
                    if not _is_valid_uuid_value(value)
                )
                if invalid_uuid_count:
                    issues.append(f"{table.name}.{col.name}: {invalid_uuid_count} invalid UUID value(s)")
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


def _is_valid_uuid_value(value: Any) -> bool:
    try:
        uuid.UUID(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return False
    return True


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
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        raise RuntimeError("Data Designer standalone SDK is not importable.") from exc

    runtime = _digital_twin_data_designer_runtime()
    _require_data_designer_api_key(runtime)
    preflight_slm_endpoint(runtime)
    skip_health_check = _data_designer_skip_health_check()
    model_alias = "pdf-value-generator"
    prompt = _pdf_data_designer_prompt()
    model_config = build_data_designer_model_config(
        dd,
        model_alias,
        workflow="pdf-value-batch",
        runtime=runtime,
        temperature=float(os.getenv("SP_PDF_DATA_DESIGNER_TEMPERATURE", os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", "0.2"))),
        top_p=float(os.getenv("SP_PDF_DATA_DESIGNER_TOP_P", os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", "0.8"))),
        timeout_env="SP_PDF_DATA_DESIGNER_TIMEOUT",
        parallel_env="SP_PDF_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
        extra_body=_data_designer_chat_extra_body(runtime),
        skip_health_check=skip_health_check,
    )
    data_designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
    merged_sdk_values: dict[str, Any] = {"values": {}}
    for chunk_index, plan_chunk in enumerate(_pdf_generation_plan_chunks(plan), start=1):
        seed_frame = _pdf_binding_seed_frame(plan_chunk)
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="synthetic_value",
                prompt=prompt,
                system_prompt=(
                    "Return exactly one short replacement value. No explanation, JSON, Markdown, labels, "
                    "or repeated binding metadata."
                ),
                model_alias=model_alias,
            )
        )
        builder.add_column(
            dd.ValidationColumnConfig(
                name="synthetic_value_validation",
                target_columns=["synthetic_value"],
                validator_type=dd.ValidatorType.LOCAL_CALLABLE,
                validator_params=dd.LocalCallableValidatorParams(
                    validation_function=_validate_data_designer_pdf_values
                ),
            )
        )
        try:
            preview = data_designer.preview(builder, num_records=len(seed_frame))
        except Exception as exc:
            chunk_size = len(plan_chunk.get("bindings") or [])
            raise RuntimeError(
                "NeMo Data Designer PDF value generation failed "
                f"for binding batch {chunk_index} containing {chunk_size} bindings."
            ) from exc
        dataset = getattr(preview, "dataset", preview)
        merged_sdk_values["values"].update(_pdf_values_from_data_designer_rows(dataset, chunk_index))
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
    chunk_size = max(1, int(os.getenv("SP_PDF_DATA_DESIGNER_BINDINGS_PER_BATCH", "10")))
    return [{"bindings": bindings[index : index + chunk_size]} for index in range(0, len(bindings), chunk_size)]


def _pdf_table_cell_binding_id(region_id: str, row_index: int, column_index: int) -> str:
    return f"{region_id}__r{row_index}__c{column_index}"


def _pdf_inline_span_binding_id(region_id: str, span_index: int) -> str:
    return f"{region_id}__s{span_index}"


def _pdf_data_designer_prompt() -> str:
    return (
        "Generate one privacy-safe synthetic replacement value for a PDF binding.\n"
        "General instructions: {{ user_prompt }}\n"
        "Binding id: {{ binding_id }}\n"
        "Binding type: {{ binding_type }}\n"
        "Label: {{ label }}\n"
        "Semantic role: {{ semantic_role }}\n"
        "Generator strategy: {{ generator_strategy }}\n"
        "Shape pattern: {{ shape_pattern }}\n"
        "Value type: {{ value_type }}\n"
        "Return only the replacement value as plain text, preferably under 80 characters. "
        "Do not include JSON, Markdown, field labels, binding ids, explanations, or real personally "
        "identifiable information."
    )


def _pdf_binding_seed_frame(plan: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for binding in plan.get("bindings") or []:
        row = dict(binding)
        row["user_prompt"] = _data_designer_user_prompt()
        for key in (
            "binding_id",
            "binding_type",
            "label",
            "semantic_role",
            "generator_strategy",
            "shape_pattern",
            "value_type",
        ):
            value = row.get(key)
            row[key] = "" if value is None else str(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _pdf_values_from_data_designer_rows(dataset: Any, chunk_index: int) -> dict[str, str]:
    frame = _data_designer_dataset_to_frame(dataset, "PDF")
    required = {"binding_id", "synthetic_value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "NeMo Data Designer PDF output missing required column(s) "
            f"{missing} for binding batch {chunk_index}."
        )
    output: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        validation = record.get("synthetic_value_validation")
        if not _data_designer_validation_passed(validation):
            detail = _data_designer_validation_error(validation) or record.get("synthetic_value")
            raise RuntimeError(f"NeMo Data Designer PDF value failed SDK validation: {detail}")
        binding_id = str(record.get("binding_id") or "").strip()
        value = _repair_pdf_synthetic_value(record.get("synthetic_value"))
        if not binding_id:
            raise RuntimeError(f"NeMo Data Designer PDF output has an empty binding_id in batch {chunk_index}.")
        if not value:
            raise RuntimeError(f"NeMo Data Designer PDF output has an empty value for `{binding_id}`.")
        output[binding_id] = value
    return output


def _validate_data_designer_pdf_values(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [_validate_data_designer_pdf_value(row.get("synthetic_value")) for row in frame.to_dict(orient="records")]
    )


def _validate_data_designer_pdf_value(value: Any) -> dict[str, Any]:
    text = _repair_pdf_synthetic_value(value)
    if not text:
        return {"is_valid": False, "error_messages": "empty_pdf_replacement_value"}
    lowered = text.lower()
    if any(marker in lowered for marker in ("binding id:", "semantic role:", "generator strategy:", "```")):
        return {"is_valid": False, "error_messages": "contains_generation_scaffold"}
    if len(text) > int(os.getenv("SP_PDF_DATA_DESIGNER_MAX_VALUE_CHARS", "240")):
        return {"is_valid": False, "error_messages": "pdf_replacement_value_too_long"}
    return {"is_valid": True, "error_messages": None}


def _clean_pdf_synthetic_value(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return " ".join(text.strip().strip('"').strip("'").split())


def _repair_pdf_synthetic_value(value: Any) -> str:
    """Extract one renderable PDF replacement value from a chatty local SLM reply."""
    if isinstance(value, dict):
        for key in ("value", "replacement_value", "synthetic_value", "text"):
            if value.get(key) is not None:
                return _repair_pdf_synthetic_value(value.get(key))
    raw_text = "" if value is None else str(value).strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:text|json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    if not raw_text:
        return ""

    text = _clean_pdf_synthetic_value(raw_text)
    parsed = _parse_json_object_if_present(text)
    if isinstance(parsed, dict):
        for key in ("value", "replacement_value", "synthetic_value", "text"):
            if parsed.get(key) is not None:
                return _repair_pdf_synthetic_value(parsed.get(key))
        values = parsed.get("values")
        if isinstance(values, dict) and values:
            return _repair_pdf_synthetic_value(next(iter(values.values())))

    lines = [line.strip(" -:\t") for line in re.split(r"[\r\n]+", raw_text) if line.strip()]
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in ("binding id:", "semantic role:", "generator strategy:")):
            continue
        labelled = re.match(
            r"^(?:replacement value|synthetic value|value|answer)\s*[:=]\s*(?P<value>.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if labelled:
            return _clip_pdf_value(_clean_pdf_synthetic_value(labelled.group("value")))
        if len(line) <= int(os.getenv("SP_PDF_DATA_DESIGNER_MAX_VALUE_CHARS", "240")):
            return _clean_pdf_synthetic_value(line)
    return _clip_pdf_value(text)


def _parse_json_object_if_present(text: str) -> dict[str, Any] | None:
    candidate = text
    if not candidate.startswith("{"):
        extracted = _extract_first_json_object(candidate)
        if extracted is None:
            return None
        candidate = extracted
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clip_pdf_value(text: str) -> str:
    limit = int(os.getenv("SP_PDF_DATA_DESIGNER_MAX_VALUE_CHARS", "240"))
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return clipped or text[:limit]


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

