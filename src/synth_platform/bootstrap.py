"""Composition root for adapters and shared use cases."""

from __future__ import annotations

from pathlib import Path

from synth_platform.engine.generation.service import GenerationService
from synth_platform.infrastructure.artifacts.store import SynthpkgStore
from synth_platform.infrastructure.documents.reportlab_renderer import ReportlabRenderer
from synth_platform.infrastructure.extraction.document_pass import (
    DocumentExtractionPass,
)
from synth_platform.infrastructure.sources.postgres import PostgresSource
from synth_platform.infrastructure.sources.sqlite import SqliteSource
from synth_platform.settings import Settings


def build_artifact_store() -> SynthpkgStore:
    return SynthpkgStore()


def build_sqlite_source(db_path: str) -> SqliteSource:
    return SqliteSource(db_path)


def build_source_connector(config):
    kind = config.get("kind")
    if kind == "sqlite":
        return SqliteSource(config["path"])
    if kind in {"postgresql", "postgresql+psycopg"}:
        return PostgresSource(
            config["url"],
            tuple(config.get("allowed_schemas", ("public",))),
            tuple(config["allowed_tables"]) if config.get("allowed_tables") else None,
        )
    raise ValueError(f"unsupported source connector: {kind!r}")


def build_extraction_pass(source_kind, config):
    if str(source_kind).lower().endswith("pdf"):
        return DocumentExtractionPass(config["document"], config["target_schema"])
    raise ValueError(f"unsupported extraction source kind: {source_kind!r}")


def build_generator():
    return GenerationService()


def build_validator():
    from synth_platform.application.use_cases.validate_dataset import validate_dataset

    return validate_dataset


def build_document_renderer():
    return ReportlabRenderer()


def build_run_repository():
    from synth_platform.infrastructure.persistence.metadata_repository import (
        InMemoryMetadataRepository,
    )

    return InMemoryMetadataRepository()


def default_settings() -> Settings:
    return Settings.from_env()


def workspace_root_path(settings: Settings | None = None) -> Path:
    """Return the local-only metadata/run root selected by deployment settings."""

    runtime = settings or default_settings()
    return Path(runtime.approved_output_root).expanduser().resolve() / "workspace"


def build_workspace_service(
    root: str | Path | None = None,
    *,
    settings: Settings | None = None,
):
    """Compose atomic local repositories for the shared product workspace."""

    from synth_platform.application.services.workspace import (
        WorkspaceRepositories,
        WorkspaceService,
    )
    from synth_platform.infrastructure.persistence.local_workspace import (
        BuiltInSchemaTemplateRepository,
        LocalArtifactCatalog,
        LocalProjectRepository,
        LocalResultRepository,
        LocalSessionRepository,
        LocalSettingsRepository,
        LocalWorkspaceCommitter,
    )

    runtime = settings or default_settings()
    storage_root = (
        Path(root).expanduser().resolve() if root else workspace_root_path(runtime)
    )
    storage_root.mkdir(parents=True, exist_ok=True)
    repositories = WorkspaceRepositories(
        sessions=LocalSessionRepository(storage_root),
        results=LocalResultRepository(storage_root),
        projects=LocalProjectRepository(storage_root),
        templates=BuiltInSchemaTemplateRepository(),
        settings=LocalSettingsRepository(storage_root),
        artifacts=LocalArtifactCatalog(
            storage_root,
            allowed_roots=[storage_root, Path(runtime.approved_output_root)],
        ),
        committer=LocalWorkspaceCommitter(storage_root),
    )
    return WorkspaceService(repositories, runtime)



def build_guarded_chat_model(
    profile: str,
    *,
    model: str = "qwen3:8b",
    settings: Settings | None = None,
):
    """Compose a loopback-only ChatModel behind deterministic guardrails."""

    from synth_platform.domain.guardrails.models import GuardrailPolicy
    from synth_platform.engine.security.text_guardrails import (
        DeterministicModelGuardrails,
    )
    from synth_platform.infrastructure.llm.guarded_chat import GuardedChatModel
    from synth_platform.infrastructure.llm.ollama_chat import OllamaChatModel

    runtime = settings or default_settings()
    profiles = {
        "interaction_semantics": GuardrailPolicy(
            name="interaction_semantics",
            version=runtime.guardrail_policy_version,
            max_input_characters=runtime.guardrail_max_input_characters,
            max_output_characters=runtime.guardrail_max_output_characters,
            required_input_markers=(
                "<sanitized_transcript>",
                "</sanitized_transcript>",
            ),
            allowed_json_keys=(
                "topics",
                "issue_codes",
                "action_codes",
                "resolution_status",
                "initial_customer_sentiment",
                "final_customer_sentiment",
            ),
        ),
        "synthetic_text": GuardrailPolicy(
            name="synthetic_text",
            version=runtime.guardrail_policy_version,
            max_input_characters=runtime.guardrail_max_input_characters,
            max_output_characters=runtime.guardrail_max_output_characters,
            require_json_object=False,
            required_input_markers=("JSON array",),
        ),
        "schema_guided_extraction": GuardrailPolicy(
            name="schema_guided_extraction",
            version=runtime.guardrail_policy_version,
            max_input_characters=runtime.guardrail_max_input_characters,
            max_output_characters=runtime.guardrail_max_output_characters,
            required_input_markers=("Entities and fields to extract:", "Document:"),
        ),
    }
    try:
        policy = profiles[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported model guardrail profile: {profile!r}") from exc
    delegate = OllamaChatModel(model=model, host=runtime.local_model_host)
    return GuardedChatModel(delegate, DeterministicModelGuardrails(), policy)
