"""Stable SDK for the two isolated source tracks and shared runtime."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from synth_platform.infrastructure.artifacts.store import SynthpkgStore
from synth_platform.infrastructure.documents.pdf_router import PdfRouter
from synth_platform.infrastructure.documents.reportlab_renderer import ReportlabRenderer
from synth_platform.infrastructure.documents.tesseract_ocr import TesseractOcrAdapter
from synth_platform.infrastructure.extraction.document_pass import DocumentExtractionPass
from synth_platform.infrastructure.sinks.postgres_sink import PostgresSink
from synth_platform.infrastructure.sources.postgres import PostgresSource
from synth_platform.infrastructure.sources.sqlite import SqliteSource
from synth_platform.application.dto.commands import GenerationRequest, TrainingRequest
from synth_platform.application.use_cases.export_artifact import export_artifact
from synth_platform.application.use_cases.generate_dataset import generate_dataset
from synth_platform.application.use_cases.load_artifact import load_artifact
from synth_platform.application.use_cases.publish_dataset import publish_dataset
from synth_platform.application.use_cases.propose_pdf_schema import propose_pdf_schema
from synth_platform.application.use_cases.render_synthetic_pdfs import (
    render_synthetic_pdf_package,
)
from synth_platform.application.use_cases.train_from_pdf import (
    TrainPdfCommand, train_from_pdf,
)
from synth_platform.application.use_cases.train_model import train_model
from synth_platform.application.use_cases.validate_dataset import validate_dataset
from synth_platform.application.use_cases.validate_extraction import validate_extraction
from synth_platform.domain.artifacts.guards import require_source_kind
from synth_platform.domain.extraction.target_schema import TargetSchema
from synth_platform.domain.runs.models import SourceKind
from synth_platform.errors import UnsupportedSourceError
from synth_platform.settings import Settings


def build_sqlite_source(source: str, **_):
    path = source[len("sqlite:///"):] if source.startswith("sqlite:///") else source
    return SqliteSource(path)


def build_postgres_source(source: str, **kwargs):
    return PostgresSource(
        source,
        allowed_schemas=tuple(kwargs.get("allowed_schemas") or ("public",)),
        allowed_tables=(tuple(kwargs["allowed_tables"])
                        if kwargs.get("allowed_tables") is not None else None),
    )


CONNECTOR_FACTORIES = {
    "sqlite": build_sqlite_source,
    "postgresql": build_postgres_source,
    "postgresql+psycopg": build_postgres_source,
}


def _connector_for(source: str, **kwargs):
    parsed = urlparse(source)
    if not parsed.scheme:
        return build_sqlite_source(source, **kwargs)
    factory = CONNECTOR_FACTORIES.get(parsed.scheme.lower())
    if factory is None:
        raise UnsupportedSourceError(f"unsupported source scheme: {parsed.scheme!r}")
    return factory(source, **kwargs)


class _PdfDeps:
    def __init__(self, router, store):
        self.pdf_router = router
        self.document_extractor_factory = DocumentExtractionPass
        self.artifact_store = store


class SyntheticDataPlatform:
    def __init__(self, settings: Settings | None = None, store=None,
                 pdf_router: PdfRouter | None = None):
        self.settings = settings or Settings()
        self.store = store or SynthpkgStore()
        self.pdf_router = pdf_router or PdfRouter(ocr_adapter=TesseractOcrAdapter())
        self.renderer = ReportlabRenderer()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "SyntheticDataPlatform":
        return cls(settings=settings)

    # Compatibility database training method.
    def train(self, request: TrainingRequest):
        conn = _connector_for(request.source)
        try:
            conn.health_check()
            if hasattr(conn, "assert_read_only"):
                conn.assert_read_only()
            return train_model(conn, sample_size=request.sample_size, seed=request.seed,
                               rare_threshold=request.rare_category_threshold)
        finally:
            conn.close()

    def test_database_connection(self, source: str, *, allowed_schemas=("public",),
                                 allowed_tables=None):
        conn = _connector_for(source, allowed_schemas=allowed_schemas,
                              allowed_tables=allowed_tables)
        try:
            health = conn.health_check()
            conn.assert_read_only()
            return health
        finally:
            conn.close()

    def discover_database(self, source: str, *, allowed_schemas=("public",),
                          allowed_tables=None):
        conn = _connector_for(source, allowed_schemas=allowed_schemas,
                              allowed_tables=allowed_tables)
        try:
            conn.assert_read_only()
            return conn.discover_schema()
        finally:
            conn.close()

    def train_database(self, source: str, *, sample_size=50_000, seed=42,
                       rare_threshold=10, allowed_schemas=("public",),
                       allowed_tables=None):
        conn = _connector_for(source, allowed_schemas=allowed_schemas,
                              allowed_tables=allowed_tables)
        try:
            conn.assert_read_only()
            return train_model(conn, sample_size=sample_size, seed=seed,
                               rare_threshold=rare_threshold)
        finally:
            conn.close()

    def ingest_pdf(self, source: bytes | str, extraction_mode: str = "auto"):
        return self.pdf_router.load(source, mode=extraction_mode)

    def propose_pdf_schema(self, document) -> TargetSchema:
        return propose_pdf_schema(document)

    def confirm_pdf_schema(self, schema: TargetSchema | dict) -> TargetSchema:
        confirmed = schema if isinstance(schema, TargetSchema) else TargetSchema.model_validate(schema)
        if not confirmed.entities:
            raise ValueError("confirmed PDF target schema must contain at least one entity")
        return confirmed

    def train_pdf(self, source: bytes | str, target_schema: TargetSchema | dict,
                  artifact_path: str | Path, *, extraction_mode="auto", seed=42,
                  rare_threshold=10, document_template=None):
        command = TrainPdfCommand(
            pdf_source=source, extraction_mode=extraction_mode,
            confirmed_target_schema=self.confirm_pdf_schema(target_schema),
            artifact_path=str(artifact_path), seed=seed,
            rare_threshold=rare_threshold, document_template=document_template)
        return train_from_pdf(command, _PdfDeps(self.pdf_router, self.store))

    def export(self, artifact, path: str | Path) -> Path:
        return export_artifact(self.store, artifact, path)

    def load(self, path: str | Path):
        return load_artifact(self.store, path)

    load_artifact = load

    def generate(self, artifact, request: GenerationRequest) -> dict:
        return generate_dataset(artifact, request)

    def validate(self, artifact, tables: dict, evaluator=None):
        extra = evaluator.privacy_utility_checks(tables) if evaluator is not None else None
        return validate_dataset(artifact, tables, extra)

    def publish_to_database(self, artifact, tables, report, target_url: str,
                            target_schema: str, dataset_id: str, if_exists="fail"):
        expected_prefix = ("db_track_" if artifact.manifest.source_kind == SourceKind.DATABASE
                           else "pdf_track_")
        if not target_schema.startswith(expected_prefix):
            raise ValueError(
                f"{artifact.manifest.source_kind.value} artifact requires {expected_prefix}* schema")
        sink = PostgresSink(target_url, target_schema, if_exists=if_exists)
        return publish_dataset(artifact, tables, report, sink, dataset_id)

    def render_synthetic_pdfs(self, artifact, tables, output_path: str | Path,
                              generation_run_id: str, document_id: str | None = None):
        require_source_kind(artifact, SourceKind.PDF)
        return self.renderer.render(artifact, tables, output_path,
                                    generation_run_id, document_id)

    def render_synthetic_pdf_package(self, artifact, tables, report, output_dir: str | Path,
                                     generation_run_id: str, document_count: int = 1):
        return render_synthetic_pdf_package(
            artifact, tables, report, self.renderer, output_dir,
            generation_run_id, document_count)
