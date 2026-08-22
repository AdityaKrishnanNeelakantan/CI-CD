"""Composition root for adapters and shared use cases."""
from __future__ import annotations

from synth_platform.infrastructure.artifacts.store import SynthpkgStore
from synth_platform.infrastructure.documents.pdf_router import PdfRouter
from synth_platform.infrastructure.documents.reportlab_renderer import ReportlabRenderer
from synth_platform.infrastructure.documents.tesseract_ocr import TesseractOcrAdapter
from synth_platform.infrastructure.extraction.document_pass import DocumentExtractionPass
from synth_platform.infrastructure.sinks.registry import build_output_sink
from synth_platform.infrastructure.sources.postgres import PostgresSource
from synth_platform.infrastructure.sources.sqlite import SqliteSource
from synth_platform.engine.generation.service import GenerationService
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
        return PostgresSource(config["url"], tuple(config.get("allowed_schemas", ("public",))),
                              tuple(config["allowed_tables"]) if config.get("allowed_tables") else None)
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
    from synth_platform.infrastructure.persistence.metadata_repository import InMemoryMetadataRepository
    return InMemoryMetadataRepository()


def default_settings() -> Settings:
    return Settings.from_env()
