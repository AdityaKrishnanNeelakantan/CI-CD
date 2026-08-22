"""PDF-track extraction, quality gating, and shared-dataset training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.application.use_cases.train_model import train_relational_dataset
from synth_platform.application.use_cases.validate_extraction import validate_extraction
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.documents.models import (
    CalculationRule, DocumentTemplateIR, FieldBinding, RepeatingTableBinding,
    SectionTemplate,
)
from synth_platform.domain.extraction.target_schema import TargetSchema
from synth_platform.domain.profiling.models import SamplingRecord
from synth_platform.domain.runs.models import SourceKind


class TrainPdfCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    pdf_source: bytes | str
    extraction_mode: str = "auto"
    confirmed_target_schema: TargetSchema
    seed: int = 42
    rare_threshold: int = 10
    artifact_path: str
    source_run_id: str | None = None
    document_template: DocumentTemplateIR | None = None


class PdfDeps(Protocol):
    pdf_router: object
    document_extractor_factory: object
    artifact_store: object


class PdfTrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    artifact: SynthArtifact
    artifact_path: str
    dataset: RelationalDataset
    extraction_report: object
    source_disconnected: bool = True



def _default_template(target: TargetSchema) -> DocumentTemplateIR:
    sections = []
    fields = []
    repeating = []
    for entity in target.entities:
        section_name = entity.name
        sections.append(SectionTemplate(name=section_name, title=entity.name.replace("_", " ").title()))
        if entity.strategy.value == "singleton":
            for field in entity.fields:
                fields.append(FieldBinding(section=section_name, label=field.name.replace("_", " ").title(),
                                           table=entity.name, column=field.name))
        else:
            columns = [field.name for field in entity.fields]
            repeating.append(RepeatingTableBinding(section=section_name, table=entity.name,
                                                    columns=columns, labels=[c.replace("_", " ").title() for c in columns]))
    all_fields = {field.name for entity in target.entities for field in entity.fields}
    calculations = []
    required = {"ending_balance", "beginning_balance", "credits", "debits", "fees", "interest"}
    if required.issubset(all_fields):
        calculations.append(CalculationRule(
            name="statement_balance_reconciliation",
            expression="ending_balance = beginning_balance + credits - debits - fees + interest"))
    return DocumentTemplateIR(document_type="synthetic_document", sections=sections,
                              field_bindings=fields, repeating_tables=repeating,
                              calculations=calculations)

def train_from_pdf(cmd: TrainPdfCommand, deps) -> PdfTrainingResult:
    loaded = deps.pdf_router.load(cmd.pdf_source, mode=cmd.extraction_mode)
    extractor = deps.document_extractor_factory(
        loaded.document, cmd.confirmed_target_schema)
    dataset = extractor.extract()
    report = validate_extraction(
        loaded.document, dataset, cmd.confirmed_target_schema,
        raise_on_failure=True)
    sampling = {
        name: SamplingRecord(
            strategy=f"pdf_{loaded.route}", seed=cmd.seed,
            population_count=len(frame), sample_count=len(frame),
            bias_warning="document extraction is evidence-bounded")
        for name, frame in dataset.tables.items()
    }
    artifact = train_relational_dataset(
        dataset, source_kind=SourceKind.PDF,
        source_fingerprint=loaded.source_fingerprint,
        sample_records=sampling, seed=cmd.seed,
        rare_threshold=cmd.rare_threshold,
        source_run_id=cmd.source_run_id,
        document_template=cmd.document_template or _default_template(cmd.confirmed_target_schema))
    path = deps.artifact_store.write_signed(artifact, cmd.artifact_path)
    # No Document or original bytes are retained in the returned artifact/runtime.
    return PdfTrainingResult(artifact=artifact, artifact_path=str(path),
                             dataset=dataset, extraction_report=report,
                             source_disconnected=True)
