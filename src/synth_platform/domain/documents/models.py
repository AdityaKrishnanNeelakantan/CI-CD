"""Source-free document template and rendered document contracts."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SectionTemplate(_Base):
    name: str
    title: str
    static_text: str = ""


class FieldBinding(_Base):
    section: str
    label: str
    table: str
    column: str


class RepeatingTableBinding(_Base):
    section: str
    table: str
    columns: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class CalculationRule(_Base):
    name: str
    expression: str
    tolerance: float = 1e-6


class DocumentTemplateIR(_Base):
    document_type: str = "synthetic_report"
    sections: list[SectionTemplate] = Field(default_factory=list)
    field_bindings: list[FieldBinding] = Field(default_factory=list)
    repeating_tables: list[RepeatingTableBinding] = Field(default_factory=list)
    calculations: list[CalculationRule] = Field(default_factory=list)
    page_size: str = "LETTER"


class RenderedSection(_Base):
    name: str
    content: list[str] = Field(default_factory=list)


class SyntheticDocument(_Base):
    document_id: str
    artifact_id: str
    generation_run_id: str
    bound_entity_ids: dict[str, str] = Field(default_factory=dict)
    sections: list[RenderedSection] = Field(default_factory=list)
