"""TargetSchema: the caller-supplied contract that guides extraction.

This is how an arbitrary source (structured or unstructured) is lifted into the
canonical relational IR. The caller declares the entities, fields, keys, and
parent links they want; a schema-guided ExtractionPass fills the rows. The
platform hardcodes NO domain entities — the TargetSchema is always supplied as
input, exactly like a database is input. A self-describing source (e.g. SQLite)
may derive its own schema and ignore the target. Topological
ordering is NOT defined here — it is delegated to the one canonical
graph module (domain/relational/dag.py::build_graph_plan).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_NAME = "pii_name"
    PII_SSN = "pii_ssn"
    FREE_TEXT = "free_text"


class TargetField(_Base):
    name: str
    type: FieldType = FieldType.STRING
    required: bool = False
    aliases: list[str] = Field(default_factory=list)  # labels to match in source


class TargetParent(_Base):
    """Declares an explicit FK from this entity to a parent entity's PK.

    The link column is named explicitly (fk_field) — the platform never guesses
    `{table}_id`, which was a root-cause defect in the legacy code.
    """
    entity: str
    fk_field: str


class ExtractionStrategy(str, Enum):
    ROWS_PER_TABLE_ROW = "rows_per_table_row"      # each source table row -> entity row
    ROWS_PER_LIST_ITEM = "rows_per_list_item"      # each list/bullet item -> entity row
    ROWS_PER_SECTION = "rows_per_section"          # each matching section -> entity row
    SINGLETON = "singleton"                        # one row for the whole document


class TargetEntity(_Base):
    name: str
    primary_key: str | None = None
    fields: list[TargetField] = Field(default_factory=list)
    parent: TargetParent | None = None
    aliases: list[str] = Field(default_factory=list)  # section headings to locate
    strategy: ExtractionStrategy = ExtractionStrategy.ROWS_PER_LIST_ITEM

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


class TargetSchema(_Base):
    entities: list[TargetEntity] = Field(default_factory=list)

    def entity(self, name: str) -> TargetEntity | None:
        return next((e for e in self.entities if e.name == name), None)

