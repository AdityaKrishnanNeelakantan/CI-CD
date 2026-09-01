"""Schema domain models: tables, columns, keys, constraints, and metadata."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_FORMAT_VERSION = "1.1.0"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class TableRef(_Base):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str
    schema_name: str | None = Field(default=None, alias="schema", serialization_alias="schema")

    @property
    def schema(self) -> str | None:
        return self.schema_name

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.name}" if self.schema_name else self.name


class ConnectionHealth(_Base):
    server_version: str = ""
    current_database: str = ""
    current_user: str = ""
    latency_ms: float = 0.0
    transaction_read_only: bool = False


class ColumnSchema(_Base):
    name: str
    physical_type: str = "unknown"
    logical_type: str | None = None
    nullable: bool = True
    default: str | None = None


class ForeignKey(_Base):
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    confirmed: bool = True
    evidence: str = "declared"
    name: str | None = None
    parent_columns: list[str] = Field(default_factory=list)
    child_columns: list[str] = Field(default_factory=list)


class CompositeKey(_Base):
    table: str
    columns: list[str]


class UniqueConstraint(_Base):
    name: str | None = None
    columns: list[str] = Field(default_factory=list)


class CheckConstraint(_Base):
    name: str | None = None
    expression: str


class IndexSchema(_Base):
    name: str
    columns: list[str] = Field(default_factory=list)
    unique: bool = False
    expression: str | None = None


class TableSchema(_Base):
    name: str
    schema_name: str | None = None
    primary_key: str | None = None
    primary_key_columns: list[str] = Field(default_factory=list)
    primary_key_confirmed: bool = True
    columns: list[ColumnSchema] = Field(default_factory=list)
    unique_constraints: list[UniqueConstraint] = Field(default_factory=list)
    check_constraints: list[CheckConstraint] = Field(default_factory=list)
    indexes: list[IndexSchema] = Field(default_factory=list)
    row_count: int = 0
    estimated_row_count: int | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class DatabaseSchema(_Base):
    format_version: str = SCHEMA_FORMAT_VERSION
    source_kind: str
    tables: dict[str, TableSchema] = Field(default_factory=dict)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    composite_keys: list[CompositeKey] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
