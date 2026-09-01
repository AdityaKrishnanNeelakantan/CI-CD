"""Privacy policy models (pure)."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrivacyAction(str, Enum):
    PRESERVE = "preserve"        # keep safe business enums
    BUCKET = "bucket"            # bucket rare values
    SUPPRESS = "suppress"        # drop genuinely risky values
    TRANSFORM = "transform"      # synthesize identifiers, never raw
    EXCLUDE = "exclude"          # free text, initially excluded


class ColumnPolicy(_Base):
    table: str
    column: str
    action: PrivacyAction
    reason: str = ""


class PrivacyPolicy(_Base):
    rare_category_threshold: int = 10
    columns: list[ColumnPolicy] = Field(default_factory=list)
    formal_dp: bool = False   # only true with a real accountant present
