"""Compiled conditional distributions (pure, source-free).

A conditional generator stores P(child | parent-context) as a table of marginals
keyed by the discretized parent context, plus an unconditional fallback for
unseen contexts. This preserves cross-column correlation deterministically
without any trained network — the honest baseline that satisfies "never generate
columns independently" (ARCHITECTURE_AUDIT RC-1 / A-01).

Storage is generic over value kind:
  - categorical child: each bucket holds (values, probabilities),
  - numeric child:     each bucket holds an empirical quantile grid.
The parent context is a tuple of discretized parent values (categorical values
as-is; numeric parents binned into quantile bands at compile time).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CategoricalConditional(_Base):
    values: list[str]
    probabilities: list[float]


class NumericConditional(_Base):
    quantiles: list[float]
    minimum: float
    maximum: float


class ConditionalModel(_Base):
    """P(child | context). `by_context` maps a context key (joined discretized
    parent values) to the child's conditional distribution; `fallback` is the
    unconditional marginal used for unseen contexts."""
    parents: list[str] = Field(default_factory=list)
    kind: str  # "categorical" | "numeric"
    categorical_by_context: dict[str, CategoricalConditional] = Field(default_factory=dict)
    numeric_by_context: dict[str, NumericConditional] = Field(default_factory=dict)
    categorical_fallback: CategoricalConditional | None = None
    numeric_fallback: NumericConditional | None = None

    @property
    def is_conditional(self) -> bool:
        return bool(self.parents)
