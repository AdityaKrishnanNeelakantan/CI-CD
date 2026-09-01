"""Metric protocol + evaluation context (pure domain contract).

A Metric is a self-contained validation unit: it declares what it needs and
produces CheckResults. The engine drives a registry of these; nothing else
re-implements per-check thresholding (ARCHITECTURE_AUDIT A-05). Adding a metric
is a new registry entry, never an edit to a growing service function (OCP).

The context is passed by the engine; the artifact/tables types are `Any` here so
the domain stays pandas-free (the concrete frames live in outer layers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from synth_platform.domain.validation.models import CheckResult
from synth_platform.domain.validation.thresholds import Thresholds


@dataclass
class ValidationContext:
    artifact: Any                       # compiled SynthArtifact
    tables: dict[str, Any]              # generated frames
    thresholds: Thresholds = field(default_factory=Thresholds)
    extras: dict[str, Any] = field(default_factory=dict)  # e.g. holdout split


@runtime_checkable
class Metric(Protocol):
    name: str
    dimension: str

    def evaluate(self, ctx: ValidationContext) -> list[CheckResult]:
        """Pure: derive CheckResults from the context. Never mutates it."""
        ...
