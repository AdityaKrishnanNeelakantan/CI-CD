"""Pipeline stage timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class PipelineStageTimings:
    """Elapsed seconds per pipeline stage (both generation modes)."""

    load_seconds: float = 0.0
    profile_seconds: float = 0.0
    plan_seconds: float = 0.0
    recommend_seconds: float = 0.0
    metadata_seconds: float = 0.0
    fit_seconds: float = 0.0
    generate_seconds: float = 0.0
    sample_seconds: float = 0.0
    pii_overlay_seconds: float = 0.0
    rules_seconds: float = 0.0
    duplicate_repair_seconds: float = 0.0
    validate_seconds: float = 0.0
    fidelity_seconds: float = 0.0
    metrics_seconds: float = 0.0
    export_seconds: float = 0.0
    render_seconds: float = 0.0
    total_seconds: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def elapsed_total(self) -> float:
        if self.total_seconds > 0:
            return self.total_seconds
        parts = [
            self.load_seconds,
            self.profile_seconds,
            self.plan_seconds,
            self.recommend_seconds,
            self.metadata_seconds,
            self.fit_seconds,
            self.generate_seconds or self.sample_seconds,
            self.pii_overlay_seconds,
            self.rules_seconds,
            self.duplicate_repair_seconds,
            self.validate_seconds,
            self.fidelity_seconds,
            self.metrics_seconds,
            self.export_seconds,
            self.render_seconds,
        ]
        parts.extend(self.extra.values())
        return sum(parts)

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["total_seconds"] = round(self.elapsed_total(), 6)
        for key, value in self.extra.items():
            payload[key] = round(float(value), 6)
        return {k: round(float(v), 6) if isinstance(v, (int, float)) else v for k, v in payload.items()}


@contextmanager
def time_stage(seconds_holder: PipelineStageTimings, attr: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if hasattr(seconds_holder, attr):
            setattr(seconds_holder, attr, getattr(seconds_holder, attr) + elapsed)
        else:
            seconds_holder.extra[attr] = seconds_holder.extra.get(attr, 0.0) + elapsed


def run_timed(callable_fn: Callable[[], T], *, stage: PipelineStageTimings, attr: str) -> T:
    start = time.perf_counter()
    try:
        return callable_fn()
    finally:
        elapsed = time.perf_counter() - start
        if hasattr(stage, attr):
            setattr(stage, attr, getattr(stage, attr) + elapsed)
        else:
            stage.extra[attr] = stage.extra.get(attr, 0.0) + elapsed
