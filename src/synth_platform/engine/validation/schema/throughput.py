"""Throughput tracking and bounded-memory generation helpers."""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

try:
    import resource  # POSIX-only; absent on Windows (see _peak_memory_mb's guard below)
except ImportError:  # pragma: no cover - exercised on Windows CI/dev machines
    resource = None

T = TypeVar("T")


def _peak_memory_mb() -> float:
    """Return peak RSS in MB (platform-dependent; best-effort)."""
    if resource is None:
        return 0.0
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss is KB; macOS: bytes
        rss = float(usage.ru_maxrss)
        if rss > 10_000_000:  # likely bytes (macOS)
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return 0.0


@dataclass
class StageTimings:
    profile_seconds: float = 0.0
    fit_seconds: float = 0.0
    sample_seconds: float = 0.0
    export_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PerformanceReport:
    """Summary metrics for high-volume generation runs."""

    generation_mode: str
    target_rows: int
    rows_generated: int
    chunk_size: int
    rows_per_second: float
    peak_memory_mb: float
    memory_bounded: bool
    export_format: str
    export_path: Optional[str] = None
    stages: StageTimings = field(default_factory=StageTimings)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = self.stages.to_dict()
        return payload


@contextmanager
def track_peak_memory() -> Iterator[List[float]]:
    """Track peak traced memory (MB) during a block."""
    peaks: List[float] = [0.0]
    tracemalloc.start()
    try:
        yield peaks
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks[0] = max(peaks[0], peak / (1024 * 1024), _peak_memory_mb())


@contextmanager
def time_stage(seconds_holder: StageTimings, attr: str) -> Iterator[None]:
    """Context manager that records elapsed seconds into *seconds_holder*."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        setattr(seconds_holder, attr, getattr(seconds_holder, attr) + elapsed)


def run_timed(callable_fn: Callable[[], T], *, stage: StageTimings, attr: str) -> T:
    """Run callable and accumulate timing."""
    start = time.perf_counter()
    try:
        return callable_fn()
    finally:
        setattr(stage, attr, getattr(stage, attr) + (time.perf_counter() - start))


def estimate_memory_bounded(
    *,
    target_rows: int,
    chunk_size: int,
    peak_memory_mb: float,
    max_preview_rows: int = 5000,
) -> bool:
    """Heuristic: streaming stayed bounded if chunk_size < target and peak is modest."""
    if target_rows <= max_preview_rows:
        return True
    return chunk_size < target_rows and peak_memory_mb < max(512.0, target_rows / 1000)


def build_performance_report(
    *,
    generation_mode: str,
    target_rows: int,
    rows_generated: int,
    chunk_size: int,
    stages: StageTimings,
    peak_memory_mb: float,
    export_format: str,
    export_path: Optional[Path] = None,
    memory_bounded: Optional[bool] = None,
    notes: Optional[List[str]] = None,
) -> PerformanceReport:
    total = stages.total_seconds or (
        stages.profile_seconds + stages.fit_seconds + stages.sample_seconds + stages.export_seconds
    )
    rows_per_second = rows_generated / total if total > 0 else 0.0
    bounded = memory_bounded
    if bounded is None:
        bounded = estimate_memory_bounded(
            target_rows=target_rows,
            chunk_size=chunk_size,
            peak_memory_mb=peak_memory_mb,
        )
    return PerformanceReport(
        generation_mode=generation_mode,
        target_rows=target_rows,
        rows_generated=rows_generated,
        chunk_size=chunk_size,
        rows_per_second=round(rows_per_second, 2),
        peak_memory_mb=round(peak_memory_mb, 2),
        memory_bounded=bool(bounded),
        export_format=export_format,
        export_path=str(export_path) if export_path else None,
        stages=stages,
        notes=list(notes or []),
    )
