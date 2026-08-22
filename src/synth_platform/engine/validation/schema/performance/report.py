"""Performance report payloads for pipeline runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from synth_platform.engine.validation.schema.performance.timing import PipelineStageTimings


@dataclass
class PipelinePerformanceReport:
    generation_mode: str
    target_rows: int
    rows_generated: int
    chunk_size: int
    rows_per_second: float
    generation_time_seconds: float
    total_time_seconds: float
    peak_memory_mb: float
    memory_bounded: bool
    export_format: str
    output_size_mb: float = 0.0
    write_time_seconds: float = 0.0
    write_mb_per_second: float = 0.0
    chunk_count: int = 0
    cache_hit: bool = False
    profile_cache_hit: bool = False
    model_reused: bool = False
    refit_performed: bool = True
    refit_reason: Optional[str] = None
    chunked_mode: bool = False
    output_materialized: bool = True
    output_streamed: bool = False
    fidelity_mode: str = "exact"
    sampled_fidelity_size: Optional[int] = None
    stages: PipelineStageTimings = field(default_factory=PipelineStageTimings)
    notes: List[str] = field(default_factory=list)
    export_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = self.stages.to_dict()
        return payload


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024 * 1024)


def build_pipeline_performance_report(
    *,
    generation_mode: str,
    target_rows: int,
    rows_generated: int,
    chunk_size: int,
    stages: PipelineStageTimings,
    peak_memory_mb: float,
    export_format: str,
    export_path: Optional[Path] = None,
    memory_bounded: Optional[bool] = None,
    chunk_count: int = 0,
    cache_hit: bool = False,
    profile_cache_hit: bool = False,
    model_reused: bool = False,
    refit_performed: bool = True,
    refit_reason: Optional[str] = None,
    fidelity_mode: str = "exact",
    sampled_fidelity_size: Optional[int] = None,
    notes: Optional[List[str]] = None,
) -> PipelinePerformanceReport:
    total = stages.elapsed_total()
    generation_time = stages.extra.get("sdv_sample_seconds", 0.0) + stages.pii_overlay_seconds
    if generation_time <= 0:
        generation_time = stages.generate_seconds or stages.sample_seconds or total
    rows_per_second = rows_generated / generation_time if generation_time > 0 else 0.0
    write_time = stages.export_seconds
    output_size = _dir_size_mb(export_path) if export_path else 0.0
    write_mbps = output_size / write_time if write_time > 0 else 0.0
    if memory_bounded is None:
        memory_bounded = chunk_size < target_rows and peak_memory_mb < max(512.0, target_rows / 1000)
    return PipelinePerformanceReport(
        generation_mode=generation_mode,
        target_rows=target_rows,
        rows_generated=rows_generated,
        chunk_size=chunk_size,
        rows_per_second=round(rows_per_second, 2),
        generation_time_seconds=round(generation_time, 4),
        total_time_seconds=round(total, 4),
        peak_memory_mb=round(peak_memory_mb, 2),
        memory_bounded=bool(memory_bounded),
        export_format=export_format,
        output_size_mb=round(output_size, 3),
        write_time_seconds=round(write_time, 4),
        write_mb_per_second=round(write_mbps, 3),
        chunk_count=int(chunk_count),
        cache_hit=cache_hit,
        profile_cache_hit=profile_cache_hit,
        model_reused=model_reused,
        refit_performed=refit_performed,
        refit_reason=refit_reason,
        chunked_mode=chunk_size < target_rows,
        output_materialized=chunk_count <= 1 and target_rows <= chunk_size,
        output_streamed=chunk_size < target_rows,
        fidelity_mode=fidelity_mode,
        sampled_fidelity_size=sampled_fidelity_size,
        stages=stages,
        notes=list(notes or []),
        export_path=str(export_path) if export_path else None,
    )
