"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PipelineConfig:
    generation_mode: str  # schema_driven | source_driven
    seed: int = 42
    preview_rows: int = 100
    full_rows: Optional[int] = None
    chunk_size: int = 10_000
    source_chunk_size: Optional[int] = None
    export_format: str = "parquet"
    output_dir: Optional[Path] = None
    table_name: str = "source_table"
    model_type: Optional[str] = None
    fit_max_rows: int = 5_000
    profile_max_rows: int = 5_000
    epochs: int = 50
    reuse_artifacts_dir: Optional[Path] = None
    rules_text: str = ""
    llm_text_enabled: bool = False
    llm_full_enabled: bool = False
    max_llm_rows: int = 50
    fidelity_mode: str = "auto"  # auto | exact | sampled
    fidelity_sample_size: int = 5_000
    large_row_threshold: int = 50_000
    preview_only: bool = False
    write_reports: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineProgress:
    stage: str = "idle"
    message: str = ""
    percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "message": self.message, "percent": self.percent}
