"""Pipeline run results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from synth_platform.engine.validation.schema.performance.report import PipelinePerformanceReport
from synth_platform.application.orchestration.schema.config import PipelineProgress


@dataclass
class PipelineResult:
    generation_mode: str
    preview_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    export_paths: Dict[str, Path] = field(default_factory=dict)
    artifact_dir: Optional[Path] = None
    performance: Optional[PipelinePerformanceReport] = None
    validation_report: Dict[str, Any] = field(default_factory=dict)
    fidelity_report: Dict[str, Any] = field(default_factory=dict)
    preview_rule_evidence: List[Dict[str, Any]] = field(default_factory=list)
    final_rule_evidence: List[Dict[str, Any]] = field(default_factory=list)
    rule_evidence: List[Dict[str, Any]] = field(default_factory=list)
    validation_contract: Dict[str, Any] = field(default_factory=dict)
    export_validation: Dict[str, Any] = field(default_factory=dict)
    progress: PipelineProgress = field(default_factory=PipelineProgress)
    profile: Any = None
    schema_summary: str = ""
    row_counts: Dict[str, int] = field(default_factory=dict)
    cache_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_mode": self.generation_mode,
            "row_counts": self.row_counts,
            "export_paths": {k: str(v) for k, v in self.export_paths.items()},
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "performance": self.performance.to_dict() if self.performance else {},
            "validation_report": self.validation_report,
            "fidelity_report": self.fidelity_report,
            "preview_rule_evidence": self.preview_rule_evidence,
            "final_rule_evidence": self.final_rule_evidence,
            "rule_evidence": self.final_rule_evidence or self.rule_evidence,
            "validation_contract": self.validation_contract,
            "export_validation": self.export_validation,
            "progress": self.progress.to_dict(),
            "cache_evidence": self.cache_evidence,
            "schema_summary": self.schema_summary,
        }
