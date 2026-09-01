"""Generation quarantine: isolate synthetic output; promote only on PASS."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from synth_platform.domain.validation.models import Status, ValidationReport
from synth_platform.errors import QuarantineBlockedError


@dataclass
class QuarantineResult:
    status: Status
    staging_dir: Path
    report: ValidationReport
    promoted_dir: Path | None = None

    @property
    def promotable(self) -> bool:
        return self.status == Status.PASS


class Quarantine:
    def __init__(self, staging_root: str | Path):
        self.staging_root = Path(staging_root)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def submit(self, tables: dict, report: ValidationReport, run_id: str = "run") -> QuarantineResult:
        d = self.staging_root / run_id
        d.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(d / f"{name}.csv", index=False)
        (d / "validation_report.json").write_text(report.model_dump_json(indent=2))
        (d / "STATUS").write_text(report.overall.value)
        return QuarantineResult(status=report.overall, staging_dir=d, report=report)

    def promote(self, result: QuarantineResult, destination: str | Path) -> QuarantineResult:
        if result.status != Status.PASS:
            from synth_platform.domain.validation.release_gate import decide
            failed = decide(result.report).blocking  # canonical block reasons
            raise QuarantineBlockedError(
                f"promotion blocked: overall={result.status.value}; blocking: {failed}")
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        for csv in result.staging_dir.glob("*.csv"):
            shutil.copy2(csv, dest / csv.name)
        (dest / "validation_report.json").write_text(result.report.model_dump_json(indent=2))
        result.promoted_dir = dest
        return result
