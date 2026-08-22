"""Run manifest: immutable execution evidence for a single pipeline run.

Each run gets its own directory under <runs_dir>/<run_id>/. Stage outputs
(discovery.json, profile.json, ...) are written into that same directory,
and every stage's StageResult is appended to run_manifest.json as it
completes. A run directory is never reused across runs and existing stage
entries in the manifest are never rewritten - new stages are appended.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.observability import log_stage_result
from synth_platform.engine.common.database.core.stage_result import StageResult

MANIFEST_FILENAME = "run_manifest.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short_id = uuid.uuid4().hex[:8]
    return f"{timestamp}-{short_id}"


def _detect_code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class RunManifest:
    run_id: str
    run_dir: Path
    created_at: str
    code_version: str
    config_hash: str | None = None
    source_fingerprint: str | None = None
    model_version: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        runs_dir: str | Path,
        config_hash: str | None = None,
        run_id: str | None = None,
    ) -> RunManifest:
        """Create a fresh, uniquely-directoried run.

        `run_id` is normally left to auto-generate; it is exposed so tests
        can force a collision and verify a run directory is never reused.
        """
        run_id = run_id or _new_run_id()
        run_dir = Path(runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = cls(
            run_id=run_id,
            run_dir=run_dir,
            created_at=_utc_now_iso(),
            code_version=_detect_code_version(),
            config_hash=config_hash,
        )
        manifest.write()
        return manifest

    def record_stage(self, result: StageResult) -> None:
        entry = result.to_dict()
        entry["recorded_at"] = _utc_now_iso()
        self.stages.append(entry)
        self.write()
        log_stage_result(self.run_id, result)

    def output_path(self, filename: str) -> Path:
        return self.run_dir / filename

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "source_fingerprint": self.source_fingerprint,
            "model_version": self.model_version,
            "stages": self.stages,
        }

    def write(self) -> Path:
        manifest_path = self.run_dir / MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return manifest_path
