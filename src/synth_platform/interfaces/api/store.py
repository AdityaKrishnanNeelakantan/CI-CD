"""File-backed state for the local FastAPI proof-of-concept."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.settings import Settings


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _state_root() -> Path:
    override = os.environ.get("SP_API_STATE_DIR")
    if override:
        return Path(override)
    return Path(Settings.from_env().staging_root) / "api"


class ApiStateStore:
    """Persist sessions, jobs, downloads, and generated payloads as JSON/files."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _state_root()
        self.sessions_dir = self.root / "sessions"
        self.jobs_dir = self.root / "jobs"
        self.downloads_dir = self.root / "downloads"
        self.blobs_dir = self.root / "blobs"
        self.results_dir = self.root / "results"
        self._lock = threading.RLock()
        for path in (self.sessions_dir, self.jobs_dir, self.downloads_dir, self.blobs_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)

    def create_session(self, workflow: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utc_now()
        record = {
            "id": uuid.uuid4().hex,
            "workflow": workflow,
            "created_at": now,
            "updated_at": now,
            "state": state or {},
        }
        self.save_session(record)
        return record

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._read_json(self.sessions_dir / f"{session_id}.json")

    def save_session(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        record["updated_at"] = utc_now()
        self._write_json(self.sessions_dir / f"{record['id']}.json", record)
        return record

    def create_job(self, *, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utc_now()
        record = {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "status": "queued",
            "progress": ["queued"],
            "error": None,
            "result": None,
            "payload": payload or {},
            "created_at": now,
            "updated_at": now,
        }
        self.save_job(record)
        return record

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._read_json(self.jobs_dir / f"{job_id}.json")

    def save_job(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        record["updated_at"] = utc_now()
        self._write_json(self.jobs_dir / f"{record['id']}.json", record)
        return record

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        record = self.get_job(job_id)
        record.update(updates)
        return self.save_job(record)

    def append_progress(self, job_id: str, message: str) -> dict[str, Any]:
        record = self.get_job(job_id)
        progress = list(record.get("progress") or [])
        progress.append(message)
        record["progress"] = progress
        return self.save_job(record)

    def write_blob(self, name: str, data: bytes) -> Path:
        blob_id = f"{uuid.uuid4().hex}_{Path(name).name}"
        path = self.blobs_dir / blob_id
        path.write_bytes(data)
        return path

    def write_result(self, session_id: str, payload: dict[str, Any]) -> Path:
        path = self.results_dir / f"{session_id}.json"
        self._write_json(path, payload)
        return path

    def read_result(self, session_id: str) -> dict[str, Any]:
        return self._read_json(self.results_dir / f"{session_id}.json")

    def create_download(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        record = {
            "id": uuid.uuid4().hex,
            "created_at": now,
            "updated_at": now,
            **payload,
        }
        self._write_json(self.downloads_dir / f"{record['id']}.json", record)
        return record

    def get_download(self, download_id: str) -> dict[str, Any]:
        return self._read_json(self.downloads_dir / f"{download_id}.json")

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise KeyError(path.stem)
        with self._lock:
            data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"state file is not an object: {path}")
        return data

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
            temp.replace(path)


def get_api_store() -> ApiStateStore:
    return ApiStateStore()
