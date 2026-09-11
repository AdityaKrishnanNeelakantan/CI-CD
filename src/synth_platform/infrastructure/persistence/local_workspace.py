"""Atomic local persistence adapters for shared workspace metadata.

Only application DTOs are stored. Workflow source payloads, transcript text,
and database credentials are never accepted by these adapters.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from synth_platform.application.dto.workspace import (
    ArtifactRef,
    ArtifactRegistration,
    Project,
    Template,
    UserPreferences,
    WorkflowKind,
    WorkflowResultView,
    WorkflowSession,
)
from synth_platform.engine.generation.schema.templates.library import (
    list_templates as list_schema_templates,
)
from synth_platform.engine.generation.schema.templates.library import (
    load_template as load_schema_template,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _safe_id(value: str) -> str:
    if not value or any(not (char.isalnum() or char in "_.-") for char in value):
        raise ValueError("identifier is not path-safe")
    return value


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_file_lock(path: Path, *, timeout_seconds: float = 10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                stale = time.time() - path.stat().st_mtime > timeout_seconds
            except OSError:
                stale = False
            if stale:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("workspace commit lock is busy") from None
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target:
            shutil.copyfileobj(source_handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class _JsonCollection(Generic[ModelT]):
    def __init__(self, directory: Path, model_type: type[ModelT], id_field: str):
        self._directory = directory
        self._model_type = model_type
        self._id_field = id_field
        self._lock = threading.RLock()
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(self, model: ModelT) -> None:
        identifier = _safe_id(str(getattr(model, self._id_field)))
        with self._lock:
            _atomic_json_write(
                self._directory / f"{identifier}.json",
                model.model_dump(mode="json"),
            )

    def get(self, identifier: str) -> ModelT | None:
        path = self._directory / f"{_safe_id(identifier)}.json"
        with self._lock:
            if not path.is_file():
                return None
            return self._model_type.model_validate_json(
                path.read_text(encoding="utf-8")
            )

    def list(self) -> list[ModelT]:
        with self._lock:
            return [
                self._model_type.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(self._directory.glob("*.json"))
            ]

    def delete(self, identifier: str) -> None:
        with self._lock:
            (self._directory / f"{_safe_id(identifier)}.json").unlink(missing_ok=True)


class LocalSessionRepository:
    def __init__(self, root: str | Path):
        self._store = _JsonCollection(
            Path(root) / "sessions", WorkflowSession, "session_id"
        )

    def save(self, session: WorkflowSession) -> None:
        self._store.save(session)

    def get(self, session_id: str) -> WorkflowSession | None:
        return self._store.get(session_id)

    def list(self) -> list[WorkflowSession]:
        return self._store.list()

    def delete(self, session_id: str) -> None:
        self._store.delete(session_id)


class LocalResultRepository:
    def __init__(self, root: str | Path):
        self._store = _JsonCollection(
            Path(root) / "results", WorkflowResultView, "result_id"
        )

    def save(self, result: WorkflowResultView) -> None:
        self._store.save(result)

    def get(self, result_id: str) -> WorkflowResultView | None:
        return self._store.get(result_id)

    def list(self) -> list[WorkflowResultView]:
        return self._store.list()

    def delete(self, result_id: str) -> None:
        self._store.delete(result_id)


class LocalProjectRepository:
    def __init__(self, root: str | Path):
        self._store = _JsonCollection(Path(root) / "projects", Project, "project_id")

    def save(self, project: Project) -> None:
        self._store.save(project)

    def get(self, project_id: str) -> Project | None:
        return self._store.get(project_id)

    def list(self) -> list[Project]:
        return self._store.list()

    def delete(self, project_id: str) -> None:
        self._store.delete(project_id)


class LocalSettingsRepository:
    def __init__(self, root: str | Path):
        self._path = Path(root) / "preferences.json"
        self._lock = threading.RLock()

    def get(self) -> UserPreferences:
        with self._lock:
            if not self._path.is_file():
                return UserPreferences()
            return UserPreferences.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )

    def save(self, preferences: UserPreferences) -> None:
        with self._lock:
            _atomic_json_write(self._path, preferences.model_dump(mode="json"))


class LocalWorkspaceCommitter:
    """Recoverable multi-record completion commit with a cross-process lock."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._journal = self._root / ".completion-journal.json"
        self._lock = self._root / ".completion.lock"
        if self._journal.is_file():
            with _exclusive_file_lock(self._lock):
                self._recover_locked()

    def _apply_locked(self, payload: dict) -> None:
        result = WorkflowResultView.model_validate(payload["result"])
        session = WorkflowSession.model_validate(payload["session"])
        _atomic_json_write(
            self._root / "results" / f"{_safe_id(result.result_id)}.json",
            result.model_dump(mode="json"),
        )
        _atomic_json_write(
            self._root / "sessions" / f"{_safe_id(session.session_id)}.json",
            session.model_dump(mode="json"),
        )
        if payload.get("project") is not None:
            project = Project.model_validate(payload["project"])
            _atomic_json_write(
                self._root / "projects" / f"{_safe_id(project.project_id)}.json",
                project.model_dump(mode="json"),
            )

    def _recover_locked(self) -> None:
        if not self._journal.is_file():
            return
        payload = json.loads(self._journal.read_text(encoding="utf-8"))
        self._apply_locked(payload)
        self._journal.unlink(missing_ok=True)
        _fsync_directory(self._root)

    def commit_completion(
        self,
        result: WorkflowResultView,
        session: WorkflowSession,
        project: Project | None,
    ) -> None:
        payload = {
            "result": result.model_dump(mode="json"),
            "session": session.model_dump(mode="json"),
            "project": project.model_dump(mode="json") if project else None,
        }
        with _exclusive_file_lock(self._lock):
            if self._journal.is_file():
                self._recover_locked()
            _atomic_json_write(self._journal, payload)
            self._apply_locked(payload)
            self._journal.unlink(missing_ok=True)
            _fsync_directory(self._root)


class BuiltInSchemaTemplateRepository:
    """Expose the real schema template library without inventing other assets."""

    def _template(self, name: str) -> Template:
        schema = load_schema_template(name)
        return Template(
            template_id=f"schema.{name}.v1",
            name=schema.name,
            description=schema.description or "",
            workflow=WorkflowKind.SCHEMA,
            version="1.0.0",
            tags=["built-in", "schema", name],
            definition=schema.model_dump(mode="json"),
            built_in=True,
        )

    def get(self, template_id: str) -> Template | None:
        for template in self.list():
            if template.template_id == template_id:
                return template
        return None

    def list(self) -> list[Template]:
        return [self._template(name) for name in list_schema_templates()]


class LocalArtifactCatalog:
    def __init__(self, root: str | Path, *, allowed_roots: list[str | Path]):
        root_path = Path(root)
        self._store = _JsonCollection(
            root_path / "artifacts", ArtifactRef, "artifact_id"
        )
        self._blob_root = root_path / "artifact_blobs" / "sha256"
        self._allowed_roots = [
            Path(item).expanduser().resolve() for item in allowed_roots
        ]
        if not self._allowed_roots:
            raise ValueError("at least one artifact root must be allowed")

    def register(
        self, session_id: str, registration: ArtifactRegistration
    ) -> ArtifactRef:
        path = Path(registration.path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("artifact path must identify a file")
        if not any(path.is_relative_to(root) for root in self._allowed_roots):
            raise ValueError("artifact path is outside approved local roots")
        checksum = _sha256_file(path)
        snapshot_path = self._blob_root / f"{checksum}{path.suffix.lower()}"
        if snapshot_path.exists():
            if _sha256_file(snapshot_path) != checksum:
                raise ValueError("content-addressed artifact snapshot is corrupt")
        else:
            _atomic_copy(path, snapshot_path)
        for existing in self.list(session_id=session_id):
            if (
                existing.local_path == str(snapshot_path)
                and existing.sha256 == checksum
            ):
                return existing
        artifact = ArtifactRef(
            session_id=session_id,
            label=registration.label,
            media_type=registration.media_type,
            format=registration.format,
            local_path=str(snapshot_path),
            size_bytes=snapshot_path.stat().st_size,
            sha256=checksum,
            downloadable=registration.downloadable,
        )
        self._store.save(artifact)
        return artifact

    def get(self, artifact_id: str) -> ArtifactRef | None:
        return self._store.get(artifact_id)

    def read_bytes(self, artifact_id: str) -> bytes:
        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(f"artifact not found: {artifact_id}")
        path = Path(artifact.local_path)
        if not path.is_file() or path.stat().st_size != artifact.size_bytes:
            raise ValueError("artifact snapshot is missing or has an invalid size")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise ValueError("artifact snapshot checksum verification failed")
        return content

    def list(self, *, session_id: str | None = None) -> list[ArtifactRef]:
        artifacts = self._store.list()
        if session_id is not None:
            artifacts = [item for item in artifacts if item.session_id == session_id]
        return artifacts

    def delete(self, artifact_id: str) -> None:
        self._store.delete(artifact_id)
