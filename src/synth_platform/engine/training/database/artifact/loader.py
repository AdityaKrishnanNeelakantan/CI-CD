"""Loads a portable artifact with checksum verification and zip-safety
checks, before trusting any of its content.

The loading sequence verifies structure and integrity (zip entry paths,
then every checksum) before parsing anything as JSON or deserialising a
model, so a corrupted or tampered archive fails closed rather than
partially loading. No source database connection is ever required here -
that is the whole point of a portable artifact.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_platform.engine.training.database.artifact.builder import (
    CHECKSUMS_FILENAME,
    MANIFEST_FILENAME,
    SELF_TEST_FILENAME,
)
from synth_platform.engine.training.database.artifact.errors import (
    ArtifactLoadError,
    ChecksumMismatchError,
    IncompatibleArtifactError,
    PathTraversalError,
    UntrustedModelFormatError,
)
from synth_platform.engine.training.database.artifact.manifest import ARTIFACT_FORMAT, ARTIFACT_FORMAT_VERSION
from synth_platform.engine.training.database.base import SynthesizerAdapter
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class


def _validate_member_path(name: str) -> None:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized:
        raise PathTraversalError(f"unsafe archive entry path: {name!r}")
    parts = normalized.split("/")
    if any(part == ".." for part in parts):
        raise PathTraversalError(f"unsafe archive entry path: {name!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactLoadError(f"artifact is missing required file: {path.name}")
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(f"artifact file is not valid JSON: {path.name}") from exc
    if not isinstance(data, dict):
        raise ArtifactLoadError(f"artifact file must contain a JSON object: {path.name}")
    return data


def _verify_checksums(extract_root: Path, checksums_path: Path) -> None:
    with checksums_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    for line in lines:
        digest, _, rel_path = line.partition("  ")
        if not digest or not rel_path:
            raise ArtifactLoadError(f"malformed checksums entry: {line!r}")
        file_path = extract_root / rel_path
        if not file_path.is_file():
            raise ChecksumMismatchError(f"file listed in checksums but missing from archive: {rel_path}")
        actual = _sha256_file(file_path)
        if actual != digest:
            raise ChecksumMismatchError(
                f"checksum mismatch for {rel_path}: expected {digest}, got {actual}"
            )


@dataclass
class LoadedArtifact:
    manifest: dict[str, Any]
    database_catalog: dict[str, Any]
    dataset_contract: dict[str, Any]
    reference_profile: dict[str, Any]
    self_test_spec: dict[str, Any]
    extract_dir: Path = field(repr=False, compare=False)

    def get_model(self, table_name: str, *, allow_cloudpickle_models: bool = False) -> SynthesizerAdapter:
        if table_name not in self.manifest["tables"]:
            raise ArtifactLoadError(f"no trained model for table {table_name!r} in this artifact")
        table_entry = self.manifest["tables"][table_name]
        # Fail closed by default: deserializing a cloudpickle model can
        # execute arbitrary code, so a caller must explicitly opt in after
        # confirming the artifact's origin is trusted.
        if table_entry["model_serialization_format"] == "cloudpickle" and not allow_cloudpickle_models:
            raise UntrustedModelFormatError(
                f"table {table_name!r} uses cloudpickle serialization, which can execute "
                "arbitrary code on load. Pass allow_cloudpickle_models=True only if this "
                "artifact's origin is trusted."
            )
        model_path = self.extract_dir / table_entry["model_path"]
        adapter_cls = get_synthesizer_adapter_class(table_entry["model_type"])
        return adapter_cls.load(model_path)


def load_artifact(path: str | Path, extract_dir: str | Path | None = None) -> LoadedArtifact:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise ArtifactLoadError(f"artifact not found: {archive_path}")

    extract_root = Path(extract_dir) if extract_dir else Path(tempfile.mkdtemp(prefix="artifact_"))
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            for name in names:
                _validate_member_path(name)
            zf.extractall(extract_root)
    except zipfile.BadZipFile as exc:
        raise ArtifactLoadError(f"artifact is not a valid zip file: {archive_path}") from exc

    checksums_path = extract_root / CHECKSUMS_FILENAME
    if not checksums_path.is_file():
        raise ArtifactLoadError(f"artifact is missing {CHECKSUMS_FILENAME}")
    _verify_checksums(extract_root, checksums_path)

    manifest = _read_json(extract_root / MANIFEST_FILENAME)

    if manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise IncompatibleArtifactError(
            f"unrecognised artifact_format: {manifest.get('artifact_format')!r}"
        )
    if manifest.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise IncompatibleArtifactError(
            f"unsupported artifact_format_version: {manifest.get('artifact_format_version')!r}"
        )

    return LoadedArtifact(
        manifest=manifest,
        database_catalog=_read_json(extract_root / "schema" / "database_catalog.json"),
        dataset_contract=_read_json(extract_root / "semantics" / "dataset_contract.json"),
        reference_profile=_read_json(extract_root / "validation" / "reference_profile.json"),
        self_test_spec=_read_json(extract_root / SELF_TEST_FILENAME),
        extract_dir=extract_root,
    )
