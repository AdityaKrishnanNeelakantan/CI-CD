"""Permission-restricted, bounded-lifetime staging for raw UI source files."""

from __future__ import annotations

import atexit
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path
from uuid import uuid4

_STAGING_ROOT = Path(tempfile.gettempdir()) / "synth-platform-source-staging"
_MAX_AGE_SECONDS = 24 * 60 * 60
_ACTIVE_DIRECTORIES: set[Path] = set()


def _ensure_private_staging_root() -> None:
    """Create and validate the fixed temporary root without following symlinks."""

    try:
        _STAGING_ROOT.mkdir(mode=0o700)
    except FileExistsError:
        pass
    root_info = _STAGING_ROOT.lstat()
    effective_uid = getattr(os, "geteuid", lambda: root_info.st_uid)()
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != effective_uid:
        raise PermissionError("source staging root must be an owned directory")
    _STAGING_ROOT.chmod(0o700)
    root_info = _STAGING_ROOT.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode) != 0o700:
        raise PermissionError("source staging root is not private")


def _secure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        path_info = path.lstat()
        if not stat.S_ISDIR(path_info.st_mode):
            raise PermissionError("source staging path is not a directory")
        if stat.S_IMODE(path_info.st_mode) != 0o700:
            raise PermissionError("source staging directory is not private")
    except OSError:
        try:
            path_info = path.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(path_info.st_mode):
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise


def cleanup_stale_source_staging(*, max_age_seconds: int = _MAX_AGE_SECONDS) -> int:
    """Remove abandoned source directories older than the bounded retention age."""

    _ensure_private_staging_root()
    cutoff = time.time() - max(60, max_age_seconds)
    removed = 0
    for candidate in _STAGING_ROOT.iterdir():
        candidate_info = candidate.lstat()
        if (
            not stat.S_ISDIR(candidate_info.st_mode)
            or candidate_info.st_mtime >= cutoff
        ):
            continue
        try:
            shutil.rmtree(candidate)
        except OSError:
            continue
        removed += 1
    return removed


def create_source_staging(prefix: str) -> Path:
    """Create a private staging directory and register process-exit cleanup."""

    cleanup_stale_source_staging()
    safe_prefix = "".join(
        char for char in prefix.lower() if char.isalnum() or char == "-"
    )
    path = _STAGING_ROOT / f"{safe_prefix or 'source'}-{uuid4().hex}"
    _secure_directory(path)
    _ACTIVE_DIRECTORIES.add(path)
    return path


def touch_source_staging(path: str | Path | None) -> None:
    """Refresh an active directory so a concurrent stale sweep leaves it alone."""

    if path is None:
        return
    staging_path = Path(path)
    if staging_path.is_dir():
        os.utime(staging_path, None)
        _ACTIVE_DIRECTORIES.add(staging_path)


def remove_source_staging(
    path: str | Path | None,
    *,
    ignore_errors: bool = False,
) -> None:
    """Remove one managed source directory and verify that it is gone."""

    if path is None:
        return
    staging_path = Path(path)
    if staging_path.parent != _STAGING_ROOT:
        raise ValueError("source staging path is outside the managed root")
    try:
        shutil.rmtree(staging_path)
    except FileNotFoundError:
        pass
    except OSError:
        if ignore_errors:
            return
        raise
    if staging_path.exists():
        if ignore_errors:
            return
        raise OSError("source staging directory cleanup could not be verified")
    _ACTIVE_DIRECTORIES.discard(staging_path)


def _cleanup_process_staging() -> None:
    for path in tuple(_ACTIVE_DIRECTORIES):
        remove_source_staging(path, ignore_errors=True)


atexit.register(_cleanup_process_staging)
