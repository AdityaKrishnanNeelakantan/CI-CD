"""Exceptions for building and loading portable artifacts."""

from __future__ import annotations


class ArtifactError(Exception):
    """Base class for all artifact build/load errors."""


class ArtifactBuildError(ArtifactError):
    """Raised when an artifact cannot be assembled from its inputs."""


class ArtifactLoadError(ArtifactError):
    """Raised when an artifact file is missing, unreadable, or malformed."""


class ChecksumMismatchError(ArtifactLoadError):
    """Raised when a file's content does not match its recorded checksum - the
    archive has been corrupted or tampered with since it was built."""


class PathTraversalError(ArtifactLoadError):
    """Raised when a zip entry would extract outside the target directory."""


class IncompatibleArtifactError(ArtifactLoadError):
    """Raised when the artifact's format version is not supported by this loader."""


class UntrustedModelFormatError(ArtifactLoadError):
    """Raised when a table's model is serialized with cloudpickle (arbitrary
    code execution on load) and the caller has not explicitly opted in via
    allow_cloudpickle_models=True."""
