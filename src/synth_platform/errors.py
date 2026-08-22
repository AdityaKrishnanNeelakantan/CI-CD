"""Typed, catchable errors. No bare Exception in core paths."""
from __future__ import annotations


class SynthPlatformError(Exception):
    """Base for all platform errors."""


class SourceUnavailableError(SynthPlatformError):
    """A source connector failed its health check."""


class UnsafeIdentifierError(SynthPlatformError):
    """A table/column identifier failed strict validation."""


class ReadOnlyViolationError(SynthPlatformError):
    """A write was attempted through a read-only connector."""


class UnsupportedCycleError(SynthPlatformError):
    """The relational graph has a cycle with no configured resolution."""


class ArtifactIntegrityError(SynthPlatformError):
    """Checksum mismatch, missing member, or forbidden payload in an artifact."""


class ArtifactCompatibilityError(SynthPlatformError):
    """Artifact format version is outside the supported range."""


class ForbiddenPayloadError(ArtifactIntegrityError):
    """Artifact contained a credential, URL, or raw source row."""


class UnsafePathError(ArtifactIntegrityError):
    """A package member path escaped the extraction root (path traversal)."""


class PickleRejectedError(ArtifactIntegrityError):
    """A pickled payload was encountered and refused."""


class QuarantineBlockedError(SynthPlatformError):
    """Promotion refused because the release verdict was not PASS."""


class PrivacyLeakError(SynthPlatformError):
    """A raw value from a non-public column survived into the compiled artifact.
    Raised at compile time by the privacy verifier — redaction is incomplete."""


class ExtractionError(SynthPlatformError):
    """An extraction pass failed to produce a valid dataset (e.g. an LLM adapter
    could not reach its backend or returned unparseable output)."""

class UnsupportedSourceError(SynthPlatformError):
    """The requested source URL scheme has no registered connector."""


class ExtractionQualityError(ExtractionError):
    """A PDF/document extraction failed a critical quality gate."""


class PublicationBlockedError(SynthPlatformError):
    """Publication was refused because the release decision was not PASS."""


class ArtifactSourceKindError(SynthPlatformError):
    """An artifact was used by the wrong source-track operation."""
