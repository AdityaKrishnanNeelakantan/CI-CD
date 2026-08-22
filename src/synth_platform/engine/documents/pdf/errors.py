"""Exceptions shared by all DocumentAdapter implementations."""

from __future__ import annotations


class DocumentAdapterError(Exception):
    """Base class for all document adapter errors."""


class UnsupportedFileTypeError(DocumentAdapterError):
    """Raised when a file's extension is not supported by the adapter."""


class FileTooLargeError(DocumentAdapterError):
    """Raised when a file exceeds the configured maximum size."""


class ExtractionError(DocumentAdapterError):
    """Raised when text extraction fails or the file is unreadable/corrupt."""


class OCRUnavailableError(DocumentAdapterError):
    """Raised when an OCR fallback is needed but the OCR toolchain (Tesseract/poppler) is missing."""


class DoclingUnavailableError(DocumentAdapterError):
    """Raised when Docling extraction is explicitly requested but unavailable."""


class DocumentIndexError(DocumentAdapterError):
    """Raised when the cross-run document hash index (duplicates.py) cannot be
    read or written - e.g. corrupted JSON on disk, or a filesystem failure
    (disk full, permission denied) while persisting it.
    """
