"""Exceptions shared by all SourceAdapter implementations.

Adapters raise these instead of letting driver-specific exceptions (e.g.
sqlite3.OperationalError) leak past the adapter boundary, so callers can
handle source errors generically regardless of which adapter is in use.
"""

from __future__ import annotations


class SourceAdapterError(Exception):
    """Base class for all source adapter errors."""


class SourceConfigError(SourceAdapterError):
    """Raised when the source configuration is invalid or incomplete."""


class SourceConnectionError(SourceAdapterError):
    """Raised when a connection or health check against the source fails."""


class UnknownTableError(SourceAdapterError):
    """Raised when an operation references a table the source does not have."""
