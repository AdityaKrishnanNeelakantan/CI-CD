"""Protocol-independent contracts for selecting a synthetic-twin capability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CapabilityId(StrEnum):
    """Stable product capability identifiers."""

    SCHEMA_TWIN = "schema"
    DATABASE_TWIN = "database"
    DOCUMENT_TWIN = "document"
    INTERACTION_TWIN = "interaction"


class CapabilityStatus(StrEnum):
    """Whether a capability can currently be invoked."""

    AVAILABLE = "available"
    PLANNED = "planned"


class RouteReason(StrEnum):
    """Auditable reason for a coordinator decision."""

    EXPLICIT_SELECTION = "explicit_selection"
    EXPLICIT_COMMAND = "explicit_command"
    ATTACHMENT_TYPE = "attachment_type"
    AMBIGUOUS_ATTACHMENTS = "ambiguous_attachments"
    UNSUPPORTED_ATTACHMENT = "unsupported_attachment"
    NEEDS_CAPABILITY = "needs_capability"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """User-facing metadata without interface or workflow dependencies."""

    capability_id: CapabilityId
    display_name: str
    description: str
    command: str
    accepted_extensions: tuple[str, ...]
    status: CapabilityStatus = CapabilityStatus.AVAILABLE


@dataclass(frozen=True)
class RouteRequest:
    """Normalized conversational input consumed by the coordinator."""

    message: str = ""
    explicit_capability: CapabilityId | None = None
    attachment_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    """A coordinator outcome; workflow execution remains outside this contract."""

    capability_id: CapabilityId | None
    status: CapabilityStatus | None
    reason: RouteReason
    needs_confirmation: bool
    user_message: str

    @property
    def can_invoke(self) -> bool:
        return (
            self.capability_id is not None and self.status is CapabilityStatus.AVAILABLE
        )
