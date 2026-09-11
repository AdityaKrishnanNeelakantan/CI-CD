"""Deterministic routing policy for the unified synthetic-twin entry point."""

from __future__ import annotations

from pathlib import PurePath

from synth_platform.application.coordinator.contracts import (
    CapabilityDescriptor,
    CapabilityId,
    CapabilityStatus,
    RouteDecision,
    RouteReason,
    RouteRequest,
)

CAPABILITIES: tuple[CapabilityDescriptor, ...] = (
    CapabilityDescriptor(
        capability_id=CapabilityId.SCHEMA_TWIN,
        display_name="Schema Twin",
        description="Create validated relational data from a schema or description.",
        command="/schema",
        accepted_extensions=(".sql", ".json", ".yaml", ".yml"),
    ),
    CapabilityDescriptor(
        capability_id=CapabilityId.DATABASE_TWIN,
        display_name="Database Twin",
        description="Learn from SQLite data and create a portable synthetic twin.",
        command="/database",
        accepted_extensions=(".db", ".sqlite", ".sqlite3"),
    ),
    CapabilityDescriptor(
        capability_id=CapabilityId.DOCUMENT_TWIN,
        display_name="Document Twin",
        description="Create a validated synthetic PDF while preserving useful structure.",
        command="/document",
        accepted_extensions=(".pdf",),
    ),
    CapabilityDescriptor(
        capability_id=CapabilityId.INTERACTION_TWIN,
        display_name="Customer Interaction Twin",
        description="Create a structured, privacy-safe twin of an interaction.",
        command="/interaction",
        accepted_extensions=(".txt", ".log"),
    ),
)

_CAPABILITIES_BY_ID = {item.capability_id: item for item in CAPABILITIES}
_COMMANDS = {
    command: capability_id
    for capability_id, commands in {
        CapabilityId.SCHEMA_TWIN: ("/schema",),
        CapabilityId.DATABASE_TWIN: ("/database", "/db"),
        CapabilityId.DOCUMENT_TWIN: ("/document", "/pdf"),
        CapabilityId.INTERACTION_TWIN: ("/interaction", "/customer"),
    }.items()
    for command in commands
}
_EXTENSIONS = {
    extension: item.capability_id
    for item in CAPABILITIES
    for extension in item.accepted_extensions
}


def get_capability(capability_id: CapabilityId) -> CapabilityDescriptor:
    """Return registered metadata for a stable capability ID."""

    return _CAPABILITIES_BY_ID[capability_id]


def route_request(request: RouteRequest) -> RouteDecision:
    """Select a capability without generating data or invoking a workflow."""

    if request.explicit_capability is not None:
        return _decision_for(
            request.explicit_capability,
            RouteReason.EXPLICIT_SELECTION,
        )

    command = _leading_command(request.message)
    if command is not None:
        return _decision_for(command, RouteReason.EXPLICIT_COMMAND)

    if request.attachment_names:
        matched, unsupported = _attachment_candidates(request.attachment_names)
        if len(matched) == 1 and not unsupported:
            return _decision_for(next(iter(matched)), RouteReason.ATTACHMENT_TYPE)
        if len(matched) > 1 or (matched and unsupported):
            return RouteDecision(
                capability_id=None,
                status=None,
                reason=RouteReason.AMBIGUOUS_ATTACHMENTS,
                needs_confirmation=True,
                user_message=(
                    "The attachments map to different capabilities or include an unsupported "
                    "type. Choose Schema, Database, Document, or Interaction explicitly."
                ),
            )
        return RouteDecision(
            capability_id=None,
            status=None,
            reason=RouteReason.UNSUPPORTED_ATTACHMENT,
            needs_confirmation=True,
            user_message=(
                "I cannot route those attachment types yet. Choose a capability explicitly; "
                "the selected workflow will validate its input."
            ),
        )

    return RouteDecision(
        capability_id=None,
        status=None,
        reason=RouteReason.NEEDS_CAPABILITY,
        needs_confirmation=True,
        user_message=(
            "Choose Schema, Database, Document, or Interaction so I can hand your request "
            "to the correct specialized workflow."
        ),
    )


def _leading_command(message: str) -> CapabilityId | None:
    tokens = message.strip().lower().split(maxsplit=1)
    if not tokens:
        return None
    return _COMMANDS.get(tokens[0])


def _attachment_candidates(
    attachment_names: tuple[str, ...],
) -> tuple[set[CapabilityId], set[str]]:
    matched: set[CapabilityId] = set()
    unsupported: set[str] = set()
    for name in attachment_names:
        extension = PurePath(name).suffix.lower()
        capability_id = _EXTENSIONS.get(extension)
        if capability_id is None:
            unsupported.add(extension or "<no extension>")
        else:
            matched.add(capability_id)
    return matched, unsupported


def _decision_for(capability_id: CapabilityId, reason: RouteReason) -> RouteDecision:
    capability = get_capability(capability_id)
    if capability.status is CapabilityStatus.PLANNED:
        return RouteDecision(
            capability_id=capability_id,
            status=capability.status,
            reason=reason,
            needs_confirmation=False,
            user_message=(
                f"{capability.display_name} is represented in the target architecture but is "
                "not implemented in this repository yet. I will not substitute another workflow."
            ),
        )
    return RouteDecision(
        capability_id=capability_id,
        status=capability.status,
        reason=reason,
        needs_confirmation=False,
        user_message=(
            f"I'll hand this request to {capability.display_name}. The existing workflow "
            "continues to own generation, validation, and artifacts."
        ),
    )
