"""Thin product-capability coordination for conversational interfaces."""

from synth_platform.application.coordinator.contracts import (
    CapabilityDescriptor,
    CapabilityId,
    CapabilityStatus,
    RouteDecision,
    RouteReason,
    RouteRequest,
)
from synth_platform.application.coordinator.router import (
    CAPABILITIES,
    get_capability,
    route_request,
)

__all__ = [
    "CAPABILITIES",
    "CapabilityDescriptor",
    "CapabilityId",
    "CapabilityStatus",
    "RouteDecision",
    "RouteReason",
    "RouteRequest",
    "get_capability",
    "route_request",
]
