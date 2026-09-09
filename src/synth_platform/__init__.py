"""synth_platform: clean-architecture source-free synthetic data platform."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "GenerationRequest",
    "SyntheticDataPlatform",
    "TrainingRequest",
    "ValidationPolicy",
]
__version__ = "3.0.2"

_LAZY_EXPORTS = {
    "SyntheticDataPlatform": (
        "synth_platform.interfaces.sdk.client",
        "SyntheticDataPlatform",
    ),
    "TrainingRequest": (
        "synth_platform.application.dto.commands",
        "TrainingRequest",
    ),
    "GenerationRequest": (
        "synth_platform.application.dto.commands",
        "GenerationRequest",
    ),
    "ValidationPolicy": (
        "synth_platform.application.dto.commands",
        "ValidationPolicy",
    ),
}


def __getattr__(name: str) -> Any:
    """Load public runtime objects only when callers request them."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))


if TYPE_CHECKING:
    from synth_platform.application.dto.commands import (
        GenerationRequest,
        TrainingRequest,
        ValidationPolicy,
    )
    from synth_platform.interfaces.sdk.client import SyntheticDataPlatform
