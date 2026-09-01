"""synth_platform: clean-architecture source-free synthetic data platform."""
from synth_platform.application.dto.commands import (
    GenerationRequest, TrainingRequest, ValidationPolicy,
)

__all__ = ["SyntheticDataPlatform", "TrainingRequest", "GenerationRequest", "ValidationPolicy"]
__version__ = "3.0.2"


def __getattr__(name: str):
    if name == "SyntheticDataPlatform":
        from synth_platform.interfaces.sdk.client import SyntheticDataPlatform

        return SyntheticDataPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
