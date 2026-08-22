"""synth_platform: clean-architecture source-free synthetic data platform."""
from synth_platform.application.dto.commands import (
    GenerationRequest, TrainingRequest, ValidationPolicy,
)
from synth_platform.interfaces.sdk.client import SyntheticDataPlatform

__all__ = ["SyntheticDataPlatform", "TrainingRequest", "GenerationRequest", "ValidationPolicy"]
__version__ = "3.0.0"
