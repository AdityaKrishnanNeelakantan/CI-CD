"""Compile the deterministic learning plan (engine planner)."""
from __future__ import annotations

from synth_platform.domain.planning.models import LearningPlan
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.engine.training.service import compile_plan


def compile_learning_plan(profile: DataProfile) -> LearningPlan:
    return compile_plan(profile)
