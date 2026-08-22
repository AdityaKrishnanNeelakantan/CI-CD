"""Deterministic learning-plan compiler. Every choice is explainable."""
from __future__ import annotations

from synth_platform.domain.planning.models import LearningPlan, LearningTask
from synth_platform.domain.profiling.models import DataProfile

_STRATEGY = {
    "numeric": ("statistical", "empirical_quantile_inverse", "numeric -> inverse-CDF"),
    "categorical": ("statistical", "weighted_categorical", "low-cardinality -> weighted sampling"),
    "datetime": ("statistical", "empirical_quantile_inverse", "datetime -> epoch quantiles"),
    "identifier": ("statistical", "deterministic_key", "unique id -> key generator"),
    "pii": ("statistical", "synthetic_generator", "direct PII -> synthetic, no raw values"),
    "text": ("statistical", "placeholder_text", "free text -> placeholder"),
}


def compile_plan(profile: DataProfile) -> LearningPlan:
    tasks = []
    for tprof in profile.tables.values():
        for col, cprof in tprof.columns.items():
            backend, strat, reason = _STRATEGY.get(
                cprof.physical_type, ("statistical", "weighted_categorical", "fallback"))
            tasks.append(LearningTask(table=tprof.table_name, column=col, backend=backend,
                                      strategy=strat, reason=reason, decision_basis="heuristic"))
    return LearningPlan(tasks=tasks)


def backend_registry() -> dict[str, str]:
    """SCAFFOLD: statistical implemented inline; forest/neural are plugins (B14)."""
    return {"statistical": "engine.generation.column_generator"}
