"""Cycle policy (pure). Multi-table cycles require an explicit strategy."""
from __future__ import annotations

from synth_platform.domain.relational.dag import GraphPlan
from synth_platform.errors import UnsupportedCycleError


def enforce_cycle_policy(plan: GraphPlan, allow_deferred: bool = False) -> None:
    if plan.unsupported_cycles and not allow_deferred:
        raise UnsupportedCycleError(
            f"multi-table cycle(s) with no deferred-key strategy: "
            f"{plan.unsupported_cycles}"
        )
