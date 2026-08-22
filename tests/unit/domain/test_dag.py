from synth_platform.domain.relational.cycles import enforce_cycle_policy
from synth_platform.domain.relational.dag import build_graph_plan
from synth_platform.domain.schema.models import ForeignKey
from synth_platform.errors import UnsupportedCycleError
import pytest


def _fk(p, c):
    return ForeignKey(parent_table=p, parent_column="id", child_table=c, child_column=f"{p}_id")


def test_acyclic_levels_parent_before_child():
    plan = build_graph_plan(["users", "accounts"], [_fk("users", "accounts")])
    assert plan.order.index("users") < plan.order.index("accounts")
    assert not plan.unsupported_cycles


def test_self_reference_detected_not_cycle():
    plan = build_graph_plan(["employees"], [_fk("employees", "employees")])
    assert plan.self_references == ["employees"]
    assert not plan.unsupported_cycles
    enforce_cycle_policy(plan)  # allowed


def test_multi_table_cycle_is_unsupported():
    fks = [_fk("a", "b"), _fk("b", "a")]
    plan = build_graph_plan(["a", "b"], fks)
    assert plan.unsupported_cycles
    with pytest.raises(UnsupportedCycleError):
        enforce_cycle_policy(plan)
