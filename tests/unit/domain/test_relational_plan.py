"""RelationalPlan is the single home of the relationship graph (W1 / A-04, A-09)."""
from synth_platform.domain.relational.dag import build_graph_plan
from synth_platform.domain.relational.models import GraphEdge, RelationalPlan
from synth_platform.domain.schema.models import ForeignKey


def _fk(p, c):
    return ForeignKey(parent_table=p, parent_column="id", child_table=c, child_column=f"{p}_id")


def test_plan_order_is_parents_before_children():
    plan = RelationalPlan(
        levels=[["a"], ["b"], ["c"]],
        edges=[GraphEdge(parent_table="a", parent_column="id", child_table="b", child_column="a_id"),
               GraphEdge(parent_table="b", parent_column="id", child_table="c", child_column="b_id")])
    assert plan.order == ["a", "b", "c"]


def test_edges_into_returns_inbound_fks():
    e1 = GraphEdge(parent_table="a", parent_column="id", child_table="c", child_column="a_id")
    e2 = GraphEdge(parent_table="b", parent_column="id", child_table="c", child_column="b_id")
    plan = RelationalPlan(levels=[["a", "b"], ["c"]], edges=[e1, e2])
    into_c = plan.edges_into("c")
    assert {e.child_column for e in into_c} == {"a_id", "b_id"}
    assert plan.edges_into("a") == []


def test_graph_plan_order_matches_tarjan_levels():
    # the canonical dag is the only toposort; plan mirrors its output
    g = build_graph_plan(["users", "accounts"], [_fk("users", "accounts")])
    assert g.order.index("users") < g.order.index("accounts")


def test_target_schema_has_no_duplicate_toposort():
    # A-09: the ad-hoc ordering was deleted; only the canonical dag remains.
    from synth_platform.domain.extraction import target_schema
    assert not hasattr(target_schema.TargetSchema, "topological_order")
