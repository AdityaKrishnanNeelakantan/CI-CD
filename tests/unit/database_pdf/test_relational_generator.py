from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.generation.database.relational_generator import (
    RelationalGenerationError,
    compute_fk_validity,
    generate_relational_dataset,
)
from synth_platform.engine.training.database.base import SynthesizerAdapter

pytestmark = pytest.mark.unit


class _FakeAdapter(SynthesizerAdapter):
    """Deterministic stand-in for a real SynthesizerAdapter - returns rows
    with a sequential primary key so FK-assignment behaviour can be
    checked exactly, without needing a real fitted statistical model.
    """

    model_type = "fake"
    file_extension = ".json"
    serialization_format = "json"

    def __init__(self, pk_column: str, other_columns: dict[str, list] | None = None) -> None:
        self._pk_column = pk_column
        self._other_columns = other_columns or {}

    def fit(self, df, table_name, table_contract, seed):
        return {}

    def sample(self, num_rows: int, seed: int | None = None) -> pd.DataFrame:
        data = {self._pk_column: [f"{self._pk_column}-{i}" for i in range(num_rows)]}
        for name, values in self._other_columns.items():
            data[name] = [values[i % len(values)] for i in range(num_rows)]
        return pd.DataFrame(data)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls("id")


def _schema_graph():
    return {
        "nodes": ["customers", "orders"],
        "edges": [{"parent": "customers", "child": "orders", "parent_key": "customer_id", "child_key": "customer_id"}],
        "self_referencing_edges": [],
        "generation_order": ["customers", "orders"],
    }


def test_child_foreign_keys_only_ever_reference_generated_parent_rows():
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    tables = generate_relational_dataset(_schema_graph(), adapters, {"customers": 5, "orders": 30}, seed=1)

    parent_ids = set(tables["customers"]["customer_id"])
    assert set(tables["orders"]["customer_id"]) <= parent_ids
    assert len(tables["orders"]) == 30
    assert len(tables["customers"]) == 5


def test_fk_validity_is_100_percent_by_construction():
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    tables = generate_relational_dataset(_schema_graph(), adapters, {"customers": 5, "orders": 30}, seed=1)
    graph = _schema_graph()
    validity = compute_fk_validity(tables, graph)
    assert validity["overall_fk_validity"] == 1.0
    edge_key = "orders.customer_id->customers.customer_id"
    assert validity["edges"][edge_key]["fk_validity"] == 1.0
    assert validity["edges"][edge_key]["total_count"] == 30


def test_same_seed_produces_identical_fk_assignment():
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    graph = _schema_graph()
    a = generate_relational_dataset(graph, adapters, {"customers": 5, "orders": 30}, seed=7)
    b = generate_relational_dataset(graph, adapters, {"customers": 5, "orders": 30}, seed=7)
    assert list(a["orders"]["customer_id"]) == list(b["orders"]["customer_id"])


def test_missing_adapter_for_a_table_raises():
    with pytest.raises(RelationalGenerationError):
        generate_relational_dataset(_schema_graph(), {"customers": _FakeAdapter("customer_id")}, {"customers": 5, "orders": 10})


def test_missing_row_count_for_a_table_raises():
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    with pytest.raises(RelationalGenerationError):
        generate_relational_dataset(_schema_graph(), adapters, {"customers": 5})


def test_zero_row_parent_table_raises_rather_than_silently_producing_invalid_fks():
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    with pytest.raises(RelationalGenerationError):
        generate_relational_dataset(_schema_graph(), adapters, {"customers": 0, "orders": 10})


def test_fk_validity_detects_a_genuine_mismatch():
    """If a caller hands compute_fk_validity a dataset where the FK column
    was NOT assigned via generate_relational_dataset (e.g. hand-built),
    it must catch a real mismatch rather than always reporting 1.0.
    """
    tables = {
        "customers": pd.DataFrame({"customer_id": ["a", "b"]}),
        "orders": pd.DataFrame({"order_id": [1, 2], "customer_id": ["a", "does-not-exist"]}),
    }
    validity = compute_fk_validity(tables, _schema_graph())
    assert validity["overall_fk_validity"] == 0.5


def test_category_overrides_are_forwarded_to_the_right_table_adapter():
    """generate_relational_dataset() must thread category_overrides_by_table
    through to the specific table's own sample() call - proving the
    "tweak without retraining" capability (src/synthesis/adapters/
    category_rebalancing.py) actually works through the full relational
    pipeline, not just when calling an adapter directly in isolation.
    """
    captured_calls: list[dict] = []

    class _RecordingAdapter(_FakeAdapter):
        def sample(self, num_rows, seed=None, category_overrides=None):
            captured_calls.append({"table": self._pk_column, "overrides": category_overrides})
            return super().sample(num_rows, seed=seed)

    adapters = {"customers": _RecordingAdapter("customer_id"), "orders": _RecordingAdapter("order_id")}
    generate_relational_dataset(
        _schema_graph(), adapters, {"customers": 5, "orders": 10}, seed=1,
        category_overrides_by_table={"customers": {"gender": {"male": 2, "female": 3}}},
    )

    customers_call = next(c for c in captured_calls if c["table"] == "customer_id")
    orders_call = next(c for c in captured_calls if c["table"] == "order_id")
    assert customers_call["overrides"] == {"gender": {"male": 2, "female": 3}}
    assert orders_call["overrides"] is None


def test_independent_tables_with_no_edges_all_generate_full_row_counts():
    graph = {
        "nodes": ["lookup"],
        "edges": [],
        "self_referencing_edges": [],
        "generation_order": ["lookup"],
    }
    tables = generate_relational_dataset(graph, {"lookup": _FakeAdapter("code")}, {"lookup": 10}, seed=1)
    assert len(tables["lookup"]) == 10
