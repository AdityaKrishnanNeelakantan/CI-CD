from __future__ import annotations

import pytest

from synth_platform.engine.generation.database.schema_graph import build_schema_graph

pytestmark = pytest.mark.unit


def _fk(column, references_table, references_column):
    return {"column": column, "references_table": references_table, "references_column": references_column}


def _table(primary_key, foreign_keys=None):
    return {"primary_key": primary_key, "foreign_keys": foreign_keys or []}


def test_two_table_parent_child_orders_parent_first():
    contract = {
        "tables": {
            "orders": {"primary_key": ["order_id"], "foreign_keys": [_fk("customer_id", "customers", "customer_id")]},
            "customers": _table(["customer_id"]),
        }
    }
    graph = build_schema_graph(contract)
    assert graph["generation_order"] == ["customers", "orders"]
    assert graph["edges"] == [{"parent": "customers", "child": "orders", "parent_key": "customer_id", "child_key": "customer_id"}]


def test_three_table_chain_orders_correctly():
    contract = {
        "tables": {
            "transaction": {"primary_key": ["id"], "foreign_keys": [_fk("account_id", "account", "id")]},
            "account": {"primary_key": ["id"], "foreign_keys": [_fk("customer_id", "customer", "id")]},
            "customer": _table(["id"]),
        }
    }
    graph = build_schema_graph(contract)
    assert graph["generation_order"] == ["customer", "account", "transaction"]


def test_table_with_no_foreign_keys_and_no_dependents_is_independent():
    contract = {"tables": {"lookup": _table(["code"])}}
    graph = build_schema_graph(contract)
    assert graph["generation_order"] == ["lookup"]
    assert graph["edges"] == []


def test_two_foreign_keys_to_the_same_parent_do_not_double_count_dependency():
    contract = {
        "tables": {
            "shipment": {
                "primary_key": ["id"],
                "foreign_keys": [
                    _fk("origin_id", "warehouse", "id"),
                    _fk("dest_id", "warehouse", "id"),
                ],
            },
            "warehouse": _table(["id"]),
        }
    }
    graph = build_schema_graph(contract)
    assert graph["generation_order"] == ["warehouse", "shipment"]
    assert len(graph["edges"]) == 2


def test_self_referencing_foreign_key_is_excluded_from_cycle_detection():
    contract = {
        "tables": {
            "employees": {
                "primary_key": ["employee_id"],
                "foreign_keys": [_fk("manager_id", "employees", "employee_id")],
            }
        }
    }
    graph = build_schema_graph(contract)
    assert graph["generation_order"] == ["employees"]
    assert graph["edges"] == []
    assert len(graph["self_referencing_edges"]) == 1


def test_cross_table_cycle_is_handled_via_scc_condensation():
    contract = {
        "tables": {
            "a": {"primary_key": ["id"], "foreign_keys": [_fk("b_id", "b", "id")]},
            "b": {"primary_key": ["id"], "foreign_keys": [_fk("a_id", "a", "id")]},
        }
    }
    graph = build_schema_graph(contract)
    assert set(graph["generation_order"]) == {"a", "b"}
    assert len(graph["sccs"]) == 1
    assert set(graph["sccs"][0]) == {"a", "b"}
    assert graph["condensed_order"] == [["a", "b"]]


def test_foreign_key_referencing_a_table_outside_the_contract_is_ignored():
    contract = {
        "tables": {
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [_fk("customer_id", "customers_not_in_contract", "customer_id")],
            }
        }
    }
    graph = build_schema_graph(contract)
    assert graph["generation_order"] == ["orders"]
    assert graph["edges"] == []
