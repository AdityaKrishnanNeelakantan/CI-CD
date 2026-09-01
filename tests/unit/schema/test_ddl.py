"""Regression tests for SQL DDL → SchemaConfig parsing."""

from __future__ import annotations

import warnings

from synth_platform.engine.inference.schema.ddl import from_ddl


def test_primary_key_id_columns_are_not_typed_as_foreign_key():
    ddl = """
    CREATE TABLE customers(
     customer_id VARCHAR(20) PRIMARY KEY,
     email VARCHAR(100)
    );
    CREATE TABLE accounts(
     account_id VARCHAR(20) PRIMARY KEY,
     customer_id VARCHAR(20) NOT NULL,
     balance_usd DECIMAL(12,2),
     FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schema = from_ddl(ddl, infer_fks=True, default_rows=10)

    customers = {c.name: c for c in schema.columns["customers"]}
    accounts = {c.name: c for c in schema.columns["accounts"]}

    assert customers["customer_id"].type != "foreign_key"
    assert accounts["account_id"].type != "foreign_key"
    assert accounts["customer_id"].type == "foreign_key"
    assert any(
        r.parent_table == "customers"
        and r.child_table == "accounts"
        and r.parent_key == "customer_id"
        for r in schema.relationships
    )


def test_dropped_inferred_fks_do_not_leave_orphan_foreign_key_types():
    ddl = """
    CREATE TABLE widgets(
      widget_id VARCHAR(20) PRIMARY KEY,
      label VARCHAR(50)
    );
    """
    with warnings.catch_warnings(record=True):
        schema = from_ddl(ddl, infer_fks=True)
    widget_id = next(c for c in schema.columns["widgets"] if c.name == "widget_id")
    assert widget_id.type != "foreign_key"
