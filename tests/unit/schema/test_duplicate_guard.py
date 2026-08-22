import pandas as pd

from synth_platform.engine.generation.schema.duplicate_guard import enforce_duplicate_policy
from synth_platform.engine.validation.schema.quality import check_quality
from synth_platform.engine.inference.schema.schema import Column, Relationship, SchemaConfig, Table
from synth_platform.engine.validation.schema.validation import validate_data


def _schema():
    return SchemaConfig(
        name="Duplicate Guard Test",
        tables=[Table(name="customers", row_count=4), Table(name="orders", row_count=4)],
        columns={
            "customers": [
                Column(name="customer_id", type="int", unique=True),
                Column(name="full_name", type="text", distribution_params={"text_type": "name"}),
                Column(name="risk_tier", type="categorical", distribution_params={"choices": ["low", "high"]}),
            ],
            "orders": [
                Column(name="order_id", type="int", unique=True),
                Column(name="customer_id", type="foreign_key"),
                Column(name="amount", type="money", distribution_params={"min": 1, "max": 10}),
            ],
        },
        relationships=[
            Relationship(parent_table="customers", parent_key="customer_id", child_table="orders", child_key="customer_id")
        ],
        seed=42,
    )


def test_duplicate_guard_repairs_safe_business_fingerprints_without_touching_fk_keys():
    schema = _schema()
    tables = {
        "customers": pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4],
                "full_name": ["Alex Lee", "Alex Lee", "Maya Rao", "Maya Rao"],
                "risk_tier": ["high", "high", "low", "low"],
            }
        ),
        "orders": pd.DataFrame(
            {
                "order_id": [10, 11, 12, 13],
                "customer_id": [1, 1, 2, 2],
                "amount": [5.0, 5.0, 7.0, 8.0],
            }
        ),
    }

    repaired, report = enforce_duplicate_policy(tables, schema, seed=42)

    assert report.repaired_rows > 0
    assert repaired["orders"]["customer_id"].tolist() == [1, 1, 2, 2]
    assert repaired["customers"].duplicated(subset=["full_name", "risk_tier"]).sum() == 0
    assert validate_data(repaired, schema).has_errors is False


def test_validation_blocks_duplicate_unique_columns():
    schema = _schema()
    tables = {
        "customers": pd.DataFrame(
            {
                "customer_id": [1, 1, 3, 4],
                "full_name": ["A", "B", "C", "D"],
                "risk_tier": ["low", "low", "high", "high"],
            }
        ),
        "orders": pd.DataFrame(
            {
                "order_id": [10, 11, 12, 13],
                "customer_id": [1, 3, 4, 4],
                "amount": [5.0, 6.0, 7.0, 8.0],
            }
        ),
    }

    report = validate_data(tables, schema)

    assert report.has_errors
    assert any("duplicate values" in issue.message for issue in report.issues)


def test_quality_warns_on_repeated_text_but_not_normal_categories():
    schema = _schema()
    tables = {
        "customers": pd.DataFrame(
            {
                "customer_id": list(range(1, 101)),
                "full_name": ["Alex Lee"] * 30 + [f"Person {i}" for i in range(70)],
                "risk_tier": ["low"] * 80 + ["high"] * 20,
            }
        ),
        "orders": pd.DataFrame(
            {
                "order_id": list(range(1, 101)),
                "customer_id": list(range(1, 101)),
                "amount": [10.0] * 100,
            }
        ),
    }

    report = check_quality(tables, relationships=schema.relationships, schema=schema)

    assert any(i.category == "duplicate_density" and i.column == "full_name" for i in report.issues)
    assert not any(i.category == "duplicate_density" and i.column == "risk_tier" for i in report.issues)
