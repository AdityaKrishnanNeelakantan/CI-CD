"""Smoke tests for the banking/KYC extension.

These tests are intentionally small and fast. They verify that the added
banking schema config, smart value generators, FK generation, validators,
quality checks, structured Oracle report, and CLI demo path work together.
"""

import pandas as pd
from synth_platform.engine.validation.schema.quality import check_quality
from synth_platform.engine.validation.schema.reporting import build_validation_report
from synth_platform.engine.inference.schema.schema import BankingConfig, Column, Relationship, SchemaConfig, Table
from synth_platform.engine.generation.schema.simulator import DataSimulator
from synth_platform.engine.validation.schema.validation import validate_data


def _banking_schema(country="US"):
    return SchemaConfig(
        name="banking_pytest_smoke",
        domain="banking",
        seed=123,
        banking=BankingConfig(
            country=country,
            currency="INR" if country == "IN" else "USD",
            routing_number_format="ifsc" if country == "IN" else "aba",
            account_number_digits=12,
        ),
        tables=[
            Table(name="customers", row_count=12),
            Table(name="accounts", row_count=18),
            Table(name="transactions", row_count=40),
        ],
        columns={
            "customers": [
                Column(name="customer_id", type="int", unique=True, distribution_params={"min": 1, "max": 10000}),
                Column(name="ssn", type="ssn", unique=True),
                Column(name="aadhaar", type="aadhaar", unique=True),
            ],
            "accounts": [
                Column(name="account_id", type="int", unique=True, distribution_params={"min": 1, "max": 10000}),
                Column(name="customer_id", type="foreign_key"),
                Column(name="account_number", type="account_number", unique=True),
                Column(name="routing_number", type="routing_number" if country != "IN" else "ifsc_code"),
                Column(name="balance", type="money", distribution_params={"min": 0, "max": 10000, "decimals": 2}),
            ],
            "transactions": [
                Column(name="transaction_id", type="int", unique=True, distribution_params={"min": 1, "max": 10000}),
                Column(name="account_id", type="foreign_key"),
                Column(name="amount", type="money", distribution_params={"min": 1, "max": 500, "decimals": 2}),
            ],
        },
        relationships=[
            Relationship(parent_table="customers", parent_key="customer_id", child_table="accounts", child_key="customer_id"),
            Relationship(parent_table="accounts", parent_key="account_id", child_table="transactions", child_key="account_id"),
        ],
    )


def _generate_tables(schema):
    simulator = DataSimulator(schema)
    tables = {}
    for table_name, batch in simulator.generate_all():
        tables[table_name] = batch if table_name not in tables else pd.concat([tables[table_name], batch], ignore_index=True)
    return tables


def test_banking_schema_generation_validation_quality_and_validation_report():
    schema = _banking_schema(country="US")
    tables = _generate_tables(schema)

    assert set(tables) == {"customers", "accounts", "transactions"}
    assert len(tables["customers"]) == 12
    assert len(tables["accounts"]) == 18
    assert len(tables["transactions"]) == 40

    assert tables["accounts"]["account_number"].astype(str).str.len().eq(12).all()
    assert tables["customers"]["ssn"].astype(str).str.match(r"^\d{3}-\d{2}-\d{4}$").all()
    assert tables["accounts"]["routing_number"].astype(str).str.match(r"^\d{9}$").all()

    assert set(tables["accounts"]["customer_id"]).issubset(set(tables["customers"]["customer_id"]))
    assert set(tables["transactions"]["account_id"]).issubset(set(tables["accounts"]["account_id"]))

    validation_report = validate_data(tables, schema)
    assert validation_report.is_clean

    quality_report = check_quality(tables, relationships=schema.relationships, schema=schema)
    assert quality_report.passed

    validation_report_doc = build_validation_report(tables, schema, validation_report=validation_report, quality_report=quality_report)
    assert validation_report_doc["mvp_report"] == "validation"
    assert validation_report_doc["passed"] is True
    assert validation_report_doc["summary"]["hard_guarantees_passed"] is True
    assert validation_report_doc["summary"]["total_rows"] == 70
    assert validation_report_doc["guarantees"]["row_count_fulfillment"]["passed"] is True


def test_india_banking_generates_ifsc_and_aadhaar_formats():
    schema = _banking_schema(country="IN")
    tables = _generate_tables(schema)

    assert tables["accounts"]["routing_number"].astype(str).str.match(r"^[A-Z]{4}0[A-Z0-9]{6}$").all()
    assert tables["customers"]["aadhaar"].astype(str).str.match(r"^\d{4} \d{4} \d{4}$").all()


def test_safe_banking_case_note_uses_locale_fallback_without_pii(monkeypatch):
    """The one LLM-backed banking text column must never leak numbers or PII-like text."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    schema = _banking_schema(country="IN")
    schema.columns["transactions"].append(
        Column(
            name="case_note",
            type="text",
            distribution_params={
                "text_type": "banking_case_note",
                "llm_text": True,
                "llm_enabled": True,
                "country": "IN",
                "context": "UPI and card transaction support review",
            },
        )
    )

    tables = _generate_tables(schema)
    notes = tables["transactions"]["case_note"].astype(str)

    assert len(notes) == 40
    assert notes.str.len().gt(24).all()
    assert not notes.str.contains(r"\d", regex=True).any()
    assert not notes.str.contains(r"@", regex=True).any()
    assert not notes.str.contains(r"phone|email|aadhaar|ssn|account number|routing number", case=False, regex=True).any()
    assert notes.str.contains(r"upi|transfer|merchant|settlement|mobile|wallet|bank", case=False, regex=True).any()
