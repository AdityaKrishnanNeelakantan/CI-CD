"""Tests for pre-input Deterministic Faker Masker + context field kinds."""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.context_fields import (
    ContextFieldKind,
    make_faker,
    render_context_value,
    render_identifier_value,
    resolve_context_kind,
)
from synth_platform.engine.common.database.privacy.deterministic_masker import DeterministicFakerMasker

pytestmark = pytest.mark.unit


def _approved(semantic_type: str) -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved", "physical_type": "TEXT"}


def test_resolve_context_kind_covers_all_context_aware_fields():
    assert resolve_context_kind("full_name", "person_name") is ContextFieldKind.PERSON_NAME
    assert resolve_context_kind("email", "email") is ContextFieldKind.EMAIL
    assert resolve_context_kind("phone", "phone_number") is ContextFieldKind.PHONE
    assert resolve_context_kind("street_address", "free_text") is ContextFieldKind.ADDRESS
    assert resolve_context_kind("city", "free_text") is ContextFieldKind.CITY
    assert resolve_context_kind("state", "free_text") is ContextFieldKind.STATE
    assert resolve_context_kind("postal_code", "free_text") is ContextFieldKind.POSTAL_CODE
    assert resolve_context_kind("country", "free_text") is ContextFieldKind.COUNTRY
    assert resolve_context_kind("company", "free_text") is ContextFieldKind.COMPANY
    assert resolve_context_kind("merchant_name", "free_text") is ContextFieldKind.COMPANY
    assert resolve_context_kind("customer_id", "identifier") is ContextFieldKind.IDENTIFIER
    assert resolve_context_kind("record_uuid", "identifier") is ContextFieldKind.UUID
    assert resolve_context_kind("age", "numerical") is ContextFieldKind.PASSTHROUGH
    assert resolve_context_kind("segment", "category") is ContextFieldKind.PASSTHROUGH


def test_render_identifier_value_uses_column_aware_faker_patterns():
    faker = make_faker(seed=11)
    assert render_identifier_value(faker, "customer_id").startswith("CUST-")
    assert render_identifier_value(faker, "account_id").startswith("ACC-")
    assert render_identifier_value(faker, "loan_id").startswith("LOAN-")
    assert render_identifier_value(faker, "transaction_id").startswith("TXN-")
    card = render_identifier_value(faker, "card_id")
    assert not card.startswith("SYN-")
    assert card.isdigit() and len(card) >= 13
    via_kind = str(render_context_value(ContextFieldKind.IDENTIFIER, faker, "account_id"))
    assert via_kind.startswith("ACC-")
    assert not via_kind.startswith("SYN-")


def test_deterministic_masker_is_stable_and_strips_raw_context_values():
    df = pd.DataFrame(
        {
            "full_name": ["Ada Lovelace", "Alan Turing", "Ada Lovelace"],
            "email": ["ada@example.com", "alan@example.com", "ada@example.com"],
            "phone": ["+1-202-555-0100", "+1-202-555-0101", "+1-202-555-0100"],
            "street_address": ["1 Main St", "2 Oak Ave", "1 Main St"],
            "city": ["London", "Manchester", "London"],
            "company": ["Analytical Engines", "Bletchley Park", "Analytical Engines"],
            "customer_id": ["CUST-0001", "CUST-0002", "CUST-0001"],
            "age": [36, 41, 36],
            "segment": ["gold", "silver", "gold"],
        }
    )
    contract = {
        "columns": {
            "full_name": _approved("person_name"),
            "email": _approved("email"),
            "phone": _approved("phone_number"),
            "street_address": _approved("free_text"),
            "city": _approved("free_text"),
            "company": _approved("free_text"),
            "customer_id": _approved("identifier"),
            "age": _approved("numerical"),
            "segment": _approved("category"),
        }
    }
    masker = DeterministicFakerMasker(seed=42)
    masked_a, report_a = masker.mask_table(df, "customers", contract)
    masked_b, _report_b = masker.mask_table(df, "customers", contract)

    # Determinism: same seed + same inputs => identical masked frame.
    pd.testing.assert_frame_equal(masked_a, masked_b)

    # Same source value => same masked stand-in.
    assert masked_a.loc[0, "full_name"] == masked_a.loc[2, "full_name"]
    assert masked_a.loc[0, "email"] == masked_a.loc[2, "email"]

    # Raw production values must not survive.
    assert "Ada Lovelace" not in set(masked_a["full_name"])
    assert "ada@example.com" not in set(masked_a["email"])
    assert "+1-202-555-0100" not in set(masked_a["phone"])
    assert "1 Main St" not in set(masked_a["street_address"])
    assert "Analytical Engines" not in set(masked_a["company"])
    assert "CUST-0001" not in set(masked_a["customer_id"])
    # Identifiers use Faker domain patterns — never SYN- placeholders.
    assert not any(str(v).startswith("SYN-") for v in masked_a["customer_id"])
    assert all(str(v).startswith("CUST-") for v in masked_a["customer_id"])

    # Statistical columns passthrough.
    assert list(masked_a["age"]) == [36, 41, 36]
    assert list(masked_a["segment"]) == ["gold", "silver", "gold"]

    # Masked values look realistic / non-placeholder.
    assert all("@" in str(v) for v in masked_a["email"])
    assert not any(str(v).startswith("NAME-") for v in masked_a["full_name"])
    assert not any(str(v).startswith("CITY-") for v in masked_a["city"])
    assert "full_name" in report_a["context_aware_columns"]
    assert "age" not in report_a["context_aware_columns"]
