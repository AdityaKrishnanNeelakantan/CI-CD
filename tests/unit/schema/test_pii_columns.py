"""Tests for PII-aware column handling."""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.generation.schema.pii_columns import (
    column_requires_fresh_generation,
    generate_fresh_column,
    generate_fresh_column_batch,
    is_true_identifier_column,
    resolve_column_semantic,
    should_prefer_schema_engine,
    validate_no_pii_replay,
)


def test_resolve_column_semantic_for_pii_names():
    assert resolve_column_semantic("ssn") == "ssn"
    assert resolve_column_semantic("aadhaar") == "aadhaar"
    assert resolve_column_semantic("full_name") == "name"
    assert resolve_column_semantic("street_address") == "address"
    assert resolve_column_semantic("city") == "city"
    assert resolve_column_semantic("state") == "state"
    assert resolve_column_semantic("postal_code") == "postal_code"
    assert resolve_column_semantic("country") == "country"
    assert resolve_column_semantic("company_name") == "company"
    assert resolve_column_semantic("medical_record_number") == "national_id"


def test_should_prefer_schema_for_small_dataset():
    df = pd.DataFrame({"amount": [1, 2, 3], "segment": ["a", "b", "a"]})
    prefer, reason = should_prefer_schema_engine(df)
    assert prefer is True
    assert "200" in reason


def test_generate_fresh_column_never_replays_source_ssn():
    source = pd.Series(["111-22-3333", "444-55-6666"])
    fresh = generate_fresh_column("ssn", source, 5, seed=1)
    assert set(source.astype(str)).isdisjoint(set(fresh.astype(str)))


def test_validate_no_pii_replay_detects_overlap():
    source = pd.DataFrame({"ssn": ["111-22-3333", "444-55-6666"]})
    bad = pd.DataFrame({"ssn": ["111-22-3333", "999-88-7777"]})
    good = pd.DataFrame({"ssn": ["222-33-4444", "999-88-7777"]})
    assert validate_no_pii_replay(source, bad, ["ssn"])["passed"] is False
    assert validate_no_pii_replay(source, good, ["ssn"])["passed"] is True


def test_generate_fresh_names_exclude_protected_source_values():
    """Faker can emit common names that exist in source — generation must scrub them."""
    source = pd.Series(["John Smith"] * 50 + [f"Unique User {i}" for i in range(450)])
    fresh = generate_fresh_column_batch("full_name", source, 200, seed=42, semantic="name")
    source_keys = set(source.astype(str).str.strip().str.lower())
    syn_keys = set(fresh.dropna().astype(str).str.strip().str.lower())
    assert source_keys.isdisjoint(syn_keys)


def test_is_true_identifier_ignores_business_float_columns():
    balances = pd.Series([100.5, 200.0, 300.25, 400.0])
    assert is_true_identifier_column("Balance", balances) is False
    assert is_true_identifier_column("Credit Score", pd.Series([700, 710, 720])) is False
    assert is_true_identifier_column("customer_id", pd.Series([1, 2, 3, 4, 5])) is True


def test_phone_number_column_detected_as_pii():
    assert resolve_column_semantic("Phone Number") == "phone"
    assert column_requires_fresh_generation(
        "Phone Number", pd.Series(["+1-555-0100", "+1-555-0101"])
    )


def test_generate_fresh_emails_exclude_protected_source_values():
    source = pd.Series([f"user{i}@example.com" for i in range(300)])
    fresh = generate_fresh_column_batch("email", source, 150, seed=7, semantic="email")
    source_keys = set(source.astype(str).str.strip().str.lower())
    syn_keys = set(fresh.dropna().astype(str).str.strip().str.lower())
    assert source_keys.isdisjoint(syn_keys)


def test_contextual_sensitive_fields_use_faker_and_do_not_replay_source_values():
    cases = {
        "street_address": pd.Series(["1 Real Patient Way", "2 Real Patient Way"]),
        "city": pd.Series(["Sourceville", "Rawtown"]),
        "state": pd.Series(["AA", "BB"]),
        "postal_code": pd.Series(["12345", "67890"]),
        "country": pd.Series(["Source Country", "Raw Country"]),
        "company_name": pd.Series(["Source Hospital", "Raw Clinic"]),
    }
    for column_name, source in cases.items():
        fresh = generate_fresh_column_batch(column_name, source, 20, seed=11)
        source_keys = set(source.astype(str).str.strip().str.lower())
        syn_keys = set(fresh.dropna().astype(str).str.strip().str.lower())
        assert source_keys.isdisjoint(syn_keys), column_name
        assert fresh.notna().all(), column_name
        assert not any(str(v).startswith("synthetic_") for v in fresh), column_name


def test_string_identifier_uses_domain_identifier_renderer_not_placeholder():
    source = pd.Series([f"C{i:05d}" for i in range(20)])
    fresh = generate_fresh_column_batch("customer_id", source, 50, seed=12, semantic="id_like")
    assert fresh.nunique() == 50
    assert not set(source.astype(str).str.lower()) & set(fresh.astype(str).str.lower())
    assert not any(str(v).startswith("synthetic_customer_id_") for v in fresh)
    assert all(str(v).startswith("CUST-") for v in fresh)
