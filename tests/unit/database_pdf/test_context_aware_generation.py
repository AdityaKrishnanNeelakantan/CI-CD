"""Unit tests for context-aware Database Twin value generation helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.context_generators import (
    fake_free_text,
    generate_identifier_values,
    generate_pii_columns,
    learn_identifier_format,
    make_faker,
)
from synth_platform.engine.training.database.adapters.safe_copula_adapter import SafeCopulaSynthesizerAdapter

pytestmark = pytest.mark.unit


def approved(semantic_type: str, physical_type: str = "TEXT") -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved", "physical_type": physical_type}


def test_learn_identifier_format_preserves_prefix_and_width_without_source_ids():
    values = [f"CUST-{i:05d}" for i in range(1, 40)]
    fmt = learn_identifier_format(values)
    assert fmt is not None
    assert fmt["separator"] == "-"
    assert fmt["segments"][0] == {"type": "literal", "value": "CUST"}
    assert fmt["segments"][1] == {"type": "numeric", "width": 5}
    assert fmt["start_at"] == 0
    assert fmt["start_at_mode"] == "after_observed_range"
    assert fmt["synthetic_start_at"] > 39
    serialized = str(fmt)
    assert "CUST-00001" not in serialized
    assert "CUST-00039" not in serialized


def test_learn_identifier_format_rejects_legacy_syn_placeholders():
    syn_values = [f"SYN-ACCOUNT_ID-{i:08d}" for i in range(20)]
    assert learn_identifier_format(syn_values) is None


def test_generate_identifier_values_ignores_syn_format_and_uses_faker_patterns():
    syn_fmt = {
        "kind": "segmented",
        "separator": "-",
        "segments": [
            {"type": "literal", "value": "SYN"},
            {"type": "literal", "value": "ACCOUNT_ID"},
            {"type": "numeric", "width": 8},
        ],
        "start_at": 1,
    }
    rng = np.random.default_rng(0)
    values = generate_identifier_values(
        5, column_name="account_id", format_spec=syn_fmt, sequential=False, rng=rng
    )
    assert len(values) == 5
    assert not any(str(v).startswith("SYN-") for v in values)
    assert all(str(v).startswith("ACC-") for v in values)

    card_values = generate_identifier_values(
        3, column_name="card_id", format_spec=syn_fmt, sequential=False, rng=rng
    )
    assert not any(str(v).startswith("SYN-") for v in card_values)
    assert all(str(v).isdigit() and len(str(v)) >= 13 for v in card_values)


def test_generate_identifier_values_uses_learned_format_and_stays_unique():
    fmt = learn_identifier_format([f"CUST-{i:05d}" for i in range(20)])
    rng = np.random.default_rng(0)
    values = generate_identifier_values(
        15, column_name="customer_id", format_spec=fmt, sequential=True, rng=rng
    )
    assert len(set(values)) == 15
    assert all(re.fullmatch(r"CUST-\d{5,}", v) for v in values)
    assert not set(values) & {f"CUST-{i:05d}" for i in range(20)}


def test_identifier_generation_keeps_merchant_shape_but_moves_outside_source_range():
    source_values = [f"MER-{i}" for i in range(51960, 51970)]
    fmt = learn_identifier_format(source_values)
    rng = np.random.default_rng(0)
    values = generate_identifier_values(
        10, column_name="merchant_id", format_spec=fmt, sequential=True, rng=rng
    )
    assert len(set(values)) == 10
    assert all(re.fullmatch(r"MER-\d{5,}", v) for v in values)
    assert not set(values) & set(source_values)
    assert min(int(v.split("-")[1]) for v in values) > 51969


def test_non_primary_identifier_columns_use_learned_shape_and_safe_range():
    fmt = learn_identifier_format([f"ACC-00000-{i}" for i in range(181436, 181446)])
    rng = np.random.default_rng(0)
    data = generate_pii_columns(
        {
            "account_id": {
                "semantic_type": "identifier",
                "physical_type": "TEXT",
                "identifier_format": fmt,
            }
        },
        num_rows=10,
        primary_key=None,
        seed=7,
        rng=rng,
    )
    values = data["account_id"]
    assert len(set(values)) == 10
    assert all(re.fullmatch(r"ACC-00000-\d{6,}", str(v)) for v in values)
    assert not set(values) & {f"ACC-00000-{i}" for i in range(181436, 181446)}
    assert min(int(str(v).rsplit("-", 1)[1]) for v in values) > 181445


def test_compact_alphanumeric_identifiers_preserve_ssot_shape_without_replay():
    source_values = [
        "TXN00005HFWN2XNY3",
        "TXN00009897MR3AF0",
        "TXN0002DXC6LENBZN",
        "TXN0003B7WOPHYMBL",
        "TXN0003LEFVFA0Y95",
    ]
    fmt = learn_identifier_format(source_values)
    rng = np.random.default_rng(0)
    values = generate_identifier_values(
        20, column_name="transaction_id", format_spec=fmt, sequential=True, rng=rng
    )
    assert fmt is not None
    assert fmt["kind"] == "compact"
    assert len(set(values)) == 20
    assert all(re.fullmatch(r"TXN\d{4}[A-Z0-9]{10}", str(v)) for v in values)
    assert not set(values) & set(source_values)


def test_compact_account_and_merchant_identifiers_keep_prefix_and_tail_shape():
    source_by_column = {
        "account_id": [
            "ACCP1YZO9D6NWHS",
            "ACC39NBWQ11B0YW",
            "ACCVEP600JTRPHJ",
            "ACC9DARN097DNX",
        ],
        "merchant_id": [
            "MER1ECHZUV9E1WF",
            "MERBTO9OOBMKTWM",
            "MERI68JP0JLHMNF",
            "MERN72ZNR7VK3E6",
        ],
    }
    rng = np.random.default_rng(1)
    prefix_by_column = {"account_id": "ACC", "merchant_id": "MER"}
    for column_name, source_values in source_by_column.items():
        fmt = learn_identifier_format(source_values)
        values = generate_identifier_values(
            10, column_name=column_name, format_spec=fmt, sequential=True, rng=rng
        )
        prefix = prefix_by_column[column_name]
        assert fmt is not None
        assert fmt["kind"] == "compact"
        assert all(re.fullmatch(rf"{prefix}[A-Z0-9]{{12}}", str(v)) for v in values)
        assert not set(values) & set(source_values)


def test_sequential_identifiers_do_not_wrap_when_exceeding_learned_width():
    """Regression: width from cust-1..3 is 1; generating past 9 must stay unique."""
    fmt = learn_identifier_format(["cust-1", "cust-2", "cust-3"])
    assert fmt is not None
    assert fmt["start_at"] == 0
    rng = np.random.default_rng(0)
    values = generate_identifier_values(
        12, column_name="customer_id", format_spec=fmt, sequential=True, rng=rng
    )
    assert len(values) == 12
    assert len(set(values)) == 12
    assert not set(values) & {"cust-1", "cust-2", "cust-3"}


def test_person_name_and_email_are_context_consistent_and_not_placeholders():
    rng = np.random.default_rng(7)
    data = generate_pii_columns(
        {
            "name": {"semantic_type": "person_name", "physical_type": "TEXT"},
            "email": {"semantic_type": "email", "physical_type": "TEXT"},
        },
        num_rows=5,
        primary_key=None,
        seed=7,
        rng=rng,
    )
    assert not any(str(v).startswith("NAME-") for v in data["name"])
    assert all(" " in str(v) or str(v).isalpha() for v in data["name"])
    assert all("@" in str(v) for v in data["email"])
    for name, email in zip(data["name"], data["email"]):
        local = str(email).split("@", 1)[0]
        first = str(name).split()[0].lower()
        assert first[:3] in local


def test_free_text_address_and_company_use_realistic_renderers():
    faker = make_faker(seed=3)
    address = fake_free_text(faker, "street_address")
    company = fake_free_text(faker, "company")
    sentence = fake_free_text(faker, "notes")
    assert address and not address.startswith("STREET_ADDRESS-")
    assert company and not company.startswith("COMPANY-")
    assert sentence.endswith(".") or len(sentence.split()) > 1


def test_safe_adapter_artifact_roundtrip_keeps_context_aware_generation(tmp_path):
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(30)],
            "name": [f"Person {i} Example" for i in range(30)],
            "email": [f"person{i}@example.com" for i in range(30)],
            "phone": [f"+1-202-555-{i:04d}" for i in range(30)],
            "address": [f"{100 + i} Main Street" for i in range(30)],
            "company": [f"Acme {i}" for i in range(30)],
            "age": [20 + (i % 40) for i in range(30)],
            "segment": (["gold", "silver", "bronze"] * 10),
        }
    )
    contract = {
        "primary_key": ["customer_id"],
        "columns": {
            "customer_id": approved("identifier"),
            "name": approved("person_name"),
            "email": approved("email"),
            "phone": approved("phone_number"),
            "address": approved("free_text"),
            "company": approved("free_text"),
            "age": approved("numerical", "INTEGER"),
            "segment": approved("category"),
        },
    }
    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=2)
    adapter.fit(df, "customers", contract, seed=11)
    path = tmp_path / "model.json"
    adapter.save(path)
    loaded = SafeCopulaSynthesizerAdapter.load(path)

    assert "identifier_format" in loaded._pii_columns["customer_id"]
    generated = loaded.sample(20, seed=11)
    assert all(re.fullmatch(r"CUST-\d{5}", v) for v in generated["customer_id"])
    assert not set(generated["customer_id"]) & set(df["customer_id"])
    assert not any(str(v).startswith("NAME-") for v in generated["name"])
    assert not set(generated["name"]) & set(df["name"])
    assert not set(generated["email"]) & set(df["email"])
    assert generated["phone"].notna().all()
    assert generated["address"].notna().all()
    assert generated["company"].notna().all()
    assert generated["segment"].isin(["gold", "silver", "bronze"]).all()
    assert generated["age"].between(0, 120).all() or generated["age"].notna().all()

    again = loaded.sample(20, seed=11)
    pd.testing.assert_frame_equal(generated, again)
