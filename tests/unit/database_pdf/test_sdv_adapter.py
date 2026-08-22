from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.sdv_adapter import SDVSynthesizerAdapter
from synth_platform.engine.training.database.base import SynthesisError

pytestmark = pytest.mark.unit


def approved(semantic_type: str) -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved"}


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(20)],
            "age": [20 + (i % 40) for i in range(20)],
            "status": ["active", "inactive", "pending"] * 6 + ["active", "active"],
            "email": [f"user{i}@example.com" for i in range(20)],
            "bio": [f"Free text bio number {i} with unique wording." for i in range(20)],
        }
    )


def sample_contract() -> dict:
    return {
        "primary_key": ["customer_id"],
        "columns": {
            "customer_id": approved("identifier"),
            "age": approved("numerical"),
            "status": approved("category"),
            "email": approved("email"),
            "bio": approved("free_text"),
        },
    }


def test_repeating_identifier_column_is_trained_without_uniqueness_error():
    """Regression test: a foreign-key-shaped column (semantic_type
    "identifier" but legitimately repeating, e.g. orders.customer_id) must
    not be declared an SDV alternate key, which requires uniqueness -
    caught by a real two-table e2e run, not a hand-written fixture.
    """
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": ["c1", "c1", "c2", "c3", "c2"],  # repeats - not unique
            "amount": [10.0, 20.0, 15.0, 5.0, 8.0],
        }
    )
    contract = {
        "primary_key": ["order_id"],
        "columns": {
            "order_id": approved("identifier"),
            "customer_id": approved("identifier"),
            "amount": approved("numerical"),
        },
    }
    adapter = SDVSynthesizerAdapter()
    evidence = adapter.fit(df, "orders", contract, seed=1)  # must not raise
    assert evidence["primary_key"] == "order_id"
    assert "customer_id" not in evidence["alternate_keys"]

    generated = adapter.sample(10, seed=1)
    assert len(generated) == 10
    assert generated["order_id"].is_unique


def test_fit_returns_evidence_with_correct_trained_and_excluded_columns():
    adapter = SDVSynthesizerAdapter()
    evidence = adapter.fit(sample_df(), "customers", sample_contract(), seed=1)

    # free_text ("bio") is present in output (sdtype="text", pii=True -
    # Faker/placeholder-generated, never statistically modelled) rather
    # than excluded outright - a NOT NULL free_text column being silently
    # dropped from generated output broke the target-database write with
    # a raw sqlite3.IntegrityError, a confirmed real bug found running
    # this pipeline against a real medical fixture (dataset_metadata.value).
    assert set(evidence["trained_columns"]) == {"customer_id", "age", "status", "email", "bio"}
    assert evidence["primary_key"] == "customer_id"
    assert evidence["row_count"] == 20
    assert not evidence["excluded_columns"]


def test_sample_returns_requested_row_count_and_only_trained_columns():
    adapter = SDVSynthesizerAdapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(15, seed=1)

    assert len(generated) == 15
    assert set(generated.columns) == {"customer_id", "age", "status", "email", "bio"}
    assert generated["bio"].notna().all()


def test_free_text_column_never_leaks_real_values():
    adapter = SDVSynthesizerAdapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(20, seed=1)
    assert not set(generated["bio"]) & set(sample_df()["bio"])


def test_synthetic_identifiers_are_unique_and_not_the_real_values():
    adapter = SDVSynthesizerAdapter()
    real_df = sample_df()
    adapter.fit(real_df, "customers", sample_contract(), seed=1)
    generated = adapter.sample(30, seed=1)

    assert generated["customer_id"].is_unique
    assert set(generated["customer_id"]) & set(real_df["customer_id"]) == set()


def test_synthetic_emails_are_fake_and_never_the_real_training_values():
    adapter = SDVSynthesizerAdapter()
    real_df = sample_df()
    adapter.fit(real_df, "customers", sample_contract(), seed=1)
    generated = adapter.sample(30, seed=1)

    assert set(generated["email"]) & set(real_df["email"]) == set()


def test_synthetic_categories_stay_within_the_observed_value_set():
    adapter = SDVSynthesizerAdapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(50, seed=1)
    assert set(generated["status"]).issubset({"active", "inactive", "pending"})


def test_same_seed_is_deterministic_for_modeled_columns():
    """Reproducibility applies to statistically-modeled columns (numerical/
    category/boolean/datetime). Identifier and PII (email) columns are
    Faker-generated with their own independent RNG state that SDV does not
    expose a public reset for - and are meant to be fresh each call by
    design anyway, so exact reproduction of a "fake" id/email isn't a
    property worth chasing. Matches the spec's own "where supported"
    qualifier on this requirement.
    """
    adapter = SDVSynthesizerAdapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)

    sample_a1 = adapter.sample(10, seed=42)
    sample_a2 = adapter.sample(10, seed=42)
    sample_b = adapter.sample(10, seed=999)

    modeled_columns = ["age", "status"]
    assert sample_a1[modeled_columns].equals(sample_a2[modeled_columns])
    assert not sample_a1[modeled_columns].equals(sample_b[modeled_columns])


def test_same_seed_is_fully_deterministic_without_id_or_pii_columns():
    contract = {
        "primary_key": [],
        "columns": {"age": approved("numerical"), "status": approved("category")},
    }
    df = sample_df()[["age", "status"]]

    adapter = SDVSynthesizerAdapter()
    adapter.fit(df, "customers", contract, seed=1)

    sample_a1 = adapter.sample(10, seed=42)
    sample_a2 = adapter.sample(10, seed=42)
    sample_b = adapter.sample(10, seed=999)

    assert sample_a1.equals(sample_a2)
    assert not sample_a1.equals(sample_b)


def test_sample_before_fit_raises():
    adapter = SDVSynthesizerAdapter()
    with pytest.raises(SynthesisError):
        adapter.sample(5)


def test_category_overrides_raises_type_error_not_accepted_by_this_adapter():
    """Regression test: SynthesizerAdapter.sample()'s documented contract
    (src/synthesis/base.py) is that an adapter unable to support
    category_overrides simply omits the parameter, so callers get a plain
    TypeError - not an adapter-specific exception type.
    """
    df = sample_df()
    adapter = SDVSynthesizerAdapter()
    adapter.fit(df, "customers", sample_contract(), seed=1)
    with pytest.raises(TypeError):
        adapter.sample(5, category_overrides={"status": {"active": 2}})


def test_fit_with_no_trainable_columns_raises():
    # free_text is now trainable (generated as sdtype="text", pii=True -
    # see test_fit_returns_evidence_with_correct_trained_and_excluded_columns),
    # so an unmapped semantic type is needed to exercise "no trainable
    # columns" here instead.
    adapter = SDVSynthesizerAdapter()
    contract = {
        "primary_key": [],
        "columns": {"bio": approved("totally_unmapped_semantic_type")},
    }
    with pytest.raises(SynthesisError):
        adapter.fit(sample_df()[["bio"]], "t", contract, seed=1)


def test_fit_with_unapproved_column_excludes_it():
    adapter = SDVSynthesizerAdapter()
    contract = sample_contract()
    contract["columns"]["status"] = {"semantic_type": "category", "inference_status": "review_required"}

    evidence = adapter.fit(sample_df(), "customers", contract, seed=1)
    assert "status" not in evidence["trained_columns"]
    reasons = {e["column"]: e["reason"] for e in evidence["excluded_columns"]}
    assert reasons["status"] == "inference_status=review_required"


def test_fit_raises_when_contract_column_missing_from_dataframe():
    adapter = SDVSynthesizerAdapter()
    contract = sample_contract()
    contract["columns"]["missing_col"] = approved("numerical")
    with pytest.raises(SynthesisError):
        adapter.fit(sample_df(), "customers", contract, seed=1)


def test_save_and_load_round_trip_in_a_fresh_instance(tmp_path: Path):
    adapter = SDVSynthesizerAdapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.pkl"
    adapter.save(model_path)

    loaded = SDVSynthesizerAdapter.load(model_path)
    generated = loaded.sample(5, seed=1)
    assert len(generated) == 5
    assert set(generated.columns) == {"customer_id", "age", "status", "email", "bio"}


def test_save_before_fit_raises(tmp_path: Path):
    adapter = SDVSynthesizerAdapter()
    with pytest.raises(SynthesisError):
        adapter.save(tmp_path / "model.pkl")


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(SynthesisError):
        SDVSynthesizerAdapter.load(tmp_path / "does_not_exist.pkl")
