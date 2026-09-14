from __future__ import annotations

import json
import pickletools
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.safe_copula_adapter import SafeCopulaSynthesizerAdapter
from synth_platform.engine.training.database.base import SynthesisError

pytestmark = pytest.mark.unit


def approved(semantic_type: str, physical_type: str = "TEXT") -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved", "physical_type": physical_type}


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


def small_adapter() -> SafeCopulaSynthesizerAdapter:
    # This project's own test fixtures are ~20 rows; the real default
    # minimum_support (20) would suppress every category in a fixture
    # this small into __RARE__, which still works but tests nothing
    # meaningful about categorical encoding - lower it just for these
    # small-fixture tests.
    return SafeCopulaSynthesizerAdapter(category_minimum_support=2)


def test_fit_returns_evidence_and_never_statistically_trains_free_text():
    # free_text ("bio") is present in output (Faker-generated placeholder
    # prose, never statistically modelled) rather than excluded outright -
    # a NOT NULL free_text column being silently dropped from generated
    # output broke the target-database write with a raw
    # sqlite3.IntegrityError, a confirmed real bug found running this
    # pipeline against a real medical fixture (dataset_metadata.value).
    adapter = small_adapter()
    evidence = adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    assert set(evidence["trained_columns"]) == {"customer_id", "age", "status", "email", "bio"}
    assert not evidence["excluded_columns"]

    generated = adapter.sample(20, seed=1)
    assert generated["bio"].notna().all()
    assert not set(generated["bio"]) & set(sample_df()["bio"])


def test_sample_returns_requested_row_count_and_columns():
    adapter = small_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(10, seed=1)
    assert len(generated) == 10
    assert set(generated.columns) == {"customer_id", "age", "status", "email", "bio"}


def test_generated_pii_columns_never_equal_source_values():
    adapter = small_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(20, seed=1)
    assert not set(generated["customer_id"]) & set(sample_df()["customer_id"])
    assert not set(generated["email"]) & set(sample_df()["email"])


def test_person_name_columns_are_never_learned_from_or_equal_to_source_values():
    """Regression: found running the full pipeline against a real medical
    SQLite fixture - patients.last_name/first_name, classified "category"
    by the (now-fixed) inference engine, were learned as an empirical
    frequency model over the real observed names and then sampled back
    out verbatim: 100% of generated last names and 96% of generated first
    names in that run were real patient names. person_name must behave
    exactly like email/identifier - Faker-generated, zero statistical
    influence from the real column - not like an ordinary category.
    """
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(20)],
            # A small, closed set of common names repeating - exactly the
            # shape that used to score higher as "category" than
            # "person_name" (low cardinality, low distinct_ratio).
            "first_name": ["Ada", "Grace", "Alan", "Linus"] * 5,
            "last_name": ["Lovelace", "Hopper", "Turing", "Torvalds"] * 5,
        }
    )
    contract = {
        "primary_key": ["customer_id"],
        "columns": {
            "customer_id": approved("identifier"),
            "first_name": approved("person_name"),
            "last_name": approved("person_name"),
        },
    }
    adapter = small_adapter()
    evidence = adapter.fit(df, "customers", contract, seed=1)
    # "trained_columns" here means "present in output" (matches how email
    # is already reported, per test_fit_returns_evidence_and_excludes_free_text
    # above) - person_name columns appear in output but are never
    # statistically fit, same as email/identifier.
    assert set(evidence["trained_columns"]) == {"customer_id", "first_name", "last_name"}
    assert not evidence["excluded_columns"]

    generated = adapter.sample(50, seed=1)
    assert not set(generated["first_name"]) & set(df["first_name"])
    assert not set(generated["last_name"]) & set(df["last_name"])


def test_manager_and_branch_names_use_context_generators_not_generic_descriptions():
    df = pd.DataFrame(
        {
            "branch_id": [f"BR-{i:03d}" for i in range(20)],
            "branch_name": ["Downtown Branch", "Northside Office", "West End Branch", "Market Street Office"] * 5,
            "manager_name": ["Ada Lovelace", "Grace Hopper", "Alan Turing", "Katherine Johnson"] * 5,
        }
    )
    contract = {
        "primary_key": ["branch_id"],
        "columns": {
            "branch_id": approved("identifier"),
            "branch_name": approved("category"),
            "manager_name": approved("person_name"),
        },
    }
    adapter = small_adapter()
    evidence = adapter.fit(df, "branches", contract, seed=1)
    assert not evidence["excluded_columns"]

    generated = adapter.sample(30, seed=1)
    assert not set(generated["manager_name"]) & set(df["manager_name"])
    assert not set(generated["branch_name"]) & set(df["branch_name"])
    assert generated["manager_name"].str.split().map(len).ge(2).all()
    assert generated["branch_name"].str.contains("Branch|Office|Financial Center|Service Center").all()
    assert not generated["branch_name"].str.contains("performance|materials|features", case=False).any()


def test_phone_number_columns_are_never_learned_from_or_equal_to_source_values():
    """Regression: found the same way as person_name above - a phone
    column with a handful of real numbers repeated across rows (found on
    the real medical fixture: a shared office line appearing 5+ times)
    scored higher as "category" than "phone_number" and leaked real
    phone numbers back out verbatim.
    """
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(20)],
            "phone": ["+1-202-555-1002", "+1-202-555-1198", "+1-202-555-1197", "+1-202-555-1200"] * 5,
        }
    )
    contract = {
        "primary_key": ["customer_id"],
        "columns": {
            "customer_id": approved("identifier"),
            "phone": approved("phone_number"),
        },
    }
    adapter = small_adapter()
    evidence = adapter.fit(df, "customers", contract, seed=1)
    assert not evidence["excluded_columns"]

    generated = adapter.sample(50, seed=1)
    assert not set(generated["phone"]) & set(df["phone"])


def test_generated_categorical_values_come_from_the_learned_vocabulary():
    adapter = small_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(30, seed=1)
    assert set(generated["status"]) <= {"active", "inactive", "pending", "__RARE__"}


def test_rare_categories_never_appear_verbatim_in_generated_output():
    df = sample_df()
    df["status"] = ["common"] * 19 + ["one_off_singleton"]
    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=5)
    adapter.fit(df, "customers", sample_contract(), seed=1)
    generated = adapter.sample(50, seed=1)
    assert "one_off_singleton" not in set(generated["status"])


def test_same_seed_is_fully_deterministic():
    contract = {"primary_key": [], "columns": {"age": approved("numerical"), "status": approved("category")}}
    df = sample_df()[["age", "status"]]

    adapter = small_adapter()
    adapter.fit(df, "customers", contract, seed=1)

    sample_a1 = adapter.sample(10, seed=42)
    sample_a2 = adapter.sample(10, seed=42)
    sample_b = adapter.sample(10, seed=999)

    assert sample_a1.equals(sample_a2)
    assert not sample_a1.equals(sample_b)


def test_null_rate_is_learned_and_reproduced_approximately():
    df = pd.DataFrame({"amount": [1.0, 2.0, None, 4.0, None, 6.0, 7.0, 8.0, None, 10.0] * 5})
    contract = {"primary_key": [], "columns": {"amount": approved("numerical")}}
    adapter = small_adapter()
    adapter.fit(df, "t", contract, seed=1)
    generated = adapter.sample(500, seed=1)
    null_rate = generated["amount"].isna().mean()
    assert 0.15 < null_rate < 0.45  # true rate is 0.3, generous tolerance for a random draw


def test_integer_typed_primary_key_generates_plain_integers_not_formatted_strings():
    """Regression test: order_id was declared INTEGER PRIMARY KEY in a
    real source SQLite schema (a strict rowid alias), but this adapter
    always generated a formatted string ("ORDER_ID-000000") for every
    identifier column regardless of its real physical type - writing
    that into an actual target database raised sqlite3.IntegrityError:
    datatype mismatch, since a rowid-alias column strictly requires an
    integer value. Caught via a real end-to-end target-database write,
    not a hand-written fixture.
    """
    df = pd.DataFrame({"order_id": list(range(20)), "amount": [10.0 + i for i in range(20)]})
    contract = {
        "primary_key": ["order_id"],
        "columns": {
            "order_id": approved("identifier", physical_type="INTEGER"),
            "amount": approved("numerical"),
        },
    }
    adapter = small_adapter()
    adapter.fit(df, "orders", contract, seed=1)
    generated = adapter.sample(10, seed=1)
    assert all(isinstance(v, int) for v in generated["order_id"])
    assert list(generated["order_id"]) == list(range(1, 11))


def test_non_primary_key_integer_typed_identifier_generates_plain_integers():
    df = pd.DataFrame({"order_id": [f"o{i}" for i in range(20)], "customer_ref": list(range(20))})
    contract = {
        "primary_key": ["order_id"],
        "columns": {
            "order_id": approved("identifier"),
            "customer_ref": approved("identifier", physical_type="INTEGER"),
        },
    }
    adapter = small_adapter()
    adapter.fit(df, "orders", contract, seed=1)
    generated = adapter.sample(10, seed=1)
    assert all(isinstance(v, int) for v in generated["customer_ref"])


def test_sample_before_fit_raises():
    adapter = SafeCopulaSynthesizerAdapter()
    with pytest.raises(SynthesisError):
        adapter.sample(5)


def test_fit_with_no_trainable_columns_raises():
    # free_text is now Faker-generated output (never excluded outright -
    # see test_fit_returns_evidence_and_never_statistically_trains_free_text),
    # so an unmapped semantic type is needed to exercise "no trainable/
    # generatable columns" here instead.
    adapter = small_adapter()
    contract = {"primary_key": [], "columns": {"bio": approved("totally_unmapped_semantic_type")}}
    with pytest.raises(SynthesisError):
        adapter.fit(sample_df()[["bio"]], "t", contract, seed=1)


def test_fit_with_unapproved_column_excludes_it():
    adapter = small_adapter()
    contract = sample_contract()
    contract["columns"]["status"] = {"semantic_type": "category", "inference_status": "review_required"}

    evidence = adapter.fit(sample_df(), "customers", contract, seed=1)
    assert "status" not in evidence["trained_columns"]
    reasons = {e["column"]: e["reason"] for e in evidence["excluded_columns"]}
    assert reasons["status"] == "inference_status=review_required"


def test_fit_raises_when_contract_column_missing_from_dataframe():
    adapter = small_adapter()
    contract = sample_contract()
    contract["columns"]["missing_col"] = approved("numerical")
    with pytest.raises(SynthesisError):
        adapter.fit(sample_df(), "customers", contract, seed=1)


def test_save_and_load_round_trip_in_a_fresh_instance(tmp_path: Path):
    adapter = small_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)

    loaded = SafeCopulaSynthesizerAdapter.load(model_path)
    generated = loaded.sample(5, seed=1)
    assert len(generated) == 5
    assert set(generated.columns) == {"customer_id", "age", "status", "email", "bio"}


def test_loaded_model_generates_identical_output_to_original():
    adapter = small_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.json"
        adapter.save(model_path)
        loaded = SafeCopulaSynthesizerAdapter.load(model_path)

    original_sample = adapter.sample(10, seed=7)
    loaded_sample = loaded.sample(10, seed=7)
    assert original_sample.equals(loaded_sample)


def test_save_before_fit_raises(tmp_path: Path):
    adapter = SafeCopulaSynthesizerAdapter()
    with pytest.raises(SynthesisError):
        adapter.save(tmp_path / "model.json")


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(SynthesisError):
        SafeCopulaSynthesizerAdapter.load(tmp_path / "does_not_exist.json")


def test_saved_model_file_is_valid_plain_json_with_no_pickle_opcodes(tmp_path: Path):
    """The whole point of this adapter: the saved file must be loadable
    with a plain, strict JSON parser (proving no pickle/cloudpickle
    framing survived anywhere in it) and must not even parse as a valid
    pickle stream.
    """
    adapter = small_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)

    raw_bytes = model_path.read_bytes()
    json.loads(raw_bytes.decode("utf-8"))  # must not raise

    with pytest.raises(Exception):
        list(pickletools.genops(raw_bytes))
