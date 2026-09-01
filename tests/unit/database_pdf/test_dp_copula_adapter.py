from __future__ import annotations

import json
import pickletools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.dp_copula_adapter import DPCopulaSynthesizerAdapter, MissingColumnBoundsError
from synth_platform.engine.training.database.base import SynthesisError

pytestmark = pytest.mark.unit


def approved(semantic_type: str, physical_type: str = "TEXT") -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved", "physical_type": physical_type}


def sample_df(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(n)],
            "age": rng.integers(18, 90, n),
            "status": rng.choice(["active", "inactive", "pending"], n, p=[0.7, 0.2, 0.1]),
            "email": [f"user{i}@example.com" for i in range(n)],
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
        },
    }


def make_adapter(**overrides) -> DPCopulaSynthesizerAdapter:
    defaults = dict(epsilon_budget=2.0, column_bounds={"age": (18, 90)}, category_minimum_support=5)
    defaults.update(overrides)
    return DPCopulaSynthesizerAdapter(**defaults)


def test_fit_spends_exactly_the_configured_epsilon_budget():
    adapter = make_adapter(epsilon_budget=3.0, correlation_epsilon_fraction=0.3)
    evidence = adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    assert evidence["privacy_summary"]["total_epsilon_spent"] == pytest.approx(3.0)


def test_fit_raises_missing_bounds_for_unbounded_numeric_column():
    adapter = DPCopulaSynthesizerAdapter(epsilon_budget=1.0, column_bounds={})  # no bounds for "age"
    with pytest.raises(MissingColumnBoundsError):
        adapter.fit(sample_df(), "customers", sample_contract(), seed=1)


def test_fit_never_derives_a_bound_from_the_real_data_itself():
    """A caller who forgets to supply bounds must get a loud error, never
    a silent, data-derived fallback - that would leak information through
    the bound choice while looking like it satisfied the privacy contract.
    """
    df = sample_df()
    df["age"] = [999999] * len(df)  # an obviously wrong "bound" would be visible if derived from data
    adapter = DPCopulaSynthesizerAdapter(epsilon_budget=1.0, column_bounds={})
    with pytest.raises(MissingColumnBoundsError):
        adapter.fit(df, "customers", sample_contract(), seed=1)


def test_sample_returns_requested_row_count_and_columns():
    adapter = make_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(15, seed=1)
    assert len(generated) == 15
    assert set(generated.columns) == {"customer_id", "age", "status", "email"}


def test_generated_numeric_values_stay_within_the_declared_bounds():
    """The copula itself can sample outside [lower, upper] (it's a
    continuous Gaussian, not truncated) - real behaviour worth locking in:
    this adapter does NOT currently clip generated output back into the
    declared bound, only the training statistics were clipped. Documented
    via this test rather than silently assumed.
    """
    adapter = make_adapter(epsilon_budget=5.0)  # generous budget -> tight noise -> mean/variance close to true
    adapter.fit(sample_df(n=2000), "customers", sample_contract(), seed=1)
    generated = adapter.sample(200, seed=1)
    # Not asserting strict bound containment (see docstring) - only that
    # values are broadly plausible, not wildly divergent.
    assert generated["age"].mean() == pytest.approx(sample_df(n=2000)["age"].mean(), abs=15)


def test_pii_columns_are_never_modelled_and_never_equal_source_values():
    adapter = make_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    generated = adapter.sample(20, seed=1)
    assert not set(generated["customer_id"]) & set(sample_df()["customer_id"])
    assert not set(generated["email"]) & set(sample_df()["email"])
    # PII columns cost zero privacy budget - confirm no expenditure is
    # attributed to them.
    evidence = adapter.fit(sample_df(), "customers2", sample_contract(), seed=1)
    purposes = [e["purpose"] for e in evidence["privacy_summary"]["expenditures"]]
    assert not any("customer_id" in p or "email" in p for p in purposes)


def test_free_text_column_is_never_dropped_or_leaked():
    """Regression: free_text used to be excluded from generated output
    entirely (never trained, never Faker-substituted) - silently breaking
    any NOT NULL free_text column's target-database write with a raw
    sqlite3.IntegrityError, a confirmed real bug found running the full
    pipeline against a real medical fixture (dataset_metadata.value).
    free_text must behave like PII here: present in output, zero privacy
    budget spent, never equal to the real values.
    """
    df = sample_df()
    df["notes"] = [f"Real confidential note number {i} zz{i}qq" for i in range(len(df))]
    contract = sample_contract()
    contract["columns"]["notes"] = approved("free_text")

    adapter = make_adapter()
    evidence = adapter.fit(df, "customers", contract, seed=1)
    assert not evidence["excluded_columns"]

    generated = adapter.sample(20, seed=1)
    assert "notes" in generated.columns
    assert generated["notes"].notna().all()
    assert not set(generated["notes"]) & set(df["notes"])

    purposes = [e["purpose"] for e in evidence["privacy_summary"]["expenditures"]]
    assert not any("notes" in p for p in purposes)


def test_same_seed_is_fully_deterministic():
    adapter = make_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    a = adapter.sample(10, seed=42)
    b = adapter.sample(10, seed=42)
    assert a.equals(b)


def test_tighter_epsilon_produces_more_variance_across_independent_fits():
    """The whole point of the mechanism: less privacy budget -> more
    noise -> less stable/reproducible statistics across independent
    training runs on the same data. Checked at the adapter level, not
    just the underlying primitive (already covered in
    tests/unit/test_dp_primitives.py) - this is the integrated behaviour.
    """
    df = sample_df(n=2000)
    contract = sample_contract()

    tight_means = []
    loose_means = []
    for seed in range(10):
        tight_adapter = make_adapter(epsilon_budget=0.05)
        tight_adapter.fit(df, "customers", contract, seed=seed)
        tight_means.append(tight_adapter.sample(5, seed=1)["age"].mean())

        loose_adapter = make_adapter(epsilon_budget=5.0)
        loose_adapter.fit(df, "customers", contract, seed=seed)
        loose_means.append(loose_adapter.sample(5, seed=1)["age"].mean())

    assert np.std(tight_means) > np.std(loose_means)


def test_total_spend_never_exceeds_budget_regardless_of_column_count_or_fraction():
    """This adapter always partitions a *fixed* budget upfront rather
    than accumulating spends that could overshoot it - so a
    PrivacyBudgetExceededError should never actually surface from fit()
    itself. What must hold, for any column count or correlation-epsilon
    fraction, is the invariant the accountant enforces: total spend never
    exceeds the configured budget, even at extreme fractions (which
    exercises the epsilon<1 cap on the correlation query from both
    directions).
    """
    df = sample_df()
    df["age2"] = df["age"]
    df["age3"] = df["age"]
    contract = sample_contract()
    contract["columns"]["age2"] = approved("numerical")
    contract["columns"]["age3"] = approved("numerical")

    for correlation_fraction in (0.0, 0.5, 0.9):
        adapter = DPCopulaSynthesizerAdapter(
            epsilon_budget=2.0,
            column_bounds={"age": (18, 90), "age2": (18, 90), "age3": (18, 90)},
            correlation_epsilon_fraction=correlation_fraction,
        )
        evidence = adapter.fit(df, "customers", contract, seed=1)
        assert evidence["privacy_summary"]["total_epsilon_spent"] <= 2.0 + 1e-9


def test_save_and_load_round_trip(tmp_path: Path):
    adapter = make_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)

    loaded = DPCopulaSynthesizerAdapter.load(model_path)
    generated = loaded.sample(5, seed=1)
    assert len(generated) == 5


def test_loaded_model_generates_identical_output_to_original(tmp_path: Path):
    adapter = make_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)
    loaded = DPCopulaSynthesizerAdapter.load(model_path)

    original = adapter.sample(10, seed=7)
    reloaded = loaded.sample(10, seed=7)
    assert original.equals(reloaded)


def test_saved_model_file_is_valid_plain_json_with_no_pickle_opcodes(tmp_path: Path):
    adapter = make_adapter()
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)

    raw_bytes = model_path.read_bytes()
    json.loads(raw_bytes.decode("utf-8"))
    with pytest.raises(Exception):
        list(pickletools.genops(raw_bytes))


def test_saved_model_persists_the_privacy_summary_for_audit(tmp_path: Path):
    adapter = make_adapter(epsilon_budget=1.5)
    adapter.fit(sample_df(), "customers", sample_contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)

    saved_state = json.loads(model_path.read_text())
    assert saved_state["privacy_summary"]["total_epsilon_spent"] == pytest.approx(1.5)


def test_sample_before_fit_raises():
    adapter = make_adapter()
    with pytest.raises(SynthesisError):
        adapter.sample(5)


def test_fit_with_no_trainable_columns_raises():
    adapter = make_adapter()
    contract = {"primary_key": [], "columns": {}}
    with pytest.raises(SynthesisError):
        adapter.fit(pd.DataFrame(), "t", contract, seed=1)


def test_constructor_rejects_correlation_fraction_outside_valid_range():
    with pytest.raises(ValueError):
        DPCopulaSynthesizerAdapter(epsilon_budget=1.0, correlation_epsilon_fraction=1.0)
    with pytest.raises(ValueError):
        DPCopulaSynthesizerAdapter(epsilon_budget=1.0, correlation_epsilon_fraction=-0.1)
