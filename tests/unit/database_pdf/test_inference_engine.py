"""Unit tests: SemanticInferenceEngine against known example columns.

Covers the spec's explicit test list: IDs, postcodes, email addresses,
dates, categories and continuous measurements, plus a deliberately
ambiguous column that must be marked REVIEW_REQUIRED rather than guessed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.inference.database.engine import (
    STATUS_PROPOSED,
    STATUS_REVIEW_REQUIRED,
    SemanticInferenceEngine,
)

pytestmark = pytest.mark.unit


def infer_single_column(name: str, values: list) -> dict:
    df = pd.DataFrame({name: values})
    result = SemanticInferenceEngine().infer_table(df, "t")
    return result[name]


def test_identifier_column_is_confidently_proposed():
    candidate = infer_single_column(
        "customer_id", ["CUST-0001", "CUST-0002", "CUST-0003", "CUST-0004", "CUST-0005"]
    )
    assert candidate["semantic_type"] == "identifier"
    assert candidate["status"] == STATUS_PROPOSED
    assert candidate["confidence"] >= 0.5
    assert "name_pattern=identifier" in candidate["evidence"]
    assert any(e.startswith("distinct_ratio=") for e in candidate["evidence"])


def test_email_column_is_confidently_proposed():
    candidate = infer_single_column(
        "email",
        ["ada@example.com", "alan@example.com", "grace@example.com", "linus@example.com"],
    )
    assert candidate["semantic_type"] == "email"
    assert candidate["status"] == STATUS_PROPOSED
    assert "name_pattern=email" in candidate["evidence"]
    assert any(e.startswith("value_pattern_match_rate(email)=") for e in candidate["evidence"])


def test_datetime_column_is_confidently_proposed():
    candidate = infer_single_column(
        "created_at", ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-05"]
    )
    assert candidate["semantic_type"] == "datetime"
    assert candidate["status"] == STATUS_PROPOSED
    assert "name_pattern=datetime" in candidate["evidence"]


def test_category_column_is_confidently_proposed():
    candidate = infer_single_column(
        "status",
        ["active", "inactive", "active", "pending", "active", "inactive"] * 4,
    )
    assert candidate["semantic_type"] == "category"
    assert candidate["status"] == STATUS_PROPOSED


def test_last_name_column_is_person_name_not_category():
    """Regression: found by running this engine against a real medical
    SQLite fixture - a small sample of a person-name column naturally
    repeats common surnames (a real 2000-row sample had only 16 distinct
    last names), which used to outscore the name-pattern signal and get
    classified "category". Category columns are learned as an empirical
    frequency model over the real observed values (by design, for
    genuine categoricals like blood_type/state), so classifying a name
    column that way meant real patient surnames were sampled back out
    verbatim in the synthetic output - a confirmed privacy leak.
    """
    candidate = infer_single_column(
        "last_name",
        ["Garcia", "Walker", "Wilson", "Morris", "Clark", "Garcia", "Walker", "Wilson"] * 20,
    )
    assert candidate["semantic_type"] == "person_name"
    assert candidate["status"] == STATUS_PROPOSED
    assert "name_pattern=person_name" in candidate["evidence"]


def test_first_name_column_is_person_name_not_category():
    candidate = infer_single_column(
        "first_name",
        ["Ada", "Grace", "Alan", "Linus", "Ada", "Grace"] * 20,
    )
    assert candidate["semantic_type"] == "person_name"
    assert candidate["status"] == STATUS_PROPOSED


def test_phone_column_with_repeated_real_numbers_is_phone_number_not_category():
    """Regression: found the same way as the person_name gap - a real
    phone column with a handful of numbers repeated across patients (a
    shared office line, family members, ...) scored higher as "category"
    than "phone_number" and got frequency-modelled from the real values,
    leaking real phone numbers back out verbatim.
    """
    candidate = infer_single_column(
        "phone",
        ["+1-202-555-1002", "+1-202-555-1198", "+1-202-555-1002", "+1-202-555-1197"] * 10,
    )
    assert candidate["semantic_type"] == "phone_number"
    assert candidate["status"] == STATUS_PROPOSED


def test_phone_number_detected_from_value_pattern_alone_without_a_name_hint():
    candidate = infer_single_column(
        "contact",
        ["+1-202-555-1002", "+1-202-555-1198", "+1-202-555-1197", "+1-202-555-1200"] * 10,
    )
    assert candidate["semantic_type"] == "phone_number"


def test_unformatted_phone_number_detected_without_a_name_hint():
    """Regression: found the same way as the formatted-phone gap above -
    a real phone column with no separators (e.g. exported as a bare
    digit string) and no phone-like column name previously fell through
    every check (name pattern, formatted-value pattern) straight to
    "category" and leaked real phone numbers verbatim, same as the
    formatted case this engine already guards against.
    """
    candidate = infer_single_column(
        "contact",
        ["2025551002", "2025551198", "2025551002", "2025551197"] * 10,
    )
    assert candidate["semantic_type"] == "phone_number"
    assert candidate["status"] == STATUS_PROPOSED


def test_unformatted_digits_on_an_identifier_named_column_stay_identifier():
    """Regression guard: the relaxed unformatted-digit phone check must
    not fire for a column already named like an identifier (id/uuid/
    guid/key) - a sequential numeric ID is shape-identical to a bare
    phone number, and without this guard it would lose its previously
    confident "identifier" classification to a false phone signal.
    """
    candidate = infer_single_column(
        "customer_id",
        [f"{100000000 + i}" for i in range(20)],
    )
    assert candidate["semantic_type"] == "identifier"
    assert candidate["status"] == STATUS_PROPOSED


def test_bare_name_column_is_person_name_not_identifier():
    """Regression: the demo/sample Database Twin path used a bare `name`
    column of distinct person names. Without an exact-name / value-pattern
    person_name signal, near-uniqueness alone scored it as identifier and
    generation emitted NAME-<digits> placeholders.
    """
    candidate = infer_single_column(
        "name",
        ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Linus Torvalds", "Katherine Johnson"] * 4,
    )
    assert candidate["semantic_type"] == "person_name"
    assert candidate["status"] == STATUS_PROPOSED
    assert (
        "name_pattern=person_name" in candidate["evidence"]
        or any(e.startswith("value_pattern_match_rate(person_name)=") for e in candidate["evidence"])
    )


def test_organisation_name_column_is_unaffected_by_the_person_name_pattern():
    """provider_name/facility_name/company_name must NOT match the
    person-name pattern - only well-established person-name field
    conventions (first_name, last_name, surname, ...) should."""
    candidate = infer_single_column(
        "facility_name",
        ["General Hospital", "St Mary's Clinic", "General Hospital", "City Medical"] * 10,
    )
    assert candidate["semantic_type"] != "person_name"


def test_continuous_measurement_is_confidently_numerical():
    candidate = infer_single_column(
        "amount", [19.99, 42.5, 7.25, 100.0, 3.33, 88.1, 15.6, 27.4, 51.9, 12.0]
    )
    assert candidate["semantic_type"] == "numerical"
    assert candidate["status"] == STATUS_PROPOSED


def test_boolean_column_is_confidently_proposed():
    candidate = infer_single_column("is_active", [True, False, True, True, False])
    assert candidate["semantic_type"] == "boolean"
    assert candidate["status"] == STATUS_PROPOSED
    assert "dtype=boolean" in candidate["evidence"]


def test_free_text_column_is_confidently_proposed():
    candidate = infer_single_column(
        "comments",
        [
            "Customer requested a refund for late delivery",
            "Left a positive review about packaging",
            "Asked about bulk order discounts",
            "Reported a damaged item on arrival",
        ],
    )
    assert candidate["semantic_type"] == "free_text"
    assert candidate["status"] == STATUS_PROPOSED
    assert "name_pattern=free_text" in candidate["evidence"]


def test_postcode_like_category_column_is_proposed():
    candidate = infer_single_column(
        "postal_code",
        ["10001", "10001", "94105", "94105", "73301", "73301"],
    )
    assert candidate["semantic_type"] == "category"
    assert candidate["status"] == STATUS_PROPOSED
    assert any(e.startswith("value_pattern_match_rate(postcode)=") for e in candidate["evidence"])


def test_bare_near_unique_numeric_defaults_to_numerical_not_identifier():
    """Uniqueness alone, with no name/PK hint, must not default to
    "identifier" - per the spec's own worked example, identifier confidence
    requires uniqueness *combined with* a naming or structural hint.
    """
    candidate = infer_single_column("ref_num", list(range(1000, 1010)))
    assert candidate["semantic_type"] == "numerical"
    assert candidate["status"] == STATUS_PROPOSED


def test_ambiguous_short_token_column_without_person_name_cue_requires_review():
    """A short near-unique string column that is not named like a person
    field and does not look like multi-token person names remains
    REVIEW_REQUIRED rather than being forced to identifier.
    """
    candidate = infer_single_column("label", ["Ada", "Alan", "Grace", "Linus"])
    assert candidate["status"] == STATUS_REVIEW_REQUIRED
    assert candidate["confidence"] < 0.5


def test_no_signal_column_falls_back_to_review_required():
    candidate = infer_single_column("weird", [None, None, None])
    assert candidate["status"] == STATUS_REVIEW_REQUIRED
    assert candidate["confidence"] == 0.0
    assert "no_signal" in candidate["evidence"]


def test_primary_key_hint_boosts_identifier_even_without_name_pattern():
    df = pd.DataFrame({"ref": ["a1", "b2", "c3", "d4"]})
    engine = SemanticInferenceEngine()
    result = engine.infer_table(df, "t", discovery_table={"primary_key": ["ref"]})
    assert result["ref"]["semantic_type"] == "identifier"
    assert "primary_key=true" in result["ref"]["evidence"]


def test_alternatives_are_reported_for_every_candidate():
    candidate = infer_single_column(
        "customer_id", ["CUST-0001", "CUST-0002", "CUST-0003", "CUST-0004", "CUST-0005"]
    )
    assert isinstance(candidate["alternatives"], list)
    for alt in candidate["alternatives"]:
        assert set(alt) == {"semantic_type", "confidence"}


def test_raw_values_never_appear_in_candidate_output():
    candidate = infer_single_column(
        "email", ["secret.user@example.com", "another.person@example.com"]
    )
    serialized = str(candidate)
    assert "secret.user@example.com" not in serialized
    assert "another.person@example.com" not in serialized
