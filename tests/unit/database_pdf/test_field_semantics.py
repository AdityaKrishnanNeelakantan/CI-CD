from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.field_semantics import STATUS_PROPOSED, STATUS_REVIEW_REQUIRED, infer_field_semantics

pytestmark = pytest.mark.unit


def test_value_type_alone_resolves_role_without_a_label():
    result = infer_field_semantics(None, "PERSON")
    assert result["semantic_role"] == "person_name"
    assert result["status"] == STATUS_PROPOSED


def test_label_keyword_resolves_role_without_a_strong_value_type():
    result = infer_field_semantics("Total Due", "numeric")
    assert result["semantic_role"] == "currency_amount"
    assert result["status"] == STATUS_PROPOSED


def test_account_number_does_not_tie_with_reference_code():
    """Regression test: 'number' used to be a reference_code keyword and
    collided with 'Account Number' (2.0-2.0 tie against account_number),
    forcing it into REVIEW_REQUIRED with 0.5 confidence for a label that
    should resolve cleanly.
    """
    result = infer_field_semantics("Account Number", "numeric")
    assert result["semantic_role"] == "account_number"
    assert result["status"] == STATUS_PROPOSED
    assert result["confidence"] == 1.0


def test_phone_number_label_does_not_fall_back_to_reference_code():
    result = infer_field_semantics("Phone Number", "text")
    assert result["semantic_role"] == "phone_number"


def test_invoice_number_resolves_to_reference_code():
    result = infer_field_semantics("Invoice Number", "text")
    assert result["semantic_role"] == "reference_code"


def test_value_type_and_label_agreeing_gives_full_confidence():
    result = infer_field_semantics("Customer Name", "PERSON")
    assert result["semantic_role"] == "person_name"
    assert result["confidence"] == 1.0


def test_no_signal_falls_back_to_generic_and_flags_review():
    result = infer_field_semantics("Some Unrecognised Label", "text")
    assert result["semantic_role"] == "generic_text"
    assert result["status"] == STATUS_REVIEW_REQUIRED
    assert result["confidence"] == 0.0


@pytest.mark.parametrize(
    ("label", "expected_role"),
    [
        ("Sex", "biological_sex"),
        ("Gender", "biological_sex"),
        ("Smoking Status", "smoking_status"),
        ("Alcohol Consumption", "alcohol_consumption"),
        ("Diet Preference", "diet_preference"),
        ("Exercise Habits", "exercise_habits"),
        ("Description", "transaction_description"),
        ("Reason", "medical_reason"),
        ("Medication Summary", "medication_name"),
        ("Dosage", "dosage"),
        ("Frequency", "frequency"),
        ("Account Holder", "person_name"),
    ],
)
def test_small_vocabulary_health_fields_resolve_to_categorical_roles(label, expected_role):
    """Regression: these fields used to have no label-keyword signal at
    all, so they fell through to generic_text - whose generator strategy
    refills the value's shape_pattern with random characters, producing
    unreadable gibberish ("Hhexdv", "Yzfwn") instead of a plausible value.
    Found by inspecting an actual generated synthetic twin.
    """
    result = infer_field_semantics(label, "text")
    assert result["semantic_role"] == expected_role


def test_unlabelled_numeric_field_falls_back_to_generic_numeric():
    result = infer_field_semantics(None, "numeric")
    assert result["semantic_role"] == "generic_numeric"
    assert result["status"] == STATUS_REVIEW_REQUIRED
