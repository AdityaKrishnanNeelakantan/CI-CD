from __future__ import annotations

import pytest

from synth_platform.engine.common.database.privacy.profile_sanitizer import sanitize_profile_for_export

pytestmark = pytest.mark.unit


def _profile_with_column(column: dict) -> dict:
    return {
        "table_name": "t",
        "row_count": 100,
        "columns": {"balance": column},
        "correlations": [],
        "warnings": [],
    }


def test_exact_minimum_and_maximum_are_removed():
    profile = _profile_with_column(
        {
            "minimum": 1.0,
            "maximum": 987654.32,
            "generation_lower_bound": 10.0,
            "generation_upper_bound": 5000.0,
        }
    )
    sanitized = sanitize_profile_for_export(profile)
    col = sanitized["columns"]["balance"]
    assert "minimum" not in col
    assert "maximum" not in col
    assert col["generation_lower_bound"] == 10.0
    assert col["generation_upper_bound"] == 5000.0


def test_exact_category_frequencies_are_removed_but_safe_companion_kept():
    profile = _profile_with_column(
        {
            "category_frequencies": {"active": 30, "rare_customer_x": 1},
            "safe_category_frequencies": {"active": 30, "__RARE__": 1},
        }
    )
    sanitized = sanitize_profile_for_export(profile)
    col = sanitized["columns"]["balance"]
    assert "category_frequencies" not in col
    assert col["safe_category_frequencies"] == {"active": 30, "__RARE__": 1}


def test_representative_examples_are_removed_with_no_replacement():
    profile = _profile_with_column({"representative_examples": ["Gr***e", "Al***n"]})
    sanitized = sanitize_profile_for_export(profile)
    assert "representative_examples" not in sanitized["columns"]["balance"]


def test_safe_fields_and_other_evidence_survive_untouched():
    profile = _profile_with_column(
        {
            "row_count": 100,
            "null_percentage": 0.5,
            "distinct_ratio": 0.9,
            "mean": 42.0,
            "quantiles": {"p25": 1.0, "p50": 2.0, "p75": 3.0},
            "warnings": ["near_unique_column"],
        }
    )
    sanitized = sanitize_profile_for_export(profile)
    col = sanitized["columns"]["balance"]
    assert col["row_count"] == 100
    assert col["mean"] == 42.0
    assert col["quantiles"] == {"p25": 1.0, "p50": 2.0, "p75": 3.0}
    assert col["warnings"] == ["near_unique_column"]


def test_table_level_fields_outside_columns_are_preserved():
    profile = _profile_with_column({})
    sanitized = sanitize_profile_for_export(profile)
    assert sanitized["table_name"] == "t"
    assert sanitized["row_count"] == 100


def test_multi_table_run_profiling_wrapper_shape_is_sanitized():
    """The real shape passed as reference_profile in this project is
    run_profiling()'s multi-table wrapper, not a single profile_table()
    result - this is the shape that actually matters.
    """
    profile = {
        "profiled_at": "2026-01-01T00:00:00Z",
        "tables": {
            "customers": {
                "table_name": "customers",
                "columns": {
                    "signup_amount": {"minimum": 1.0, "maximum": 987654.32, "generation_upper_bound": 500.0}
                },
            }
        },
    }
    sanitized = sanitize_profile_for_export(profile)
    col = sanitized["tables"]["customers"]["columns"]["signup_amount"]
    assert "minimum" not in col
    assert "maximum" not in col
    assert col["generation_upper_bound"] == 500.0
    assert sanitized["profiled_at"] == "2026-01-01T00:00:00Z"


def test_safe_category_frequencies_is_stripped_for_a_column_the_contract_marks_sensitive():
    """Regression: found running the full pipeline against a real medical
    fixture - patients.last_name (semantic_type=person_name in the
    approved contract) had its full real vocabulary (16 real surnames,
    each with an exact occurrence count) survive verbatim into the
    exported artifact's reference_profile.json via safe_category_frequencies.
    That field is only "safe" from rare-value overexposure - it has no
    idea a column is a person-name/phone-number/email/identifier, since
    profiling runs before semantic inference. The dataset_contract (the
    final approved semantic type) is what this function now needs to
    know which columns additionally need that field stripped.
    """
    profile = {
        "tables": {
            "patients": {
                "table_name": "patients",
                "columns": {
                    "last_name": {
                        "safe_category_frequencies": {"Garcia": 137, "Wilson": 138},
                        "distinct_count": 16,
                    },
                    "blood_type": {
                        "safe_category_frequencies": {"O+": 300, "A+": 250},
                        "distinct_count": 4,
                    },
                },
            }
        }
    }
    contract = {
        "tables": {
            "patients": {
                "columns": {
                    "last_name": {"semantic_type": "person_name", "inference_status": "approved"},
                    "blood_type": {"semantic_type": "category", "inference_status": "approved"},
                }
            }
        }
    }
    sanitized = sanitize_profile_for_export(profile, contract)
    cols = sanitized["tables"]["patients"]["columns"]
    assert "safe_category_frequencies" not in cols["last_name"]
    assert cols["last_name"]["distinct_count"] == 16  # non-vocabulary evidence still survives
    # An ordinary category (not sensitive) keeps its safe vocabulary -
    # this is exactly the field the QA fidelity ("known_vocabulary")
    # check needs to work at all.
    assert cols["blood_type"]["safe_category_frequencies"] == {"O+": 300, "A+": 250}


def test_sanitize_without_a_contract_is_unaffected_backward_compatible():
    profile = {
        "tables": {
            "patients": {"columns": {"last_name": {"safe_category_frequencies": {"Garcia": 137}}}}
        }
    }
    sanitized = sanitize_profile_for_export(profile)
    assert sanitized["tables"]["patients"]["columns"]["last_name"]["safe_category_frequencies"] == {"Garcia": 137}
