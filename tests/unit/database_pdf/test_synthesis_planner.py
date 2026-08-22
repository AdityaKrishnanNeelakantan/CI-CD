from __future__ import annotations

import pytest

from synth_platform.engine.training.database.planner import (
    RECOMMENDED_MODEL_TYPE_DEFAULT,
    RECOMMENDED_MODEL_TYPE_SENSITIVE,
    recommend_synthesizer,
)

pytestmark = pytest.mark.unit


def _column(semantic_type: str, inference_status: str = "approved") -> dict:
    return {"semantic_type": semantic_type, "inference_status": inference_status}


def test_recommends_default_when_no_sensitive_columns():
    contract = {
        "tables": {
            "orders": {
                "columns": {
                    "amount": _column("numerical"),
                    "status": _column("category"),
                }
            }
        }
    }
    result = recommend_synthesizer(contract)
    assert result["recommended_model_type"] == RECOMMENDED_MODEL_TYPE_DEFAULT
    assert result["sensitive_columns"] == []


def test_recommends_dp_when_any_sensitive_column_present():
    contract = {
        "tables": {
            "customers": {
                "columns": {
                    "email": _column("email"),
                    "signup_date": _column("datetime"),
                }
            }
        }
    }
    result = recommend_synthesizer(contract)
    assert result["recommended_model_type"] == RECOMMENDED_MODEL_TYPE_SENSITIVE
    assert result["sensitive_columns"] == ["customers.email"]


def test_ignores_sensitive_semantic_type_when_not_yet_approved():
    contract = {
        "tables": {
            "customers": {
                "columns": {
                    "email": _column("email", inference_status="review_required"),
                }
            }
        }
    }
    result = recommend_synthesizer(contract)
    assert result["recommended_model_type"] == RECOMMENDED_MODEL_TYPE_DEFAULT
    assert result["sensitive_columns"] == []


def test_collects_sensitive_columns_across_multiple_tables():
    contract = {
        "tables": {
            "customers": {"columns": {"name": _column("person_name")}},
            "support_tickets": {"columns": {"caller_phone": _column("phone_number")}},
        }
    }
    result = recommend_synthesizer(contract)
    assert result["recommended_model_type"] == RECOMMENDED_MODEL_TYPE_SENSITIVE
    assert sorted(result["sensitive_columns"]) == ["customers.name", "support_tickets.caller_phone"]


def test_recommendation_includes_human_readable_reason():
    contract = {"tables": {"t": {"columns": {"c": _column("identifier")}}}}
    result = recommend_synthesizer(contract)
    assert result["reasons"]
    assert isinstance(result["reasons"][0], str)
