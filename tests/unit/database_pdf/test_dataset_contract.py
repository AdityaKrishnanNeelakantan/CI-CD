from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.inference.database.contract import (
    build_or_update_dataset_contract,
    load_dataset_contract,
    save_dataset_contract,
)

pytestmark = pytest.mark.unit


DISCOVERY = {
    "tables": {
        "customers": {
            "primary_key": ["customer_id"],
            "columns": [
                {"name": "customer_id", "database_type": "TEXT", "nullable": False},
                {"name": "age", "database_type": "INTEGER", "nullable": True},
            ],
        }
    }
}


def candidates(customer_id_type="identifier", customer_id_status="proposed"):
    return {
        "customers": {
            "customer_id": {
                "semantic_type": customer_id_type,
                "status": "REVIEW_REQUIRED" if customer_id_status != "proposed" else "proposed",
                "confidence": 0.9,
                "evidence": ["name_pattern=identifier"],
                "alternatives": [],
            },
            "age": {
                "semantic_type": "numerical",
                "status": "proposed",
                "confidence": 0.95,
                "evidence": ["dtype=numeric"],
                "alternatives": [],
            },
        }
    }


def test_fresh_contract_has_no_approved_columns_without_decisions():
    contract = build_or_update_dataset_contract(
        "ds", "sha256:abc", DISCOVERY, candidates(), existing_contract=None, decisions=None
    )
    assert contract["revision"] == 1
    col = contract["tables"]["customers"]["columns"]["customer_id"]
    assert col["inference_status"] == "proposed"
    assert col["semantic_type"] == "identifier"


def test_decision_marks_column_approved_with_override_evidence():
    contract = build_or_update_dataset_contract(
        "ds",
        "sha256:abc",
        DISCOVERY,
        candidates(),
        decisions={"customers": {"customer_id": "category"}},  # disagrees with candidate
    )
    col = contract["tables"]["customers"]["columns"]["customer_id"]
    assert col["inference_status"] == "approved"
    assert col["semantic_type"] == "category"
    assert "human_override" in col["evidence"]
    assert col["sensitive"] is True  # ID-like column names stay sensitive even after override.


def test_decision_matching_candidate_is_recorded_as_accepted_not_override():
    contract = build_or_update_dataset_contract(
        "ds",
        "sha256:abc",
        DISCOVERY,
        candidates(),
        decisions={"customers": {"customer_id": "identifier"}},  # agrees with candidate
    )
    col = contract["tables"]["customers"]["columns"]["customer_id"]
    assert "human_accepted_proposal" in col["evidence"]
    assert "human_override" not in col["evidence"]
    assert col["sensitive"] is True  # "identifier" is sensitive by default


def test_approved_override_survives_a_rerun_with_disagreeing_candidate():
    first = build_or_update_dataset_contract(
        "ds",
        "sha256:abc",
        DISCOVERY,
        candidates(),
        decisions={"customers": {"customer_id": "category"}},
    )
    assert first["tables"]["customers"]["columns"]["customer_id"]["semantic_type"] == "category"

    # Simulate a rerun: fresh candidates now confidently propose "identifier"
    # again, and no new decision is supplied - the prior approval must win.
    second = build_or_update_dataset_contract(
        "ds",
        "sha256:abc",
        DISCOVERY,
        candidates(customer_id_type="identifier"),
        existing_contract=first,
        decisions=None,
    )
    col = second["tables"]["customers"]["columns"]["customer_id"]
    assert col["semantic_type"] == "category"
    assert col["inference_status"] == "approved"
    assert second["revision"] == 2


def test_controlled_location_columns_do_not_persist_as_free_text_and_stay_sensitive():
    discovery = {
        "tables": {
            "branches": {
                "primary_key": [],
                "columns": [
                    {"name": "country", "database_type": "TEXT", "nullable": False},
                    {"name": "city", "database_type": "TEXT", "nullable": False},
                ],
            }
        }
    }
    candidate = {
        "branches": {
            "country": {
                "semantic_type": "free_text",
                "status": "proposed",
                "confidence": 0.9,
                "evidence": ["legacy_bad_candidate"],
                "alternatives": [],
            },
            "city": {
                "semantic_type": "free_text",
                "status": "proposed",
                "confidence": 0.9,
                "evidence": ["legacy_bad_candidate"],
                "alternatives": [],
            }
        }
    }

    contract = build_or_update_dataset_contract(
        "ds",
        "sha256:abc",
        discovery,
        candidate,
        decisions={"branches": {"country": "free_text", "city": "free_text"}},
    )

    for column_name in ("country", "city"):
        col = contract["tables"]["branches"]["columns"][column_name]
        assert col["semantic_type"] == "category"
        assert col["sensitive"] is True
        assert "normalized_controlled_text_to_category" in col["evidence"]


def test_unapproved_column_is_still_updated_by_fresh_candidates_on_rerun():
    first = build_or_update_dataset_contract("ds", "sha256:abc", DISCOVERY, candidates())
    # "age" was never approved or overridden, so a rerun should refresh it.
    second = build_or_update_dataset_contract(
        "ds", "sha256:abc", DISCOVERY, candidates(), existing_contract=first
    )
    col = second["tables"]["customers"]["columns"]["age"]
    assert col["inference_status"] == "proposed"
    assert col["semantic_type"] == "numerical"


def test_save_and_load_round_trip(tmp_path: Path):
    contract = build_or_update_dataset_contract("ds", "sha256:abc", DISCOVERY, candidates())
    save_dataset_contract(contract, tmp_path)

    loaded = load_dataset_contract(tmp_path)
    assert loaded == contract
    assert (tmp_path / "history" / "dataset_contract.rev1.json").exists()


def test_load_returns_none_when_no_contract_exists(tmp_path: Path):
    assert load_dataset_contract(tmp_path) is None
