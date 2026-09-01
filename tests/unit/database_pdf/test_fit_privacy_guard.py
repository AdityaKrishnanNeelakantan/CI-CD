"""Hard privacy: raw context/PII must never enter model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.context_fields import ContextFieldKind, resolve_context_kind
from synth_platform.engine.common.database.privacy.deterministic_masker import DeterministicFakerMasker
from synth_platform.engine.common.database.privacy.fit_guard import (
    FitPrivacyError,
    assert_no_raw_context_in_fit_frame,
    blob_contains_any,
    context_aware_columns,
)
from synth_platform.engine.training.database.adapters.safe_copula_adapter import SafeCopulaSynthesizerAdapter

pytestmark = pytest.mark.unit


def _approved(semantic_type: str, physical_type: str = "TEXT") -> dict:
    return {
        "semantic_type": semantic_type,
        "inference_status": "approved",
        "physical_type": physical_type,
    }


def test_column_name_pii_beats_mislabelled_category_semantic():
    """Fail closed: email/name columns marked category still mask/exclude."""
    assert resolve_context_kind("email", "category") is ContextFieldKind.EMAIL
    assert resolve_context_kind("full_name", "category") is ContextFieldKind.PERSON_NAME
    assert resolve_context_kind("customer_id", "numerical") is ContextFieldKind.IDENTIFIER
    assert resolve_context_kind("age", "numerical") is ContextFieldKind.PASSTHROUGH
    assert resolve_context_kind("segment", "category") is ContextFieldKind.PASSTHROUGH


def test_masker_strips_raw_even_when_semantic_is_category():
    df = pd.DataFrame(
        {
            "email": ["canary.leak.unique@example.com", "other.canary@example.com"],
            "segment": ["gold", "silver"],
        }
    )
    contract = {
        "columns": {
            "email": _approved("category"),
            "segment": _approved("category"),
        }
    }
    masked, report = DeterministicFakerMasker(seed=7).mask_table(df, "customers", contract)
    assert "canary.leak.unique@example.com" not in set(masked["email"].astype(str))
    assert "email" in report["context_aware_columns"]
    assert list(masked["segment"]) == ["gold", "silver"]
    assert_no_raw_context_in_fit_frame(df, masked, contract, table_name="customers")


def test_fit_guard_raises_if_raw_context_survives():
    source = pd.DataFrame({"email": ["raw@example.com", "other@example.com"], "age": [40, 41]})
    unfit = source.copy()
    contract = {"columns": {"email": _approved("email"), "age": _approved("numerical")}}
    with pytest.raises(FitPrivacyError, match="unchanged after mask"):
        assert_no_raw_context_in_fit_frame(source, unfit, contract, table_name="t")


def test_safe_copula_never_stores_raw_pii_in_model_json(tmp_path: Path):
    canary_email = "ZZZ_CANARY_EMAIL_9f3a2b@leak-test.example"
    canary_name = "ZZZ_CANARY_PERSON_9f3a2b"
    canary_id = "ZZZ-CANARY-ID-9f3a2b"
    df = pd.DataFrame(
        {
            "customer_id": [canary_id, "CUS-000002", "CUS-000003"],
            "full_name": [canary_name, "Ada Lovelace", "Alan Turing"],
            "email": [canary_email, "ada@example.com", "alan@example.com"],
            "age": [41, 36, 41],
            "segment": ["gold", "silver", "gold"],
        }
    )
    contract = {
        "columns": {
            "customer_id": _approved("identifier"),
            "full_name": _approved("person_name"),
            "email": _approved("email"),
            "age": _approved("numerical", "INTEGER"),
            "segment": _approved("category"),
        },
        "primary_key": ["customer_id"],
    }
    masker = DeterministicFakerMasker(seed=11)
    safe_df, _ = masker.mask_table(df, "customers", contract)
    assert_no_raw_context_in_fit_frame(df, safe_df, contract, table_name="customers")

    synth = SafeCopulaSynthesizerAdapter()
    synth.fit(safe_df, "customers", contract, seed=11)
    model_path = tmp_path / "customers.json"
    synth.save(model_path)
    blob = model_path.read_text(encoding="utf-8")
    hits = blob_contains_any(blob, [canary_email, canary_name, canary_id])
    assert hits == [], f"raw canaries leaked into model artifact: {hits}"

    sample = synth.sample(5, seed=11)
    sample_blob = sample.to_csv(index=False)
    hits = blob_contains_any(sample_blob, [canary_email, canary_name, canary_id])
    assert hits == [], f"raw canaries leaked into samples: {hits}"


def test_training_path_fingerprint_uses_masked_frame_not_raw(tmp_path: Path):
    """DNA-only fit: fingerprint/report must reference masked context columns."""
    canary = "ZZZ_SERVICE_CANARY_email_7c1d@leak.example"
    df = pd.DataFrame(
        {
            "email": [canary, "a@example.com", "b@example.com"],
            "age": [20, 30, 40],
            "segment": ["a", "b", "a"],
        }
    )
    contract = {
        "columns": {
            "email": _approved("email"),
            "age": _approved("numerical", "INTEGER"),
            "segment": _approved("category"),
        },
        "primary_key": [],
    }
    masker = DeterministicFakerMasker(seed=3)
    safe_df, report = masker.mask_table(df, "customers", contract)
    assert_no_raw_context_in_fit_frame(df, safe_df, contract, table_name="customers")
    assert canary not in set(safe_df["email"].astype(str))
    assert "email" in report["context_aware_columns"]
    assert context_aware_columns(contract) == ["email"]

    synth = SafeCopulaSynthesizerAdapter()
    evidence = synth.fit(safe_df, "customers", contract, seed=3)
    # Present in output schema, but never in the statistical encoder/copula.
    assert "email" in evidence.get("trained_columns", [])
    assert "email" in synth._pii_columns
    assert "email" not in synth._encoders
    assert "age" in synth._encoders
    model_path = tmp_path / "customers.json"
    synth.save(model_path)
    assert canary not in model_path.read_text(encoding="utf-8")


def test_pdf_template_never_persists_raw_field_values():
    from synth_platform.engine.documents.pdf.template_compiler import compile_document_template

    canary = "ZZZ_PDF_CANARY_Kimberly_9f3a2b"
    layout = {
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    {
                        "region_id": "p1_r0",
                        "region_type": "field",
                        "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 0.1},
                        "reading_order_index": 0,
                        "lines": [{"text": f"Name: {canary}", "cells": ["Name:", canary]}],
                    },
                    {
                        "region_id": "p1_r1",
                        "region_type": "paragraph",
                        "bbox": {"x0": 0, "y0": 0.2, "x1": 1, "y1": 0.3},
                        "reading_order_index": 1,
                        "lines": [{"text": f"Patient {canary} attended clinic.", "cells": ["x"]}],
                    },
                ],
            }
        ],
    }
    template = compile_document_template(layout)
    blob = json.dumps(template)
    assert canary not in blob
    field = template["pages"][0]["regions"][0]
    assert "masked_preview" in field
    assert canary not in field["masked_preview"]
