"""Tests for generic LLM text generation layer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table
from synth_platform.engine.generation.schema.simulator import DataSimulator
from synth_platform.engine.generation.text.context import build_safe_row_context
from synth_platform.engine.generation.text.eligibility import assess_column_eligibility, list_eligible_columns
from synth_platform.engine.generation.text.evidence import TextGenerationEvidence, merge_evidence
from synth_platform.engine.generation.text.fallback import fallback_text
from synth_platform.engine.generation.text.generator import TextGenerationConfig, TextGenerationEngine, generate_text_column
from synth_platform.engine.generation.text.prompts import build_text_generation_prompt
from synth_platform.engine.generation.text.validators import validate_generated_text


def _schema(*columns: Column, row_count: int = 10) -> SchemaConfig:
    return SchemaConfig(
        name="generic text",
        tables=[Table(name="records", row_count=row_count)],
        columns={"records": list(columns)},
    )


@pytest.mark.parametrize(
    "column,expected",
    [
        (Column(name="long_description", type="text", distribution_params={"text_type": "long_text"}), True),
        (Column(name="support_notes", type="text", distribution_params={"text_type": "notes"}), True),
        (Column(name="ticket_message", type="text", distribution_params={"text_type": "ticket_message"}), True),
        (Column(name="amount", type="float"), False),
        (Column(name="status", type="categorical", distribution_params={"choices": ["A", "B"]}), False),
        (Column(name="created_at", type="datetime"), False),
        (Column(name="is_active", type="boolean"), False),
    ],
)
def test_eligibility_blocks_structured_types(column: Column, expected: bool):
    assert assess_column_eligibility(column).eligible is expected


@pytest.mark.parametrize(
    "name,col_type",
    [
        ("contact_email", "email"),
        ("full_name", "text"),
        ("phone_number", "phone"),
        ("home_address", "text"),
        ("record_id", "int"),
        ("customer_id", "int"),
        ("account_number", "text"),
    ],
)
def test_eligibility_blocks_pii_and_id_columns(name: str, col_type: str):
    column = Column(name=name, type=col_type, unique=(name.endswith("_id")))
    assert assess_column_eligibility(column).eligible is False


def test_eligibility_allows_notes_blocks_email_and_id():
    notes = Column(name="support_notes", type="text", distribution_params={"text_type": "notes"})
    email = Column(name="contact_email", type="email")
    pk = Column(name="record_id", type="int", unique=True)
    status = Column(name="status", type="categorical", distribution_params={"choices": ["A", "B"]})

    assert assess_column_eligibility(notes).eligible is True
    assert assess_column_eligibility(email).eligible is False
    assert assess_column_eligibility(pk).eligible is False
    assert assess_column_eligibility(status).eligible is False


def test_list_eligible_columns_is_schema_driven():
    schema = _schema(
        Column(name="feedback_text", type="text", distribution_params={"text_type": "feedback"}),
        Column(name="email", type="email"),
    )
    rows = list_eligible_columns(schema)
    by_name = {r["column"]: r for r in rows}
    assert by_name["feedback_text"]["eligible"] is True
    assert by_name["email"]["eligible"] is False


def test_safe_context_includes_status_excludes_email():
    row = {"status": "delayed", "priority": "high", "contact_email": "a@example.com", "notes": "secret"}
    ctx = build_safe_row_context(
        row,
        target_column="notes",
        columns=[
            Column(name="status", type="categorical", distribution_params={"choices": ["delayed", "open"]}),
            Column(name="priority", type="categorical", distribution_params={"choices": ["high", "low"]}),
            Column(name="contact_email", type="email"),
            Column(name="notes", type="text"),
        ],
    )
    assert "status" in ctx
    assert "priority" in ctx
    assert "contact_email" not in ctx
    assert "notes" not in ctx
    assert "a@example.com" not in " ".join(ctx.values())


def test_prompt_includes_privacy_rules_and_not_protected_values():
    prompt = build_text_generation_prompt(
        table_name="records",
        column_name="case_notes",
        text_role="notes",
        rows=3,
        safe_context={"status": "delayed"},
        max_words=30,
    )
    assert "Do not include PII" in prompt or "privacy-safe" in prompt.lower()
    assert "delayed" in prompt
    assert "a@example.com" not in prompt
    assert "secret" not in prompt


def test_validation_rejects_email_phone_replay_and_long_output():
    assert validate_generated_text("hello world test", max_words=10).passed
    assert not validate_generated_text("contact me at a@example.com now please thanks", max_words=20).passed
    assert not validate_generated_text("call +1 555 010 9999 today please thanks", max_words=20).passed
    assert not validate_generated_text("exact source line", source_values={"exact source line"}, max_words=10).passed
    long = " ".join(["word"] * 100)
    assert not validate_generated_text(long, max_words=20).passed


def test_fallback_on_empty_and_nullable():
    fb = fallback_text(index=0, text_role="notes", domain="generic", nullable=False, seed=7)
    assert fb
    assert len(fb.split()) >= 3
    null = fallback_text(index=0, text_role="notes", nullable=True, null_probability=1.0, rng_value=0.0)
    assert null is None


def test_context_aware_fallback_uses_safe_context_without_pii():
    from synth_platform.engine.generation.text.faker_fallback import context_aware_fallback_text

    text = context_aware_fallback_text(
        index=2,
        text_role="ticket_message",
        safe_context={"status": "open", "priority": "high"},
        domain="saas",
        seed=11,
    )
    assert isinstance(text, str) and text
    assert "@" not in text
    assert "open" in text.lower() or "priority" in text.lower() or "support" in text.lower()


def test_generate_text_column_without_llm_uses_fallback():
    column = Column(
        name="ticket_message",
        type="text",
        distribution_params={"text_type": "ticket_message", "llm_text": True, "domain": "saas"},
    )
    ctx = pd.DataFrame({"status": ["open", "closed"], "priority": ["high", "low"]})
    result = generate_text_column(
        table_name="records",
        column=column,
        size=2,
        table_data=ctx,
        config=TextGenerationConfig(llm_enabled=False, is_preview=True),
    )
    assert len(result.values) == 2
    assert all(isinstance(v, str) and v for v in result.values)
    assert result.evidence.fallback_count >= 1
    assert result.evidence.llm_used is False


def test_generate_text_column_for_ineligible_integer_returns_fallback_values():
    column = Column(name="retry_count", type="int", nullable=True, distribution_params={"null_probability": 0.0})

    result = generate_text_column(
        table_name="records",
        column=column,
        size=3,
        config=TextGenerationConfig(llm_enabled=False, is_preview=True, seed=17),
    )

    assert len(result.values) == 3
    assert all(isinstance(value, str) and value for value in result.values)
    assert result.evidence.fallback_count == 3
    assert any("ineligible" in warning for warning in result.evidence.warnings)


def test_unsafe_llm_output_falls_back():
    column = Column(
        name="review_text",
        type="text",
        distribution_params={"text_type": "review", "llm_text": True},
    )
    engine = TextGenerationEngine(TextGenerationConfig(llm_enabled=True, is_preview=True, max_llm_rows=2, seed=7))
    with patch.object(
        engine,
        "_call_provider",
        return_value=["Email me at leak@example.com for more details please thanks"],
    ):
        result = engine.generate(table_name="records", column=column, size=1)
    assert result.evidence.privacy_fail_count >= 1
    assert result.evidence.fallback_count >= 1
    assert result.values[0]
    assert "@" not in str(result.values[0])


def test_provider_failure_falls_back():
    column = Column(
        name="summary_text",
        type="text",
        distribution_params={"text_type": "summary", "llm_text": True},
    )
    engine = TextGenerationEngine(TextGenerationConfig(llm_enabled=True, is_preview=True, max_llm_rows=2, seed=3))
    with patch.object(engine, "_call_provider", side_effect=RuntimeError("provider down")):
        result = engine.generate(table_name="records", column=column, size=2)
    assert result.provider_error
    assert result.evidence.fallback_count >= 2
    assert all(isinstance(v, str) and v for v in result.values if v is not None)


def test_source_replay_is_rejected():
    column = Column(
        name="feedback",
        type="text",
        distribution_params={"text_type": "feedback", "llm_text": True},
    )
    engine = TextGenerationEngine(TextGenerationConfig(llm_enabled=True, is_preview=True, max_llm_rows=1, seed=11))
    with patch.object(engine, "_call_provider", return_value=["exact source line copied verbatim here"]):
        result = engine.generate(
            table_name="records",
            column=column,
            size=1,
            source_series=pd.Series(["exact source line copied verbatim here"]),
        )
    assert result.evidence.source_replay_count >= 1
    assert result.evidence.fallback_count >= 1


def test_null_probability_is_preserved():
    column = Column(
        name="optional_notes",
        type="text",
        nullable=True,
        distribution_params={"text_type": "notes", "llm_text": True, "null_probability": 1.0},
    )
    result = generate_text_column(
        table_name="records",
        column=column,
        size=20,
        config=TextGenerationConfig(llm_enabled=False, seed=99),
    )
    assert all(v is None for v in result.values)


def test_max_llm_rows_enforced_for_full_generation():
    column = Column(
        name="description",
        type="text",
        distribution_params={"text_type": "description", "llm_text": True},
    )
    config = TextGenerationConfig(llm_enabled=True, is_preview=False, max_llm_rows=10)
    engine = TextGenerationEngine(config)
    assert engine._effective_llm_cap(1_000_000) <= 200


def test_large_full_generation_does_not_call_llm_by_default():
    config = TextGenerationConfig(llm_enabled=False, is_preview=False, max_llm_rows=50)
    engine = TextGenerationEngine(config)
    assert engine._effective_llm_cap(2_000_000) == 0


def test_evidence_merge_tracks_counts():
    items = [
        TextGenerationEvidence(
            table="records",
            column="notes",
            rows_requested=5,
            rows_generated=5,
            fallback_count=2,
            privacy_fail_count=1,
            source_replay_count=0,
        )
    ]
    merged = merge_evidence(items)
    assert merged["summary"]["total_fallback_count"] == 2
    assert merged["summary"]["total_privacy_fail_count"] == 1


def test_simulator_records_text_generation_evidence():
    schema = _schema(
        Column(
            name="support_message",
            type="text",
            distribution_params={"text_type": "support_message", "llm_text": True, "domain": "saas"},
        ),
        row_count=5,
    )
    simulator = DataSimulator(schema, llm_text_enabled=True, is_preview=True, max_llm_rows=5)
    tables = {}
    for table_name, batch in simulator.generate_all():
        tables[table_name] = batch
    evidence = merge_evidence(simulator._text_generation_evidence)
    assert evidence["summary"]["column_count"] >= 1
    assert "support_message" in tables["records"].columns
