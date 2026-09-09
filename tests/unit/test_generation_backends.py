from __future__ import annotations

import json
import importlib.machinery
import importlib.util
import os
import sys
import types

import pandas as pd
import pytest

from synth_platform.engine.generation.backends import GeneratorFactory, NemoSchemaGenerationError
from synth_platform.engine.generation import backends
from synth_platform.application.workflows.schema_twin import load_schema_bytes
from synth_platform.engine.transcripts import build_transcript_contract, generate_synthetic_transcript


def _schema_contract():
    return load_schema_bytes(
        b"""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            email TEXT
        );
        """,
        "schema.sql",
        seed=1,
    )


def _install_fake_data_designer(monkeypatch, rows_json: str):
    monkeypatch.setenv("SP_PLATFORM_SLM_PREFLIGHT", "0")
    monkeypatch.setenv("SP_PLATFORM_SLM_MODEL", os.getenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm"))
    monkeypatch.setenv("SP_PLATFORM_SLM_PROVIDER", os.getenv("SP_PLATFORM_SLM_PROVIDER", "internal"))
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", os.getenv("SP_PLATFORM_SLM_ENDPOINT", "http://localhost:11434/v1"))
    captured: dict[str, object] = {}

    class FakeModelProvider:
        def __init__(self, **kwargs):
            captured["model_provider"] = kwargs

    class FakeModelConfig:
        def __init__(self, **kwargs):
            captured["model_config"] = kwargs

    class FakeChatCompletionInferenceParams:
        def __init__(self, **kwargs):
            captured["inference_parameters"] = kwargs

    class FakeDataFrameSeedSource:
        def __init__(self, df):
            captured["seed_frame"] = df

    class FakeLLMTextColumnConfig:
        def __init__(self, **kwargs):
            captured["llm_column"] = kwargs
            self.kwargs = kwargs

    class FakeLLMStructuredColumnConfig:
        def __init__(self, **kwargs):
            captured["llm_structured_column"] = kwargs
            self.kwargs = kwargs

    class FakeLocalCallableValidatorParams:
        def __init__(self, **kwargs):
            captured["validator_params"] = kwargs
            self.validation_function = kwargs["validation_function"]

    class FakeValidationColumnConfig:
        def __init__(self, **kwargs):
            captured["validation_column"] = kwargs
            self.kwargs = kwargs

    class FakeDataDesignerConfigBuilder:
        def __init__(self, model_configs=None):
            captured["builder_model_configs"] = model_configs
            self.columns = []

        def with_seed_dataset(self, source):
            captured["seed_source"] = source
            return self

        def add_column(self, column):
            self.columns.append(column)
            captured["column"] = column
            if isinstance(column, FakeLLMTextColumnConfig):
                captured["llm_column_instance"] = column
            if isinstance(column, FakeLLMStructuredColumnConfig):
                captured["llm_structured_column_instance"] = column
            if isinstance(column, FakeValidationColumnConfig):
                captured["validation_column_instance"] = column
            return self

    class FakeDataDesigner:
        def __init__(self, **kwargs):
            captured["data_designer"] = kwargs

        def preview(self, builder, *, num_records):
            captured["preview_num_records"] = num_records
            captured.setdefault("preview_calls", []).append(
                {
                    "num_records": num_records,
                    "seed_frame": captured.get("seed_frame"),
                    "column": captured.get("column"),
                }
            )
            column = (
                captured.get("llm_column_instance")
                or captured.get("llm_structured_column_instance")
                or captured.get("column")
            )
            column_name = getattr(column, "kwargs", {}).get("name", "rows_json")
            seed_frame = captured.get("seed_frame")
            if column_name in {"synthetic_message", "synthetic_value"} and isinstance(seed_frame, pd.DataFrame):
                values = rows_json if isinstance(rows_json, (dict, list)) else [str(rows_json)]
                records = seed_frame.head(num_records).to_dict(orient="records")
                for index, record in enumerate(records):
                    if isinstance(values, dict):
                        record[column_name] = values.get(record.get("binding_id"), "")
                    else:
                        record[column_name] = values[index % len(values)]
                validation_column = captured.get("validation_column_instance")
                if validation_column is not None:
                    validator_params = validation_column.kwargs["validator_params"]
                    validation = validator_params.validation_function(pd.DataFrame(records)).to_dict(orient="records")
                    for record, result in zip(records, validation):
                        record[validation_column.kwargs["name"]] = result
                return types.SimpleNamespace(dataset=pd.DataFrame(records))
            return types.SimpleNamespace(dataset=pd.DataFrame([{column_name: rows_json}]))

    fake_config = types.SimpleNamespace(
        ModelProvider=FakeModelProvider,
        ModelConfig=FakeModelConfig,
        ChatCompletionInferenceParams=FakeChatCompletionInferenceParams,
        DataFrameSeedSource=FakeDataFrameSeedSource,
        LLMTextColumnConfig=FakeLLMTextColumnConfig,
        LLMStructuredColumnConfig=FakeLLMStructuredColumnConfig,
        LocalCallableValidatorParams=FakeLocalCallableValidatorParams,
        ValidationColumnConfig=FakeValidationColumnConfig,
        ValidatorType=types.SimpleNamespace(LOCAL_CALLABLE="local_callable"),
        DataDesignerConfigBuilder=FakeDataDesignerConfigBuilder,
    )
    fake_interface = types.SimpleNamespace(DataDesigner=FakeDataDesigner)
    fake_package = types.ModuleType("data_designer")
    fake_package.__path__ = []
    fake_package.__spec__ = importlib.machinery.ModuleSpec("data_designer", loader=None, is_package=True)
    fake_package.config = fake_config
    fake_package.interface = fake_interface
    monkeypatch.setitem(sys.modules, "data_designer", fake_package)
    monkeypatch.setitem(sys.modules, "data_designer.config", fake_config)
    monkeypatch.setitem(sys.modules, "data_designer.interface", fake_interface)
    return captured


def test_current_backend_preserves_existing_transcript_generation():
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    backend = GeneratorFactory.create("current")

    assert backend.generate_transcript(contract, turn_count=4, seed=1) == generate_synthetic_transcript(
        contract,
        turn_count=4,
        seed=1,
    )


def test_nemo_backend_does_not_fake_generation_when_sdk_is_missing(monkeypatch):
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )
    backend = GeneratorFactory.create("nemo")

    monkeypatch.delenv("SP_PLATFORM_SLM_ENDPOINT", raising=False)
    monkeypatch.delenv("SP_PLATFORM_SLM_API_KEY_ENV", raising=False)
    monkeypatch.delenv("SP_PLATFORM_SLM_MODEL", raising=False)
    monkeypatch.delenv("SP_PLATFORM_SLM_PROVIDER", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "https://internal-llm.example.com/v1")
    monkeypatch.setattr(
        backends.util,
        "find_spec",
        lambda name: None if name in {"data_designer", "nemo_microservices"} else importlib.util.find_spec(name),
    )

    with pytest.raises((RuntimeError, NotImplementedError)):
        backend.generate_transcript(contract, turn_count=4, seed=1)


def test_nemo_backend_mock_mode_produces_distinct_transcript(monkeypatch):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )
    backend = GeneratorFactory.create("nemo")

    rows = backend.generate_transcript(contract, turn_count=4, seed=1)

    assert len(rows) == 4
    assert rows[0]["speaker"] == "Agent"
    assert rows[1]["speaker"] == "Customer"
    assert all(row["text"] for row in rows)
    assert "speaker_" not in " ".join(row["speaker"] for row in rows)


def test_nemo_backend_transcript_defaults_to_single_structured_data_designer_call(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.delenv("SP_TRANSCRIPT_GENERATION_MODE", raising=False)
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "turns": [
                {"turn": 1, "speaker": "Agent", "timestamp": "", "text": "I can review that synthetic claim."},
                {"turn": 2, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
            ]
        },
    )
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    rows = GeneratorFactory.create("nemo").generate_transcript(contract, turn_count=2, seed=1)

    column = captured["llm_structured_column"]
    model_config = captured["model_config"]
    inference = captured["inference_parameters"]
    assert column["name"] == "synthetic_transcript_json"
    assert column["model_alias"] == "transcript-generator"
    assert captured["preview_num_records"] == 1
    assert len(captured["seed_frame"]) == 1
    assert "turn_plan" in captured["seed_frame"].iloc[0]["source_contract_json"]
    assert inference["max_tokens"] == 768
    assert inference["timeout"] == 300
    assert inference["max_parallel_requests"] == 1
    assert captured["model_provider"]["api_key"] is None
    assert captured["model_provider"]["endpoint"] == "http://127.0.0.1:11434/v1"
    assert model_config["model"] == "synth-platform-slm"
    assert model_config["provider"] == "internal"
    assert rows == [
        {"turn": 1, "speaker": "Agent", "timestamp": "", "text": "I can review that synthetic claim."},
        {"turn": 2, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
    ]


def test_data_designer_transcript_twin_generates_ssot_and_turns_in_one_call(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_TRANSCRIPT_TWIN_OUTPUT_MODE", "conversation")
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "structured_ssot": {
                "metadata": {"source_transcript_id": "chat.txt", "channel": "customer_interaction", "version": "1.0.0"},
                "entities": {"participants": [{"speaker": "Agent", "role": "support_representative"}, {"speaker": "Customer", "role": "requester"}]},
                "support_context": {
                    "source": "sanitized_transcript_contract",
                    "issue_type": "claim_support",
                    "issue_summary": "Customer needed help with a claim.",
                    "topic_terms": ["claim", "support"],
                },
                "resolved_issues": [{"issue_type": "claim_support", "resolution_status": "in_progress", "resolution_summary": "Claim support remained in progress."}],
                "actions_taken": [{"action": "reviewed_claim", "detail": "Reviewed sanitized claim context."}],
                "account_mutations": [{"action": "support_state_update", "new_value": "claim_review_pending"}],
                "sentiment_analysis": {"initial_customer_sentiment": "needs_help", "final_customer_sentiment": "neutral"},
                "privacy_validation": {"raw_source_text_used": False, "source_identifiers_removed": True, "generated_from_sanitized_turns": True},
            },
            "turns": [
                {"turn": 1, "speaker": "Agent", "timestamp": "", "text": "I can review that synthetic claim."},
                {"turn": 2, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
            ],
        },
    )
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    result = backends.generate_transcript_twin_with_data_designer(contract, turn_count=2, seed=1)

    assert captured["preview_num_records"] == 1
    assert captured["llm_structured_column"]["name"] == "synthetic_twin_json"
    turns_schema = captured["llm_structured_column"]["output_format"]["properties"]["turns"]
    assert turns_schema["minItems"] == 2
    assert turns_schema["maxItems"] == 2
    assert result.metadata["mode"] == "transcript_twin_single_structured_preview"
    assert result.metadata["sdk_preview_calls"] == 1
    assert result.structured_ssot["status"] == "generated"
    assert result.turns == [
        {"turn": 1, "speaker": "Agent", "timestamp": "", "text": "I can review that synthetic claim."},
        {"turn": 2, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
    ]


def test_data_designer_transcript_twin_ssot_mode_does_not_require_turns(monkeypatch):
    captured = _install_fake_data_designer(monkeypatch, {})
    monkeypatch.setenv("SP_TRANSCRIPT_TWIN_OUTPUT_MODE", "ssot")
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
        nvidia_options={"enabled": True, "build_structured_ssot": False},
    )

    result = backends.generate_transcript_twin_with_data_designer(contract, turn_count=2, seed=1)

    assert result.metadata["mode"] == "ssot_only_sdk_plus_platform_turns"
    assert result.structured_ssot["status"] == "generated"
    assert len(result.turns) == 2
    assert captured["llm_structured_column"]["name"] == "synthetic_twin_ssot"
    assert "synthetic_twin_json" not in str(captured)


def test_transcript_json_schema_can_pin_requested_turn_count():
    schema = backends._transcript_twin_json_schema(12)

    turns_schema = schema["properties"]["turns"]
    assert turns_schema["minItems"] == 12
    assert turns_schema["maxItems"] == 12


def test_nemo_backend_transcript_can_use_data_designer_text_column_per_turn(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_TRANSCRIPT_GENERATION_MODE", "row")
    captured = _install_fake_data_designer(
        monkeypatch,
        ["I can review that synthetic claim.", "I need help with my synthetic claim."],
    )
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    rows = GeneratorFactory.create("nemo").generate_transcript(contract, turn_count=2, seed=1)

    column = captured["llm_column"]
    assert column["name"] == "synthetic_message"
    assert captured["preview_num_records"] == 2
    assert len(captured["seed_frame"]) == 2
    assert list(captured["seed_frame"]["speaker"]) == ["Agent", "Customer"]
    assert captured["validation_column"]["target_columns"] == ["synthetic_message"]
    assert rows == [
        {"turn": 1, "speaker": "Agent", "timestamp": "", "text": "I can review that synthetic claim."},
        {"turn": 2, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
    ]


def test_data_designer_transcript_rows_reject_repetitive_local_slm_output():
    rows = [
        {"turn": 1, "speaker": "Customer", "timestamp": "", "text": "I need help resetting my password in Chrome."},
        {"turn": 2, "speaker": "Agent", "timestamp": "", "text": "I need help resetting my password in Chrome."},
    ]

    with pytest.raises(RuntimeError, match="repeated_message"):
        backends._validate_transcript_row_batch_quality(rows, expected_rows=2)


def test_data_designer_transcript_validation_rejects_role_inverted_output():
    result = backends._validate_data_designer_transcript_message(
        {
            "speaker": "Agent",
            "speaker_role": "agent",
            "synthetic_message": "Can you please check why my recent order was charged twice?",
        }
    )

    assert result["is_valid"] is False
    assert "speaker_role_conflict" in result["error_messages"]


def test_data_designer_transcript_validation_rejects_source_replay():
    source_text = "Can you please check why my recent order was charged twice?"
    result = backends._validate_data_designer_transcript_message(
        {
            "speaker": "Customer",
            "speaker_role": "customer",
            "synthetic_message": source_text,
            "source_turn_hashes_json": json.dumps([backends._transcript_text_hash(source_text)]),
            "source_ngram_hashes_json": "[]",
        }
    )

    assert result["is_valid"] is False
    assert "source_replay" in result["error_messages"]


def test_transcript_seed_frame_turns_anonymized_speakers_into_customer_agent_dialogue():
    contract = build_transcript_contract(
        "Alice: I cannot log in.\nBob: I can help with the password reset.\nAlice: Thanks.",
        source_name="chat.txt",
    )

    frame = backends._transcript_turn_seed_frame(contract, target_turns=4, seed=1)

    assert list(frame["speaker"]) == ["Customer", "Agent", "Customer", "Agent"]
    assert set(frame["speaker_role"]) == {"customer", "agent"}
    assert set(frame["target_locale"]) == {"en_US"}
    assert "source_turn_hashes_json" in frame.columns
    assert "source_ngram_hashes_json" in frame.columns
    assert "MOCK_" not in frame.to_json()


def test_transcript_seed_frame_preserves_internal_external_role_signal():
    contract = build_transcript_contract(
        "internal : Thank you for calling support. How can I help?\n"
        "external : I need help with a card charge.\n"
        "internal : Let me review the charge.",
        source_name="call.txt",
    )

    frame = backends._transcript_turn_seed_frame(contract, target_turns=4, seed=1)

    assert list(frame["speaker"]) == ["Agent", "Customer", "Agent", "Customer"]
    assert list(frame["speaker_role"]) == ["agent", "customer", "agent", "customer"]
    assert set(frame["conversation_topic"]) == {"charge, card, how, review"}


def test_transcript_seed_frame_uses_structured_ssot_without_locale_env_override(monkeypatch):
    monkeypatch.setenv("SP_TRANSCRIPT_TARGET_LOCALE", "fr_FR")
    contract = build_transcript_contract(
        "Customer: I need help with an order charge.\nAgent: I can review the charge.",
        source_name="chat.txt",
    )
    metadata = contract.entities[0].metadata
    metadata["structured_ssot"] = {
        "status": "generated",
        "support_context": {
            "issue_type": "payment_charge_investigation",
            "issue_summary": "Customer asked about a possible duplicate order charge.",
            "topic_terms": "order, charge, payment",
        },
    }

    frame = backends._transcript_turn_seed_frame(contract, target_turns=2, seed=1)

    assert set(frame["conversation_topic"]) == {"Customer asked about a possible duplicate order charge."}
    assert set(frame["topic_terms"]) == {"order, charge, payment"}
    assert set(frame["target_locale"]) == {"en_US"}


def test_nemo_backend_transcript_allows_local_model_without_api_key(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "http://localhost:11434/v1")
    monkeypatch.setenv("SP_PLATFORM_SLM_PROVIDER", "local")
    monkeypatch.setenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm")
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "turns": [
                {"turn": 1, "speaker": "Agent", "timestamp": "", "text": "I can review that synthetic claim."},
                {"turn": 2, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
            ]
        },
    )
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    rows = GeneratorFactory.create("nemo").generate_transcript(contract, turn_count=2, seed=1)

    assert len(rows) == 2
    assert captured["preview_num_records"] == 1
    assert "turn_plan" in captured["seed_frame"].iloc[0]["source_contract_json"]
    assert captured["model_provider"]["api_key"] is None
    assert captured["model_provider"]["endpoint"] == "http://127.0.0.1:11434/v1"
    assert captured["model_config"]["provider"] == "local"
    assert captured["model_config"]["model"] == "synth-platform-slm"
    assert captured["inference_parameters"]["extra_body"] is None


def test_nemo_backend_transcript_rejects_model_scaffolding_with_sdk_validation(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "turns": [
                {"turn": 1, "speaker": "Customer", "text": "Speaker 4 says, \"I want to understand the plan benefits.\""},
                {"turn": 2, "speaker": "Agent", "text": "Hello from [Company Name]. I can help with your account."},
                {"turn": 3, "speaker": "Customer", "text": "Speaker: speaker_2"},
            ]
        },
    )
    contract = build_transcript_contract(
        "Customer: I am trying to activate my account.\n"
        "Agent: I can help with that account activation.\n"
        "Customer: What plan benefits are included?",
        source_name="chat.txt",
    )

    with pytest.raises(RuntimeError, match="failed SDK validation"):
        GeneratorFactory.create("nemo").generate_transcript(contract, turn_count=3, seed=1)

    assert captured["llm_structured_column"]["name"] == "synthetic_transcript_json"
    assert "turn_plan" in captured["seed_frame"].iloc[0]["source_contract_json"]


def test_nemo_backend_schema_mock_generates_tables(monkeypatch, tmp_path):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    schema = load_schema_bytes(
        b"""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            email TEXT
        );
        """,
        "schema.sql",
        seed=1,
    )
    backend = GeneratorFactory.create("nemo")

    result = backend.generate_schema(
        schema,
        row_count=3,
        table_row_counts={"users": 3},
        seed=1,
        output_dir=tmp_path,
        export_format="csv",
    )

    assert result.hard_checks_passed is True
    assert result.row_counts == {"users": 3}
    assert (tmp_path / "users.csv").exists()


def test_current_backend_schema_generation_is_not_backend_implemented():
    backend = GeneratorFactory.create("current")

    with pytest.raises(NotImplementedError):
        backend.generate_schema(_schema_contract())


def test_nemo_backend_schema_uses_data_designer_runtime_model(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test-key")
    monkeypatch.setenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm")
    monkeypatch.setenv("SP_SCHEMA_ROW_GENERATION_MODE", "llm")
    captured = _install_fake_data_designer(
        monkeypatch,
        json.dumps(
            [
                {"user_id": 1, "email": "one@example.com"},
                {"user_id": 2, "email": "two@example.com"},
            ]
        ),
    )
    backend = GeneratorFactory.create("nemo")

    result = backend.generate_schema(
        _schema_contract(),
        row_count=2,
        table_row_counts={"users": 2},
        seed=1,
        output_dir=tmp_path,
        export_format="csv",
    )

    model_config = captured["model_config"]
    assert model_config["alias"] == "schema-row-generator"
    assert model_config["provider"] == "internal"
    assert model_config["model"] == "synth-platform-slm"
    assert result.metadata["data_designer"]["model_alias"] == "schema-row-generator"
    assert result.metadata["data_designer"]["provider"] == "internal"
    assert result.row_counts == {"users": 2}


def test_nemo_backend_schema_invalid_json_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test-key")
    monkeypatch.setenv("SP_SCHEMA_ROW_GENERATION_MODE", "llm")
    _install_fake_data_designer(monkeypatch, "not json")
    backend = GeneratorFactory.create("nemo")

    with pytest.raises(NemoSchemaGenerationError, match="valid rows_json"):
        backend.generate_schema(
            _schema_contract(),
            row_count=2,
            table_row_counts={"users": 2},
            seed=1,
            output_dir=tmp_path,
            export_format="csv",
        )


def test_nemo_backend_schema_missing_column_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test-key")
    monkeypatch.setenv("SP_SCHEMA_ROW_GENERATION_MODE", "llm")
    _install_fake_data_designer(monkeypatch, json.dumps([{"user_id": 1}, {"user_id": 2}]))
    backend = GeneratorFactory.create("nemo")

    with pytest.raises(NemoSchemaGenerationError, match="missing required column"):
        backend.generate_schema(
            _schema_contract(),
            row_count=2,
            table_row_counts={"users": 2},
            seed=1,
            output_dir=tmp_path,
            export_format="csv",
        )


def test_nemo_backend_schema_trims_extra_rows_with_metadata(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test-key")
    monkeypatch.setenv("SP_SCHEMA_ROW_GENERATION_MODE", "llm")
    _install_fake_data_designer(
        monkeypatch,
        json.dumps(
            [
                {"user_id": idx + 1, "email": f"user{idx + 1}@example.com"}
                for idx in range(5)
            ]
        ),
    )
    backend = GeneratorFactory.create("nemo")

    result = backend.generate_schema(
        _schema_contract(),
        row_count=2,
        table_row_counts={"users": 2},
        seed=1,
        output_dir=tmp_path,
        export_format="csv",
    )

    repairs = result.metadata["data_designer"]["row_count_repairs"]
    assert result.row_counts == {"users": 2}
    assert repairs == [
        {
            "table": "users",
            "requested_rows": 2,
            "returned_rows": 5,
            "action": "trimmed_extra_rows",
        }
    ]
    assert (tmp_path / "users.csv").exists()


def test_nemo_backend_schema_missing_table_fails_before_export(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test-key")
    monkeypatch.setenv("SP_SCHEMA_ROW_GENERATION_MODE", "llm")
    _install_fake_data_designer(monkeypatch, "[]")
    backend = GeneratorFactory.create("nemo")

    with pytest.raises(NemoSchemaGenerationError, match="expected 2"):
        backend.generate_schema(
            _schema_contract(),
            row_count=2,
            table_row_counts={"users": 2},
            seed=1,
            output_dir=tmp_path,
            export_format="csv",
        )
    assert not (tmp_path / "users.csv").exists()


def test_nemo_backend_schema_defaults_to_deterministic_rows(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_SCHEMA_ROW_GENERATION_MODE", raising=False)
    backend = GeneratorFactory.create("nemo")

    result = backend.generate_schema(
        _schema_contract(),
        row_count=2,
        table_row_counts={"users": 2},
        seed=1,
        output_dir=tmp_path,
        export_format="csv",
    )

    assert result.hard_checks_passed is True
    assert result.row_counts == {"users": 2}
    assert result.metadata["data_designer"]["mode"] == "schema_draft_plus_deterministic_rows"
    assert result.metadata["data_designer"]["row_generation"] == "deterministic_platform_generator"


def test_nemo_backend_schema_deterministic_rows_sort_fk_parents(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_SCHEMA_ROW_GENERATION_MODE", raising=False)
    schema = load_schema_bytes(
        json.dumps(
            {
                "name": "school",
                "tables": [
                    {
                        "name": "courses",
                        "row_count": 5,
                        "columns": [
                            {"name": "course_id", "type": "uuid", "unique": True},
                            {"name": "teacher_id", "type": "foreign_key"},
                        ],
                    },
                    {
                        "name": "enrollments",
                        "row_count": 5,
                        "columns": [
                            {"name": "enrollment_id", "type": "uuid", "unique": True},
                            {"name": "student_id", "type": "foreign_key"},
                            {"name": "course_id", "type": "foreign_key"},
                        ],
                    },
                    {
                        "name": "students",
                        "row_count": 5,
                        "columns": [{"name": "student_id", "type": "uuid", "unique": True}],
                    },
                    {
                        "name": "teachers",
                        "row_count": 5,
                        "columns": [{"name": "teacher_id", "type": "uuid", "unique": True}],
                    },
                ],
                "relationships": [
                    {
                        "parent_table": "students",
                        "parent_key": "student_id",
                        "child_table": "enrollments",
                        "child_key": "student_id",
                    },
                    {
                        "parent_table": "courses",
                        "parent_key": "course_id",
                        "child_table": "enrollments",
                        "child_key": "course_id",
                    },
                    {
                        "parent_table": "teachers",
                        "parent_key": "teacher_id",
                        "child_table": "courses",
                        "child_key": "teacher_id",
                    },
                ],
            }
        ).encode("utf-8"),
        "school.json",
        seed=7,
    )

    result = GeneratorFactory.create("nemo").generate_schema(
        schema,
        row_count=20,
        table_row_counts={"students": 5, "teachers": 5, "courses": 5, "enrollments": 5},
        seed=7,
        output_dir=tmp_path,
        export_format="csv",
    )

    assert result.hard_checks_passed is True
    assert result.row_counts == {"students": 5, "teachers": 5, "courses": 5, "enrollments": 5}


def test_nemo_backend_schema_deterministic_rows_follow_categorical_probabilities(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_SCHEMA_ROW_GENERATION_MODE", raising=False)
    schema = load_schema_bytes(
        json.dumps(
            {
                "name": "weighted_status",
                "tables": [
                    {
                        "name": "orders",
                        "row_count": 50,
                        "columns": [
                            {"name": "order_id", "type": "uuid", "unique": True},
                            {
                                "name": "status",
                                "type": "categorical",
                                "choices": ["common", "rare"],
                                "probabilities": [0.9, 0.1],
                            },
                        ],
                    }
                ],
            }
        ).encode("utf-8"),
        "weighted_status.json",
        seed=1,
    )

    result = GeneratorFactory.create("nemo").generate_schema(
        schema,
        row_count=50,
        table_row_counts={"orders": 50},
        seed=1,
        output_dir=tmp_path,
        export_format="csv",
    )

    frame = pd.read_csv(tmp_path / "orders.csv")
    assert result.hard_checks_passed is True
    counts = frame["status"].value_counts().to_dict()
    assert counts["common"] > counts["rare"]


def test_nemo_schema_validation_reports_missing_table():
    report = backends._validate_schema_tables(_schema_contract(), {})

    assert report["hard_checks_passed"] is False
    assert "users: missing table" in report["issues"]


def test_nemo_backend_database_mock_writes_fk_safe_samples(monkeypatch, tmp_path):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    backend = GeneratorFactory.create("nemo")

    class Manifest:
        run_id = "test"
        run_dir = tmp_path

        def output_path(self, value: str):
            return tmp_path / value

        def record_stage(self, _result):
            return None

    contract = {
        "tables": {
            "customer": {
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "columns": {
                    "customer_id": {"physical_type": "TEXT", "semantic_type": "identifier", "nullable": False, "sensitive": True},
                    "email": {"physical_type": "TEXT", "semantic_type": "email", "nullable": False, "sensitive": True},
                },
                "business_rules": [],
            },
            "account": {
                "primary_key": ["account_id"],
                "foreign_keys": [
                    {"column": "customer_id", "references_table": "customer", "references_column": "customer_id"}
                ],
                "columns": {
                    "account_id": {"physical_type": "TEXT", "semantic_type": "identifier", "nullable": False, "sensitive": True},
                    "customer_id": {"physical_type": "TEXT", "semantic_type": "identifier", "nullable": False, "sensitive": True},
                    "balance": {"physical_type": "REAL", "semantic_type": "numerical", "nullable": False, "sensitive": False},
                },
                "business_rules": [],
            },
        }
    }

    result = backend.generate_database(
        contract,
        manifest=Manifest(),
        rows=10,
        seed=1,
        row_plan={"customer": 5, "account": 5},
    )

    assert result.metrics["overall_fk_validity"] == 1.0
    assert result.metrics["total_rows"] == 10
    assert (tmp_path / "generated_samples" / "customer.csv").exists()
    assert (tmp_path / "generated_samples" / "account.csv").exists()


def test_nemo_backend_pdf_mock_writes_values(monkeypatch, tmp_path):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    monkeypatch.setenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm")
    monkeypatch.setenv("SP_PLATFORM_SLM_PROVIDER", "internal")
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "http://localhost:11434/v1")
    backend = GeneratorFactory.create("nemo")

    class Manifest:
        run_id = "test"

        def output_path(self, value: str):
            return tmp_path / value

        def record_stage(self, _result):
            return None

    template = {
        "pages": [
            {
                "regions": [
                    {
                        "region_id": "field_1",
                        "region_type": "field",
                        "shape_pattern": "llll",
                    }
                ]
            }
        ]
    }
    binding_map = {
        "bindings": [
            {
                "binding_type": "field",
                "region_id": "field_1",
                "label": "Name",
                "semantic_role": "generic_text",
                "generator_strategy": "fake_generic_text",
            }
        ]
    }

    values = backend.generate_pdf(
        {
            "template": template,
            "binding_map": binding_map,
            "doc_id": "doc",
            "manifest": Manifest(),
            "binding_map_reference": "binding.json",
        },
        seed=1,
    )

    assert values["fields"]["field_1"]["value"].startswith("sdk_mock_")
    assert values["model_family"] == "open_source_slm"
    assert values["model"] == "synth-platform-slm"
    assert values["model_provider"] == "internal"
    assert (tmp_path / "documents" / "doc" / "document_synthetic_values.json").exists()


def test_nemo_pdf_sdk_output_must_include_required_field_values():
    template = {
        "pages": [{"regions": [{"region_id": "field_1", "region_type": "field"}]}],
    }
    binding_map = {
        "bindings": [
            {
                "binding_type": "field",
                "region_id": "field_1",
                "label": "Name",
                "semantic_role": "generic_text",
                "generator_strategy": "fake_generic_text",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="SDK output missing value"):
        backends._coerce_pdf_values_from_sdk(template, binding_map, {"fields": {}})


def test_nemo_pdf_values_schema_requires_discovered_binding_ids():
    plan = {
        "bindings": [
            {"binding_id": "field_1"},
            {"binding_id": "inline_1__s0"},
            {"binding_id": "table_1__r0__c0"},
        ]
    }

    schema = backends._pdf_values_json_schema(plan)
    values_schema = schema["properties"]["values"]

    assert values_schema["required"] == ["field_1", "inline_1__s0", "table_1__r0__c0"]
    assert set(values_schema["properties"]) == {"field_1", "inline_1__s0", "table_1__r0__c0"}
    assert values_schema["properties"]["field_1"] == {"type": "string"}
    assert values_schema["additionalProperties"] is False


def test_nemo_pdf_empty_binding_value_map_fails_without_legacy_fallback():
    template = {
        "pages": [{"regions": [{"region_id": "inline_1", "inline_variable_spans": [{"shape_pattern": "dddd"}]}]}],
    }
    binding_map = {
        "bindings": [
            {
                "binding_type": "inline_spans",
                "region_id": "inline_1",
                "spans": [{"start": 0, "end": 4, "semantic_role": "reference_code"}],
            }
        ]
    }

    with pytest.raises(RuntimeError, match=r"values\.inline_1__s0"):
        backends._coerce_pdf_values_from_sdk(template, binding_map, {"values": {}})


def test_nemo_pdf_sdk_output_accepts_region_id_arrays():
    template = {
        "pages": [
            {
                "regions": [
                    {"region_id": "field_1", "region_type": "field"},
                    {
                        "region_id": "inline_1",
                        "region_type": "paragraph",
                        "inline_variable_spans": [{"shape_pattern": "dddd"}],
                    },
                    {
                        "region_id": "table_1",
                        "region_type": "table",
                        "header_row_count": 1,
                        "rows": [
                            [{"text": "Code"}],
                            [{"text": "1234"}],
                        ],
                    },
                ]
            }
        ],
    }
    binding_map = {
        "bindings": [
            {
                "binding_type": "field",
                "region_id": "field_1",
                "label": "Name",
                "semantic_role": "person_name",
            },
            {
                "binding_type": "inline_spans",
                "region_id": "inline_1",
                "spans": [{"start": 0, "end": 4, "semantic_role": "reference_code"}],
            },
            {
                "binding_type": "table",
                "region_id": "table_1",
                "columns": [{"column_index": 0, "semantic_role": "reference_code"}],
            },
        ]
    }
    sdk_values = {
        "fields": [{"region_id": "field_1", "value": "Alex Morgan", "semantic_role": "person_name"}],
        "inline_spans": [{"region_id": "inline_1", "spans": [{"value": "A123"}]}],
        "tables": [{"region_id": "table_1", "rows": [[{"value": "B456"}]]}],
    }

    values = backends._coerce_pdf_values_from_sdk(template, binding_map, sdk_values)

    assert values["fields"]["field_1"]["value"] == "Alex Morgan"
    assert values["inline_spans"]["inline_1"][0]["value"] == "A123"
    assert values["tables"]["table_1"][1][0]["value"] == "B456"


def test_nemo_pdf_sdk_output_accepts_flat_table_cells():
    template = {
        "pages": [
            {
                "regions": [
                    {
                        "region_id": "table_1",
                        "region_type": "table",
                        "header_row_count": 1,
                        "rows": [
                            [{"text": "Date"}, {"text": "Amount"}],
                            [{"text": "2023-01-01"}, {"text": "10.00"}],
                        ],
                    },
                ]
            }
        ],
    }
    binding_map = {
        "bindings": [
            {
                "binding_type": "table",
                "region_id": "table_1",
                "columns": [
                    {"column_index": 0, "semantic_role": "date"},
                    {"column_index": 1, "semantic_role": "amount"},
                ],
            },
        ]
    }
    sdk_values = {
        "fields": [],
        "inline_spans": [],
        "tables": [
            {
                "region_id": "table_1",
                "cells": [
                    {"row_index": 0, "column_index": 0, "value": "2023-04-15", "semantic_role": "date"},
                    {"row_index": 0, "column_index": 1, "value": "25.50", "semantic_role": "amount"},
                ],
            }
        ],
    }

    values = backends._coerce_pdf_values_from_sdk(template, binding_map, sdk_values)

    assert values["tables"]["table_1"][1][0]["value"] == "2023-04-15"
    assert values["tables"]["table_1"][1][1]["value"] == "25.50"


def test_nemo_pdf_sdk_output_accepts_binding_value_map():
    template = {
        "pages": [
            {
                "regions": [
                    {"region_id": "field_1", "region_type": "field"},
                    {
                        "region_id": "inline_1",
                        "region_type": "paragraph",
                        "inline_variable_spans": [{"shape_pattern": "dddd"}],
                    },
                    {
                        "region_id": "table_1",
                        "region_type": "table",
                        "header_row_count": 1,
                        "rows": [
                            [{"text": "Date"}, {"text": "Amount"}],
                            [{"text": "2023-01-01"}, {"text": "10.00"}],
                        ],
                    },
                ]
            }
        ],
    }
    binding_map = {
        "bindings": [
            {"binding_type": "field", "region_id": "field_1", "label": "Name", "semantic_role": "person_name"},
            {
                "binding_type": "inline_spans",
                "region_id": "inline_1",
                "spans": [{"start": 0, "end": 4, "semantic_role": "reference_code"}],
            },
            {
                "binding_type": "table",
                "region_id": "table_1",
                "columns": [
                    {"column_index": 0, "semantic_role": "date"},
                    {"column_index": 1, "semantic_role": "amount"},
                ],
            },
        ]
    }
    sdk_values = {
        "values": {
            "field_1": "Alex Morgan",
            "inline_1__s0": "A123",
            "table_1__r0__c0": "2023-04-15",
            "table_1__r0__c1": "25.50",
        }
    }

    values = backends._coerce_pdf_values_from_sdk(template, binding_map, sdk_values)

    assert values["fields"]["field_1"]["value"] == "Alex Morgan"
    assert values["inline_spans"]["inline_1"][0]["value"] == "A123"
    assert values["tables"]["table_1"][1][0]["value"] == "2023-04-15"
    assert values["tables"]["table_1"][1][1]["value"] == "25.50"


def test_nemo_pdf_data_designer_batches_large_binding_plans(monkeypatch):
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "field_1": "Alex Morgan",
            "inline_1__s0": "A123",
            "table_1__r0__c0": "2023-04-15",
        },
    )
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test")
    monkeypatch.setenv("SP_PDF_DATA_DESIGNER_BINDINGS_PER_BATCH", "1")
    template = {
        "pages": [
            {
                "regions": [
                    {"region_id": "field_1", "region_type": "field"},
                    {
                        "region_id": "inline_1",
                        "region_type": "paragraph",
                        "inline_variable_spans": [{"shape_pattern": "dddd"}],
                    },
                    {
                        "region_id": "table_1",
                        "region_type": "table",
                        "header_row_count": 1,
                        "rows": [
                            [{"text": "Date"}],
                            [{"text": "2023-01-01"}],
                        ],
                    },
                ]
            }
        ],
    }
    binding_map = {
        "bindings": [
            {"binding_type": "field", "region_id": "field_1", "label": "Name", "semantic_role": "person_name"},
            {
                "binding_type": "inline_spans",
                "region_id": "inline_1",
                "spans": [{"start": 0, "end": 4, "semantic_role": "reference_code"}],
            },
            {
                "binding_type": "table",
                "region_id": "table_1",
                "columns": [{"column_index": 0, "semantic_role": "date"}],
            },
        ]
    }

    values = backends._generate_pdf_values_with_data_designer(template, binding_map, seed=1)

    assert len(captured["preview_calls"]) == 3
    assert captured["llm_column"]["name"] == "synthetic_value"
    assert captured["validation_column"]["target_columns"] == ["synthetic_value"]
    assert values["fields"]["field_1"]["value"] == "Alex Morgan"
    assert values["inline_spans"]["inline_1"][0]["value"] == "A123"
    assert values["tables"]["table_1"][1][0]["value"] == "2023-04-15"


def test_nemo_pdf_validation_repairs_chatty_local_slm_values(monkeypatch):
    monkeypatch.setenv("SP_PDF_DATA_DESIGNER_MAX_VALUE_CHARS", "40")

    validation = backends._validate_data_designer_pdf_value(
        "Binding id: p5_r3\nReplacement value: Synthetic Account Update Complete"
    )

    assert validation["is_valid"] is True
    assert backends._repair_pdf_synthetic_value(
        '{"value": "2026-04-18", "semantic_role": "date"}'
    ) == "2026-04-18"


def test_nemo_pdf_value_rows_use_repaired_values(monkeypatch):
    monkeypatch.setenv("SP_PDF_DATA_DESIGNER_MAX_VALUE_CHARS", "32")
    frame = pd.DataFrame(
        [
            {
                "binding_id": "field_1",
                "synthetic_value": "Here is a safe synthetic replacement value that is far too verbose for rendering.",
                "synthetic_value_validation": {"is_valid": True},
            }
        ]
    )

    values = backends._pdf_values_from_data_designer_rows(frame, chunk_index=1)

    assert values["field_1"]
    assert len(values["field_1"]) <= 32


def test_unknown_generation_backend_is_rejected():
    with pytest.raises(ValueError):
        GeneratorFactory.create("unknown")
