from __future__ import annotations

import json
import importlib.machinery
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

    class FakeDataDesignerConfigBuilder:
        def __init__(self, model_configs=None):
            captured["builder_model_configs"] = model_configs

        def with_seed_dataset(self, source):
            captured["seed_source"] = source
            return self

        def add_column(self, column):
            captured["column"] = column
            return self

    class FakeDataDesigner:
        def __init__(self, **kwargs):
            captured["data_designer"] = kwargs

        def preview(self, _builder, *, num_records):
            captured["preview_num_records"] = num_records
            captured.setdefault("preview_calls", []).append(
                {
                    "num_records": num_records,
                    "seed_frame": captured.get("seed_frame"),
                    "column": captured.get("column"),
                }
            )
            column = captured.get("column")
            column_name = getattr(column, "kwargs", {}).get("name", "rows_json")
            return types.SimpleNamespace(dataset=pd.DataFrame([{column_name: rows_json}]))

    fake_config = types.SimpleNamespace(
        ModelProvider=FakeModelProvider,
        ModelConfig=FakeModelConfig,
        ChatCompletionInferenceParams=FakeChatCompletionInferenceParams,
        DataFrameSeedSource=FakeDataFrameSeedSource,
        LLMTextColumnConfig=FakeLLMTextColumnConfig,
        LLMStructuredColumnConfig=FakeLLMStructuredColumnConfig,
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

    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_ENDPOINT", "https://internal-llm.example.com/v1")

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
    assert rows[0]["speaker"] == "Customer"
    assert rows[1]["speaker"] == "Agent"
    assert all(row["text"] for row in rows)


def test_nemo_backend_transcript_uses_data_designer_structured_column(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "turns": [
                {"turn": 1, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
                {"turn": 2, "speaker": "Agent", "timestamp": "", "text": "I can help with that synthetic claim."},
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
    assert column["name"] == "synthetic_transcript_json"
    assert column["model_alias"] == "transcript-generator"
    assert column["output_format"]["required"] == ["turns"]
    assert captured["model_provider"]["api_key"] is None
    assert captured["model_provider"]["endpoint"] == "http://localhost:8000/v1"
    assert model_config["model"] == "local/slm"
    assert model_config["provider"] == "internal"
    assert rows == [
        {"turn": 1, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
        {"turn": 2, "speaker": "Agent", "timestamp": "", "text": "I can help with that synthetic claim."},
    ]


def test_nemo_backend_transcript_allows_local_model_without_api_key(monkeypatch):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_PROVIDER", "local")
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MODEL", "local/slm")
    captured = _install_fake_data_designer(
        monkeypatch,
        {
            "turns": [
                {"turn": 1, "speaker": "Customer", "timestamp": "", "text": "I need help with my synthetic claim."},
                {"turn": 2, "speaker": "Agent", "timestamp": "", "text": "I can help with that synthetic claim."},
            ]
        },
    )
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    rows = GeneratorFactory.create("nemo").generate_transcript(contract, turn_count=2, seed=1)

    assert len(rows) == 2
    assert captured["model_provider"]["api_key"] is None
    assert captured["model_provider"]["endpoint"] == "http://localhost:8000/v1"
    assert captured["model_config"]["provider"] == "local"
    assert captured["model_config"]["model"] == "local/slm"
    assert captured["inference_parameters"]["extra_body"] is None


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
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MODEL", "nvidia/custom-runtime-model")
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
    assert model_config["model"] == "nvidia/custom-runtime-model"
    assert result.metadata["data_designer"]["model_alias"] == "schema-row-generator"
    assert result.metadata["data_designer"]["provider"] == "internal"
    assert result.row_counts == {"users": 2}


def test_nemo_backend_schema_invalid_json_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_API_KEY", "test-key")
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
    assert values["model"] == "local/slm"
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
        json.dumps(
            {
                "values": {
                    "field_1": "Alex Morgan",
                    "inline_1__s0": "A123",
                    "table_1__r0__c0": "2023-04-15",
                }
            }
        ),
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
    assert values["fields"]["field_1"]["value"] == "Alex Morgan"
    assert values["inline_spans"]["inline_1"][0]["value"] == "A123"
    assert values["tables"]["table_1"][1][0]["value"] == "2023-04-15"


def test_unknown_generation_backend_is_rejected():
    with pytest.raises(ValueError):
        GeneratorFactory.create("unknown")
