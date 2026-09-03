"""AppTest coverage for Schema Mode UI."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/schema_twin.py"
SAMPLE_SCHEMA = Path("tests/fixtures/schema/schema_twin_minimal.sql")


def _install_fake_data_designer(monkeypatch, responses: list[str], calls: list[dict]) -> None:
    monkeypatch.setenv("SP_PLATFORM_SLM_PREFLIGHT", "0")
    original_find_spec = importlib.util.find_spec

    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeBuilder:
        def __init__(self, **kwargs):
            self.model_configs = kwargs.get("model_configs") or []
            self.columns = []
            calls.append({"builder_model_configs": self.model_configs})

        def add_column(self, config):
            self.columns.append(config)

        def with_seed_dataset(self, seed_source):
            self.seed_source = seed_source
            return self

        def build(self):
            return {
                "model_configs": [config.__dict__ for config in self.model_configs],
                "columns": [column.__dict__ for column in self.columns],
            }

    class FakeDataDesigner:
        def __init__(self, *, model_providers):
            self.model_providers = model_providers

        def preview(self, builder, *, num_records):
            llm_column = next((column for column in builder.columns if getattr(column, "name", None) == "schema_json"), None)
            calls.append(
                {
                    "model_alias": builder.model_configs[0].alias,
                    "provider": builder.model_configs[0].provider,
                    "model": builder.model_configs[0].model,
                    "skip_health_check": builder.model_configs[0].skip_health_check,
                    "prompt": getattr(llm_column, "prompt", None),
                    "column_type": getattr(llm_column, "column_type", None),
                    "output_format": getattr(llm_column, "output_format", None),
                    "columns": builder.columns,
                    "num_records": num_records,
                }
            )
            if llm_column is not None:
                return [{"schema_json": responses.pop(0)}]
            return [
                {getattr(column, "name", f"column_{idx}"): f"value_{row}_{idx}" for idx, column in enumerate(builder.columns)}
                for row in range(num_records)
            ]

    fake_dd = types.SimpleNamespace(
        DataDesignerConfigBuilder=FakeBuilder,
        DataFrameSeedSource=FakeConfig,
        ModelConfig=FakeConfig,
        ChatCompletionInferenceParams=FakeConfig,
        ModelProvider=FakeConfig,
        SamplerColumnConfig=FakeConfig,
        SamplerType=types.SimpleNamespace(
            UUID="uuid",
            UNIFORM="uniform",
            BERNOULLI="bernoulli",
            DATETIME="datetime",
            CATEGORY="category",
        ),
        UUIDSamplerParams=FakeConfig,
        UniformSamplerParams=FakeConfig,
        BernoulliSamplerParams=FakeConfig,
        DatetimeSamplerParams=FakeConfig,
        CategorySamplerParams=FakeConfig,
        LLMTextColumnConfig=FakeConfig,
        LLMStructuredColumnConfig=FakeConfig,
    )
    fake_interface = types.SimpleNamespace(DataDesigner=FakeDataDesigner)

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "data_designer.config" else original_find_spec(name),
    )
    monkeypatch.setitem(sys.modules, "data_designer.config", fake_dd)
    monkeypatch.setitem(sys.modules, "data_designer.interface", fake_interface)


def test_schema_twin_collects_schema_rules_and_generation_config(monkeypatch):
    calls = []
    _install_fake_data_designer(monkeypatch, [], calls)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    uploader = at.get("file_uploader")[0]
    uploader.upload(
        SAMPLE_SCHEMA.name,
        SAMPLE_SCHEMA.read_bytes(),
        "application/sql",
    )
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["schema_config"] is not None

    body = "\n".join(str(item.value) for item in [*at.markdown, *at.subheader])
    assert "Ask AI to define the schema" in body
    assert "Define rules" in body
    assert "Configure preview" in body
    assert any(area.label == "Describe the dataset you need" for area in at.text_area)
    assert "AI schema draft settings" not in body
    assert sum(1 for b in at.button if b.label == "Test AI provider") == 0
    assert not any(b.label == "Data Designer preflight" for b in at.button)
    assert any(b.label == "Ask AI to draft schema" for b in at.button)

    rule_area = next(area for area in at.text_area if area.label == "Global generation rules")
    at = rule_area.set_value("Use realistic customer names and paid/pending/cancelled order status values.").run()
    assert not at.exception, [str(e) for e in at.exception]

    assert at.session_state["schema_rules"]
    assert any(n.label == "users" for n in at.number_input)
    assert any(n.label == "orders" for n in at.number_input)
    assert not any(s.label == "Model provider" for s in at.selectbox)
    assert not any(t.label == "Model" for t in at.text_input)
    assert not any(t.label == "API key" for t in at.text_input)
    assert any(b.label == "Preview records" for b in at.button)

    preview_button = next(button for button in at.button if button.label == "Preview records")
    at = preview_button.click().run()
    assert not at.exception, [str(e) for e in at.exception]

    result = at.session_state["schema_result"]
    assert result["preview_tables"]["users"].shape[0] > 0
    assert result["preview_tables"]["orders"].shape[0] > 0
    assert result["data_designer_configs"]["schema_contract"]["validation"]["hard_checks_passed"] is True


def test_schema_twin_ai_draft_uses_generic_prompt_workflow(monkeypatch):
    calls = []
    responses = [
        """
        {
          "name": "tiny_ecommerce",
          "description": "Tiny ecommerce schema",
          "domain": "ecommerce",
          "tables": [
            {
              "name": "users",
              "row_count": 2,
              "columns": [
                {"name": "user_id", "type": "int", "unique": true, "nullable": false},
                {"name": "email", "type": "email", "unique": true, "nullable": false}
              ]
            },
            {
              "name": "orders",
              "row_count": 2,
              "columns": [
                {"name": "order_id", "type": "int", "unique": true, "nullable": false},
                {"name": "user_id", "type": "foreign_key", "nullable": false},
                {"name": "amount", "type": "float", "params": {"min": 1, "max": 100}}
              ]
            }
          ],
          "relationships": [
            {"parent_table": "users", "parent_key": "user_id", "child_table": "orders", "child_key": "user_id"}
          ]
        }
        """.strip(),
    ]

    _install_fake_data_designer(monkeypatch, responses, calls)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_SCHEMA_DRAFT_CACHE", "0")

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    prompt = next(area for area in at.text_area if area.label == "Describe the dataset you need")
    at = prompt.set_value(
        "Create an ecommerce schema with users and orders. Include user to order relationships."
    ).run()
    assert not at.exception, [str(e) for e in at.exception]

    draft_button = next(button for button in at.button if button.label == "Ask AI to draft schema")
    assert not draft_button.disabled
    at = draft_button.click().run()
    assert not at.exception, [str(e) for e in at.exception]

    assert calls
    assert '"name": "tiny_ecommerce"' in at.session_state["schema_ai_ddl"]
    assert '"users"' in at.session_state["schema_ai_ddl"]
    assert '"orders"' in at.session_state["schema_ai_ddl"]
    preview_calls = [call for call in calls if call.get("num_records") == 1]
    assert len(preview_calls) == 1
    assert preview_calls[0]["model_alias"] == "schema-draft-generator"
    assert preview_calls[0]["provider"] == "internal"
    assert preview_calls[0]["model"] == "synth-platform-slm"
    assert preview_calls[0]["output_format"] is None
    assert "Required table names inferred from the user request" in preview_calls[0]["prompt"]
