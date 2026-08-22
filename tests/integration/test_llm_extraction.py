"""LLM-backed extraction is a sibling ExtractionPass producing the canonical IR.

Uses a deterministic FAKE ChatModel so tests run fully offline — no network, no
Ollama. Proves: the LLM path yields the same IR shape; untrusted LLM output is
validated against the schema (unknown entities/fields dropped, malformed JSON
rejected); a hostile response cannot inject tables or break key/FK integrity;
and the full deterministic pipeline runs on LLM-extracted data.
"""
import json

import pytest

from synth_platform.infrastructure.extraction.llm_pass import LlmExtractionPass
from synth_platform.infrastructure.sources.frame_connector import FrameConnector
from synth_platform.application.use_cases.generate_dataset import generate_dataset
from synth_platform.application.use_cases.train_model import train_model
from synth_platform.domain.extraction.target_schema import (
    ExtractionStrategy, FieldType, TargetEntity, TargetField, TargetParent, TargetSchema,
)
from synth_platform.domain.generation.models import GenerationRequest
from synth_platform.errors import ExtractionError


class FakeChatModel:
    """Returns a canned JSON completion — the deterministic stand-in for Ollama."""
    name = "fake"

    def __init__(self, completion: str):
        self._completion = completion

    def complete(self, system, user, *, json_only=True):
        return self._completion


def _schema():
    return TargetSchema(entities=[
        TargetEntity(name="study", primary_key="study_id",
                     strategy=ExtractionStrategy.SINGLETON,
                     fields=[TargetField(name="study_number", type=FieldType.IDENTIFIER)]),
        TargetEntity(name="criterion", primary_key="criterion_id",
                     strategy=ExtractionStrategy.ROWS_PER_LIST_ITEM,
                     parent=TargetParent(entity="study", fk_field="study_id"),
                     fields=[TargetField(name="text", type=FieldType.FREE_TEXT)]),
    ])


def test_llm_path_produces_canonical_ir():
    completion = json.dumps({
        "study": [{"study_number": "ABC-1"}],
        "criterion": [{"text": "adult"}, {"text": "consented"}],
    })
    ds = LlmExtractionPass(FakeChatModel(completion), "irrelevant doc text", _schema()).extract()
    ds.validate()
    assert ds.tables["study"].iloc[0]["study_number"] == "ABC-1"
    assert len(ds.tables["criterion"]) == 2
    assert set(ds.tables["criterion"]["study_id"]) == {1}          # parent FK wired
    assert {(fk.parent_table, fk.child_table) for fk in ds.schema.foreign_keys} == {
        ("study", "criterion")}


def test_hostile_output_cannot_inject_tables_or_fields():
    # LLM tries to add an unknown entity + unknown field + break the key.
    completion = json.dumps({
        "study": [{"study_number": "X", "id": 999, "__proto__": "evil"}],
        "criterion": [{"text": "ok"}],
        "secret_admin_table": [{"password": "hunter2"}],
    })
    ds = LlmExtractionPass(FakeChatModel(completion), "doc", _schema()).extract()
    ds.validate()
    # unknown entity dropped; schema is authoritative
    assert set(ds.tables) == {"study", "criterion"}
    # unknown fields dropped; only schema fields + allocated keys survive
    assert set(ds.tables["study"].columns) == {"study_id", "study_number"}
    # the model could NOT set the PK — it is allocated deterministically
    assert ds.tables["study"].iloc[0]["study_id"] == 1


def test_malformed_json_raises_typed_error():
    with pytest.raises(ExtractionError):
        LlmExtractionPass(FakeChatModel("not json at all"), "doc", _schema()).extract()


def test_non_object_output_raises():
    with pytest.raises(ExtractionError):
        LlmExtractionPass(FakeChatModel("[1,2,3]"), "doc", _schema()).extract()


def test_llm_extracted_data_flows_through_full_pipeline():
    completion = json.dumps({
        "study": [{"study_number": "S1"}],
        "criterion": [{"text": f"item {i}"} for i in range(6)],
    })
    ds = LlmExtractionPass(FakeChatModel(completion), "doc", _schema()).extract()
    art = train_model(FrameConnector(ds), sample_size=100, seed=1)
    del ds
    syn = generate_dataset(art, GenerationRequest(root_table_rows={"study": 1}, seed=1))
    study_ids = set(syn["study"]["study_id"])
    assert syn["study"]["study_id"].is_unique
    assert syn["criterion"]["study_id"].isin(study_ids).all()   # zero orphans
