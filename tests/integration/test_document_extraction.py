"""Document -> canonical IR closes the compiler front-end (RC-8).

A parsed Document + a caller-supplied TargetSchema lifts into a RelationalDataset
with correct entities, keys, and parent FKs — driven entirely by the schema,
with zero hardcoded domain entities. Then the SAME back-end a database uses
trains/generates from it (PDF is a first-class source).
"""
from synth_platform.infrastructure.documents.model import Document, Line, Table
from synth_platform.infrastructure.extraction.document_pass import DocumentExtractionPass
from synth_platform.infrastructure.sources.frame_connector import FrameConnector
from synth_platform.domain.extraction.target_schema import (
    ExtractionStrategy, FieldType, TargetEntity, TargetField, TargetParent, TargetSchema,
)
from synth_platform.application.use_cases.generate_dataset import generate_dataset
from synth_platform.application.use_cases.train_model import train_model
from synth_platform.domain.generation.models import GenerationRequest


def _doc():
    return Document(lines=[
        Line("Study Number: ABC-123"),
        Line("Objectives"),
        Line("• first objective text"),
        Line("• second objective text"),
        Line("• third objective text"),
        Line("Inclusion criteria"),
        Line("a. must be an adult"),
        Line("b. signed consent"),
    ])


def _schema():
    return TargetSchema(entities=[
        TargetEntity(name="study", primary_key="study_id",
                     strategy=ExtractionStrategy.SINGLETON, aliases=[],
                     fields=[TargetField(name="study_number", type=FieldType.IDENTIFIER,
                                         aliases=["Study Number"])]),
        TargetEntity(name="objective", primary_key="objective_id",
                     strategy=ExtractionStrategy.ROWS_PER_LIST_ITEM, aliases=["Objectives"],
                     parent=TargetParent(entity="study", fk_field="study_id"),
                     fields=[TargetField(name="text", type=FieldType.FREE_TEXT)]),
        TargetEntity(name="inclusion", primary_key="inclusion_id",
                     strategy=ExtractionStrategy.ROWS_PER_LIST_ITEM, aliases=["Inclusion criteria"],
                     parent=TargetParent(entity="study", fk_field="study_id"),
                     fields=[TargetField(name="text", type=FieldType.FREE_TEXT)]),
    ])


def test_document_lifts_into_ir_with_keys_and_fks():
    ds = DocumentExtractionPass(_doc(), _schema()).extract()
    ds.validate()
    assert len(ds.tables["study"]) == 1
    assert ds.tables["study"].iloc[0]["study_number"] == "ABC-123"
    assert len(ds.tables["objective"]) == 3
    assert len(ds.tables["inclusion"]) == 2
    # parent FK present and pointing at the one study row
    assert set(ds.tables["objective"]["study_id"]) == {1}
    assert {(fk.parent_table, fk.child_table) for fk in ds.schema.foreign_keys} == {
        ("study", "objective"), ("study", "inclusion")}


def test_pdf_derived_ir_trains_and_generates_source_free():
    ds = DocumentExtractionPass(_doc(), _schema()).extract()
    conn = FrameConnector(ds)
    art = train_model(conn, sample_size=100, seed=1)
    del ds, conn   # source gone; runtime is source-free
    syn = generate_dataset(art, GenerationRequest(root_table_rows={"study": 1}, seed=1))
    study_ids = set(syn["study"]["study_id"])
    assert syn["study"]["study_id"].is_unique
    assert syn["objective"]["study_id"].isin(study_ids).all()   # zero orphans


def test_extraction_is_schema_driven_not_hardcoded():
    # the same document, a DIFFERENT schema -> different entities. No platform
    # knowledge of "study"/"objective" — it's all in the supplied TargetSchema.
    alt = TargetSchema(entities=[
        TargetEntity(name="thing", primary_key="thing_id",
                     strategy=ExtractionStrategy.ROWS_PER_LIST_ITEM, aliases=["Objectives"],
                     fields=[TargetField(name="body", type=FieldType.FREE_TEXT)])])
    ds = DocumentExtractionPass(_doc(), alt).extract()
    assert list(ds.tables.keys()) == ["thing"]
    assert len(ds.tables["thing"]) == 3
