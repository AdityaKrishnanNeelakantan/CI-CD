from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from reportlab.pdfgen import canvas

from synth_platform import GenerationRequest, SyntheticDataPlatform, TrainingRequest
from synth_platform.infrastructure.documents.pdf_router import PdfRouter
from synth_platform.infrastructure.sinks.database_sink import SqliteSink
from synth_platform.infrastructure.sources.frame_connector import FrameConnector
from synth_platform.infrastructure.sources.sqlite import SqliteSource
from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.application.use_cases.publish_dataset import publish_dataset
from synth_platform.application.use_cases.train_model import train_relational_dataset
from synth_platform.application.use_cases.validate_extraction import validate_extraction
from synth_platform.domain.artifacts.manifest import ArtifactManifest
from synth_platform.domain.extraction.target_schema import (
    ExtractionStrategy, FieldType, TargetEntity, TargetField, TargetSchema,
)
from synth_platform.domain.profiling.models import SamplingRecord
from synth_platform.domain.runs.models import SourceKind
from synth_platform.domain.schema.models import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from synth_platform.domain.validation.models import CheckResult, Status, ValidationReport
from synth_platform.errors import ArtifactIntegrityError, ExtractionQualityError, PublicationBlockedError


def test_sqlite_sampling_is_seeded_and_bounded(banking_db):
    source = SqliteSource(banking_db)
    try:
        a = source.sample_table("users", 25, seed=11)
        b = source.sample_table("users", 25, seed=11)
        c = source.sample_table("users", 25, seed=12)
    finally:
        source.close()
    assert len(a) == len(b) == len(c) == 25
    assert a["id"].tolist() == b["id"].tolist()
    assert a["id"].tolist() != c["id"].tolist()


def test_artifact_is_signed_has_tensors_and_source_identity(banking_db, tmp_path):
    platform = SyntheticDataPlatform.from_settings()
    artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", seed=4))
    path = platform.export(artifact, tmp_path / "db.synthpkg")
    assert artifact.manifest.source_kind == SourceKind.DATABASE
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        payload = b"\n".join(archive.read(name) for name in names)
    assert {"weights/model.safetensors", "signature.ed25519", "checksums.sha256"} <= names
    assert banking_db.encode() not in payload
    assert b"sqlite:///" not in payload


def test_signature_rejects_recomputed_checksums_without_trusted_key(banking_db, tmp_path):
    platform = SyntheticDataPlatform.from_settings()
    artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", seed=4))
    path = platform.export(artifact, tmp_path / "db.synthpkg")
    with zipfile.ZipFile(path) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    data["schema.json"] = data["schema.json"].replace(b'"source_kind":"sqlite_ro"', b'"source_kind":"tampered"')
    checks = []
    for name, blob in sorted(data.items()):
        if name not in {"checksums.sha256", "signature.ed25519"}:
            checks.append(f"{hashlib.sha256(blob).hexdigest()}  {name}")
    data["checksums.sha256"] = ("\n".join(checks) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, blob in data.items():
            archive.writestr(name, blob)
    with pytest.raises(ArtifactIntegrityError, match="signature"):
        platform.load(path)


def test_generated_child_counts_follow_generated_parents(banking_db):
    platform = SyntheticDataPlatform.from_settings()
    artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", seed=7))
    tables = platform.generate(artifact, GenerationRequest(root_table_rows={"users": 150}, seed=7))
    source_max = artifact.relational_plan.cardinality_models["accounts->users"].maximum
    counts = tables["accounts"].groupby("user_id").size()
    assert len(tables["users"]) == 150
    assert len(tables["accounts"]) > 0
    assert int(counts.max()) <= source_max
    assert tables["accounts"]["user_id"].isin(set(tables["users"]["id"])).all()


def test_zero_row_child_stays_zero():
    schema = DatabaseSchema(source_kind="test", tables={
        "parent": TableSchema(name="parent", primary_key="id", columns=[
            ColumnSchema(name="id", physical_type="integer", nullable=False)]),
        "child": TableSchema(name="child", primary_key="id", columns=[
            ColumnSchema(name="id", physical_type="integer", nullable=False),
            ColumnSchema(name="parent_id", physical_type="integer", nullable=False)]),
    }, foreign_keys=[ForeignKey(parent_table="parent", parent_column="id",
                                child_table="child", child_column="parent_id")])
    dataset = RelationalDataset(schema=schema, tables={
        "parent": pd.DataFrame({"id": [1, 2, 3]}),
        "child": pd.DataFrame({"id": pd.Series(dtype=int),
                               "parent_id": pd.Series(dtype=int)}),
    }).finalize_counts()
    artifact = train_relational_dataset(
        dataset, source_kind=SourceKind.DATABASE, source_fingerprint="a" * 64,
        sample_records={name: SamplingRecord(population_count=len(frame), sample_count=len(frame))
                        for name, frame in dataset.tables.items()},
        seed=1, rare_threshold=10)
    tables = SyntheticDataPlatform.from_settings().generate(
        artifact, GenerationRequest(root_table_rows={"parent": 10}, seed=1))
    assert len(tables["parent"]) == 10
    assert len(tables["child"]) == 0


def _simple_pdf(path: Path):
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, "Account Number: SOURCE-123")
    pdf.drawString(72, 700, "Beginning Balance: 1000.00")
    pdf.drawString(72, 680, "Ending Balance: 1000.00")
    pdf.save()


def _pdf_schema():
    return TargetSchema(entities=[TargetEntity(
        name="account", primary_key="account_id",
        strategy=ExtractionStrategy.SINGLETON,
        fields=[
            TargetField(name="account_number", type=FieldType.IDENTIFIER,
                        required=True, aliases=["Account Number"]),
            TargetField(name="beginning_balance", type=FieldType.NUMBER,
                        required=True, aliases=["Beginning Balance"]),
            TargetField(name="ending_balance", type=FieldType.NUMBER,
                        required=True, aliases=["Ending Balance"]),
        ])])


def test_pdf_track_is_independent_and_renders_only_pdf_artifacts(tmp_path, banking_db):
    pdf = tmp_path / "input.pdf"; _simple_pdf(pdf)
    platform = SyntheticDataPlatform.from_settings()
    result = platform.train_pdf(pdf.read_bytes(), _pdf_schema(), tmp_path / "pdf.synthpkg",
                                extraction_mode="native", seed=5)
    assert result.artifact.manifest.source_kind == SourceKind.PDF
    assert result.artifact.manifest.artifact_role == "document_relational"
    assert result.extraction_report.passed
    tables = platform.generate(result.artifact,
                               GenerationRequest(root_table_rows={"account": 2}, seed=5))
    report = platform.validate(result.artifact, tables)
    assert report.overall == Status.PASS
    rendered = platform.render_synthetic_pdfs(
        result.artifact, tables, tmp_path / "synthetic.pdf", "generation-1")
    assert rendered.valid and Path(rendered.path).exists()
    db_artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}"))
    with pytest.raises(ValueError, match="source kind mismatch"):
        platform.render_synthetic_pdfs(db_artifact, {}, tmp_path / "bad.pdf", "generation-2")


def test_image_only_pdf_fails_closed_without_ocr(tmp_path):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (800, 300), "white")
    ImageDraw.Draw(image).text((40, 80), "IMAGE ONLY STATEMENT", fill="black")
    png = tmp_path / "page.png"; image.save(png)
    pdf = tmp_path / "image-only.pdf"
    c = canvas.Canvas(str(pdf)); c.drawImage(str(png), 20, 500, width=550, height=200); c.save()
    with pytest.raises(ExtractionQualityError, match="OCR is required"):
        PdfRouter(ocr_adapter=None).load(pdf, mode="auto")


def test_release_gate_is_only_publication_door(banking_db, tmp_path):
    platform = SyntheticDataPlatform.from_settings()
    artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", seed=2))
    tables = platform.generate(artifact, GenerationRequest(root_table_rows={"users": 20}, seed=2))
    report = platform.validate(artifact, tables)
    assert report.overall == Status.PASS
    destination = tmp_path / "approved.db"
    publish_dataset(artifact, tables, report, SqliteSink(destination), "dataset-1")
    conn = sqlite3.connect(destination)
    try:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 20
    finally:
        conn.close()
    failed = ValidationReport(checks=[CheckResult(
        name="critical", dimension="structural", status=Status.FAIL,
        ran=True, critical=True)])
    with pytest.raises(PublicationBlockedError):
        publish_dataset(artifact, tables, failed, SqliteSink(tmp_path / "blocked.db"), "dataset-2")
    assert not (tmp_path / "blocked.db").exists()


def test_source_free_generation_in_separate_process(banking_db, tmp_path):
    platform = SyntheticDataPlatform.from_settings()
    artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", seed=9))
    package = platform.export(artifact, tmp_path / "model.synthpkg")
    os.remove(banking_db)
    code = (
        "from synth_platform import SyntheticDataPlatform,GenerationRequest;"
        f"a=SyntheticDataPlatform.from_settings().load(r'{package}');"
        "t=SyntheticDataPlatform.from_settings().generate(a,GenerationRequest(root_table_rows={'users':12},seed=9));"
        "print(len(t['users']),len(t['accounts']))"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.split()[0] == "12"


def test_manifest_rejects_source_urls():
    with pytest.raises(ValueError):
        ArtifactManifest(source_schema_fingerprint="postgresql://user:secret@host/db")


def test_complex_document_headers_negatives_and_footnotes_are_preserved():
    from synth_platform.infrastructure.documents.model import Document, Table
    from synth_platform.infrastructure.extraction.document_pass import DocumentExtractionPass

    document = Document(tables=[Table(rows=[
        ["Category", "2008", ""],
        ["Name", "Entered", "Won"],
        ["Alpha", "(1,234)", "7*"],
        ["Beta", "250", "3"],
    ], page=0, title="Competition results")])
    schema = TargetSchema(entities=[TargetEntity(
        name="result", primary_key="result_id",
        aliases=["Competition results"],
        strategy=ExtractionStrategy.ROWS_PER_TABLE_ROW,
        fields=[
            TargetField(name="name", type=FieldType.STRING, required=True,
                        aliases=["Name"]),
            TargetField(name="entered", type=FieldType.NUMBER, required=True,
                        aliases=["2008 Entered"]),
            TargetField(name="won", type=FieldType.INTEGER, required=True,
                        aliases=["2008 Won"]),
        ])])
    extractor = DocumentExtractionPass(document, schema)
    dataset = extractor.extract()
    report = validate_extraction(document, dataset, schema)
    assert report.passed
    assert len(dataset.tables["result"]) == 2
    assert dataset.tables["result"].iloc[0]["entered"] == -1234.0
    assert dataset.tables["result"].iloc[0]["won"] == 7
    assert any(item.get("footnote_marker") == "*"
               for item in dataset.provenance["result"])


def test_multiline_singleton_value_is_not_truncated_by_capitalization():
    from synth_platform.infrastructure.documents.model import Document, Line
    from synth_platform.infrastructure.extraction.document_pass import DocumentExtractionPass

    document = Document(lines=[
        Line("Description:"),
        Line("First line starts with a capital letter"),
        Line("Second line remains part of the value"),
        Line("Status: Active"),
    ])
    schema = TargetSchema(entities=[TargetEntity(
        name="record", primary_key="record_id",
        strategy=ExtractionStrategy.SINGLETON,
        fields=[
            TargetField(name="description", type=FieldType.FREE_TEXT,
                        required=True, aliases=["Description"]),
            TargetField(name="status", type=FieldType.CATEGORICAL,
                        required=True, aliases=["Status"]),
        ])])
    dataset = DocumentExtractionPass(document, schema).extract()
    value = dataset.tables["record"].iloc[0]["description"]
    assert value == "First line starts with a capital letter\nSecond line remains part of the value"
    assert dataset.tables["record"].iloc[0]["status"] == "Active"


def test_required_field_coverage_blocks_weak_pdf_extraction():
    from synth_platform.infrastructure.documents.model import Document, Line
    from synth_platform.infrastructure.extraction.document_pass import DocumentExtractionPass

    document = Document(lines=[Line("Account Number: A-1")])
    schema = TargetSchema(entities=[TargetEntity(
        name="account", primary_key="account_id",
        strategy=ExtractionStrategy.SINGLETON,
        fields=[
            TargetField(name="account_number", type=FieldType.IDENTIFIER,
                        required=True, aliases=["Account Number"]),
            TargetField(name="ending_balance", type=FieldType.NUMBER,
                        required=True, aliases=["Ending Balance"]),
        ])])
    dataset = DocumentExtractionPass(document, schema).extract()
    report = validate_extraction(document, dataset, schema)
    assert not report.passed
    assert "required_coverage[account.ending_balance]" in report.blocking_reasons


def test_pdf_statement_reconciliation_is_enforced_and_rendered(tmp_path):
    pdf = tmp_path / "statement.pdf"
    c = canvas.Canvas(str(pdf))
    labels = [
        ("Beginning Balance", "1000.00"),
        ("Credits", "300.00"),
        ("Debits", "125.00"),
        ("Fees", "10.00"),
        ("Interest", "5.00"),
        ("Ending Balance", "1170.00"),
    ]
    y = 740
    for label, value in labels:
        c.drawString(72, y, f"{label}: {value}")
        y -= 25
    c.save()
    schema = TargetSchema(entities=[TargetEntity(
        name="account", primary_key="account_id",
        strategy=ExtractionStrategy.SINGLETON,
        fields=[TargetField(
            name=label.lower().replace(" ", "_"), type=FieldType.NUMBER,
            required=True, aliases=[label]) for label, _ in labels],
    )])
    platform = SyntheticDataPlatform.from_settings()
    result = platform.train_pdf(
        pdf.read_bytes(), schema, tmp_path / "statement.synthpkg",
        extraction_mode="native", seed=21)
    tables = platform.generate(
        result.artifact, GenerationRequest(root_table_rows={"account": 8}, seed=21))
    account = tables["account"]
    expected = (
        account["beginning_balance"] + account["credits"] -
        account["debits"] - account["fees"] + account["interest"]
    )
    assert (account["ending_balance"] - expected).abs().max() < 1e-8
    report = platform.validate(result.artifact, tables)
    assert report.overall == Status.PASS
    package = platform.render_synthetic_pdf_package(
        result.artifact, tables, report, tmp_path / "rendered",
        generation_run_id="statement-run", document_count=3)
    assert Path(package.package_path).exists()
    assert all(document["valid"] for document in package.documents)
    bound_accounts = [document["bound_entity_ids"]["account"]
                      for document in package.documents]
    assert len(set(bound_accounts)) == 3


def test_confirmed_conditional_constraint_is_generated_and_validated():
    from synth_platform.domain.constraints.models import ConstraintDefinition, ConstraintKind

    schema = DatabaseSchema(source_kind="test", tables={
        "case": TableSchema(
            name="case", primary_key="id",
            primary_key_columns=["id"], primary_key_confirmed=True,
            columns=[
                ColumnSchema(name="id", physical_type="integer", nullable=False),
                ColumnSchema(name="status", physical_type="text", nullable=False),
                ColumnSchema(name="resolution", physical_type="text", nullable=True),
            ])
    })
    frame = pd.DataFrame({
        "id": list(range(1, 11)),
        "status": ["open", "closed"] * 5,
        "resolution": [None, "done"] * 5,
    })
    dataset = RelationalDataset(schema=schema, tables={"case": frame}).finalize_counts()
    rule = ConstraintDefinition(
        kind=ConstraintKind.CONDITIONAL, table="case",
        when_column="status", when_values=["closed"],
        required_column="resolution", source="confirmed_user_rule")
    artifact = train_relational_dataset(
        dataset, source_kind=SourceKind.DATABASE, source_fingerprint="b" * 64,
        sample_records={"case": SamplingRecord(
            population_count=len(frame), sample_count=len(frame))},
        seed=22, rare_threshold=10, confirmed_rules=[rule])
    platform = SyntheticDataPlatform.from_settings()
    tables = platform.generate(
        artifact, GenerationRequest(root_table_rows={"case": 200}, seed=22))
    generated = tables["case"]
    closed = generated["status"].astype(str).eq("closed")
    assert generated.loc[closed, "resolution"].notna().all()
    assert platform.validate(artifact, tables).overall == Status.PASS


def test_composite_primary_key_is_repaired_to_unique_values():
    schema = DatabaseSchema(source_kind="test", tables={
        "codebook": TableSchema(
            name="codebook", primary_key=None,
            primary_key_columns=["region", "code"], primary_key_confirmed=True,
            columns=[
                ColumnSchema(name="region", physical_type="text", nullable=False),
                ColumnSchema(name="code", physical_type="integer", nullable=False),
                ColumnSchema(name="label", physical_type="text", nullable=False),
            ])
    })
    frame = pd.DataFrame({
        "region": ["west", "west", "east", "east"],
        "code": [1, 2, 1, 2],
        "label": ["a", "b", "c", "d"],
    })
    dataset = RelationalDataset(schema=schema, tables={"codebook": frame}).finalize_counts()
    artifact = train_relational_dataset(
        dataset, source_kind=SourceKind.DATABASE, source_fingerprint="c" * 64,
        sample_records={"codebook": SamplingRecord(
            population_count=len(frame), sample_count=len(frame))},
        seed=23, rare_threshold=10)
    tables = SyntheticDataPlatform.from_settings().generate(
        artifact, GenerationRequest(root_table_rows={"codebook": 100}, seed=23))
    assert not tables["codebook"].duplicated(["region", "code"]).any()
    assert tables["codebook"]["code"].map(
        lambda value: isinstance(value, (int, __import__("numpy").integer))).all()
    report = SyntheticDataPlatform.from_settings().validate(artifact, tables)
    assert not [check for check in report.checks
                if check.dimension == "business" and check.status == Status.FAIL]


def test_non_allowlisted_postgres_table_is_rejected_before_sql():
    from synth_platform.infrastructure.sources.postgres import PostgresSource
    from synth_platform.domain.schema.models import TableRef
    from synth_platform.errors import UnsafeIdentifierError

    source = PostgresSource.__new__(PostgresSource)
    source.allowed_schemas = ("public",)
    source.allowed_tables = ("public.customers",)
    assert source._resolve(TableRef(schema="public", name="customers")).qualified_name == "public.customers"
    with pytest.raises(UnsafeIdentifierError, match="not allowlisted"):
        source._resolve(TableRef(schema="public", name="payments"))
    with pytest.raises(UnsafeIdentifierError, match="schema not allowlisted"):
        source._resolve(TableRef(schema="secret", name="customers"))


def test_nonlocal_postgres_source_refuses_disabled_tls_before_connecting():
    from synth_platform.infrastructure.sources.postgres import PostgresSource
    from synth_platform.errors import SourceUnavailableError

    with pytest.raises(SourceUnavailableError, match="TLS is required"):
        PostgresSource(
            "postgresql://reader:secret@db.example.com/app?sslmode=disable",
            allowed_schemas=("public",), allowed_tables=None)


def test_database_artifact_cannot_publish_into_pdf_track_schema(banking_db):
    class WrongTrackSink:
        target_schema = "pdf_track_wrong"

    platform = SyntheticDataPlatform.from_settings()
    artifact = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", seed=31))
    tables = platform.generate(
        artifact, GenerationRequest(root_table_rows={"users": 10}, seed=31))
    report = platform.validate(artifact, tables)
    with pytest.raises(PublicationBlockedError, match="expected db_track"):
        publish_dataset(artifact, tables, report, WrongTrackSink(), "wrong-track")
