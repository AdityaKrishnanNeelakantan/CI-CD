"""E2E: Database Twin context-aware generation after artifact-only load.

Proves the real workflow:

discover -> profile -> infer -> contract -> train -> save artifact
-> disconnect source -> load artifact -> generate -> validate

Names/emails/IDs must be realistic/format-preserving, numeric/category paths
unchanged, PK/FK intact, and no source PII replay.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from faker import Faker

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.validation.database.qa_service import run_qa_validation
from synth_platform.engine.generation.database.relational_service import (
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.training.database.service import run_training_and_sampling

pytestmark = pytest.mark.integration


def _build_context_db(path: Path, *, seed: int = 42, customers: int = 60) -> Path:
    faker = Faker()
    faker.seed_instance(seed)
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            address TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            postal_code TEXT NOT NULL,
            country TEXT NOT NULL,
            company TEXT NOT NULL,
            signup_date TEXT NOT NULL,
            age INTEGER NOT NULL,
            segment TEXT NOT NULL
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        """
    )
    customer_rows = []
    order_rows = []
    order_id = 1
    segments = ["gold", "silver", "bronze"]
    statuses = ["pending", "shipped", "cancelled"]
    for i in range(customers):
        first = faker.first_name()
        last = faker.last_name()
        email = faker.unique.email()
        customer_id = f"CUS-2026-{i:06d}"
        customer_rows.append(
            (
                customer_id,
                first,
                last,
                f"{first} {last}",
                email,
                faker.phone_number(),
                faker.street_address(),
                faker.city(),
                faker.state_abbr(),
                faker.postcode(),
                faker.country(),
                faker.company(),
                faker.date_between(start_date="-3y", end_date="today").isoformat(),
                20 + (i % 50),
                segments[i % 3],
            )
        )
        for _ in range(1 + (i % 3)):
            order_rows.append(
                (
                    order_id,
                    customer_id,
                    round(10 + (i * 3.5) % 500, 2),
                    statuses[i % 3],
                    faker.date_between(start_date="-2y", end_date="today").isoformat(),
                )
            )
            order_id += 1
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        customer_rows,
    )
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", order_rows)
    conn.commit()
    conn.close()
    return path


def test_database_twin_context_aware_generation_after_source_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path = _build_context_db(tmp_path / "source.db")
    with sqlite3.connect(source_path) as source_conn:
        source_customers = pd.read_sql_query("SELECT * FROM customers", source_conn)
    source_ids = set(source_customers["customer_id"])
    source_emails = set(source_customers["email"])
    source_names = set(source_customers["full_name"])

    adapter = SQLiteSourceAdapter({"path": str(source_path)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    assert run_discovery(adapter, manifest, config_path="config/project.yaml").is_success()
    discovery = json.loads(Path(manifest.output_path("discovery.json")).read_text())
    assert run_profiling(adapter, discovery, manifest, "discovery.json", sample_limit=5000).is_success()
    profile = json.loads(Path(manifest.output_path("profile.json")).read_text())
    assert run_inference(
        adapter, discovery, profile, manifest, "discovery.json", "profile.json", sample_limit=5000
    ).is_success()
    candidates = json.loads(Path(manifest.output_path("semantic_candidates.json")).read_text())

    decisions = {
        table: {col: cand["semantic_type"] for col, cand in cols.items()}
        for table, cols in candidates["tables"].items()
    }
    assert candidates["tables"]["customers"]["full_name"]["semantic_type"] == "person_name"
    assert candidates["tables"]["customers"]["email"]["semantic_type"] == "email"
    assert candidates["tables"]["customers"]["customer_id"]["semantic_type"] == "identifier"

    assert run_contract_approval(
        dataset_id="context_aware",
        source_fingerprint=discovery["source_fingerprint"],
        discovery_data=discovery,
        candidates_by_table=candidates["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference="semantic_candidates.json",
        decisions=decisions,
    ).is_success()
    contract = load_dataset_contract(metadata_dir)

    assert run_training_and_sampling(
        adapter,
        contract,
        manifest,
        "dataset_contract.json",
        sample_limit=5000,
        num_rows_to_generate=25,
        seed=9,
        model_type="safe_gaussian_copula",
    ).is_success()
    training_report = json.loads(Path(manifest.output_path("training_report.json")).read_text())

    export = run_artifact_export(
        "context_aware",
        "1.0.0",
        discovery,
        contract,
        profile,
        training_report,
        manifest.run_id,
        manifest.code_version,
        manifest,
        "training_report.json",
    )
    assert export.is_success()
    artifact_path = export.output_references[0]

    del adapter
    gc.collect()

    def _deny_source_connect(self):
        raise AssertionError(
            f"source database must not be accessed during artifact-only generation: {self._db_path}"
        )

    monkeypatch.setattr(SQLiteSourceAdapter, "_connect", _deny_source_connect)
    try:
        os.remove(source_path)
    except PermissionError:
        # Windows may retain a transient lock; monkeypatch is the hard guarantee.
        pass

    loaded = load_artifact(artifact_path)
    adapters_by_table = {
        table: loaded.get_model(table, allow_cloudpickle_models=True)
        for table in loaded.manifest["tables"]
    }
    gen_manifest = RunManifest.create(runs_dir=tmp_path / "gen_runs")
    result = run_relational_generation(
        contract,
        adapters_by_table,
        {"customers": 40, "orders": 80},
        gen_manifest,
        contract_reference="dataset_contract.json",
        seed=9,
    )
    assert result.is_success(), result.errors
    report = load_relational_generation_report(gen_manifest.output_path("relational_generation_report.json"))
    customers = pd.read_csv(report["tables"]["customers"]["path"])
    orders = pd.read_csv(report["tables"]["orders"]["path"])

    assert not any(str(v).startswith("NAME-") for v in customers["full_name"])
    assert not any(str(v).startswith("FULL_NAME-") for v in customers["full_name"])
    assert customers["full_name"].map(lambda v: isinstance(v, str) and len(v.split()) >= 2).all()
    assert customers["email"].map(lambda v: "@" in str(v)).all()
    assert all(re.fullmatch(r"CUS-2026-\d{6}", str(v)) for v in customers["customer_id"])
    assert customers["customer_id"].is_unique
    assert set(customers["customer_id"]).isdisjoint(source_ids)
    assert set(customers["email"]).isdisjoint(source_emails)
    assert set(customers["full_name"]).isdisjoint(source_names)
    assert customers["segment"].isin(["gold", "silver", "bronze"]).all()
    assert customers["age"].notna().all()
    assert set(orders["customer_id"]).issubset(set(customers["customer_id"]))
    assert report["fk_validity"]["overall_fk_validity"] == 1.0

    qa = run_qa_validation(
        contract,
        report,
        gen_manifest,
        relational_report_reference="relational_generation_report.json",
    )
    assert qa.is_success(), qa.errors
