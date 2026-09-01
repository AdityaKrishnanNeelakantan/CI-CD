"""Build a small portable twin artifact for CI delivery verification.

This script exercises the product promise rather than a single unit boundary:

1. create a tiny source database with obvious source-only canary values;
2. discover/profile/infer/approve/train/export a portable artifact;
3. remove the source database;
4. load only the artifact and generate relational synthetic data;
5. validate the generated output;
6. fail if source canaries appear in the artifact or generated CSVs.

The resulting output directory is suitable for GitHub Actions artifact upload.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.generation.database.relational_service import (
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import (
    run_contract_approval,
    run_inference,
)
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export
from synth_platform.engine.training.database.service import run_training_and_sampling
from synth_platform.engine.validation.database.qa_service import run_qa_validation


def _create_source_database(path: Path) -> dict[str, set[str]]:
    conn = sqlite3.connect(path)
    source_ids: set[str] = set()
    source_names: set[str] = set()
    source_emails: set[str] = set()
    try:
        conn.executescript(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL,
                age INTEGER NOT NULL,
                segment TEXT NOT NULL
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );

            CREATE INDEX idx_orders_customer_id ON orders(customer_id);
            """
        )
        for i in range(1, 121):
            customer_id = f"SRC-CUSTOMER-{i:04d}"
            full_name = f"Source Canary Person {i:04d}"
            email = f"source.canary.{i:04d}@example.invalid"
            source_ids.add(customer_id)
            source_names.add(full_name)
            source_emails.add(email)
            conn.execute(
                "INSERT INTO customers VALUES (?,?,?,?,?)",
                (
                    customer_id,
                    full_name,
                    email,
                    20 + (i % 45),
                    ["gold", "silver", "bronze"][i % 3],
                ),
            )
            for j in range(1, 4):
                conn.execute(
                    "INSERT INTO orders (customer_id, amount, status) VALUES (?,?,?)",
                    (
                        customer_id,
                        round(20.0 + i * 1.5 + j * 3.25, 2),
                        ["pending", "shipped", "cancelled"][(i + j) % 3],
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return {
        "ids": source_ids,
        "names": source_names,
        "emails": source_emails,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_zip_for_canaries(path: Path, canaries: set[str]) -> list[str]:
    hits: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info.filename)
            text = data.decode("utf-8", errors="ignore")
            hits.extend(sorted(value for value in canaries if value in text))
    return sorted(set(hits))


def _scan_generated_for_canaries(report: dict, canaries: set[str]) -> list[str]:
    hits: list[str] = []
    for entry in report["tables"].values():
        text = Path(entry["path"]).read_text(encoding="utf-8")
        hits.extend(sorted(value for value in canaries if value in text))
    return sorted(set(hits))


def build_delivery(output_dir: Path, work_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    source_path = work_dir / "source.db"
    source_values = _create_source_database(source_path)
    source_canaries = set().union(*source_values.values())

    adapter = SQLiteSourceAdapter({"path": str(source_path)})
    manifest = RunManifest.create(runs_dir=work_dir / "training_runs")
    metadata_dir = work_dir / "metadata"

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    assert discovery_result.is_success(), discovery_result.errors
    discovery_data = _read_json(Path(discovery_result.output_references[0]))

    profiling_result = run_profiling(
        adapter, discovery_data, manifest, "discovery.json", sample_limit=500
    )
    assert profiling_result.is_success(), profiling_result.errors
    profile_data = _read_json(Path(profiling_result.output_references[0]))

    inference_result = run_inference(
        adapter,
        discovery_data,
        profile_data,
        manifest,
        "discovery.json",
        "profile.json",
        sample_limit=500,
    )
    assert inference_result.is_success(), inference_result.errors
    candidates_data = _read_json(Path(inference_result.output_references[0]))
    decisions = {
        table: {column: candidate["semantic_type"] for column, candidate in columns.items()}
        for table, columns in candidates_data["tables"].items()
    }

    approval_result = run_contract_approval(
        dataset_id="ci_portable_twin",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    assert approval_result.is_success(), approval_result.errors
    contract = load_dataset_contract(metadata_dir)

    training_result = run_training_and_sampling(
        adapter,
        contract,
        manifest,
        "dataset_contract.json",
        sample_limit=500,
        num_rows_to_generate=25,
        seed=22,
        model_type="safe_gaussian_copula",
    )
    assert training_result.is_success(), training_result.errors
    training_report = _read_json(Path(training_result.output_references[0]))

    export_result = run_artifact_export(
        "ci_portable_twin",
        "1.0.0",
        discovery_data,
        contract,
        profile_data,
        training_report,
        manifest.run_id,
        manifest.code_version,
        manifest,
        "training_report.json",
    )
    assert export_result.is_success(), export_result.errors
    assert export_result.metrics["self_test_passed"] is True
    artifact_path = Path(export_result.output_references[0])

    adapter = None
    source_path.unlink()

    artifact_hits = _scan_zip_for_canaries(artifact_path, source_canaries)
    assert artifact_hits == [], f"source canaries leaked into artifact: {artifact_hits[:5]}"

    loaded = load_artifact(artifact_path)
    adapters_by_table = {
        table: loaded.get_model(table)
        for table in loaded.manifest["tables"]
    }

    generation_manifest = RunManifest.create(runs_dir=work_dir / "generation_runs")
    generation_result = run_relational_generation(
        contract,
        adapters_by_table,
        {"customers": 40, "orders": 80},
        generation_manifest,
        contract_reference="dataset_contract.json",
        seed=23,
    )
    assert generation_result.is_success(), generation_result.errors
    relational_report = load_relational_generation_report(
        generation_manifest.output_path("relational_generation_report.json")
    )

    generated_hits = _scan_generated_for_canaries(relational_report, source_canaries)
    assert generated_hits == [], f"source canaries leaked into generated output: {generated_hits[:5]}"

    customers = pd.read_csv(relational_report["tables"]["customers"]["path"])
    orders = pd.read_csv(relational_report["tables"]["orders"]["path"])
    assert set(customers["customer_id"]).isdisjoint(source_values["ids"])
    assert set(customers["email"]).isdisjoint(source_values["emails"])
    assert set(customers["full_name"]).isdisjoint(source_values["names"])
    assert set(orders["customer_id"]).issubset(set(customers["customer_id"]))
    assert relational_report["fk_validity"]["overall_fk_validity"] == 1.0

    qa_result = run_qa_validation(
        contract,
        relational_report,
        generation_manifest,
        relational_report_reference="relational_generation_report.json",
        reference_profile=profile_data,
    )
    assert qa_result.is_success(), qa_result.errors
    qa_report_path = Path(qa_result.output_references[0])
    qa_report = _read_json(qa_report_path)
    assert qa_report["hard_checks_passed"] is True

    delivered_artifact = output_dir / "ci_portable_twin.zip"
    shutil.copy2(artifact_path, delivered_artifact)

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    delivered_tables = {}
    for table_name, entry in relational_report["tables"].items():
        target = generated_dir / f"{table_name}.csv"
        shutil.copy2(entry["path"], target)
        delivered_tables[table_name] = {
            "path": str(target.relative_to(output_dir)),
            "row_count": entry["row_count"],
        }

    shutil.copy2(qa_report_path, output_dir / "qa_report.json")
    shutil.copy2(generation_manifest.output_path("relational_generation_report.json"), output_dir / "relational_generation_report.json")
    shutil.copy2(manifest.run_dir / "run_manifest.json", output_dir / "training_run_manifest.json")
    shutil.copy2(generation_manifest.run_dir / "run_manifest.json", output_dir / "generation_run_manifest.json")

    summary = {
        "artifact": delivered_artifact.name,
        "artifact_version": "1.0.0",
        "generated_tables": delivered_tables,
        "hard_checks_passed": qa_report["hard_checks_passed"],
        "overall_fk_validity": relational_report["fk_validity"]["overall_fk_validity"],
        "source_disconnected": not source_path.exists(),
        "source_canary_hits": {
            "artifact": artifact_hits,
            "generated": generated_hits,
        },
    }
    (output_dir / "delivery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifact-delivery")
    parser.add_argument("--work-dir", default=".artifact-delivery-work")
    args = parser.parse_args()

    summary = build_delivery(Path(args.output_dir), Path(args.work_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
