"""Checkpoint 6 orchestration: the approved dataset_contract -> a fully
relationally-linked, constraint-checked synthetic dataset - the Generate
and Constraints stages combined, since a derived field or hard
constraint is checked against the already-FK-resolved row, not before it.

Reads: the approved dataset_contract (for foreign_keys and business_rules)
and one already-fitted SynthesizerAdapter per table (Checkpoint 4's
per-table training is unchanged - this stage only decides what order to
sample them in, links them, and repairs/recalculates afterward). Writes:
one CSV per table plus relational_generation_report.json (FK validity +
constraint compliance) - never raw source rows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.generation.database.constraint_engine import (
    apply_derived_fields,
    check_constraints,
    repair_violations,
)
from synth_platform.engine.generation.database.relational_generator import (
    RelationalGenerationError,
    compute_fk_validity,
    generate_relational_dataset,
)
from synth_platform.engine.generation.database.schema_graph import CyclicSchemaError, build_schema_graph
from synth_platform.engine.training.database.base import SynthesizerAdapter

STAGE_NAME = "relational_generation"
RELATIONAL_REPORT_FILENAME = "relational_generation_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"generated_at", "generation_order", "fk_validity", "constraint_reports", "tables"}


class RelationalReportLoadError(Exception):
    """Raised when relational_generation_report.json is missing, corrupted, or malformed."""


def run_relational_generation(
    dataset_contract: dict[str, Any],
    adapters_by_table: dict[str, SynthesizerAdapter],
    row_counts_by_table: dict[str, int],
    manifest: RunManifest,
    contract_reference: str,
    seed: int | None = None,
    category_overrides_by_table: dict[str, dict[str, dict[str, float]]] | None = None,
    batch_size: int | None = None,
) -> StageResult:
    """Generate a relational synthetic dataset.

    ``batch_size=None`` keeps the legacy in-memory path (AppTest / small runs).
    When ``batch_size`` is an int, generation streams via
    :func:`generate_relational_dataset_streaming` with O(batch_size) RAM.
    """
    output_report_path = manifest.output_path(RELATIONAL_REPORT_FILENAME)
    if output_report_path.exists():
        raise RuntimeError(
            f"{RELATIONAL_REPORT_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    if batch_size is not None:
        return _run_relational_generation_streaming(
            dataset_contract=dataset_contract,
            adapters_by_table=adapters_by_table,
            row_counts_by_table=row_counts_by_table,
            manifest=manifest,
            contract_reference=contract_reference,
            seed=seed,
            category_overrides_by_table=category_overrides_by_table,
            batch_size=int(batch_size),
            output_report_path=output_report_path,
        )

    try:
        schema_graph = build_schema_graph(dataset_contract)
        tables = generate_relational_dataset(
            schema_graph, adapters_by_table, row_counts_by_table, seed=seed,
            category_overrides_by_table=category_overrides_by_table,
        )
    except (CyclicSchemaError, RelationalGenerationError) as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    constraint_reports: dict[str, Any] = {}
    warnings: list[str] = []
    for table_name, table_contract in dataset_contract["tables"].items():
        business_rules = table_contract.get("business_rules", [])
        derived_fields = [r for r in business_rules if r.get("type") == "derived"]
        constraints = [r for r in business_rules if r.get("type") == "constraint"]

        df = tables[table_name]
        if derived_fields:
            df = apply_derived_fields(df, derived_fields)
        if constraints:
            report = check_constraints(df, constraints)
            df = repair_violations(df, constraints, report)
            constraint_reports[table_name] = report
            if report["total_violations"]:
                warnings.append(f"{table_name}: {report['total_violations']} constraint violation(s) repaired")
        tables[table_name] = df

    fk_validity = compute_fk_validity(tables, schema_graph)
    if fk_validity["overall_fk_validity"] < 1.0:
        warnings.append(f"overall_fk_validity below 1.0: {fk_validity['overall_fk_validity']}")

    samples_dir = manifest.run_dir / "relational_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    table_paths: dict[str, str] = {}

    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_order": schema_graph["generation_order"],
        "fk_validity": fk_validity,
        "constraint_reports": constraint_reports,
        "tables": {},
    }
    try:
        for table_name, df in tables.items():
            table_path = samples_dir / f"{table_name}.csv"
            df.to_csv(table_path, index=False)
            table_paths[table_name] = str(table_path)

        report_data["tables"] = {
            table_name: {"row_count": len(df), "path": table_paths[table_name]}
            for table_name, df in tables.items()
        }
        with output_report_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_report_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[f"could not write {RELATIONAL_REPORT_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[contract_reference],
        output_references=[str(output_report_path)] + list(table_paths.values()),
        metrics={
            "table_count": len(tables),
            "total_rows": sum(len(df) for df in tables.values()),
            "overall_fk_validity": fk_validity["overall_fk_validity"],
        },
        warnings=warnings,
        evidence={"generated_at": report_data["generated_at"]},
    )
    manifest.record_stage(result)
    return result


def _run_relational_generation_streaming(
    *,
    dataset_contract: dict[str, Any],
    adapters_by_table: dict[str, SynthesizerAdapter],
    row_counts_by_table: dict[str, int],
    manifest: RunManifest,
    contract_reference: str,
    seed: int | None,
    category_overrides_by_table: dict[str, dict[str, dict[str, float]]] | None,
    batch_size: int,
    output_report_path: Path,
) -> StageResult:
    from synth_platform.engine.generation.database.streaming_generator import (
        generate_relational_dataset_streaming,
    )

    samples_dir = manifest.run_dir / "relational_samples"
    key_store_path = manifest.run_dir / "relational_keys.sqlite"

    try:
        schema_graph = build_schema_graph(dataset_contract)
        streamed = generate_relational_dataset_streaming(
            schema_graph,
            adapters_by_table,
            row_counts_by_table,
            dataset_contract,
            samples_dir,
            key_store_path,
            batch_size=batch_size,
            seed=seed,
            category_overrides_by_table=category_overrides_by_table,
        )
    except (CyclicSchemaError, RelationalGenerationError, ValueError, OSError) as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    warnings = list(streamed.get("warnings") or [])
    fk_validity = streamed["fk_validity"]
    if fk_validity["overall_fk_validity"] < 1.0:
        warnings.append(f"overall_fk_validity below 1.0: {fk_validity['overall_fk_validity']}")

    report_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_order": streamed["generation_order"],
        "fk_validity": fk_validity,
        "constraint_reports": streamed["constraint_reports"],
        "tables": streamed["tables"],
        "streaming": True,
        "batch_size": batch_size,
    }
    table_paths = [entry["path"] for entry in streamed["tables"].values()]
    try:
        with output_report_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_report_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[f"could not write {RELATIONAL_REPORT_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[contract_reference],
        output_references=[str(output_report_path)] + table_paths,
        metrics={
            "table_count": len(streamed["tables"]),
            "total_rows": sum(int(t["row_count"]) for t in streamed["tables"].values()),
            "overall_fk_validity": fk_validity["overall_fk_validity"],
            "batch_size": batch_size,
            "streaming": True,
        },
        warnings=warnings,
        evidence={"generated_at": report_data["generated_at"], "streaming": True},
    )
    manifest.record_stage(result)
    return result


def load_relational_generation_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a relational_generation_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise RelationalReportLoadError(f"relational_generation_report.json not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RelationalReportLoadError(
            f"relational_generation_report.json is not valid JSON: {report_path}"
        ) from exc

    if not isinstance(data, dict):
        raise RelationalReportLoadError(
            f"relational_generation_report.json did not parse to an object: {report_path}"
        )

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise RelationalReportLoadError(
            f"relational_generation_report.json missing required keys: {sorted(missing)}"
        )

    return data
