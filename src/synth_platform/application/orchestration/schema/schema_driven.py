"""Schema-driven pipeline with chunked export and bounded validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synth_platform.engine.generation.schema.chunked_export import StreamingCsvWriter, StreamingParquetWriter
from synth_platform.engine.inference.schema.distribution_rules import (
    GlobalRuleBudget,
    aggregate_rule_evidence,
    apply_distribution_rules,
    apply_distribution_rules_to_chunk,
    cross_table_rules,
    evaluate_distribution_rules,
    parse_distribution_rules,
)
from synth_platform.engine.generation.schema.duplicate_guard import enforce_duplicate_policy, locked_columns_from_distribution_rules
from synth_platform.engine.validation.schema.performance.memory import track_peak_memory
from synth_platform.engine.validation.schema.performance.report import build_pipeline_performance_report
from synth_platform.engine.validation.schema.performance.timing import PipelineStageTimings, time_stage
from synth_platform.application.orchestration.schema.config import PipelineConfig, PipelineProgress
from synth_platform.application.orchestration.schema.result import PipelineResult
from synth_platform.engine.validation.schema.export_validation import validate_export_paths
from synth_platform.engine.validation.schema.reporting import ReservoirTableSampler, build_validation_report
from synth_platform.engine.inference.schema.schema import SchemaConfig
from synth_platform.engine.generation.schema.simulator import DataSimulator
from synth_platform.engine.validation.schema.validation import validate_data


def _resolve_fidelity_mode(config: PipelineConfig, target_rows: int) -> str:
    if config.fidelity_mode in {"exact", "sampled"}:
        return config.fidelity_mode
    return "sampled" if target_rows > config.large_row_threshold else "exact"


def _scale_schema_to_rows(schema: SchemaConfig, target_rows: int) -> SchemaConfig:
    scaled = schema.model_copy(deep=True)
    base_total = sum(int(t.row_count or 0) for t in scaled.tables) or max(target_rows, 1)
    ratio = target_rows / base_total
    for table in scaled.tables:
        if table.row_count:
            table.row_count = max(1, int(round(table.row_count * ratio)))
    return scaled


def _schema_table_totals(schema: SchemaConfig) -> Dict[str, int]:
    return {table.name: int(table.row_count or 0) for table in schema.tables}


def _export_validation_inputs(schema: SchemaConfig) -> tuple[Dict[str, str], list[tuple[str, str, str, str]]]:
    pk_columns: Dict[str, str] = {}
    fk_checks: List[tuple[str, str, str, str]] = []
    for table in schema.tables:
        for column in schema.get_columns(table.name):
            if getattr(column, "unique", False) and column.type != "foreign_key":
                pk_columns[table.name] = column.name
                break
    for rel in schema.relationships:
        fk_checks.append((rel.child_table, rel.child_key, rel.parent_table, rel.parent_key))
    return pk_columns, fk_checks


def _open_table_writer(table_name: str, output_dir: Path, export_format: str):
    suffix = "parquet" if export_format == "parquet" else "csv"
    path = output_dir / f"{table_name}.{suffix}"
    if export_format == "parquet":
        return StreamingParquetWriter(path), path
    return StreamingCsvWriter(path), path


def run_schema_pipeline(
    schema: SchemaConfig,
    config: PipelineConfig,
    *,
    progress: Optional[PipelineProgress] = None,
) -> PipelineResult:
    """Generate schema-driven data with stage timings and optional chunked export."""
    progress = progress or PipelineProgress()
    stages = PipelineStageTimings()
    peak_mb = 0.0
    target_rows = config.preview_rows if config.preview_only else int(config.full_rows or config.preview_rows)
    scaled_schema = _scale_schema_to_rows(schema, target_rows)
    if config.seed is not None:
        scaled_schema.seed = config.seed

    rules = parse_distribution_rules(config.rules_text)
    locked_columns = locked_columns_from_distribution_rules(rules)
    preview_rule_evidence: List[dict] = []
    final_rule_evidence: List[dict] = []
    export_validation_payload: Dict[str, Any] = {}
    fidelity_mode = _resolve_fidelity_mode(config, target_rows)
    sample_size = min(config.fidelity_sample_size, target_rows)
    output_dir = Path(config.output_dir or Path.cwd() / "generated_data" / "schema_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_format = config.export_format.lower()
    chunk_total = 0
    table_totals = _schema_table_totals(scaled_schema)
    expected_total_rows = sum(table_totals.values())

    with track_peak_memory(target_rows=expected_total_rows) as peaks:
        progress.stage = "planning"
        with time_stage(stages, "plan_seconds"):
            simulator = DataSimulator(
                scaled_schema,
                llm_text_enabled=config.llm_text_enabled and (config.preview_only or config.llm_full_enabled),
                is_preview=config.preview_only,
                max_llm_rows=config.max_llm_rows,
            )

        progress.stage = "generation"
        use_streaming = not config.preview_only and expected_total_rows > config.chunk_size

        if config.preview_only or not use_streaming:
            with time_stage(stages, "generate_seconds"):
                gen_result = simulator.generate_with_reports(
                    include_tables=True,
                    sample_size=min(sample_size, config.preview_rows),
                )
            tables = gen_result.tables
            row_counts = dict(gen_result.table_row_counts)
            with time_stage(stages, "rules_seconds"):
                if rules:
                    tables = apply_distribution_rules(
                        tables=tables,
                        schema=scaled_schema,
                        rules=rules,
                        seed=scaled_schema.seed,
                    )
            with time_stage(stages, "duplicate_repair_seconds"):
                tables, _ = enforce_duplicate_policy(
                    tables,
                    scaled_schema,
                    seed=scaled_schema.seed,
                    locked_columns=locked_columns,
                )
            preview_tables = {name: df.head(min(config.preview_rows, len(df))) for name, df in tables.items()}
            preview_rule_evidence = (
                evaluate_distribution_rules(preview_tables, rules, schema=scaled_schema) if rules else []
            )
            final_rule_evidence = (
                evaluate_distribution_rules(tables, rules, schema=scaled_schema) if rules and not config.preview_only else preview_rule_evidence
            )
            export_paths_local: Dict[str, Path] = {}
            if not config.preview_only:
                with time_stage(stages, "export_seconds"):
                    for table_name, df in tables.items():
                        writer, path = _open_table_writer(table_name, output_dir, export_format)
                        writer.write_chunk(df)
                        writer.close()
                        export_paths_local[table_name] = path
        else:
            progress.message = "Streaming generation and export"
            row_counts: Dict[str, int] = {}
            rows_before: Dict[str, int] = {}
            export_paths_local: Dict[str, Path] = {}
            writers: Dict[str, object] = {}
            sampler = ReservoirTableSampler(
                sample_size=min(sample_size, config.preview_rows),
                rng=np.random.default_rng(int(scaled_schema.seed or 42)),
            )
            budget = GlobalRuleBudget.from_rules(rules, table_totals) if rules else None
            per_chunk_evidence: List[List[dict]] = []
            parent_table_data: Dict[str, pd.DataFrame] = {}
            parent_tables_needed = {rule.source_table for rule in cross_table_rules(rules)}

            with time_stage(stages, "generate_seconds"):
                for table_name, batch in simulator.generate_all():
                    chunk_total += 1
                    before = rows_before.get(table_name, 0)
                    if rules and budget is not None:
                        with time_stage(stages, "rules_seconds"):
                            batch = apply_distribution_rules_to_chunk(
                                batch,
                                scaled_schema,
                                rules,
                                budget,
                                table_name=table_name,
                                rows_before=before,
                                seed=scaled_schema.seed,
                            )
                            per_chunk_evidence.append(
                                evaluate_distribution_rules({table_name: batch}, rules, schema=scaled_schema)
                            )
                    child_cross_rules = [
                        rule for rule in cross_table_rules(rules) if rule.target_table == table_name
                    ]
                    if child_cross_rules and all(
                        parent in parent_table_data for parent in {rule.source_table for rule in child_cross_rules}
                    ):
                        ctx = {**parent_table_data, table_name: batch}
                        apply_distribution_rules(ctx, scaled_schema, child_cross_rules, seed=scaled_schema.seed)
                        batch = ctx[table_name]
                    if table_name in parent_tables_needed:
                        if table_name not in parent_table_data:
                            parent_table_data[table_name] = batch.copy()
                        else:
                            parent_table_data[table_name] = pd.concat(
                                [parent_table_data[table_name], batch],
                                ignore_index=True,
                            )
                    sampler.consume(table_name, batch)
                    row_counts[table_name] = row_counts.get(table_name, 0) + len(batch)
                    rows_before[table_name] = before + len(batch)

                    if table_name not in writers:
                        writers[table_name], export_paths_local[table_name] = _open_table_writer(
                            table_name,
                            output_dir,
                            export_format,
                        )
                    with time_stage(stages, "export_seconds"):
                        writers[table_name].write_chunk(batch)

            for writer in writers.values():
                writer.close()

            preview_tables = sampler.get_tables()
            tables = preview_tables
            preview_rule_evidence = (
                evaluate_distribution_rules(preview_tables, rules, schema=scaled_schema) if rules else []
            )
            with time_stage(stages, "duplicate_repair_seconds"):
                tables, _ = enforce_duplicate_policy(
                    tables,
                    scaled_schema,
                    seed=scaled_schema.seed,
                    locked_columns=locked_columns,
                )

        peak_mb = peaks[0]

    progress.stage = "validation"
    sampled = fidelity_mode == "sampled" or use_streaming
    with time_stage(stages, "validate_seconds"):
        if use_streaming and rules:
            final_rule_evidence = aggregate_rule_evidence(
                per_chunk_evidence,
                rules=rules,
                table_totals=row_counts,
            )
        elif rules:
            final_rule_evidence = evaluate_distribution_rules(tables, rules, schema=scaled_schema)
        else:
            final_rule_evidence = []
        rule_evidence = final_rule_evidence

        validation_scope = "sampled"
        if not config.preview_only and export_paths_local:
            pk_columns, fk_checks = _export_validation_inputs(scaled_schema)
            export_validation_payload = validate_export_paths(
                output_dir,
                expected_counts=row_counts,
                pk_columns=pk_columns,
                fk_checks=fk_checks,
            )
            if export_validation_payload.get("passed"):
                validation_scope = "full_export"

        validation_report_payload = build_validation_report(
            tables,
            scaled_schema,
            seed=scaled_schema.seed,
            row_counts=row_counts,
            sampled=sampled and validation_scope != "full_export",
            rule_evidence=final_rule_evidence,
            generation_mode="schema_driven",
            export_validation=export_validation_payload or None,
            validation_scope=validation_scope,
        )
        if sampled:
            validation_report_payload.setdefault("advisory", {})["validation_note"] = (
                f"Advisory metrics computed on bounded sample (up to {sample_size:,} rows per table)."
            )
            validation_report_payload["fidelity_mode"] = fidelity_mode

    performance = build_pipeline_performance_report(
        generation_mode="schema_driven",
        target_rows=target_rows,
        rows_generated=sum(row_counts.values()),
        chunk_size=config.chunk_size,
        stages=stages,
        peak_memory_mb=peak_mb,
        export_format=export_format,
        export_path=output_dir,
        chunk_count=chunk_total,
        fidelity_mode=fidelity_mode,
        sampled_fidelity_size=sample_size if sampled else None,
        notes=[f"preview_only={config.preview_only}", f"streaming={use_streaming}"],
    )

    if config.write_reports:
        (output_dir / "performance_report.json").write_text(
            json.dumps(performance.to_dict(), indent=2),
            encoding="utf-8",
        )
        (output_dir / "validation_report.json").write_text(
            json.dumps(validation_report_payload, indent=2, default=str),
            encoding="utf-8",
        )

    progress.stage = "complete"
    export_paths = export_paths_local if not config.preview_only else {}

    return PipelineResult(
        generation_mode="schema_driven",
        preview_tables=preview_tables,
        export_paths=export_paths,
        artifact_dir=output_dir,
        performance=performance,
        validation_report=validation_report_payload,
        preview_rule_evidence=preview_rule_evidence,
        final_rule_evidence=final_rule_evidence,
        rule_evidence=final_rule_evidence,
        validation_contract=validation_report_payload.get("validation_contract", {}),
        export_validation=export_validation_payload,
        progress=progress,
        schema_summary=scaled_schema.summary(),
        row_counts=row_counts,
    )
