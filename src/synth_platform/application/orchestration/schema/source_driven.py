"""Source-driven pipeline with profile cache and fitted-model reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from synth_platform.engine.generation.schema.chunked_export import StreamingCsvWriter, StreamingParquetWriter, export_table_streaming
from synth_platform.engine.inference.schema.distribution_rules import (
    GlobalRuleBudget,
    aggregate_rule_evidence,
    apply_distribution_rules,
    apply_distribution_rules_to_chunk,
    evaluate_distribution_rules,
    parse_distribution_rules,
)
from synth_platform.engine.validation.schema.export_validation import validate_single_table_export
from synth_platform.engine.validation.schema.reporting import build_validation_report
from synth_platform.engine.validation.schema.performance.memory import track_peak_memory
from synth_platform.engine.validation.schema.performance.report import build_pipeline_performance_report
from synth_platform.engine.validation.schema.performance.timing import PipelineStageTimings, time_stage
from synth_platform.application.orchestration.schema.cache import SourceProfileCache, source_dataframe_fingerprint
from synth_platform.application.orchestration.schema.config import PipelineConfig, PipelineProgress
from synth_platform.application.orchestration.schema.result import PipelineResult
from synth_platform.application.orchestration.schema.summary import source_profile_schema_summary
from synth_platform.engine.validation.schema.reporting import ReservoirTableSampler
from synth_platform.engine.inference.schema.source_driven import (
    SourceDrivenGenerator,
    build_fidelity_report,
)


DEFAULT_SOURCE_CHUNK_SIZE = 25_000


def resolve_source_chunk_size(config: PipelineConfig, target_rows: int) -> int:
    """Pick a source-driven chunk size balancing throughput and memory."""
    explicit = config.source_chunk_size or config.chunk_size
    if target_rows <= explicit:
        return target_rows
    if explicit >= DEFAULT_SOURCE_CHUNK_SIZE:
        return int(explicit)
    return DEFAULT_SOURCE_CHUNK_SIZE


def _resolve_fidelity_mode(config: PipelineConfig, target_rows: int) -> str:
    if config.fidelity_mode in {"exact", "sampled"}:
        return config.fidelity_mode
    return "sampled" if target_rows > config.large_row_threshold else "exact"


def _fit_config_matches(manifest_path: Path, *, model_type: Optional[str], seed: Optional[int]) -> bool:
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model_type and model_type not in {None, "auto"} and payload.get("model_type") != model_type:
        return False
    if seed is not None and payload.get("seed") not in {None, seed}:
        return False
    return True


def fit_or_reuse_source_generator(
    source_df: pd.DataFrame,
    config: PipelineConfig,
    *,
    stages: PipelineStageTimings,
    artifact_dir: Path,
) -> tuple[SourceDrivenGenerator, Dict[str, Any]]:
    """Fit once or load persisted model when preview artifacts are reusable."""
    evidence: Dict[str, Any] = {
        "profile_cache_hit": False,
        "model_reused": False,
        "refit_performed": True,
        "refit_reason": "initial_fit",
        "cache_hit": False,
    }
    fingerprint = source_dataframe_fingerprint(
        source_df,
        profile_max_rows=config.profile_max_rows,
        fit_max_rows=config.fit_max_rows,
        model_type=config.model_type,
        seed=config.seed,
        table_name=config.table_name,
    )
    cache = SourceProfileCache()
    cached_profile = cache.get(fingerprint)

    reuse_dir = config.reuse_artifacts_dir
    if reuse_dir and Path(reuse_dir).exists():
        manifest = Path(reuse_dir) / "manifest.json"
        if _fit_config_matches(manifest, model_type=config.model_type, seed=config.seed):
            with time_stage(stages, "load_seconds"):
                generator = SourceDrivenGenerator.load(reuse_dir)
            evidence.update(
                {
                    "model_reused": True,
                    "refit_performed": False,
                    "refit_reason": None,
                    "cache_hit": True,
                }
            )
            return generator, evidence

    generator = SourceDrivenGenerator()
    with time_stage(stages, "profile_seconds"):
        if cached_profile:
            evidence["profile_cache_hit"] = True
        generator.fit(
            source_df,
            table_name=config.table_name,
            model_type=config.model_type,
            seed=config.seed,
            epochs=config.epochs,
            fit_max_rows=config.fit_max_rows,
            profile_max_rows=config.profile_max_rows,
            stages=stages,
        )
        if generator.profile is not None and not cached_profile:
            cache.put(fingerprint, generator.profile.to_dict() if hasattr(generator.profile, "to_dict") else {})

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if artifact_dir.exists():
        import shutil

        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generator.save(artifact_dir, synthetic_rows=int(config.full_rows or config.preview_rows), seed=config.seed)
    evidence["refit_reason"] = "source_or_config_changed"
    return generator, evidence


def _chunked_generation(
    generator: SourceDrivenGenerator,
    *,
    target_rows: int,
    chunk_size: int,
    seed: Optional[int],
    stages: PipelineStageTimings,
    progress: PipelineProgress,
    export_dir: Path,
    table_name: str,
    export_format: str,
    fidelity_sample_size: int,
    rules: Optional[list] = None,
    schema_stub: Any = None,
) -> tuple[pd.DataFrame, Dict[str, Any], int, Dict[str, Any], List[Dict[str, Any]]]:
    """Stream sample → optional privacy check → export without full materialization."""
    preview_cap = min(5000, target_rows, chunk_size)
    fidelity_cap = min(fidelity_sample_size, preview_cap)
    sampler = ReservoirTableSampler(sample_size=fidelity_cap, rng=np.random.default_rng(int(seed or 42)))
    chunk_count = 0
    total_replay = 0
    privacy_checks = 0
    overlay_plan = generator._overlay_plan
    export_path = export_dir / f"{table_name}.{'parquet' if export_format == 'parquet' else 'csv'}"
    writer = (
        StreamingParquetWriter(export_path)
        if export_format.lower() == "parquet"
        else StreamingCsvWriter(export_path)
    )
    rows_written = 0
    rules = rules or []
    budget = GlobalRuleBudget.from_rules(rules, {table_name: target_rows}) if rules else None
    per_chunk_evidence: List[List[Dict[str, Any]]] = []
    rows_before = 0

    total_chunks = max(1, (target_rows + chunk_size - 1) // chunk_size)
    for chunk in generator.sample_chunks(target_rows, chunk_size, seed=seed, stages=stages):
        chunk_count += 1
        progress.stage = "generation"
        progress.message = f"Sampling chunk {chunk_count}/{total_chunks}"
        progress.percent = round(100.0 * chunk_count / total_chunks, 1)

        if rules and budget is not None and schema_stub is not None:
            with time_stage(stages, "rules_seconds"):
                chunk = apply_distribution_rules_to_chunk(
                    chunk,
                    schema_stub,
                    rules,
                    budget,
                    table_name=table_name,
                    rows_before=rows_before,
                    seed=seed,
                )
                per_chunk_evidence.append(evaluate_distribution_rules({table_name: chunk}, rules, schema=schema_stub))
            rows_before += len(chunk)

        if overlay_plan is not None and overlay_plan.privacy_columns:
            progress.stage = "pii_overlay"
            with time_stage(stages, "validate_seconds"):
                check_cols = overlay_plan.privacy_columns or overlay_plan.fresh_columns
                replay = overlay_plan.protected_cache.total_replay(chunk, check_cols)
                privacy_checks += 1
                total_replay += replay

        sampler.consume(table_name, chunk)
        progress.stage = "export"
        progress.message = f"Writing chunk {chunk_count}/{total_chunks}"
        with time_stage(stages, "export_seconds"):
            rows_written += writer.write_chunk(chunk)

    writer.close()
    export_info = {
        "table": table_name,
        "path": export_path,
        "format": export_format,
        "rows_written": rows_written,
    }

    preview = sampler.get_tables().get(table_name)
    if preview is None or preview.empty:
        preview = generator.sample(min(preview_cap, target_rows), seed=seed, stages=stages)

    privacy = {
        "passed": total_replay == 0,
        "total_replay_values": int(total_replay),
        "chunks_checked": int(privacy_checks),
        "mode": "chunk_cached_sets",
    }
    final_rule_evidence = (
        aggregate_rule_evidence(per_chunk_evidence, rules=rules, table_totals={table_name: rows_written})
        if rules and per_chunk_evidence
        else []
    )
    return preview, export_info, chunk_count, privacy, final_rule_evidence


def run_source_pipeline(
    source: Union[str, Path, pd.DataFrame],
    config: PipelineConfig,
    *,
    progress: Optional[PipelineProgress] = None,
) -> PipelineResult:
    """Source-driven run with reuse, chunking, and performance reporting."""
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    target_rows = config.preview_rows if config.preview_only else int(config.full_rows or len(df))
    chunk_size = resolve_source_chunk_size(config, target_rows)
    output_dir = Path(config.output_dir or Path.cwd() / "generated_data" / "source_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / ("preview" if config.preview_only else "full")
    progress = progress or PipelineProgress()
    stages = PipelineStageTimings()
    peak_mb = 0.0
    fidelity_mode = _resolve_fidelity_mode(config, target_rows)
    export_info: Dict[str, Any] = {}
    chunk_count = 0
    privacy_summary: Dict[str, Any] = {}
    rules = parse_distribution_rules(config.rules_text)
    preview_rule_evidence: List[Dict[str, Any]] = []
    final_rule_evidence: List[Dict[str, Any]] = []
    export_validation_payload: Dict[str, Any] = {}

    with track_peak_memory(target_rows=target_rows) as peaks:
        progress.stage = "profile"
        progress.message = "Profiling source and fitting model"
        generator, cache_evidence = fit_or_reuse_source_generator(
            df,
            config,
            stages=stages,
            artifact_dir=artifact_dir,
        )

        from synth_platform.engine.inference.schema.schema import SchemaConfig, Table
        from synth_platform.engine.inference.schema.schema_columns import column_from_source_profile

        schema_stub = SchemaConfig(
            name=config.table_name,
            tables=[Table(name=config.table_name, row_count=target_rows)],
            columns={
                config.table_name: [
                    column_from_source_profile(name, col)
                    for name, col in generator.profile.columns.items()
                ]
            },
        )

        progress.stage = "generation"
        progress.message = "Sampling synthetic rows"
        use_streaming = chunk_size < target_rows and not config.preview_only

        if use_streaming:
            preview, export_info, chunk_count, privacy_summary, final_rule_evidence = _chunked_generation(
                generator,
                target_rows=target_rows,
                chunk_size=chunk_size,
                seed=config.seed,
                stages=stages,
                progress=progress,
                export_dir=artifact_dir,
                table_name=config.table_name,
                export_format=config.export_format,
                fidelity_sample_size=config.fidelity_sample_size,
                rules=rules,
                schema_stub=schema_stub,
            )
        else:
            with time_stage(stages, "sample_seconds"):
                preview = generator.sample(target_rows, seed=config.seed, stages=stages)
            if rules:
                with time_stage(stages, "rules_seconds"):
                    tables = apply_distribution_rules(
                        {config.table_name: preview},
                        schema_stub,
                        rules,
                        seed=config.seed,
                    )
                    preview = tables[config.table_name]
            if overlay_plan := generator._overlay_plan:
                if overlay_plan.privacy_columns or overlay_plan.fresh_columns:
                    with time_stage(stages, "validate_seconds"):
                        privacy_summary = overlay_plan.replay_report(preview)
            if not config.preview_only:
                with time_stage(stages, "export_seconds"):
                    export_info = export_table_streaming(
                        config.table_name,
                        iter([preview]),
                        artifact_dir,
                        fmt=config.export_format,
                    )
            preview_rule_evidence = (
                evaluate_distribution_rules({config.table_name: preview.head(min(config.preview_rows, len(preview)))}, rules, schema=schema_stub)
                if rules
                else []
            )
            final_rule_evidence = (
                evaluate_distribution_rules({config.table_name: preview}, rules, schema=schema_stub) if rules else []
            )

        peak_mb = peaks[0]

    stages.sample_seconds = stages.extra.get("sdv_sample_seconds", 0.0) + stages.pii_overlay_seconds

    progress.stage = "validation"
    progress.message = "Building fidelity and metrics"
    fidelity_sample = preview.head(min(config.fidelity_sample_size, len(preview)))
    with time_stage(stages, "fidelity_seconds"):
        fidelity = build_fidelity_report(
            df,
            fidelity_sample,
            generator.profile,  # type: ignore[arg-type]
            fitted_model_type=generator.fitted_model_type(),
            protected_cache=getattr(generator._overlay_plan, "protected_cache", None),
        )
        fidelity["fidelity_mode"] = fidelity_mode
        fidelity["sampled"] = fidelity_mode == "sampled"
        fidelity["sampled_size"] = len(fidelity_sample)
        if privacy_summary:
            fidelity["privacy_chunk_checks"] = privacy_summary
        if fidelity_mode == "sampled":
            fidelity.setdefault(
                "advisory_note",
                "Sampled advisory fidelity, not exact full-data validation.",
            )

    preview_sample = preview.head(min(config.preview_rows, len(preview)))
    if not preview_rule_evidence and rules:
        preview_rule_evidence = evaluate_distribution_rules(
            {config.table_name: preview_sample},
            rules,
            schema=schema_stub,
        )

    validation_scope = "sampled" if fidelity_mode == "sampled" else "full_export"
    pk_column = next(
        (
            name
            for name, col in generator.profile.columns.items()
            if col.kind == "id_like"
        ),
        None,
    )
    if export_info.get("path") and not config.preview_only:
        export_validation_payload = validate_single_table_export(
            Path(export_info["path"]),
            expected_rows=int(export_info.get("rows_written", target_rows)),
            pk_column=pk_column,
        )
        if export_validation_payload.get("passed"):
            validation_scope = "full_export"

    validation_report_payload = build_validation_report(
        {config.table_name: preview_sample},
        schema_stub,
        seed=config.seed,
        row_counts={config.table_name: int(export_info.get("rows_written", len(preview)))},
        sampled=validation_scope != "full_export",
        rule_evidence=final_rule_evidence,
        generation_mode="source_driven",
        export_validation=export_validation_payload or None,
        privacy=privacy_summary or None,
        validation_scope=validation_scope,
    )

    rows_generated = int(export_info.get("rows_written", target_rows))
    generation_time = stages.extra.get("sdv_sample_seconds", 0.0) + stages.pii_overlay_seconds
    stages.extra["generation_rows_per_second"] = round(
        rows_generated / generation_time if generation_time > 0 else 0.0,
        2,
    )
    stages.extra["source_chunk_size"] = float(chunk_size)

    performance = build_pipeline_performance_report(
        generation_mode="source_driven",
        target_rows=target_rows,
        rows_generated=rows_generated,
        chunk_size=chunk_size,
        stages=stages,
        peak_memory_mb=peak_mb,
        export_format=config.export_format,
        export_path=output_dir,
        chunk_count=chunk_count,
        cache_hit=bool(cache_evidence.get("cache_hit")),
        profile_cache_hit=bool(cache_evidence.get("profile_cache_hit")),
        model_reused=bool(cache_evidence.get("model_reused")),
        refit_performed=bool(cache_evidence.get("refit_performed", True)),
        refit_reason=cache_evidence.get("refit_reason"),
        fidelity_mode=fidelity_mode,
        sampled_fidelity_size=len(fidelity_sample) if fidelity_mode == "sampled" else None,
        notes=[f"preview_only={config.preview_only}", f"streaming={use_streaming}"],
    )

    if config.write_reports:
        (output_dir / "performance_report.json").write_text(json.dumps(performance.to_dict(), indent=2), encoding="utf-8")
        (output_dir / "fidelity_report.json").write_text(json.dumps(fidelity, indent=2, default=str), encoding="utf-8")

    progress.stage = "complete"
    progress.message = "Generation complete"
    progress.percent = 100.0

    export_paths = {}
    if export_info.get("path"):
        export_paths[config.table_name] = Path(export_info["path"])

    schema_summary = source_profile_schema_summary(
        generator.profile,
        synthetic_rows=int(export_info.get("rows_written", len(preview))),
        model_type=generator.fitted_model_type(),
    )

    return PipelineResult(
        generation_mode="source_driven",
        preview_tables={config.table_name: preview.head(min(config.preview_rows, len(preview)))},
        export_paths=export_paths,
        artifact_dir=artifact_dir,
        performance=performance,
        fidelity_report=fidelity,
        validation_report=validation_report_payload,
        preview_rule_evidence=preview_rule_evidence,
        final_rule_evidence=final_rule_evidence,
        rule_evidence=final_rule_evidence,
        validation_contract=validation_report_payload.get("validation_contract", {}),
        export_validation=export_validation_payload,
        progress=progress,
        profile=generator.profile,
        schema_summary=schema_summary,
        row_counts={config.table_name: int(export_info.get("rows_written", len(preview)))},
        cache_evidence=cache_evidence,
    )
