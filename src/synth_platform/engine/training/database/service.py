"""Checkpoint 4 orchestration: train a model per table and generate rows.

Reads: the approved dataset_contract.json plus a fresh training-data read
via the same SourceAdapter used since Checkpoint 1, cleaned through the
same DataCleaner used in the Cleaning stage, then passed through the
DeterministicFakerMasker (pre-input de-identification) so context-aware
fields never enter fit() as raw production values. Writes: a saved model
file and a generated-sample CSV per table, plus training_report.json
evidence - never the real training rows themselves.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.errors import SourceAdapterError
from synth_platform.engine.profiling.database.cleaning.cleaner import DataCleaner
from synth_platform.engine.common.database.core.fingerprint import compute_dataframe_fingerprint
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.common.database.privacy.deterministic_masker import DeterministicFakerMasker
from synth_platform.engine.common.database.privacy.fit_guard import FitPrivacyError, assert_no_raw_context_in_fit_frame
from synth_platform.engine.training.database.adapters.context_generators import learn_identifier_format
from synth_platform.engine.training.database.base import SynthesisError
from synth_platform.engine.training.database.registry import create_synthesizer_adapter

STAGE_NAME = "training"
TRAINING_REPORT_FILENAME = "training_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"trained_at", "tables"}
_REQUIRED_TABLE_KEYS = {
    "model_type",
    "seed",
    "data_fingerprint",
    "fit_evidence",
    "pre_input_mask",
    "training_trace",
    "generated_row_count",
    "model_path",
    "sample_path",
    "serialization_format",
}

TrainingProgressCallback = Callable[[dict[str, Any]], None]


class TrainingReportLoadError(Exception):
    """Raised when training_report.json is missing, corrupted, or malformed."""


def run_training_and_sampling(
    adapter: SourceAdapter,
    dataset_contract: dict[str, Any],
    manifest: RunManifest,
    contract_reference: str,
    sample_limit: int,
    num_rows_to_generate: int,
    seed: int,
    model_type: str = "sdv_gaussian_copula",
    cleaner: DataCleaner | None = None,
    model_kwargs: dict[str, Any] | None = None,
    progress_callback: TrainingProgressCallback | None = None,
) -> StageResult:
    output_path = manifest.output_path(TRAINING_REPORT_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{TRAINING_REPORT_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    cleaner = cleaner or DataCleaner()
    models_dir = manifest.run_dir / "models"
    samples_dir = manifest.run_dir / "generated_samples"

    table_reports: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        total_tables = len(dataset_contract["tables"])

        for table_index, (table_name, table_contract) in enumerate(
            dataset_contract["tables"].items(),
            start=1,
        ):
            table_started = perf_counter()
            phase_durations_ms: dict[str, float] = {}

            def emit(phase: str, message: str, status: str = "running") -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        "status": status,
                        "phase": phase,
                        "message": message,
                        "table": table_name,
                        "table_index": table_index,
                        "table_count": total_tables,
                    }
                )

            print(
                f"[TRAIN] [{table_index}/{total_tables}] "
                f"{table_name}: reading sample...",
                flush=True,
            )
            emit("read_sample", "Reading source sample")
            phase_started = perf_counter()
            df = adapter.read_sample(table_name, limit=sample_limit)
            phase_durations_ms["read_sample"] = (perf_counter() - phase_started) * 1000
            print(
                f"[TRAIN] {table_name}: read {len(df)} rows "
                f"in {phase_durations_ms['read_sample'] / 1000:.2f}s",
                flush=True,
            )

            print(f"[TRAIN] {table_name}: cleaning...", flush=True)
            emit("clean", "Cleaning and typing training data")
            phase_started = perf_counter()
            cleaned_df, cleaning_report = cleaner.clean_table(df, table_name, table_contract)
            phase_durations_ms["clean"] = (perf_counter() - phase_started) * 1000
            print(
                f"[TRAIN] {table_name}: cleaning completed "
                f"in {phase_durations_ms['clean'] / 1000:.2f}s",
                flush=True,
            )
            warnings.extend(f"{table_name}: {w}" for w in cleaning_report["warnings"])

            # Learn identifier FORMAT from cleaned values before masking so
            # hash-seeded stand-ins cannot destroy source pattern metadata.
            identifier_formats: dict[str, dict] = {}
            for column_name, column_contract in table_contract.get("columns", {}).items():
                if (
                    column_contract.get("inference_status") == "approved"
                    and column_contract.get("semantic_type") == "identifier"
                    and column_name in cleaned_df.columns
                ):
                    fmt = learn_identifier_format(cleaned_df[column_name].tolist())
                    if fmt is not None:
                        identifier_formats[column_name] = fmt

# Pre-input de-identification: replace context-aware fields with
# deterministic Faker stand-ins before fit/fingerprint so the model
# learns statistical DNA only — never raw production PII/text/IDs.
            print(f"[TRAIN] {table_name}: masking sensitive fields...", flush=True)
            emit("mask", "Applying privacy mask before model fit")
            phase_started = perf_counter()
            masker = DeterministicFakerMasker(seed=seed)
            safe_df, mask_report = masker.mask_table(cleaned_df, table_name, table_contract)
            phase_durations_ms["mask"] = (perf_counter() - phase_started) * 1000
            print(
                f"[TRAIN] {table_name}: masking completed "
                f"in {phase_durations_ms['mask'] / 1000:.2f}s",
                flush=True,
            )
            # Hard gate: raw context/PII strings must never reach fit().
            assert_no_raw_context_in_fit_frame(
                cleaned_df, safe_df, table_contract, table_name=table_name
            )

            print(f"[TRAIN] {table_name}: fitting model...", flush=True)
            emit("fit_model", f"Fitting {model_type}")
            phase_started = perf_counter()
            synthesizer = create_synthesizer_adapter(model_type, **(model_kwargs or {}))
            fit_evidence = synthesizer.fit(safe_df, table_name, table_contract, seed)
            phase_durations_ms["fit_model"] = (perf_counter() - phase_started) * 1000
            print(
                f"[TRAIN] {table_name}: model fit completed "
                f"in {phase_durations_ms['fit_model'] / 1000:.2f}s",
                flush=True,
            )
            # Restore format metadata learned from pre-mask values BEFORE save/sample.
            pii_columns = getattr(synthesizer, "_pii_columns", None)
            if isinstance(pii_columns, dict):
                for column_name, fmt in identifier_formats.items():
                    if column_name in pii_columns:
                        pii_columns[column_name]["identifier_format"] = fmt

            print(f"[TRAIN] {table_name}: generating validation sample...", flush=True)
            emit("sample_model", "Generating validation sample from trained model")
            phase_started = perf_counter()
            generated_df = synthesizer.sample(num_rows_to_generate, seed=seed)
            phase_durations_ms["sample_model"] = (perf_counter() - phase_started) * 1000
            print(
                f"[TRAIN] {table_name}: sample completed "
                f"in {phase_durations_ms['sample_model'] / 1000:.2f}s",
                flush=True,
            )

            model_path = models_dir / f"{table_name}{synthesizer.file_extension}"
            emit("save_model", "Saving portable model artifact")
            phase_started = perf_counter()
            synthesizer.save(model_path)
            phase_durations_ms["save_model"] = (perf_counter() - phase_started) * 1000

            samples_dir.mkdir(parents=True, exist_ok=True)
            sample_path = samples_dir / f"{table_name}.csv"
            phase_started = perf_counter()
            generated_df.to_csv(sample_path, index=False)
            phase_durations_ms["write_sample"] = (perf_counter() - phase_started) * 1000
            total_duration_ms = (perf_counter() - table_started) * 1000

            print(
                f"[TRAIN] [{table_index}/{total_tables}] {table_name}: DONE "
                f"in {total_duration_ms / 1000:.2f}s",
                flush=True,
            )
            emit("complete", "Training completed for table", status="complete")

            table_reports[table_name] = {
                "model_type": model_type,
                "seed": seed,
                "data_fingerprint": compute_dataframe_fingerprint(safe_df),
                "fit_evidence": fit_evidence,
                "pre_input_mask": {
                    "context_aware_columns": mask_report.get("context_aware_columns", []),
                    "locale": mask_report.get("locale"),
                },
                "training_trace": {
                    "source_rows_read": int(len(df)),
                    "clean_rows": int(len(cleaned_df)),
                    "fit_rows": int(len(safe_df)),
                    "phase_durations_ms": {k: round(v, 3) for k, v in phase_durations_ms.items()},
                    "total_duration_ms": round(total_duration_ms, 3),
                },
                "generated_row_count": len(generated_df),
                "model_path": str(model_path),
                "sample_path": str(sample_path),
                "serialization_format": synthesizer.serialization_format,
            }
    except (SourceAdapterError, SynthesisError, FitPrivacyError) as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    report_data = {
        "trained_at": datetime.now(UTC).isoformat(),
        "tables": table_reports,
    }
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[f"could not write {TRAINING_REPORT_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    total_generated_rows = sum(t["generated_row_count"] for t in table_reports.values())

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[contract_reference],
        output_references=[str(output_path)]
        + [t["model_path"] for t in table_reports.values()]
        + [t["sample_path"] for t in table_reports.values()],
        metrics={
            "table_count": len(table_reports),
            "total_generated_rows": total_generated_rows,
        },
        warnings=warnings,
        evidence={"trained_at": report_data["trained_at"]},
    )
    manifest.record_stage(result)
    return result


def load_training_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a training_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise TrainingReportLoadError(f"training_report.json not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise TrainingReportLoadError(f"training_report.json is not valid JSON: {report_path}") from exc

    if not isinstance(data, dict):
        raise TrainingReportLoadError(f"training_report.json did not parse to an object: {report_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise TrainingReportLoadError(f"training_report.json missing required keys: {sorted(missing)}")

    for table_name, table in data["tables"].items():
        if not isinstance(table, dict):
            raise TrainingReportLoadError(f"training_report.json table {table_name!r} must be an object")
        missing_table_keys = _REQUIRED_TABLE_KEYS - table.keys()
        if missing_table_keys:
            raise TrainingReportLoadError(
                f"training_report.json table {table_name!r} missing keys: {sorted(missing_table_keys)}"
            )

    return data
