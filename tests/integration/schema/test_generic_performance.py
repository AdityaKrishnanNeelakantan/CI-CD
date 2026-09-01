"""Generic performance and memory policy tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.validation.schema.performance.report import build_pipeline_performance_report
from synth_platform.engine.validation.schema.performance.timing import PipelineStageTimings
from synth_platform.application.orchestration.schema.config import PipelineConfig
from synth_platform.application.orchestration.schema.session import clear_large_session_objects, store_preview_sample_only
from synth_platform.engine.generation.schema.simulator import DataSimulator
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table


pytest.importorskip("sdv")

from synth_platform.application.orchestration.schema.source_driven import run_source_pipeline


def _schema(rows: int = 500) -> SchemaConfig:
    return SchemaConfig(
        name="perf",
        seed=9,
        tables=[Table(name="records", row_count=rows)],
        columns={
            "records": [
                Column(name="record_id", type="int", unique=True, min=1, max=1000000),
                Column(name="value", type="float", min=0, max=100),
            ]
        },
    )


def test_performance_report_contains_required_fields():
    stages = PipelineStageTimings(generate_seconds=2.0, export_seconds=0.5)
    report = build_pipeline_performance_report(
        generation_mode="schema_driven",
        target_rows=1000,
        rows_generated=1000,
        chunk_size=250,
        stages=stages,
        peak_memory_mb=128.0,
        export_format="parquet",
        cache_hit=True,
        model_reused=False,
    )
    payload = report.to_dict()
    assert payload["rows_per_second"] == 500.0
    assert payload["peak_memory_mb"] == 128.0
    assert payload["cache_hit"] is True
    assert payload["chunked_mode"] is True


def test_progress_state_exists_on_pipeline_result(tmp_path: Path):
    from synth_platform.application.orchestration.schema.schema_driven import run_schema_pipeline

    result = run_schema_pipeline(
        _schema(50),
        PipelineConfig(
            generation_mode="schema_driven",
            preview_only=True,
            preview_rows=25,
            output_dir=tmp_path / "progress",
        ),
    )
    assert result.progress.stage == "complete"
    assert result.progress.to_dict()["stage"] == "complete"


def test_llm_disabled_for_large_full_generation_by_default():
    from synth_platform.engine.generation.text.generator import TextGenerationConfig, TextGenerationEngine

    engine = TextGenerationEngine(
        TextGenerationConfig(
            llm_enabled=False,
            is_preview=False,
            max_llm_rows=50,
        )
    )
    assert engine._effective_llm_cap(100_000) == 0


def test_sampled_fidelity_mode_for_large_source_run(tmp_path: Path):
    source = pd.DataFrame(
        {
            "id": np.arange(200),
            "amount": np.random.default_rng(0).normal(10, 2, 200),
            "segment": np.random.default_rng(0).choice(["a", "b"], 200),
        }
    )
    result = run_source_pipeline(
        source,
        PipelineConfig(
            generation_mode="source_driven",
            preview_only=False,
            preview_rows=50,
            full_rows=60_000,
            chunk_size=5_000,
            output_dir=tmp_path / "large",
            seed=3,
            table_name="bench",
            large_row_threshold=10_000,
            fidelity_mode="auto",
        ),
    )
    assert result.fidelity_report.get("fidelity_mode") == "sampled"
    assert result.fidelity_report.get("sampled") is True


def test_session_cleanup_removes_large_frames():
    session = SimpleNamespace()
    session.__dict__["full_tables"] = {"t": pd.DataFrame({"x": range(5000)})}
    session.__dict__["source_df"] = pd.DataFrame({"x": range(5000)})
    clear_large_session_objects(session.__dict__)
    assert "full_tables" not in session.__dict__
    assert "source_df" not in session.__dict__
