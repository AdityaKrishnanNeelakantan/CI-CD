"""Tests for source-driven synthetic generation (SDV-backed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdv")

from synth_platform.engine.profiling.schema.profiler import mimic as mimic_fn
from synth_platform.engine.profiling.schema.profiler import mimic as schema_driven_mimic
from synth_platform.engine.inference.schema.source_driven import (
    PROFILE_FILENAME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    METADATA_FILENAME,
    ColumnKind,
    SourceDataProfiler,
    SourceDrivenGenerator,
    SourceModelType,
    generate_from_source,
    load_and_sample,
    recommend_source_model,
)


def _sample_source_df(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(rows)],
            "age": rng.integers(22, 75, rows),
            "balance": np.round(rng.normal(5000, 1500, rows), 2),
            "segment": rng.choice(["retail", "premium", "business"], rows, p=[0.55, 0.30, 0.15]),
            "is_active": rng.choice([True, False], rows, p=[0.82, 0.18]),
            "signup_date": pd.date_range("2021-01-01", periods=rows, freq="D"),
        }
    )


def _simple_source_df(rows: int = 80, seed: int = 0) -> pd.DataFrame:
    """Small numeric/categorical frame for CTGAN/TVAE smoke tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(22, 75, rows),
            "balance": np.round(rng.normal(5000, 1500, rows), 2),
            "segment": rng.choice(["retail", "premium", "business"], rows, p=[0.55, 0.30, 0.15]),
            "is_active": rng.choice([True, False], rows, p=[0.82, 0.18]),
        }
    )


def test_profile_extracts_all_column_kinds():
    df = _sample_source_df()
    profile = SourceDataProfiler().profile(df, table_name="customers")

    assert profile.row_count == len(df)
    assert profile.column_count == len(df.columns)
    kinds = {col.kind for col in profile.columns.values()}
    assert ColumnKind.NUMERIC.value in kinds
    assert ColumnKind.CATEGORICAL.value in kinds
    assert ColumnKind.DATE.value in kinds
    assert ColumnKind.BOOLEAN.value in kinds
    assert ColumnKind.ID_LIKE.value in kinds

    age = profile.columns["age"]
    assert age.sdtype == "numerical"
    assert "mean" in age.marginal
    assert profile.columns["segment"].marginal["top_values"]

    saved = json.loads(json.dumps(profile.to_dict()))
    restored = type(profile).from_dict(saved)
    assert restored.columns["customer_id"].kind == ColumnKind.ID_LIKE.value


def test_recommend_source_model_prefers_copula_for_small_data():
    df = _sample_source_df(rows=80)
    profile = SourceDataProfiler().profile(df)
    assert recommend_source_model(profile.row_count, profile.columns) == SourceModelType.GAUSSIAN_COPULA.value
    assert profile.recommended_model == SourceModelType.GAUSSIAN_COPULA.value


def test_recommend_source_model_prefers_copula_for_typical_uploads():
    df = _sample_source_df(rows=5000)
    profile = SourceDataProfiler().profile(df)
    assert recommend_source_model(profile.row_count, profile.columns) == SourceModelType.GAUSSIAN_COPULA.value


def test_fit_auto_caps_large_source_for_profile_and_training():
    df = _sample_source_df(rows=12_000, seed=3)
    generator = SourceDrivenGenerator()
    profile = generator.fit(df, model_type="gaussian_copula", seed=3, fit_max_rows=800, profile_max_rows=600)
    assert profile.row_count == 12_000
    synthetic = generator.sample(50, seed=3)
    assert len(synthetic) == 50
    assert set(synthetic.columns) == set(df.columns)


@pytest.mark.parametrize("model_type", ["gaussian_copula"])
def test_model_fit_sample_and_preserve_schema(model_type: str):
    df = _sample_source_df(rows=100)
    generator = SourceDrivenGenerator()
    profile = generator.fit(df, table_name="customers", model_type=model_type, seed=11, epochs=1)
    synthetic = generator.sample(60, seed=11)

    assert profile.table_name == "customers"
    assert len(synthetic) == 60
    assert set(synthetic.columns) == set(df.columns)
    assert synthetic["segment"].dropna().isin(df["segment"].unique()).all()
    assert pd.api.types.is_bool_dtype(synthetic["is_active"])


def test_save_and_load_artifacts(tmp_path: Path):
    df = _sample_source_df(rows=90)
    generator = SourceDrivenGenerator()
    generator.fit(df, model_type="gaussian_copula", seed=3)
    artifacts = generator.save(tmp_path, synthetic_rows=40, seed=3)

    assert artifacts.profile_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.manifest_path.exists()
    assert artifacts.model_path.name == MODEL_FILENAME

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_type"] == "gaussian_copula"
    assert manifest["source_rows"] == 90
    assert manifest["seed"] == 3

    profile_payload = json.loads((tmp_path / PROFILE_FILENAME).read_text(encoding="utf-8"))
    assert profile_payload["column_count"] == len(df.columns)

    loaded = SourceDrivenGenerator.load(tmp_path)
    resampled = loaded.sample(25, seed=99)
    assert len(resampled) == 25


def test_generate_from_source_writes_csv_and_metadata(tmp_path: Path):
    df = _sample_source_df(rows=100)
    synthetic, artifacts = generate_from_source(
        df,
        rows=50,
        model_type="gaussian_copula",
        output_dir=tmp_path,
        seed=21,
        table_name="customers",
    )

    assert len(synthetic) == 50
    assert artifacts.synthetic_path is not None
    assert artifacts.synthetic_path.exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_load_and_sample_from_saved_artifacts(tmp_path: Path):
    df = _sample_source_df(rows=80)
    generate_from_source(
        df,
        rows=30,
        model_type="gaussian_copula",
        output_dir=tmp_path,
        seed=5,
    )
    resampled = load_and_sample(tmp_path, rows=20, seed=5)
    assert len(resampled) == 20


def test_reproducibility_with_seed_control(tmp_path: Path):
    df = _sample_source_df(rows=100, seed=1)

    gen_a = SourceDrivenGenerator()
    gen_a.fit(df, model_type="gaussian_copula", seed=42)
    sample_a = gen_a.sample(40, seed=42)

    gen_b = SourceDrivenGenerator()
    gen_b.fit(df, model_type="gaussian_copula", seed=42)
    sample_b = gen_b.sample(40, seed=42)

    pd.testing.assert_frame_equal(sample_a.reset_index(drop=True), sample_b.reset_index(drop=True))

    generate_from_source(df, rows=40, model_type="gaussian_copula", output_dir=tmp_path / "a", seed=42)
    resample_a = load_and_sample(tmp_path / "a", rows=40, seed=42)
    resample_b = load_and_sample(tmp_path / "a", rows=40, seed=42)
    pd.testing.assert_frame_equal(resample_a.reset_index(drop=True), resample_b.reset_index(drop=True))


def test_schema_driven_mimic_still_works():
    df = _sample_source_df(rows=40, seed=99)
    tables = schema_driven_mimic(df, rows=30, seed=99, table_name="customers", engine="schema")
    synthetic = tables["customers"]
    assert len(synthetic) == 30
    assert set(synthetic.columns) == set(df.columns)


def test_mimic_auto_uses_source_when_sdv_available():
    df = _simple_source_df(rows=60, seed=3)
    tables = schema_driven_mimic(df, rows=25, seed=11, table_name="customers", engine="auto")
    assert len(tables["customers"]) == 25


def test_build_fidelity_report_numeric_columns():
    from synth_platform.engine.inference.schema.source_driven import SourceDataProfiler, build_fidelity_report

    source = _simple_source_df(rows=80, seed=2)
    synthetic = source.copy()
    synthetic["age"] = synthetic["age"] + 1
    profile = SourceDataProfiler().profile(source, table_name="customers")
    report = build_fidelity_report(source, synthetic, profile)
    assert report["generation_mode"] == "source_driven"
    assert report["columns"]
    assert "avg_numeric_mean_delta_pct" in report
    assert "avg_correlation_delta" in report
    assert "correlation_checks" in report
    assert "missingness" in report
    assert "privacy" in report


def test_build_sdv_metadata_avoids_sdv_runtime_warnings():
    import warnings

    from synth_platform.engine.inference.schema.source_driven import (
        SourceDataProfiler,
        SourceModelType,
        _create_synthesizer,
        build_sdv_metadata,
    )

    df = pd.DataFrame(
        {
            "Account Creation Date": ["2020-01-15", "2021-03-20", "2022-06-01"] * 50,
            "Last Active Date": ["2023-01-01", "2023-06-15", "2024-01-01"] * 50,
            "Balance": np.round(np.random.default_rng(0).normal(5000, 1500, 150), 2),
        }
    )
    profile = SourceDataProfiler().profile(df, table_name="table")
    sdv_columns = [name for name, col in profile.columns.items() if col.use_sdv]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = build_sdv_metadata(df, profile)
        synthesizer = _create_synthesizer(SourceModelType.GAUSSIAN_COPULA, metadata)
        synthesizer.fit(df[sdv_columns])

    sdv_warnings = [item for item in caught if "sdv" in str(getattr(item, "filename", ""))]
    assert not sdv_warnings, [str(item.message) for item in sdv_warnings]


def test_saved_metadata_uses_new_sdv_format(tmp_path: Path):
    df = _sample_source_df(rows=90)
    generator = SourceDrivenGenerator()
    generator.fit(df, model_type="gaussian_copula", seed=3)
    artifacts = generator.save(tmp_path, synthetic_rows=40, seed=3)

    payload = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert payload.get("METADATA_SPEC_VERSION") == "V1"
    assert "tables" in payload
    table_name = generator.profile.table_name if generator.profile else "source_table"
    date_col = payload["tables"][table_name]["columns"]["signup_date"]
    assert date_col["sdtype"] == "datetime"
    assert date_col.get("datetime_format")


def _customers_like_df(rows: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [
        "Kofi Huang", "Aiko Collins", "Aaliyah Jackson", "Kwame Jackson", "Ryan Jones",
        "Suresh Campbell", "Maya Patel", "Liam O'Brien", "Noah Singh", "Emma Wilson",
        "Olivia Brown", "Ethan Davis", "Sophia Miller", "James Garcia", "Isabella Lee",
        "Mason Taylor", "Charlotte Moore", "Lucas Anderson", "Amelia Thomas", "Harper Jackson",
    ][:rows]
    return pd.DataFrame(
        {
            "customer_id": rng.integers(1, 2000, rows),
            "full_name": names if rows <= len(names) else names + [f"User {i}" for i in range(rows - len(names))],
            "ssn": [f"{rng.integers(100,999):03d}-{rng.integers(10,99):02d}-{rng.integers(1000,9999):04d}" for _ in range(rows)],
            "aadhaar": [f"{rng.integers(1000,9999):04d} {rng.integers(1000,9999):04d} {rng.integers(1000,9999):04d}" for _ in range(rows)],
            "risk_tier": rng.choice(["low", "medium", "high"], rows, p=[0.7, 0.25, 0.05]),
        }
    )


def test_source_engine_does_not_replay_pii_on_customers_like_data():
    original = _customers_like_df(rows=20)
    syn = mimic_fn(original, rows=29, seed=42, table_name="customers", engine="source")["customers"]
    assert set(original["ssn"]) & set(syn["ssn"]) == set()
    assert set(original["full_name"]) & set(syn["full_name"]) == set()
    assert set(original["aadhaar"]) & set(syn["aadhaar"]) == set()
    assert syn["customer_id"].duplicated().sum() == 0
    assert syn["ssn"].duplicated().sum() == 0


def test_auto_engine_prefers_schema_for_small_pii_dataset():
    original = _customers_like_df(rows=20)
    syn = mimic_fn(original, rows=29, seed=42, table_name="customers", engine="auto")["customers"]
    assert set(original["ssn"]) & set(syn["ssn"]) == set()
    assert syn["customer_id"].between(original["customer_id"].min(), original["customer_id"].max() + 1000).all()


def test_profiler_maps_ssn_and_aadhaar_column_types():
    from synth_platform.engine.profiling.schema.profiler import DataProfiler

    df = _customers_like_df(rows=10)
    schema = DataProfiler().profile(df, table_name="customers")
    types = {col.name: col.type for col in schema.columns["customers"]}
    assert types["ssn"] == "ssn"
    assert types["aadhaar"] == "aadhaar"
    assert types["full_name"] == "text"


@pytest.mark.parametrize("model_type", ["gaussian_copula", "ctgan", "tvae"])
def test_all_sdv_model_types_fit_and_sample(model_type: str):
    df = _simple_source_df(rows=80)
    generator = SourceDrivenGenerator()
    generator.fit(df, model_type=model_type, seed=0, epochs=1)
    synthetic = generator.sample(15, seed=0)
    assert len(synthetic) == 15
    assert not synthetic.empty
