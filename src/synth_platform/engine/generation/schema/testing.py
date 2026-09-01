"""
Pytest integration for MVP.

Provides fixture factories that make synthetic data available to test functions
without any database setup or file I/O.

Usage::

    # conftest.py
    from synth_platform.engine.generation.schema.testing import mvp_fixture

    user_tables = mvp_fixture({"name": "Users", "tables": [...], "columns": {...}}, rows=500)

    # test_my_feature.py
    def test_user_count(user_tables):
        assert len(user_tables["users"]) == 500
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False


SchemaInput = Union[Dict[str, Any], str, "SchemaConfig"]


def _load_schema(schema: SchemaInput, rows: Optional[int], seed: int):
    from synth_platform.engine.inference.schema.schema import SchemaConfig
    from synth_platform.engine.inference.schema.yaml_schema import load_yaml_schema

    if isinstance(schema, SchemaConfig):
        loaded = schema.model_copy(deep=True)
    elif isinstance(schema, str):
        loaded = load_yaml_schema(schema, rows=rows, seed=seed)
    else:
        loaded = SchemaConfig(**schema)

    if rows is not None:
        for table in loaded.tables:
            table.row_count = rows
    loaded.seed = seed
    return loaded


def _generate_tables(schema) -> Dict[str, Any]:
    from synth_platform.engine.generation.schema.simulator import DataSimulator

    simulator = DataSimulator(schema)
    tables: Dict[str, Any] = {}
    for table_name, batch in simulator.generate_all():
        if table_name in tables:
            import pandas as pd

            tables[table_name] = pd.concat([tables[table_name], batch], ignore_index=True)
        else:
            tables[table_name] = batch.reset_index(drop=True)
    return tables


def mvp_fixture(
    schema: SchemaInput,
    rows: Optional[int] = 1000,
    seed: int = 42,
    min_quality_score: Optional[float] = None,
    max_retries: int = 3,
) -> Callable:
    """Create a pytest fixture that generates MVP synthetic tables from a schema."""
    if not _PYTEST_AVAILABLE:
        raise ImportError(
            "pytest is required to use mvp.testing. "
            "Install it with: pip install pytest"
        )

    @pytest.fixture(name=None, scope="function")
    def _fixture() -> Dict[str, Any]:
        loaded = _load_schema(schema, rows=rows, seed=seed)
        attempts = max(1, max_retries if min_quality_score is not None else 1)
        last_tables: Optional[Dict[str, Any]] = None

        for attempt in range(attempts):
            attempt_schema = loaded.model_copy(deep=True)
            attempt_schema.seed = seed + attempt
            last_tables = _generate_tables(attempt_schema)
            if min_quality_score is None:
                return last_tables

            from synth_platform.engine.validation.schema.reporting import FidelityChecker

            score = FidelityChecker(attempt_schema).score(last_tables)
            if score >= min_quality_score:
                return last_tables

        return last_tables or {}

    return _fixture


def mvp_schema_fixture(
    schema: SchemaInput,
    rows: Optional[int] = 1000,
    seed: int = 42,
) -> Callable:
    """Create a pytest fixture that returns a SchemaConfig (no data generated)."""
    if not _PYTEST_AVAILABLE:
        raise ImportError(
            "pytest is required to use mvp.testing. "
            "Install it with: pip install pytest"
        )

    @pytest.fixture(name=None, scope="function")
    def _fixture():
        return _load_schema(schema, rows=rows, seed=seed)

    return _fixture


if _PYTEST_AVAILABLE:
    @pytest.fixture
    def mvp_generate():
        """Pytest fixture providing a schema-driven table generator callable."""

        def _generate(
            schema: SchemaInput,
            *,
            rows: Optional[int] = 1000,
            seed: int = 42,
        ) -> Dict[str, Any]:
            loaded = _load_schema(schema, rows=rows, seed=seed)
            return _generate_tables(loaded)

        return _generate

    @pytest.fixture
    def mvp_schema():
        """Pytest fixture providing a schema loader callable."""

        def _schema(
            schema: SchemaInput,
            *,
            rows: Optional[int] = 1000,
            seed: int = 42,
        ):
            return _load_schema(schema, rows=rows, seed=seed)

        return _schema
