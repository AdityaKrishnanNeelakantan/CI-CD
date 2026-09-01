"""Export-path validation for large generated datasets (DuckDB-backed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore


def _parquet_path(export_dir: Path, table: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = export_dir / f"{table}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _read_expr(path: Path) -> str:
    path_str = str(path).replace("'", "''")
    if path.suffix.lower() == ".parquet":
        return f"read_parquet('{path_str}')"
    return f"read_csv('{path_str}', auto_detect=true)"


def validate_export_paths(
    export_dir: Path,
    *,
    expected_counts: Mapping[str, int],
    pk_columns: Mapping[str, str],
    fk_checks: Sequence[Tuple[str, str, str, str]],
    schema_columns: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Validate row counts, PK uniqueness, and FK integrity on exported files."""
    if duckdb is None:
        return {
            "passed": False,
            "failure_reason": "duckdb not installed",
            "fix_suggestion": "Install duckdb for export-path validation.",
            "validation_scope": "full_export",
        }

    con = duckdb.connect()
    result: Dict[str, Any] = {
        "passed": True,
        "validation_scope": "full_export",
        "tables": {},
        "pk": {},
        "fk": [],
        "schema_columns": {},
        "row_count_passed": True,
        "pk_passed": True,
        "fk_passed": True,
    }

    for table, expected in expected_counts.items():
        path = _parquet_path(export_dir, table)
        if path is None:
            result["tables"][table] = {"expected": expected, "actual": None, "passed": False}
            result["passed"] = False
            result["row_count_passed"] = False
            continue
        actual = int(con.sql(f"SELECT COUNT(*) FROM {_read_expr(path)}").fetchone()[0])
        ok = actual == int(expected)
        result["tables"][table] = {"expected": expected, "actual": actual, "passed": ok, "path": str(path)}
        if not ok:
            result["passed"] = False
            result["row_count_passed"] = False

    for table, pk in pk_columns.items():
        path = _parquet_path(export_dir, table)
        if path is None:
            continue
        expr = _read_expr(path)
        total, distinct = con.sql(
            f"SELECT COUNT(*), COUNT(DISTINCT {pk}) FROM {expr}"
        ).fetchone()
        ok = int(total) == int(distinct) and int(total) > 0
        result["pk"][table] = {
            "column": pk,
            "total": int(total),
            "distinct": int(distinct),
            "passed": ok,
        }
        if not ok:
            result["passed"] = False
            result["pk_passed"] = False

    if schema_columns:
        for table, cols in schema_columns.items():
            path = _parquet_path(export_dir, table)
            if path is None:
                continue
            describe = con.sql(f"DESCRIBE SELECT * FROM {_read_expr(path)}").fetchall()
            actual_cols = {row[0] for row in describe}
            missing = [c for c in cols if c not in actual_cols]
            result["schema_columns"][table] = {"missing": missing, "passed": len(missing) == 0}
            if missing:
                result["passed"] = False

    for child_table, child_fk, parent_table, parent_pk in fk_checks:
        child_path = _parquet_path(export_dir, child_table)
        parent_path = _parquet_path(export_dir, parent_table)
        if child_path is None or parent_path is None:
            result["fk"].append(
                {
                    "check": f"{child_table}.{child_fk}->{parent_table}.{parent_pk}",
                    "passed": False,
                    "error": "missing export file",
                }
            )
            result["passed"] = False
            result["fk_passed"] = False
            continue
        orphans = int(
            con.sql(
                f"""
                SELECT COUNT(*) FROM {_read_expr(child_path)} c
                LEFT JOIN {_read_expr(parent_path)} p ON c.{child_fk} = p.{parent_pk}
                WHERE p.{parent_pk} IS NULL
                """
            ).fetchone()[0]
        )
        ok = orphans == 0
        result["fk"].append(
            {
                "check": f"{child_table}.{child_fk}->{parent_table}.{parent_pk}",
                "orphan_rows": orphans,
                "passed": ok,
            }
        )
        if not ok:
            result["passed"] = False
            result["fk_passed"] = False

    con.close()
    if not result["passed"]:
        result["failure_reason"] = "Export-path validation failed"
        result["fix_suggestion"] = "Inspect FK/PK/row-count sections in export_validation report."
    return result


def validate_single_table_export(
    path: Path,
    *,
    expected_rows: int,
    pk_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a single exported table (source-driven)."""
    if duckdb is None or not path.exists():
        return {"passed": False, "failure_reason": "missing export or duckdb"}
    con = duckdb.connect()
    expr = _read_expr(path)
    actual = int(con.sql(f"SELECT COUNT(*) FROM {expr}").fetchone()[0])
    payload: Dict[str, Any] = {
        "passed": actual == expected_rows,
        "expected_rows": expected_rows,
        "actual_rows": actual,
        "row_count_passed": actual == expected_rows,
        "validation_scope": "full_export",
        "path": str(path),
    }
    if pk_column:
        total, distinct = con.sql(
            f"SELECT COUNT(*), COUNT(DISTINCT {pk_column}) FROM {expr}"
        ).fetchone()
        payload["pk_passed"] = int(total) == int(distinct) == expected_rows
        payload["distinct_pk"] = int(distinct)
        payload["passed"] = payload["passed"] and payload["pk_passed"]
    con.close()
    return payload
