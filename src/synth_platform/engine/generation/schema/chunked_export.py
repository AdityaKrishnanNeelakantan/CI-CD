"""Chunked export for large synthetic datasets (Parquet-first, CSV second)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Parquet chunked export requires pyarrow. Install with: pip install -e '.[parquet]'"
        ) from exc


def export_csv_chunks(
    chunks: Iterator[pd.DataFrame],
    output_path: Union[str, Path],
    *,
    columns: Optional[List[str]] = None,
) -> int:
    """Append DataFrame chunks to a CSV file. Returns total rows written."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    first = True
    for chunk in chunks:
        if chunk.empty:
            continue
        frame = chunk if columns is None else chunk[columns]
        frame.to_csv(path, mode="w" if first else "a", index=False, header=first)
        first = False
        total += len(frame)
    return total


def export_parquet_chunks(
    chunks: Iterator[pd.DataFrame],
    output_path: Union[str, Path],
    *,
    compression: str = "snappy",
) -> int:
    """Write DataFrame chunks to a single Parquet file. Returns total rows written."""
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    try:
        for chunk in chunks:
            if chunk.empty:
                continue
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression)
            writer.write_table(table)
            total += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return total


def export_table_streaming(
    table_name: str,
    chunks: Iterator[pd.DataFrame],
    output_dir: Union[str, Path],
    *,
    fmt: str = "parquet",
    compression: str = "snappy",
) -> Dict[str, Any]:
    """Export one table from a chunk iterator.

    Returns dict with path, format, and rows_written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or "parquet").lower()
    if fmt == "parquet":
        path = out / f"{table_name}.parquet"
        rows = export_parquet_chunks(chunks, path, compression=compression)
    elif fmt == "csv":
        path = out / f"{table_name}.csv"
        rows = export_csv_chunks(chunks, path)
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}. Use 'parquet' or 'csv'.")
    return {"table": table_name, "path": path, "format": fmt, "rows_written": rows}


class StreamingParquetWriter:
    """Incremental Parquet writer for table batches."""

    def __init__(self, output_path: Union[str, Path], *, compression: str = "snappy") -> None:
        _require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pa = pa
        self._pq = pq
        self._compression = compression
        self._writer = None
        self.rows_written = 0

    def write_chunk(self, chunk: pd.DataFrame) -> int:
        if chunk.empty:
            return 0
        table = self._pa.Table.from_pandas(chunk, preserve_index=False)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self.path, table.schema, compression=self._compression)
        self._writer.write_table(table)
        self.rows_written += len(chunk)
        return len(chunk)

    def close(self) -> int:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.rows_written


class StreamingCsvWriter:
    """Incremental CSV writer for table batches."""

    def __init__(self, output_path: Union[str, Path]) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._first = True
        self.rows_written = 0

    def write_chunk(self, chunk: pd.DataFrame) -> int:
        if chunk.empty:
            return 0
        chunk.to_csv(self.path, mode="w" if self._first else "a", index=False, header=self._first)
        self._first = False
        self.rows_written += len(chunk)
        return len(chunk)

    def close(self) -> int:
        return self.rows_written
