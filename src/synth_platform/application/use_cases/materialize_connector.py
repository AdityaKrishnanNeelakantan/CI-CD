"""Materialize a bounded connector sample into the canonical RelationalDataset."""
from __future__ import annotations

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.profiling.models import SamplingRecord
from synth_platform.domain.schema.source import SourceConnector


def materialize_connector(
    connector: SourceConnector,
    sample_size: int,
    seed: int,
    selected_tables: list[str] | tuple[str, ...] | None = None,
) -> tuple[RelationalDataset, dict[str, SamplingRecord]]:
    schema = connector.discover_schema()
    selected = set(selected_tables or schema.tables)
    unknown = selected - set(schema.tables)
    if unknown:
        raise ValueError(f"unknown selected tables: {sorted(unknown)}")
    schema.tables = {k: v for k, v in schema.tables.items() if k in selected}
    schema.foreign_keys = [fk for fk in schema.foreign_keys
                           if fk.parent_table in selected and fk.child_table in selected]
    frames = {}
    records = {}
    for name, table in schema.tables.items():
        frame = connector.sample_table(name, max_rows=sample_size, seed=seed)
        frames[name] = frame
        supplied = getattr(connector, "last_sampling_record", None)
        record = supplied(name) if callable(supplied) else None
        records[name] = record or SamplingRecord(
            strategy="deterministic_bounded",
            seed=seed,
            population_count=int(table.row_count),
            sample_count=int(len(frame)),
            bias_warning=("" if len(frame) >= table.row_count
                          else "bounded sample; statistics are sample-based"),
        )
    dataset = RelationalDataset(schema=schema, tables=frames)
    dataset.validate()
    return dataset, records
