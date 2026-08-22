"""Maps a `source.type` string from project.yaml to a SourceAdapter class.

Adding a new source (e.g. PostgreSQL) means registering a new adapter class
here - it must never require changes to the profiler, inference engine,
trainer or validator, all of which only ever see the generic SourceAdapter
interface.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.errors import SourceConfigError
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter

_ADAPTER_REGISTRY: dict[str, type[SourceAdapter]] = {
    "sqlite": SQLiteSourceAdapter,
}


def create_source_adapter(source_type: str, connection_config: dict[str, Any]) -> SourceAdapter:
    adapter_cls = _ADAPTER_REGISTRY.get(source_type)
    if adapter_cls is None:
        known = ", ".join(sorted(_ADAPTER_REGISTRY)) or "(none registered)"
        raise SourceConfigError(
            f"Unknown source type {source_type!r}. Known types: {known}."
        )
    return adapter_cls(connection_config)
