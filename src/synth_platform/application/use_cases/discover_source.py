"""Discover the source schema through the connector port."""
from __future__ import annotations

from synth_platform.application.ports.source_connector import SourceConnector
from synth_platform.domain.schema.models import DatabaseSchema


def discover_source(connector: SourceConnector) -> DatabaseSchema:
    connector.health_check()
    return connector.discover_schema()
