"""SourceConnector port — re-exported from the domain (canonical home).

The Protocol lives in `domain/schema/source.py` so inner layers depend inward.
This module remains for import-path stability; it adds nothing.
"""
from synth_platform.domain.schema.source import SourceConnector

__all__ = ["SourceConnector"]
