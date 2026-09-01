"""PDF generation use-case entry points.

Value generation and rendering remain separate operations so validation and
artifact lineage can be inspected between stages.
"""

from synth_platform.engine.documents.pdf.generation_service import run_value_generation
from synth_platform.engine.documents.pdf.render_service import run_document_rendering

__all__ = ["run_value_generation", "run_document_rendering"]
