"""PDF Twin application workflow facade.

Stages: ingest/preflight -> extraction/profile -> template compilation ->
semantic binding -> synthetic value generation -> rendering -> validation.
"""

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.docling_engine import is_docling_available
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import (
    DOCUMENT_PROFILE_FILENAME,
    load_document_profile,
    run_document_profiling,
)
from synth_platform.engine.documents.pdf.template_service import (
    load_document_template,
    run_template_compilation,
)
from synth_platform.engine.documents.pdf.deidentification_service import (
    load_document_deidentification_report,
    run_document_deidentification,
)
from synth_platform.engine.documents.pdf.binding_service import (
    load_document_binding_map,
    run_semantic_binding,
)
from synth_platform.engine.documents.pdf.generation_service import (
    load_document_synthetic_values,
    run_value_generation,
)
from synth_platform.engine.documents.pdf.render_service import (
    load_document_ground_truth,
    run_document_rendering,
)
from synth_platform.engine.documents.pdf.validation_service import (
    load_document_validation_report,
    run_document_validation,
)

WORKFLOW_STAGES = (
    "upload",
    "document_profiling",
    "template_compilation",
    "deidentification_optional",
    "semantic_binding",
    "value_generation",
    "rendering",
    "validation",
)

__all__ = [
    "RunManifest", "PDFDocumentAdapter", "is_docling_available", "DOCUMENT_PROFILE_FILENAME",
    "load_document_profile", "run_document_profiling", "load_document_template",
    "run_template_compilation", "load_document_deidentification_report",
    "run_document_deidentification", "load_document_binding_map", "run_semantic_binding",
    "load_document_synthetic_values", "run_value_generation", "load_document_ground_truth",
    "run_document_rendering", "load_document_validation_report", "run_document_validation",
    "WORKFLOW_STAGES",
]
