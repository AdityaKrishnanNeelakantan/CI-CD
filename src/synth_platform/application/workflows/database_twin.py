"""Database Twin application workflow facade.

The Streamlit page depends on this module rather than importing individual
engine packages.  The implementation remains stage-oriented underneath:
discovery -> profiling -> inference -> training -> artifact -> generation ->
validation -> target write.
"""

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.domain.ssot.preview import SSOTPreview, SSOTPreviewItem
from synth_platform.engine.discovery.database.demo.sample_database import build_sample_database
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.training.database.planner import recommend_synthesizer
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export
from synth_platform.engine.generation.database.relational_service import (
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.generation.database.target_write_service import (
    load_target_write_report,
    run_target_write,
)
from synth_platform.engine.validation.database.qa_service import load_qa_report, run_qa_validation
from synth_platform.engine.common.database.core.run_manifest import RunManifest

WORKFLOW_STAGES = (
    "connect",
    "discovery",
    "profiling",
    "inference",
    "training",
    "artifact_export",
    "generation",
    "validation",
    "target_write",
)


def summarize_source_preview(adapter: SourceAdapter, *, sample_rows: int = 5) -> dict:
    """Return a live, bounded SSOT preview from the connected source adapter.

    This is intentionally separate from persisted discovery evidence. It lets
    presentation layers reflect the current source tables, row counts, column
    counts, and tiny samples before the immutable discovery stage is recorded.
    """
    tables: dict[str, dict] = {}
    for table_name in adapter.list_tables():
        columns = adapter.get_schema(table_name)
        rows = adapter.estimate_row_count(table_name)
        sample_df = adapter.read_sample(table_name, limit=max(1, int(sample_rows)))
        tables[table_name] = {
            "row_count": rows,
            "column_count": len(columns),
            "columns": columns,
            "sample_rows": sample_df.to_dict(orient="records"),
            "sample_columns": list(sample_df.columns),
        }
    return {
        "source_type": adapter.source_type,
        "table_count": len(tables),
        "total_rows": sum(int(table["row_count"]) for table in tables.values()),
        "total_columns": sum(int(table["column_count"]) for table in tables.values()),
        "tables": tables,
    }


def summarize_source_preview_contract(adapter: SourceAdapter, *, sample_rows: int = 5) -> SSOTPreview:
    """Typed SSOT preview for shared UI/API rendering."""
    preview = summarize_source_preview(adapter, sample_rows=sample_rows)
    return SSOTPreview(
        source_type=preview["source_type"],
        record_count=preview["total_rows"],
        field_count=preview["total_columns"],
        entity_count=preview["table_count"],
        relationship_count=None,
        items=[
            SSOTPreviewItem(
                name=table_name,
                kind="table",
                record_count=table["row_count"],
                field_count=table["column_count"],
                sample=table["sample_rows"],
                metadata={"sample_columns": table["sample_columns"]},
            )
            for table_name, table in preview["tables"].items()
        ],
    )

__all__ = [
    "SQLiteSourceAdapter", "RunManifest", "build_sample_database", "run_discovery",
    "PROFILE_FILENAME", "load_profile", "run_profiling", "load_dataset_contract",
    "run_contract_approval", "run_inference", "recommend_synthesizer",
    "get_synthesizer_adapter_class", "load_training_report", "run_training_and_sampling",
    "load_artifact", "run_artifact_export", "load_relational_generation_report",
    "run_relational_generation", "load_target_write_report", "run_target_write",
    "load_qa_report", "run_qa_validation", "summarize_source_preview",
    "summarize_source_preview_contract", "WORKFLOW_STAGES",
]
