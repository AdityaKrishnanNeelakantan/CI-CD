"""The only release-gated publication door."""
from __future__ import annotations

from synth_platform.application.use_cases.validate_dataset import exercised_capabilities
from synth_platform.domain.validation.models import Status
from synth_platform.domain.validation.release_gate import decide
from synth_platform.errors import PublicationBlockedError


def publish_dataset(artifact, tables, report, sink, dataset_id: str):
    target_schema = getattr(sink, "target_schema", None)
    if target_schema:
        from synth_platform.domain.runs.models import SourceKind
        expected = "db_track_" if artifact.manifest.source_kind == SourceKind.DATABASE else "pdf_track_"
        if not target_schema.startswith(expected):
            raise PublicationBlockedError(
                f"artifact track cannot publish to schema {target_schema!r}; expected {expected}*")
    decision = decide(report, exercised=exercised_capabilities(artifact, has_holdout=False))
    if decision.verdict != Status.PASS:
        raise PublicationBlockedError(
            f"publication blocked: {decision.verdict.value}; {decision.blocking}")
    if hasattr(sink, "configure_artifact"):
        sink.configure_artifact(artifact)
    sink.begin(dataset_id)
    try:
        for name in artifact.relational_plan.order:
            if name not in tables:
                raise PublicationBlockedError(f"generated dataset missing table {name!r}")
            sink.write_table(name, tables[name])
        sink.commit()
    except Exception:
        sink.rollback()
        raise
    return {"dataset_id": dataset_id, "tables": list(artifact.relational_plan.order)}
