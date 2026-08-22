"""Release-gated rendering and packaging of PDF-track synthetic documents."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.domain.artifacts.guards import require_source_kind
from synth_platform.domain.runs.models import SourceKind
from synth_platform.domain.validation.models import Status
from synth_platform.errors import PublicationBlockedError




def _tables_for_document(artifact, tables, index: int):
    """Bind one root record and its descendants for a single document."""
    incoming = {fk.child_table for fk in artifact.schema_.foreign_keys}
    roots = [name for name in artifact.relational_plan.order if name not in incoming]
    selected = {}
    for root in roots:
        frame = tables.get(root)
        if frame is None or frame.empty:
            selected[root] = frame.copy() if frame is not None else frame
        else:
            selected[root] = frame.iloc[[index % len(frame)]].copy()
    for table_name in artifact.relational_plan.order:
        if table_name in selected:
            continue
        frame = tables.get(table_name)
        if frame is None:
            continue
        relevant = [fk for fk in artifact.schema_.foreign_keys
                    if fk.child_table == table_name and fk.parent_table in selected]
        if not relevant:
            selected[table_name] = (
                frame.iloc[[index % len(frame)]].copy() if len(frame) else frame.copy())
            continue
        mask = None
        for fk in relevant:
            parent = selected[fk.parent_table]
            if parent is None or parent.empty:
                current = frame.index.to_series().map(lambda _: False)
            else:
                parent_values = set(parent[fk.parent_column])
                current = frame[fk.child_column].isin(parent_values)
            mask = current if mask is None else (mask & current)
        selected[table_name] = frame.loc[mask].copy() if mask is not None else frame.copy()
    return selected

class SyntheticPdfPackageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package_path: str
    manifest_path: str
    documents: list[dict] = Field(default_factory=list)


def render_synthetic_pdf_package(
    artifact,
    tables,
    report,
    renderer,
    output_dir: str | Path,
    generation_run_id: str,
    document_count: int = 1,
) -> SyntheticPdfPackageResult:
    require_source_kind(artifact, SourceKind.PDF)
    if report.overall != Status.PASS:
        raise PublicationBlockedError(
            f"synthetic PDF rendering blocked: structured validation is {report.overall.value}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    for index in range(max(1, int(document_count))):
        bound_tables = _tables_for_document(artifact, tables, index)
        result = renderer.render(
            artifact, bound_tables, output / f"synthetic_{index + 1}.pdf",
            generation_run_id=generation_run_id)
        if not result.valid:
            raise PublicationBlockedError(
                f"synthetic document validation failed: {result.checks}")
        documents.append(result.model_dump(mode="json"))
    manifest = {
        "synthetic": True,
        "artifact_id": artifact.manifest.artifact_id,
        "generation_run_id": generation_run_id,
        "documents": documents,
    }
    manifest_path = output / "document_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    package_path = output / "synthetic_pdf_package.zip"
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, manifest_path.name)
        for document in documents:
            path = Path(document["path"])
            archive.write(path, path.name)
    return SyntheticPdfPackageResult(
        package_path=str(package_path), manifest_path=str(manifest_path),
        documents=documents)
