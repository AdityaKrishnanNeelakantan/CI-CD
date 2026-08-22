"""ReportLab renderer for PDF-track synthetic tables only."""
from __future__ import annotations

import ast
import hashlib
import uuid
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from synth_platform.application.dto.documents import DocumentRenderResult
from synth_platform.domain.artifacts.guards import require_source_kind
from synth_platform.domain.runs.models import SourceKind


def _numeric_bindings(tables) -> dict[str, float]:
    values = {}
    for frame in tables.values():
        for column in frame.columns:
            numeric = __import__("pandas").to_numeric(frame[column], errors="coerce").dropna()
            if len(numeric) == 1:
                values[column] = float(numeric.iloc[0])
            elif len(numeric) > 1:
                values[column] = float(numeric.sum())
    return values


def _eval_arithmetic(
    expression: str, values: dict[str, float], tolerance: float = 1e-6
) -> bool:
    if "=" not in expression:
        return False
    lhs, rhs = [part.strip() for part in expression.split("=", 1)]
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant)
    tree = ast.parse(rhs, mode="eval")
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return False
    if lhs not in values:
        return False
    result = eval(compile(tree, "<calculation>", "eval"), {"__builtins__": {}}, values)
    return abs(values[lhs] - float(result)) <= tolerance


class ReportlabRenderer:
    def _watermark(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 38)
        canvas.setFillColor(colors.Color(0.75, 0.75, 0.75, alpha=0.35))
        canvas.translate(4.2 * inch, 5.5 * inch)
        canvas.rotate(35)
        canvas.drawCentredString(0, 0, "SYNTHETIC / DEMO")
        canvas.restoreState()
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(7.8 * inch, 0.35 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def render(self, artifact, tables, output_path: str | Path,
               generation_run_id: str, document_id: str | None = None):
        require_source_kind(artifact, SourceKind.PDF)
        if artifact.document_template is None:
            raise ValueError("PDF-track artifact has no document template")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document_id = document_id or uuid.uuid4().hex
        page_size = A4 if artifact.document_template.page_size.upper() == "A4" else LETTER
        styles = getSampleStyleSheet()
        story = [Paragraph("Synthetic Document", styles["Title"]),
                 Paragraph(f"Artifact: {artifact.manifest.artifact_id}", styles["Normal"]),
                 Paragraph(f"Generation run: {generation_run_id}", styles["Normal"]),
                 Paragraph(f"Document: {document_id}", styles["Normal"]), Spacer(1, 12)]
        for section in artifact.document_template.sections:
            story.append(Paragraph(section.title, styles["Heading2"]))
            if section.static_text:
                story.append(Paragraph(section.static_text, styles["BodyText"]))
            for binding in artifact.document_template.field_bindings:
                if binding.section != section.name or binding.table not in tables:
                    continue
                frame = tables[binding.table]
                value = "" if frame.empty or binding.column not in frame else frame.iloc[0][binding.column]
                story.append(Paragraph(f"<b>{binding.label}:</b> {value}", styles["BodyText"]))
            for repeating in artifact.document_template.repeating_tables:
                if repeating.section != section.name or repeating.table not in tables:
                    continue
                frame = tables[repeating.table]
                columns = [c for c in repeating.columns if c in frame.columns]
                if not columns:
                    continue
                labels = repeating.labels if len(repeating.labels) == len(columns) else columns
                data = [labels] + [[str(value) for value in row]
                                   for row in frame[columns].itertuples(index=False, name=None)]
                table = Table(data, repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.extend([Spacer(1, 6), table, Spacer(1, 12)])
        SimpleDocTemplate(str(output), pagesize=page_size,
                          title="Synthetic / Demo Document").build(
            story, onFirstPage=self._watermark, onLaterPages=self._watermark)

        # Attach explicit synthetic lineage metadata without retaining source text.
        reader = PdfReader(str(output))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.add_metadata({
            "/synthetic": "true",
            "/artifact_id": artifact.manifest.artifact_id,
            "/generation_run_id": generation_run_id,
            "/document_id": document_id,
            "/Title": "Synthetic / Demo Document",
        })
        temporary = output.with_suffix(".metadata.pdf")
        with temporary.open("wb") as handle:
            writer.write(handle)
        temporary.replace(output)

        final_reader = PdfReader(str(output))
        text = "\n".join(page.extract_text() or "" for page in final_reader.pages)
        required_titles = [section.title for section in artifact.document_template.sections]
        bound_values = []
        for binding in artifact.document_template.field_bindings:
            frame = tables.get(binding.table)
            if frame is not None and not frame.empty and binding.column in frame:
                bound_values.append(str(frame.iloc[0][binding.column]))
        calculations = {
            f"calculation[{rule.name}]": _eval_arithmetic(
                rule.expression, _numeric_bindings(tables), rule.tolerance)
            for rule in artifact.document_template.calculations
        }
        bound_entity_ids = {}
        for table_name, frame in tables.items():
            schema_table = artifact.schema_.tables.get(table_name)
            key = schema_table.primary_key if schema_table is not None else None
            if key and not frame.empty and key in frame:
                bound_entity_ids[table_name] = str(frame.iloc[0][key])
        checks = {
            "opens": len(final_reader.pages) > 0,
            "watermark": "SYNTHETIC / DEMO" in text,
            "no_unresolved_tokens": "{{" not in text and "}}" not in text,
            "metadata_synthetic": str(final_reader.metadata.get("/synthetic", "")).lower() == "true",
            "artifact_id_present": artifact.manifest.artifact_id in text,
            "required_sections": all(title in text for title in required_titles),
            "bound_values_match": all(value in text for value in bound_values),
            **calculations,
        }
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return DocumentRenderResult(
            path=str(output), document_id=document_id, sha256=digest,
            page_count=len(final_reader.pages), valid=all(checks.values()),
            bound_entity_ids=bound_entity_ids, checks=checks)
