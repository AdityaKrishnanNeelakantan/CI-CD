"""Schema-guided Document -> canonical RelationalDataset extraction."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from synth_platform.infrastructure.documents.model import Document, Table
from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.extraction.lowering import lower_target_schema
from synth_platform.domain.extraction.target_schema import (
    ExtractionStrategy, FieldType, TargetEntity, TargetField, TargetSchema,
)

_BULLET = re.compile(r"^\s*(?:[-*•]|\(?[0-9]+[.)]|\(?[a-zA-Z][.)]|\([ivxIVX]+\))\s+")
_FOOTNOTE = re.compile(r"(?P<value>.*?)(?:\s*[\*†‡]+|\s*\[[0-9]+\])$")
_TRUE = {"yes", "y", "true", "1", "✓", "✔", "x"}
_FALSE = {"no", "n", "false", "0", "✗", "✘", ""}


def _normalise_header(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip().lower())
    return re.sub(r"[^a-z0-9_]+", "", value).strip("_")


def _looks_heading(text: str, aliases: list[str]) -> bool:
    low = text.lower().strip().rstrip(":")
    return any(a.lower() in low for a in aliases) if aliases else False


def _is_new_heading(text: str) -> bool:
    value = text.strip()
    if not value or _BULLET.match(value):
        return False
    if ":" in value and len(value.split(":", 1)[1].strip()) > 0:
        return False
    return len(value.split()) <= 8 and (value[0].isupper() or value[0].isdigit())


def _label_matches(label: str, aliases: list[str]) -> bool:
    low = label.lower().strip().rstrip(":")
    return any(alias.lower().strip().rstrip(":") in low for alias in aliases)


def _match_value(line: str, aliases: list[str]) -> str | None:
    if ":" not in line:
        return None
    label, _, value = line.partition(":")
    if _label_matches(label, aliases):
        return value.strip() or None
    return None


def _multiline_value(
    lines: list[str],
    index: int,
    aliases: list[str],
    stop_aliases: list[str] | None = None,
) -> str | None:
    line = lines[index]
    if ":" not in line:
        return None
    label, _, value = line.partition(":")
    if not _label_matches(label, aliases):
        return None
    if value.strip():
        return value.strip()
    continuation = []
    for following in lines[index + 1:]:
        stripped = following.strip()
        if not stripped:
            continue
        if ":" in stripped:
            break
        if stop_aliases and _looks_heading(stripped, stop_aliases):
            break
        continuation.append(stripped)
    return "\n".join(continuation) or None


def _strip_footnote(value: str) -> tuple[str, str | None]:
    match = _FOOTNOTE.fullmatch(value.strip())
    if not match:
        return value.strip(), None
    cleaned = match.group("value").strip()
    marker = value.strip()[len(cleaned):].strip() or None
    return cleaned, marker


def _coerce(value, field: TargetField):
    if value is None:
        return None
    text, _ = _strip_footnote(str(value))
    if text == "":
        return None
    if field.type in {FieldType.INTEGER, FieldType.NUMBER}:
        negative = text.startswith("(") and text.endswith(")")
        cleaned = text.strip("()").replace(",", "").replace("$", "").replace("%", "")
        number = pd.to_numeric(cleaned, errors="coerce")
        if pd.isna(number):
            return None
        number = -float(number) if negative else float(number)
        return int(number) if field.type == FieldType.INTEGER else number
    if field.type == FieldType.BOOLEAN:
        low = text.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        return None
    if field.type == FieldType.DATE:
        parsed = pd.to_datetime(text, errors="coerce")
        return None if pd.isna(parsed) else parsed.date().isoformat()
    return text


def _header_rows(rows: list[list[str]]) -> tuple[list[str], int]:
    """Reconstruct one/two-row headers into stable keys."""
    if not rows:
        return [], 0
    first = list(rows[0])
    if len(rows) < 2:
        return [_normalise_header(x) for x in first], 1
    second = list(rows[1])
    # Treat row 2 as a header when row 1 contains blanks/merged groups or row 2
    # contains mostly non-numeric labels.
    second_label_rate = sum(not re.search(r"\d", str(x)) for x in second if str(x).strip()) / max(
        1, sum(bool(str(x).strip()) for x in second))
    multi = any(not str(x).strip() for x in first) or second_label_rate >= 0.7
    if not multi:
        return [_normalise_header(x) for x in first], 1
    carried = ""
    headers = []
    width = max(len(first), len(second))
    for index in range(width):
        top = str(first[index] if index < len(first) else "").strip()
        bottom = str(second[index] if index < len(second) else "").strip()
        if top:
            carried = top
        parts = [part for part in (carried, bottom) if part]
        headers.append(_normalise_header("_".join(parts)))
    return headers, 2


class DocumentExtractionPass:
    kind = "document"

    def __init__(self, document: Document, target: TargetSchema):
        self.document = document
        self.target = target
        self.provenance: dict[str, list[dict]] = {}

    def _section_lines(self, entity: TargetEntity) -> list[str]:
        lines = [line.text for line in self.document.lines]
        if not entity.aliases:
            return lines
        sibling_aliases = [e.aliases for e in self.target.entities
                           if e.name != entity.name and e.aliases]
        out, capturing = [], False
        for text in lines:
            if _looks_heading(text, entity.aliases):
                capturing = True
                continue
            if capturing:
                if any(_looks_heading(text, aliases) for aliases in sibling_aliases):
                    capturing = False
                    continue
                if _is_new_heading(text):
                    capturing = False
                    continue
                out.append(text)
        return out

    def _table_score(self, table: Table, entity: TargetEntity) -> int:
        if not table.rows:
            return 0
        header, _ = _header_rows(table.rows)
        header_set = set(header)
        score = 0
        for alias in entity.aliases:
            if alias.lower() in (table.title or "").lower():
                score += 4
        for field in entity.fields:
            for alias in field.aliases or [field.name]:
                normal = _normalise_header(alias)
                if normal in header_set:
                    score += 2
                elif any(normal and normal in h for h in header):
                    score += 1
        return score

    def _selected_tables(self, entity: TargetEntity) -> list[Table]:
        scored = [(self._table_score(table, entity), table) for table in self.document.tables]
        positive = [(score, table) for score, table in scored if score > 0]
        if not positive:
            return []
        best = max(score for score, _ in positive)
        return [table for score, table in positive if score == best]

    def _table_rows(self, entity: TargetEntity) -> list[dict]:
        rows: list[dict] = []
        for table in self._selected_tables(entity):
            headers, start = _header_rows(table.rows)
            for row_index, raw in enumerate(table.rows[start:], start=start):
                row = {field.name: None for field in entity.fields}
                matched = 0
                for field in entity.fields:
                    aliases = [_normalise_header(x) for x in (field.aliases or [field.name])]
                    index = next((i for i, header in enumerate(headers)
                                  if header in aliases or any(alias and alias in header for alias in aliases)), None)
                    if index is not None and index < len(raw):
                        cleaned, marker = _strip_footnote(str(raw[index]))
                        value = _coerce(cleaned, field)
                        if value is not None:
                            row[field.name] = value
                            matched += 1
                            evidence = {
                                "row": len(rows), "field": field.name,
                                "page": table.page, "source_row": row_index,
                                "confidence": 1.0,
                            }
                            if marker:
                                evidence["footnote_marker"] = marker
                            self.provenance.setdefault(entity.name, []).append(evidence)
                if matched:
                    rows.append(row)
        return rows

    def _rows_for(self, entity: TargetEntity) -> list[dict]:
        fields = entity.fields
        primary = fields[0].name if fields else "value"
        if entity.strategy == ExtractionStrategy.SINGLETON:
            row = {field.name: None for field in fields}
            matched = 0
            section_lines = self._section_lines(entity)
            stop_aliases = [
                alias
                for candidate in self.target.entities
                for alias in candidate.aliases
                if candidate.name != entity.name
            ]
            for index, line in enumerate(section_lines):
                for field in fields:
                    aliases = field.aliases or [field.name]
                    value = _multiline_value(
                        section_lines, index, aliases, stop_aliases=stop_aliases)
                    if value is not None:
                        cleaned, marker = _strip_footnote(value)
                        coerced = _coerce(cleaned, field)
                        if coerced is not None:
                            row[field.name] = coerced
                            matched += 1
                            evidence = {"row": 0, "field": field.name,
                                        "line": index, "confidence": 1.0}
                            if marker:
                                evidence["footnote_marker"] = marker
                            self.provenance.setdefault(entity.name, []).append(evidence)
            return [row] if matched else []
        if entity.strategy == ExtractionStrategy.ROWS_PER_SECTION:
            return [{primary: line.text.strip()} for line in self.document.lines
                    if _looks_heading(line.text, entity.aliases)]
        if entity.strategy == ExtractionStrategy.ROWS_PER_TABLE_ROW:
            return self._table_rows(entity)
        rows = []
        for text in self._section_lines(entity):
            if _BULLET.match(text):
                item = _BULLET.sub("", text).strip()
                if item:
                    field = fields[0] if fields else TargetField(name=primary)
                    rows.append({primary: _coerce(item, field)})
        return rows

    def _build_frames(self) -> dict[str, pd.DataFrame]:
        pk_values: dict[str, list[int]] = {}
        frames: dict[str, pd.DataFrame] = {}
        for entity in self.target.entities:
            rows = self._rows_for(entity)
            frame = (pd.DataFrame(rows) if rows else
                     pd.DataFrame(columns=[field.name for field in entity.fields]))
            if entity.primary_key:
                frame[entity.primary_key] = range(1, len(frame) + 1)
                pk_values[entity.name] = list(range(1, len(frame) + 1))
            frames[entity.name] = frame
        for entity in self.target.entities:
            if not entity.parent:
                continue
            parents = pk_values.get(entity.parent.entity, [])
            frame = frames[entity.name]
            if len(frame) and not parents:
                frame[entity.parent.fk_field] = None
            else:
                frame[entity.parent.fk_field] = [
                    parents[index % len(parents)] for index in range(len(frame))
                ] if parents else []
        return frames

    def extract(self) -> RelationalDataset:
        schema = lower_target_schema(self.target, self.kind)
        frames = self._build_frames()
        for name, table in schema.tables.items():
            frame = frames.get(name, pd.DataFrame())
            for column in table.column_names:
                if column not in frame.columns:
                    frame[column] = None
            frames[name] = frame[list(table.column_names)]
        dataset = RelationalDataset(
            schema=schema, tables=frames,
            metadata={"document_warning_count": len(self.document.warnings)},
            provenance=self.provenance,
        ).finalize_counts()
        dataset.validate()
        return dataset
