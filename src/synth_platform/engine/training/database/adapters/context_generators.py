"""Context-aware synthetic value helpers for Database Twin synthesizers.

Shared by SafeCopula and DPCopula adapters. Field-kind resolution and
rendering come from privacy.context_fields (same registry as the
pre-input DeterministicFakerMasker). Statistical columns are unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from synth_platform.engine.common.database.privacy.context_fields import (
    ContextFieldKind,
    is_person_name_column,
    make_faker,
    person_name_part,
    render_context_value,
    render_identifier_value,
    resolve_context_kind,
)

__all__ = [
    "build_row_name_context",
    "contextual_email",
    "fake_free_text",
    "fake_person_name",
    "generate_identifier_values",
    "generate_pii_columns",
    "learn_identifier_format",
    "make_faker",
]


def fake_person_name(faker, column_name: str) -> str:
    return str(render_context_value(ContextFieldKind.PERSON_NAME, faker, column_name))


def fake_free_text(faker, column_name: str) -> str:
    kind = resolve_context_kind(column_name, "free_text")
    return str(render_context_value(kind, faker, column_name))


def _format_has_syn_placeholder(format_spec: Mapping[str, Any] | None) -> bool:
    """True when a learned format still carries legacy SYN-… marker literals."""
    if not format_spec or format_spec.get("kind") != "segmented":
        return False
    for seg in format_spec.get("segments") or []:
        if seg.get("type") != "literal":
            continue
        value = str(seg.get("value", "")).strip().upper()
        if value == "SYN" or value.startswith("SYN"):
            return True
    return False


def learn_identifier_format(values: Sequence[Any], *, min_agreement: float = 0.8) -> dict[str, Any] | None:
    """Learn a safe identifier FORMAT from values — never stores raw IDs."""
    samples = [str(v) for v in values if v is not None and str(v).strip() != ""]
    if len(samples) < 2:
        return None

    separator_counts: dict[str, int] = {}
    for value in samples:
        for sep in ("-", "_", "/", ":"):
            if sep in value:
                separator_counts[sep] = separator_counts.get(sep, 0) + 1
    if not separator_counts:
        return None
    separator = max(separator_counts, key=separator_counts.get)
    if separator_counts[separator] / len(samples) < min_agreement:
        return None

    split_rows = [value.split(separator) for value in samples]
    widths = {len(parts) for parts in split_rows}
    if len(widths) != 1:
        return None
    segment_count = next(iter(widths))
    if segment_count < 2:
        return None

    segments: list[dict[str, Any]] = []
    numeric_max_by_index: dict[int, int] = {}
    for idx in range(segment_count):
        parts = [row[idx] for row in split_rows]
        if all(part == parts[0] for part in parts):
            segments.append({"type": "literal", "value": parts[0]})
            continue
        if all(part.isdigit() for part in parts):
            width = max(len(part) for part in parts)
            numeric_max_by_index[idx] = max(int(part) for part in parts)
            segments.append({"type": "numeric", "width": width})
            continue
        return None

    if not any(seg["type"] == "numeric" for seg in segments):
        return None
    if not any(seg["type"] == "literal" for seg in segments):
        return None

    start_at = 0
    if numeric_max_by_index:
        last_numeric_idx = max(numeric_max_by_index)
        start_at = int(numeric_max_by_index[last_numeric_idx]) + 1

    learned = {
        "kind": "segmented",
        "separator": separator,
        "segments": segments,
        "start_at": start_at,
    }
    # Never lock in legacy SYN-… stand-ins as the "source" format.
    if _format_has_syn_placeholder(learned):
        return None
    return learned


def generate_identifier_values(
    num_rows: int,
    *,
    column_name: str,
    format_spec: Mapping[str, Any] | None,
    sequential: bool,
    rng,
    row_offset: int = 0,
) -> list[str]:
    """Generate synthetic identifiers from a learned format, or a safe fallback.

    ``row_offset`` shifts sequential counters so batched sampling stays unique.
    """
    offset = max(0, int(row_offset))
    # Old artifacts may still carry SYN-… formats learned from masked training
    # data; ignore them and use column-aware Faker patterns instead.
    if _format_has_syn_placeholder(format_spec):
        format_spec = None
    if format_spec and format_spec.get("kind") == "segmented":
        separator = str(format_spec.get("separator") or "-")
        segments = list(format_spec.get("segments") or [])
        start_at = int(format_spec.get("start_at") or 0)
        values: list[str] = []
        seen: set[str] = set()
        numeric_indexes = [i for i, seg in enumerate(segments) if seg.get("type") == "numeric"]
        primary_numeric_idx = numeric_indexes[-1] if numeric_indexes else None
        for i in range(num_rows):
            parts: list[str] = []
            for seg_idx, seg in enumerate(segments):
                if seg.get("type") == "literal":
                    parts.append(str(seg.get("value", "")))
                elif seg.get("type") == "numeric":
                    width = max(1, int(seg.get("width") or 6))
                    if sequential and seg_idx == primary_numeric_idx:
                        number = start_at + offset + i
                        # Never modulo-wrap sequential counters — that reuses
                        # low values and breaks primary-key uniqueness once
                        # start_at + i exceeds 10**width (e.g. cust-1 width=1).
                        rendered = str(number)
                        if len(rendered) < width:
                            rendered = f"{number:0{width}d}"
                        parts.append(rendered)
                    elif sequential:
                        parts.append(f"{0:0{width}d}")
                    else:
                        number = int(rng.integers(0, 10**width))
                        parts.append(f"{number:0{width}d}")
                else:
                    parts.append(str(seg.get("value", "X")))
            candidate = separator.join(parts)
            attempt = 0
            while candidate in seen and attempt < 1000:
                attempt += 1
                repaired: list[str] = []
                for seg_idx, seg in enumerate(segments):
                    if seg.get("type") == "numeric":
                        width = max(1, int(seg.get("width") or 6))
                        if sequential and seg_idx == primary_numeric_idx:
                            # Keep advancing the sequential counter on collision.
                            number = start_at + offset + i + attempt
                            rendered = str(number)
                            if len(rendered) < width:
                                rendered = f"{number:0{width}d}"
                            repaired.append(rendered)
                        else:
                            repaired.append(f"{int(rng.integers(0, 10**width)):0{width}d}")
                    elif seg.get("type") == "literal":
                        repaired.append(str(seg.get("value", "")))
                    else:
                        repaired.append("X")
                candidate = separator.join(repaired)
            seen.add(candidate)
            values.append(candidate)
        return values

    if sequential:
        # Prefer learned format above; without it, use Faker-style patterns with
        # a monotonic numeric suffix so PKs stay unique across batches.
        faker = make_faker(seed=int(rng.integers(0, 2**31 - 1)) if hasattr(rng, "integers") else None)
        values: list[str] = []
        seen: set[str] = set()
        for i in range(num_rows):
            # Column-aware base pattern, then force uniqueness with offset index.
            base = render_identifier_value(faker, column_name)
            # Replace trailing digit run with a sequential counter when present.
            candidate = re.sub(r"(\d+)(?!.*\d)", lambda m: str(offset + i + 1).zfill(len(m.group(1))), base, count=1)
            if candidate == base or candidate in seen:
                candidate = f"{base}-{offset + i + 1}"
            attempt = 0
            while candidate in seen and attempt < 1000:
                attempt += 1
                candidate = f"{base}-{offset + i + 1 + attempt}"
            seen.add(candidate)
            values.append(candidate)
        return values
    faker = make_faker(seed=int(rng.integers(0, 2**31 - 1)) if hasattr(rng, "integers") else None)
    values = []
    seen: set[str] = set()
    for _ in range(num_rows):
        candidate = render_identifier_value(faker, column_name)
        attempt = 0
        while candidate in seen and attempt < 1000:
            attempt += 1
            candidate = render_identifier_value(faker, column_name)
            if attempt > 50:
                candidate = f"{candidate}-{attempt}"
        seen.add(candidate)
        values.append(candidate)
    return values


def build_row_name_context(
    columns: Mapping[str, Sequence[Any]],
    *,
    num_rows: int,
) -> list[dict[str, str]]:
    contexts = [{"first_name": "", "last_name": "", "full_name": ""} for _ in range(num_rows)]
    for column_name, values in columns.items():
        if not is_person_name_column(column_name):
            continue
        for i in range(min(num_rows, len(values))):
            value = "" if values[i] is None else str(values[i])
            parts = person_name_part(column_name, value)
            contexts[i].update({k: v for k, v in parts.items() if v})
    for ctx in contexts:
        if not ctx["full_name"] and (ctx["first_name"] or ctx["last_name"]):
            ctx["full_name"] = " ".join(p for p in (ctx["first_name"], ctx["last_name"]) if p).strip()
    return contexts


def contextual_email(faker, context: Mapping[str, str], *, used: set[str]) -> str:
    candidate = str(
        render_context_value(ContextFieldKind.EMAIL, faker, "email", row_context=context)
    )
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 1
    local, _, domain = candidate.partition("@")
    while True:
        repaired = f"{local}{suffix}@{domain}"
        if repaired not in used:
            used.add(repaired)
            return repaired
        suffix += 1


def generate_pii_columns(
    pii_columns: Mapping[str, Mapping[str, Any]],
    *,
    num_rows: int,
    primary_key: str | None,
    seed: int | None,
    rng,
    locale: str | None = None,
    row_offset: int = 0,
) -> dict[str, list[Any]]:
    """Generate all PII/unsynthesizable columns via the shared field-kind registry."""
    faker = make_faker(seed=seed, locale=locale)
    data: dict[str, list[Any]] = {}
    offset = max(0, int(row_offset))

    for column_name, pii_info in pii_columns.items():
        kind = resolve_context_kind(column_name, pii_info.get("semantic_type"))
        if kind is not ContextFieldKind.PERSON_NAME:
            continue
        data[column_name] = [
            render_context_value(kind, faker, column_name) for _ in range(num_rows)
        ]

    contexts = build_row_name_context(data, num_rows=num_rows)
    used_emails: set[str] = set()

    for column_name, pii_info in pii_columns.items():
        kind = resolve_context_kind(column_name, pii_info.get("semantic_type"))
        if kind is ContextFieldKind.PERSON_NAME:
            continue

        is_numeric_type = "INT" in str(pii_info.get("physical_type", "")).upper()
        format_spec = pii_info.get("identifier_format")

        if column_name == primary_key:
            if is_numeric_type:
                data[column_name] = list(range(1 + offset, 1 + offset + num_rows))
            else:
                data[column_name] = generate_identifier_values(
                    num_rows,
                    column_name=column_name,
                    format_spec=format_spec if isinstance(format_spec, Mapping) else None,
                    sequential=True,
                    rng=rng,
                    row_offset=offset,
                )
            continue

        if kind is ContextFieldKind.EMAIL:
            if any(ctx.get("first_name") for ctx in contexts):
                data[column_name] = [
                    contextual_email(faker, contexts[i], used=used_emails) for i in range(num_rows)
                ]
            else:
                emails = []
                for _ in range(num_rows):
                    email = faker.unique.email()
                    used_emails.add(email)
                    emails.append(email)
                data[column_name] = emails
            continue

        if kind in {
            ContextFieldKind.FREE_TEXT,
            ContextFieldKind.ADDRESS,
            ContextFieldKind.CITY,
            ContextFieldKind.STATE,
            ContextFieldKind.POSTAL_CODE,
            ContextFieldKind.COUNTRY,
            ContextFieldKind.COMPANY,
            ContextFieldKind.PHONE,
            ContextFieldKind.UUID,
        }:
            data[column_name] = [
                render_context_value(kind, faker, column_name, row_context=contexts[i])
                for i in range(num_rows)
            ]
            continue

        if is_numeric_type:
            data[column_name] = [int(rng.integers(1, 10**9)) for _ in range(num_rows)]
        else:
            data[column_name] = generate_identifier_values(
                num_rows,
                column_name=column_name,
                format_spec=format_spec if isinstance(format_spec, Mapping) else None,
                sequential=False,
                rng=rng,
                row_offset=0,
            )

    return data
