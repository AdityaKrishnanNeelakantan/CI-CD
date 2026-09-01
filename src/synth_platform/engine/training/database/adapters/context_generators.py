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


def _classify_compact_char(char: str) -> str:
    if char.isdigit():
        return "digit"
    if char.isalpha() and char.isupper():
        return "upper"
    if char.isalpha() and char.islower():
        return "lower"
    return "literal"


def _learn_compact_identifier_format(samples: Sequence[str], *, min_agreement: float) -> dict[str, Any] | None:
    alpha_prefixes = [re.match(r"^[A-Za-z]+", value) for value in samples]
    if not all(alpha_prefixes):
        return None
    prefix = str(alpha_prefixes[0].group(0))
    for match in alpha_prefixes[1:]:
        value = str(match.group(0))
        while prefix and not value.startswith(prefix):
            prefix = prefix[:-1]
    if not prefix:
        return None

    tails = [value[len(prefix):] for value in samples if value.startswith(prefix)]
    if len(tails) < 2:
        return None
    tail_lengths = [len(tail) for tail in tails]
    modal_tail_length = max(set(tail_lengths), key=tail_lengths.count)
    small_sample_agreement = min(min_agreement, 0.75)
    if tail_lengths.count(modal_tail_length) / len(tail_lengths) < small_sample_agreement:
        return None
    tails = [tail for tail in tails if len(tail) == modal_tail_length]
    if modal_tail_length < 2:
        return None

    char_classes: list[str] = []
    literal_values: list[str | None] = []
    for idx in range(modal_tail_length):
        chars = [tail[idx] for tail in tails]
        classes = [_classify_compact_char(char) for char in chars]
        majority = max(set(classes), key=classes.count)
        if set(classes) <= {"digit", "upper"} and len(set(classes)) > 1:
            char_classes.append("alnum_upper")
            literal_values.append(None)
        elif classes.count(majority) / len(classes) < min_agreement:
            return None
        elif majority == "literal" and all(char == chars[0] for char in chars):
            char_classes.append("literal")
            literal_values.append(chars[0])
        elif majority in {"digit", "upper", "lower"}:
            char_classes.append(majority)
            literal_values.append(None)
        else:
            return None

    if all(kind == "literal" for kind in char_classes):
        return None

    return {
        "kind": "compact",
        "prefix": prefix,
        "tail_classes": char_classes,
        "tail_literals": literal_values,
    }


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
        return _learn_compact_identifier_format(samples, min_agreement=min_agreement)
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
    numeric_values_by_index: dict[int, list[int]] = {}
    for idx in range(segment_count):
        parts = [row[idx] for row in split_rows]
        if all(part == parts[0] for part in parts):
            segments.append({"type": "literal", "value": parts[0]})
            continue
        if all(part.isdigit() for part in parts):
            width = max(len(part) for part in parts)
            segments.append({"type": "numeric", "width": width})
            numeric_values_by_index[idx] = [int(part) for part in parts]
            continue
        return None

    if not any(seg["type"] == "numeric" for seg in segments):
        return None
    if not any(seg["type"] == "literal" for seg in segments):
        return None

    primary_numeric_idx = max(numeric_values_by_index)
    synthetic_start_at = max(numeric_values_by_index[primary_numeric_idx]) + max(1000, len(samples) + 1)

    learned = {
        "kind": "segmented",
        "separator": separator,
        "segments": segments,
        "start_at": 0,
        "start_at_mode": "after_observed_range",
        "synthetic_start_at": synthetic_start_at,
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
    if format_spec and format_spec.get("kind") == "compact":
        prefix = str(format_spec.get("prefix") or "")
        tail_classes = list(format_spec.get("tail_classes") or [])
        tail_literals = list(format_spec.get("tail_literals") or [])
        values: list[str] = []
        seen: set[str] = set()
        alphabet_upper = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        alphabet_lower = list("abcdefghijklmnopqrstuvwxyz")
        alphabet_upper_alnum = alphabet_upper + list("0123456789")
        for _ in range(num_rows):
            attempt = 0
            while True:
                chars: list[str] = []
                for idx, char_class in enumerate(tail_classes):
                    literal = tail_literals[idx] if idx < len(tail_literals) else None
                    if char_class == "literal" and literal is not None:
                        chars.append(str(literal))
                    elif char_class == "digit":
                        chars.append(str(int(rng.integers(0, 10))))
                    elif char_class == "lower":
                        chars.append(alphabet_lower[int(rng.integers(0, len(alphabet_lower)))])
                    elif char_class == "alnum_upper":
                        chars.append(alphabet_upper_alnum[int(rng.integers(0, len(alphabet_upper_alnum)))])
                    else:
                        chars.append(alphabet_upper[int(rng.integers(0, len(alphabet_upper)))])
                candidate = prefix + "".join(chars)
                if candidate not in seen:
                    break
                attempt += 1
                if attempt > 1000:
                    candidate = f"{candidate}{offset + len(values) + 1}"
                    break
            seen.add(candidate)
            values.append(candidate)
        return values
    if format_spec and format_spec.get("kind") == "segmented":
        separator = str(format_spec.get("separator") or "-")
        segments = list(format_spec.get("segments") or [])
        values: list[str] = []
        seen: set[str] = set()
        numeric_indexes = [i for i, seg in enumerate(segments) if seg.get("type") == "numeric"]
        primary_numeric_idx = numeric_indexes[-1] if numeric_indexes else None
        primary_width = (
            max(1, int(segments[primary_numeric_idx].get("width") or 6))
            if primary_numeric_idx is not None
            else 6
        )
        configured_start = int(format_spec.get("start_at") or 0)
        if sequential and format_spec.get("start_at_mode") == "after_observed_range":
            random_start = int(format_spec.get("synthetic_start_at") or configured_start)
        elif sequential and format_spec.get("start_at_mode") == "synthetic_random":
            lower = 10 ** (primary_width - 1) if primary_width > 1 else 10
            upper = 10**primary_width if primary_width > 1 else 10**6
            random_start = int(rng.integers(lower, upper))
        else:
            random_start = configured_start if sequential else 0
        for i in range(num_rows):
            parts: list[str] = []
            for seg_idx, seg in enumerate(segments):
                if seg.get("type") == "literal":
                    parts.append(str(seg.get("value", "")))
                elif seg.get("type") == "numeric":
                    width = max(1, int(seg.get("width") or 6))
                    if sequential and seg_idx == primary_numeric_idx:
                        number = random_start + offset + i
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
                            number = random_start + offset + i + attempt
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
                sequential=kind is ContextFieldKind.IDENTIFIER,
                rng=rng,
                row_offset=offset,
            )

    return data
