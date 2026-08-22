"""Pre-input Deterministic Faker Masker for Database Twin.

Pipeline role
-------------
    Raw production sample
        -> DataCleaner
        -> DeterministicFakerMasker   <-- this module
        -> Safe context-preserved frame
        -> SynthesizerAdapter.fit()

Design
------
- Fast rules resolve each column to a ContextFieldKind.
- Hash(seed, table, column, raw_value) seeds Faker so the same source
  value always maps to the same synthetic stand-in.
- Row context is preserved: person-name columns are masked first, then
  email/company-style fields can consume that synthetic context.
- Raw PII is never written into mask reports — only counts/kinds.
- PASSTHROUGH kinds (numeric/safe-category/datetime/boolean) are untouched so
  statistical learning stays source-driven on non-PII DNA columns only.
  Column-name PII cues still win over a mis-labelled statistical type.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.privacy.context_fields import (
    CONTEXT_AWARE_KINDS,
    ContextFieldKind,
    make_faker,
    person_name_part,
    render_context_value,
    resolve_context_kind,
)

APPROVED_STATUS = "approved"


def _stable_seed(*parts: Any) -> int:
    payload = "|".join("" if p is None else str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class DeterministicFakerMasker:
    """Hash-seeded Faker masker for all context-aware field kinds."""

    def __init__(self, seed: int, locale: str | None = None) -> None:
        self.seed = int(seed)
        self.locale = locale

    def mask_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_contract: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        columns_contract = table_contract.get("columns", {})
        out = df.copy()
        column_reports: dict[str, Any] = {}
        row_contexts = [{"first_name": "", "last_name": "", "full_name": ""} for _ in range(len(out))]

        # Pass 1: person-name columns (build synthetic row context).
        for column_name, column_contract in columns_contract.items():
            if column_name not in out.columns:
                continue
            if column_contract.get("inference_status") != APPROVED_STATUS:
                continue
            kind = resolve_context_kind(column_name, column_contract.get("semantic_type"))
            if kind is not ContextFieldKind.PERSON_NAME:
                continue
            masked_values, replaced = self._mask_series(
                out[column_name],
                table_name=table_name,
                column_name=column_name,
                kind=kind,
                row_contexts=None,
            )
            out[column_name] = masked_values
            for i, value in enumerate(masked_values.tolist()):
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue
                parts = person_name_part(column_name, str(value))
                row_contexts[i].update({k: v for k, v in parts.items() if v})
            column_reports[column_name] = {
                "kind": kind.value,
                "action": "deterministic_faker_mask",
                "values_masked": replaced,
            }

        # Pass 2: all other context-aware kinds.
        for column_name, column_contract in columns_contract.items():
            if column_name not in out.columns:
                continue
            if column_contract.get("inference_status") != APPROVED_STATUS:
                continue
            kind = resolve_context_kind(column_name, column_contract.get("semantic_type"))
            if kind is ContextFieldKind.PERSON_NAME:
                continue
            if kind not in CONTEXT_AWARE_KINDS:
                column_reports[column_name] = {
                    "kind": kind.value,
                    "action": "passthrough",
                    "values_masked": 0,
                }
                continue
            masked_values, replaced = self._mask_series(
                out[column_name],
                table_name=table_name,
                column_name=column_name,
                kind=kind,
                row_contexts=row_contexts,
            )
            out[column_name] = masked_values
            column_reports[column_name] = {
                "kind": kind.value,
                "action": "deterministic_faker_mask",
                "values_masked": replaced,
            }

        report = {
            "table": table_name,
            "seed": self.seed,
            "locale": self.locale or "en_US",
            "columns": column_reports,
            "context_aware_columns": sorted(
                name for name, entry in column_reports.items() if entry.get("action") == "deterministic_faker_mask"
            ),
        }
        return out, report

    def _mask_series(
        self,
        series: pd.Series,
        *,
        table_name: str,
        column_name: str,
        kind: ContextFieldKind,
        row_contexts: list[dict[str, str]] | None,
    ) -> tuple[pd.Series, int]:
        values: list[Any] = []
        replaced = 0

        # Construct Faker once per column, then reseed it for each value.
        # This preserves deterministic raw-value -> synthetic-value mapping
        # while avoiding the very expensive creation of thousands of Faker
        # provider objects during training.
        faker = make_faker(locale=self.locale)

        for idx, raw in enumerate(series.tolist()):
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                values.append(raw)
                continue

            value_seed = _stable_seed(
                self.seed,
                table_name,
                column_name,
                raw,
            )
            faker.seed_instance(value_seed)

            ctx = row_contexts[idx] if row_contexts is not None else None

            # For email without person context, still deterministic from raw hash.
            rendered = render_context_value(
                kind,
                faker,
                column_name,
                row_context=ctx,
            )
            values.append(rendered)
            replaced += 1
        return pd.Series(values, index=series.index, dtype=object), replaced
