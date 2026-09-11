"""LlmExtractionPass: schema-guided extraction via an injected local LLM.

The LLM-first ingestion front-end. It maps an unstructured/irregular source
(PDF/DOCX/policy/manual text) into rows for a caller-supplied TargetSchema, using
a ChatModel (Ollama/Qwen3, gpt-oss, or a test fake). It is a sibling of
DocumentExtractionPass under the same ExtractionPass port — same canonical IR out.

Trust boundary (important): the LLM output is UNTRUSTED. This pass never lets the
model decide structure, keys, or integrity. It:
  1. prompts the model for JSON rows constrained to the schema's entities/fields,
  2. parses + validates the JSON against the schema (unknown entities/fields
     dropped; missing fields nulled; types coerced),
  3. routes the validated rows through the SAME deterministic lowering + PK/FK
     allocation the rule-based pass uses.
The model influences content only; the deterministic core still owns the shape.
"""
from __future__ import annotations

import json

import pandas as pd

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.application.ports.chat_model import ChatModel
from synth_platform.domain.extraction.lowering import lower_target_schema
from synth_platform.domain.extraction.target_schema import TargetSchema
from synth_platform.domain.guardrails.models import GuardrailPolicy
from synth_platform.engine.security.text_guardrails import DeterministicModelGuardrails
from synth_platform.errors import ExtractionError

_SYSTEM = (
    "You extract structured records from a document. Return ONLY JSON of the form "
    '{"<entity>": [{"<field>": <value>, ...}, ...], ...}. Use exactly the entity '
    "and field names given. Omit fields you cannot find. Invent nothing."
)


def _prompt(schema: TargetSchema, document_text: str, max_chars: int) -> str:
    spec = {e.name: [f.name for f in e.fields] for e in schema.entities}
    text = document_text[:max_chars]
    return (f"Entities and fields to extract:\n{json.dumps(spec, indent=2)}\n\n"
            f"Document:\n{text}\n\nReturn the JSON now.")


def _coerce(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)  # nested structures -> string, never dropped silently


def _normalise_evidence(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _coerce_grounded(value: object, source_evidence: str):
    coerced = _coerce(value)
    if coerced is None:
        return None
    normalized = _normalise_evidence(coerced)
    if not normalized or normalized not in source_evidence:
        return None
    return coerced


def _validate_rows(
    schema: TargetSchema,
    raw: dict,
    document_text: str,
) -> dict[str, list[dict]]:
    """Keep only known, source-grounded entity fields.

    Unknown keys and scalar values without normalized source evidence are
    discarded before deterministic lowering.
    """
    source_evidence = _normalise_evidence(document_text)
    out: dict[str, list[dict]] = {}
    for ent in schema.entities:
        field_names = {f.name for f in ent.fields}
        rows_in = raw.get(ent.name)
        rows_out: list[dict] = []
        if isinstance(rows_in, list):
            for r in rows_in:
                if not isinstance(r, dict):
                    continue
                rows_out.append(
                    {
                        fn: _coerce_grounded(r.get(fn), source_evidence)
                        for fn in field_names
                    }
                )
        elif isinstance(rows_in, dict):  # singleton returned as an object
            rows_out.append(
                {
                    fn: _coerce_grounded(rows_in.get(fn), source_evidence)
                    for fn in field_names
                }
            )
        out[ent.name] = rows_out
    return out


class LlmExtractionPass:
    kind = "llm_document"

    def __init__(self, model: ChatModel, document_text: str, target: TargetSchema,
                 max_chars: int = 12000):
        self.model = model
        self.document_text = document_text
        self.target = target
        self.max_chars = max_chars
        self.guardrails = DeterministicModelGuardrails()
        self.guardrail_policy = GuardrailPolicy(
            name="schema_guided_extraction",
            max_input_characters=max_chars + 8_000,
            max_output_characters=max_chars,
            required_input_markers=("Entities and fields to extract:", "Document:"),
            allowed_json_keys=tuple(entity.name for entity in target.entities),
        )
        self.last_input_report = None
        self.last_output_report = None

    def _extract_rows(self) -> dict[str, list[dict]]:
        prompt = _prompt(self.target, self.document_text, self.max_chars)
        guarded_prompt = self.guardrails.inspect_input(
            _SYSTEM,
            prompt,
            self.guardrail_policy,
        )
        self.last_input_report = guarded_prompt.report
        if guarded_prompt.report.blocked:
            raise ExtractionError("Document model input was blocked by guardrails.")
        completion = self.model.complete(
            guarded_prompt.system,
            guarded_prompt.user,
        )
        guarded_output = self.guardrails.inspect_output(
            completion,
            self.guardrail_policy,
            json_only=True,
        )
        self.last_output_report = guarded_output.report
        if guarded_output.report.blocked:
            raise ExtractionError("Document model output was blocked by guardrails.")
        try:
            raw = json.loads(guarded_output.text)
        except (json.JSONDecodeError, TypeError) as e:
            raise ExtractionError(f"LLM returned non-JSON output: {e}") from e
        if not isinstance(raw, dict):
            raise ExtractionError("LLM output was not a JSON object of entities")
        return _validate_rows(self.target, raw, self.document_text)

    def _build_frames(self, rows: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
        pk_values: dict[str, list[int]] = {}
        frames: dict[str, pd.DataFrame] = {}
        for ent in self.target.entities:
            df = pd.DataFrame(rows.get(ent.name) or [],
                              columns=[f.name for f in ent.fields])
            n = len(df)
            if ent.primary_key:
                df[ent.primary_key] = range(1, n + 1)
                pk_values[ent.name] = list(range(1, n + 1))
            frames[ent.name] = df
        for ent in self.target.entities:      # deterministic parent-FK allocation
            if ent.parent:
                parent_pks = pk_values.get(ent.parent.entity) or [1]
                df = frames[ent.name]
                df[ent.parent.fk_field] = [parent_pks[i % len(parent_pks)]
                                           for i in range(len(df))]
        return frames

    def extract(self) -> RelationalDataset:
        rows = self._extract_rows()
        schema = lower_target_schema(self.target, self.kind)
        frames = self._build_frames(rows)
        for name, ts in schema.tables.items():
            df = frames.get(name, pd.DataFrame())
            for col in ts.column_names:
                if col not in df.columns:
                    df[col] = None
            frames[name] = df[list(ts.column_names)]
        return RelationalDataset(schema=schema, tables=frames).finalize_counts()
