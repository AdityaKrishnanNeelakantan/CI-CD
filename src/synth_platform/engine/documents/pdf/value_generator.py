"""Generates brand-new synthetic values for every bound field/table-cell -
the document-level analogue of the structured pipeline's
RelationalGenerationEngine (Generate stage).

Reads only two already-source-disconnected artifacts: document_template.json
(value_type + shape_pattern per field/cell - never the real value) and
document_binding_map.json (semantic_role + generator_strategy per
field/column). Never touches the source file. Seeded (both the local RNG
and Faker) so the same seed reproduces the same output, as required by the
project's stated acceptance criteria for both generation tracks.

Two families of generator:
- Faker-backed / vocabulary-backed (PII, categoricals, descriptions,
  medications, narrative): produce realistic values — never random
  character salad.
- Shape-filling (numeric/alphanumeric ids, currency, dates): refill the
  field's own persisted shape_pattern so format is preserved without
  ever seeing the source value.
"""

from __future__ import annotations

import random
import re
import string
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from faker import Faker

_FAKER_METHOD_BY_STRATEGY = {
    "fake_person_name": "name",
    "fake_email": "email",
    "fake_phone_number": "phone_number",
    "fake_street_address": "street_address",
    "fake_organization": "company",
}

_DATE_TOKEN_ORDER_BY_SHAPE = {
    "NNNN-NN-NN": ("%Y", "%m", "%d"),
    "NN/NN/NNNN": ("%m", "%d", "%Y"),
    "NN-NN-NNNN": ("%m", "%d", "%Y"),
}
_DATE_GENERATION_WINDOW_DAYS = 3 * 365
# DOBs must be in the past relative to "today"; keep ages adult-ish for
# clinical/finance demos without inventing infant records by accident.
_DOB_MIN_AGE_YEARS = 18
_DOB_MAX_AGE_YEARS = 90

# Small, fixed vocabularies for fields that only ever take one of a
# handful of real-world values (see src/documents/field_semantics.py and
# src/documents/binding_engine.py) - refilling their shape_pattern with
# random characters instead (the old generic_text fallback) produces
# unreadable gibberish ("Hhexdv", "Yzfwn") rather than a plausible value,
# a real bug found by inspecting an actual generated synthetic twin.
_CATEGORICAL_VOCABULARY_BY_ROLE: dict[str, tuple[str, ...]] = {
    "biological_sex": ("Male", "Female"),
    "smoking_status": ("Never", "Former", "Current"),
    "alcohol_consumption": ("Never", "Rarely", "Socially", "Regularly", "Former"),
    "diet_preference": (
        "Standard",
        "Vegetarian",
        "Vegan",
        "Diabetic-Friendly",
        "Low-Sodium",
        "Gluten-Free",
    ),
    "exercise_habits": ("Sedentary", "Light", "Moderate", "Active", "Very Active"),
}

_TRANSACTION_DESCRIPTIONS = (
    "Grocery Mart",
    "Direct Deposit Payroll",
    "Coffee Shop",
    "Fuel Station",
    "Online Transfer",
    "ATM Withdrawal",
    "Utility Payment",
    "Pharmacy Purchase",
    "Restaurant",
    "Subscription Service",
)

_MEDICAL_REASONS = (
    "Routine Diabetes Check-up",
    "Peripheral Neuropathy Assessment",
    "Follow-up Consultation",
    "Blood Pressure Review",
    "Medication Adjustment",
    "Lab Results Discussion",
    "Annual Physical Exam",
    "Pain Management Visit",
)

_MEDICATION_NAMES = (
    "Metformin",
    "Lisinopril",
    "Atorvastatin",
    "Amlodipine",
    "Omeprazole",
    "Levothyroxine",
    "Gabapentin",
    "Metoprolol",
    "Losartan",
    "Sertraline",
)

_DOSAGES = ("500mg", "10mg", "20mg", "25mg", "40mg", "50mg", "100mg", "250mg", "5mg", "2.5mg")
_FREQUENCIES = (
    "Once daily",
    "Twice daily",
    "Three times daily",
    "Every morning",
    "Every evening",
    "As needed",
    "Weekly",
)


def _fill_shape(shape_pattern: str, rng: random.Random, *, avoid_leading_zero: bool = False) -> str:
    chars: list[str] = []
    first_digit_in_run = True
    for ch in shape_pattern:
        if ch == "U":
            chars.append(rng.choice(string.ascii_uppercase))
        elif ch == "l":
            chars.append(rng.choice(string.ascii_lowercase))
        elif ch == "N":
            if avoid_leading_zero and first_digit_in_run:
                chars.append(rng.choice("123456789"))
            else:
                chars.append(rng.choice(string.digits))
            first_digit_in_run = False
        else:
            chars.append(ch)
            first_digit_in_run = True  # any non-digit boundary resets "leading" for the next run
    return "".join(chars)


def _generate_decimal_amount(shape_pattern: str, rng: random.Random) -> dict[str, Any]:
    formatted = _fill_shape(shape_pattern, rng, avoid_leading_zero=True)
    digits_and_dots = re.sub(r"[^0-9.]", "", formatted)

    if digits_and_dots.count(".") > 1:
        # A malformed/compound shape (e.g. two amounts merged into one
        # cell upstream, seen on a real OCR'd table row) - keep every
        # digit rather than silently truncating; treat only the final dot
        # as the real decimal separator and fold earlier dots away.
        head, _, tail = digits_and_dots.rpartition(".")
        digits_and_dots = head.replace(".", "") + "." + tail

    digits_and_dots = digits_and_dots.lstrip("0") or "0"
    if digits_and_dots.startswith("."):
        digits_and_dots = "0" + digits_and_dots

    return {"formatted": formatted, "decimal": str(Decimal(digits_and_dots))}


def _generate_date(shape_pattern: str, rng: random.Random, *, as_dob: bool = False) -> str:
    if as_dob:
        age_days = rng.randint(_DOB_MIN_AGE_YEARS * 365, _DOB_MAX_AGE_YEARS * 365)
        generated = date.today() - timedelta(days=age_days)
    else:
        offset = rng.randint(0, _DATE_GENERATION_WINDOW_DAYS)
        generated = date.today() - timedelta(days=offset)
    token_order = _DATE_TOKEN_ORDER_BY_SHAPE.get(shape_pattern)
    if token_order is None:
        return generated.isoformat()
    separator = "-" if "-" in shape_pattern else "/"
    return separator.join(generated.strftime(token) for token in token_order)


def _fit_text_to_shape(text: str, shape_pattern: str | None) -> str:
    """Prefer realistic text; only trim when a short shape budget exists."""
    if not shape_pattern:
        return text
    budget = len(shape_pattern)
    if budget <= 0 or len(text) <= budget:
        return text
    return text[: max(1, budget - 1)].rstrip() + "…"


def _generate_generic_text(shape_pattern: str | None, faker: Faker, rng: random.Random) -> str:
    """Replace shaped_text letter-salad with readable Faker content.

    Length is guided by shape_pattern when present so table cells stay
    roughly the same visual width without emitting gibberish like "Yzfwnk".
    """
    budget = len(shape_pattern) if shape_pattern else 24
    if budget <= 8:
        return faker.word().capitalize()
    if budget <= 20:
        return _fit_text_to_shape(faker.catch_phrase(), shape_pattern)
    sentence = faker.sentence(nb_words=rng.randint(4, 8))
    return _fit_text_to_shape(sentence.rstrip("."), shape_pattern)


class _DocumentIdentity:
    """One coherent synthetic subject per document (name/sex/DOB).

    Without this, summary spans and demographics fields each roll their
    own Faker identity — the twin shows conflicting names, sexes, and
    impossible DOB/age pairs across sections.
    """

    def __init__(self, faker: Faker, rng: random.Random) -> None:
        self._faker = faker
        self._rng = rng
        self._name: str | None = None
        self._sex: str | None = None
        self._dob_by_shape: dict[str, str] = {}
        self._dob_iso: date | None = None

    def person_name(self) -> str:
        if self._name is None:
            self._name = self._faker.name()
        return self._name

    def biological_sex(self) -> str:
        if self._sex is None:
            self._sex = self._rng.choice(_CATEGORICAL_VOCABULARY_BY_ROLE["biological_sex"])
        return self._sex

    def dob(self, shape_pattern: str | None) -> str:
        key = shape_pattern or "NNNN-NN-NN"
        if key not in self._dob_by_shape:
            if self._dob_iso is None:
                age_days = self._rng.randint(_DOB_MIN_AGE_YEARS * 365, _DOB_MAX_AGE_YEARS * 365)
                self._dob_iso = date.today() - timedelta(days=age_days)
            generated = self._dob_iso
            token_order = _DATE_TOKEN_ORDER_BY_SHAPE.get(key)
            if token_order is None:
                rendered = generated.isoformat()
            else:
                separator = "-" if "-" in key else "/"
                rendered = separator.join(generated.strftime(token) for token in token_order)
            self._dob_by_shape[key] = rendered
        return self._dob_by_shape[key]


def _label_implies_dob(label: str | None) -> bool:
    if not label:
        return False
    lowered = label.lower()
    return "dob" in lowered or "birth" in lowered


def _generate_field_value(
    generator_strategy: str,
    shape_pattern: str | None,
    rng: random.Random,
    faker: Faker,
    semantic_role: str | None = None,
    *,
    identity: _DocumentIdentity | None = None,
    label: str | None = None,
    llm_text_enabled: bool = False,
) -> dict[str, Any]:
    # Document-scoped identity for roles that must stay consistent.
    if identity is not None:
        if generator_strategy == "fake_person_name" or semantic_role == "person_name":
            return {"value": identity.person_name()}
        if semantic_role == "biological_sex" and generator_strategy == "categorical_choice":
            return {"value": identity.biological_sex()}
        if generator_strategy == "bounded_date" and _label_implies_dob(label):
            return {"value": identity.dob(shape_pattern)}

    faker_method = _FAKER_METHOD_BY_STRATEGY.get(generator_strategy)
    if faker_method:
        return {"value": getattr(faker, faker_method)()}

    if generator_strategy == "categorical_choice":
        vocabulary = _CATEGORICAL_VOCABULARY_BY_ROLE.get(semantic_role or "", ())
        if vocabulary:
            return {"value": rng.choice(vocabulary)}
        # No vocabulary registered for this role (shouldn't happen given
        # GENERATOR_STRATEGY_BY_ROLE only maps roles that have one) - fall
        # through to readable generic text rather than emitting nothing.

    if generator_strategy == "fake_transaction_description":
        if llm_text_enabled:
            llm_value = _llm_narrative_value(
                label=label or "transaction_description",
                text_role="transaction_note",
                seed=rng.randint(0, 2**31 - 1),
            )
            if llm_value:
                return {"value": llm_value}
        return {"value": rng.choice(_TRANSACTION_DESCRIPTIONS)}
    if generator_strategy == "fake_medical_reason":
        if llm_text_enabled:
            llm_value = _llm_narrative_value(
                label=label or "medical_reason",
                text_role="narrative",
                seed=rng.randint(0, 2**31 - 1),
            )
            if llm_value:
                return {"value": llm_value}
        return {"value": rng.choice(_MEDICAL_REASONS)}
    if generator_strategy == "fake_medication_name":
        return {"value": rng.choice(_MEDICATION_NAMES)}
    if generator_strategy == "fake_dosage":
        return {"value": rng.choice(_DOSAGES)}
    if generator_strategy == "fake_frequency":
        return {"value": rng.choice(_FREQUENCIES)}
    if generator_strategy == "fake_narrative":
        if llm_text_enabled:
            llm_value = _llm_narrative_value(
                label=label or "notes",
                text_role="narrative",
                seed=rng.randint(0, 2**31 - 1),
            )
            if llm_value:
                return {"value": _fit_text_to_shape(llm_value, shape_pattern)}
        return {"value": _fit_text_to_shape(faker.paragraph(nb_sentences=2), shape_pattern)}
    if generator_strategy == "fake_generic_text":
        if llm_text_enabled:
            llm_value = _llm_narrative_value(
                label=label or "text",
                text_role="narrative",
                seed=rng.randint(0, 2**31 - 1),
            )
            if llm_value:
                return {"value": _fit_text_to_shape(llm_value, shape_pattern)}
        return {"value": _generate_generic_text(shape_pattern, faker, rng)}

    if not shape_pattern:
        if generator_strategy in {"shaped_text", "fake_generic_text"}:
            return {"value": _generate_generic_text(None, faker, rng)}
        return {"value": None}

    if generator_strategy == "bounded_decimal":
        return _generate_decimal_amount(shape_pattern, rng)
    if generator_strategy == "bounded_date":
        return {"value": _generate_date(shape_pattern, rng, as_dob=_label_implies_dob(label))}

    # shaped_numeric_id / shaped_alphanumeric_id — and legacy shaped_text
    # callers, redirected to readable text so old binding maps cannot
    # resurrect gibberish.
    if generator_strategy == "shaped_text":
        return {"value": _generate_generic_text(shape_pattern, faker, rng)}
    return {
        "value": _fill_shape(
            shape_pattern, rng, avoid_leading_zero=generator_strategy == "shaped_numeric_id"
        )
    }


def _llm_narrative_value(*, label: str, text_role: str, seed: int) -> str | None:
    """Best-effort single narrative via shared text engine; None on failure."""
    try:
        from synth_platform.engine.inference.schema.schema import Column
        from synth_platform.engine.generation.text import TextGenerationConfig, generate_text_column

        column = Column(
            name=label.replace(" ", "_").lower() or "notes",
            type="text",
            distribution_params={
                "text_type": text_role,
                "llm_enabled": True,
                "llm_text": True,
                "max_words": 28,
                "min_words": 4,
            },
        )
        result = generate_text_column(
            table_name="document",
            column=column,
            size=1,
            config=TextGenerationConfig(
                llm_enabled=True,
                is_preview=True,
                max_llm_rows=1,
                seed=seed,
            ),
        )
        value = result.values[0] if result.values else None
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except Exception:
        return None


def generate_document_values(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    seed: int | None = None,
    *,
    llm_text_enabled: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    faker = Faker()
    if seed is not None:
        faker.seed_instance(seed)
    identity = _DocumentIdentity(faker, rng)

    regions_by_id = {
        region["region_id"]: region
        for page in template["pages"]
        for region in page["regions"]
    }

    field_values: dict[str, Any] = {}
    table_values: dict[str, Any] = {}
    inline_span_values: dict[str, Any] = {}

    for binding in binding_map["bindings"]:
        region = regions_by_id[binding["region_id"]]

        if binding["binding_type"] == "field":
            field_values[binding["region_id"]] = {
                "label": binding["label"],
                "semantic_role": binding["semantic_role"],
                **_generate_field_value(
                    binding["generator_strategy"],
                    region.get("shape_pattern"),
                    rng,
                    faker,
                    semantic_role=binding["semantic_role"],
                    identity=identity,
                    label=binding.get("label"),
                    llm_text_enabled=llm_text_enabled,
                ),
            }

        elif binding["binding_type"] == "inline_spans":
            source_spans = region.get("inline_variable_spans", [])
            spans_out = []
            for span_binding, source_span in zip(binding["spans"], source_spans):
                spans_out.append(
                    {
                        "start": span_binding["start"],
                        "end": span_binding["end"],
                        "semantic_role": span_binding["semantic_role"],
                        **_generate_field_value(
                            span_binding["generator_strategy"],
                            source_span.get("shape_pattern"),
                            rng,
                            faker,
                            semantic_role=span_binding["semantic_role"],
                            identity=identity,
                            label=None,
                            llm_text_enabled=llm_text_enabled,
                        ),
                    }
                )
            inline_span_values[binding["region_id"]] = spans_out

        else:  # table
            header_row_count = region.get("header_row_count", 0)
            rows_out = []
            for row_index, row in enumerate(region["rows"]):
                if row_index < header_row_count:
                    # Static, reusable column labels ("Date", "Description",
                    # "Amount") - the same wording every time this template
                    # is rendered, never regenerated from a shape_pattern,
                    # exactly like a heading/paragraph's static text.
                    row_out = [
                        {"semantic_role": None, "value": cell.get("text", "")} for cell in row
                    ]
                    rows_out.append(row_out)
                    continue
                row_out = []
                for column in binding["columns"]:
                    cell = row[column["column_index"]] if column["column_index"] < len(row) else None
                    shape_pattern = cell["shape_pattern"] if cell else None
                    row_out.append(
                        {
                            "semantic_role": column["semantic_role"],
                            **_generate_field_value(
                                column["generator_strategy"],
                                shape_pattern,
                                rng,
                                faker,
                                semantic_role=column["semantic_role"],
                                identity=identity,
                                label=column.get("label"),
                                llm_text_enabled=llm_text_enabled,
                            ),
                        }
                    )
                rows_out.append(row_out)
            table_values[binding["region_id"]] = rows_out

    return {"fields": field_values, "tables": table_values, "inline_spans": inline_span_values}
