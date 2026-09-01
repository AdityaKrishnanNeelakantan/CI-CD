"""Infers each field's semantic role from its label text and its already-
detected value_type - the document-level analogue of
src/inference/engine.py's SemanticInferenceEngine, one layer above
src/documents/template_compiler.py's generic value_type classification
(numeric/currency/date/text/...). Same weighted-evidence,
confidence/status/evidence shape used throughout this project.

A field with no label at all (a bare table column, or a field regex
matched without a colon-delimited label) still gets a role from its
value_type alone - weaker evidence, but not nothing.
"""

from __future__ import annotations

from typing import Any

SEMANTIC_ROLES = frozenset(
    {
        "person_name",
        "email_address",
        "phone_number",
        "street_address",
        "account_number",
        "currency_amount",
        "date",
        "reference_code",
        "organization",
        "biological_sex",
        "smoking_status",
        "alcohol_consumption",
        "diet_preference",
        "exercise_habits",
        "transaction_description",
        "medical_reason",
        "medication_name",
        "dosage",
        "frequency",
        "narrative_text",
        "generic_numeric",
        "generic_text",
    }
)
STATUS_PROPOSED = "proposed"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"

# A value_type detected directly (src/documents/pii.py, entities.py, or
# template_compiler's own currency/date/numeric patterns) is strong,
# specific evidence - much stronger than a label keyword guess.
_VALUE_TYPE_WEIGHT = 3.0
_LABEL_KEYWORD_WEIGHT = 2.0

_VALUE_TYPE_TO_ROLE = {
    "email": "email_address",
    "phone_number": "phone_number",
    "street_address": "street_address",
    "PERSON": "person_name",
    "ORG": "organization",
    "date": "date",
    "currency": "currency_amount",
}

_LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "person_name": (
        "account holder",
        "patient name",
        "customer name",
        "full name",
        "contact person",
        "name",
        "customer",
        "patient",
        "attention",
        "attn",
    ),
    "email_address": ("email", "e-mail"),
    "phone_number": ("phone", "tel", "telephone", "mobile", "fax"),
    "street_address": ("address", "street", "location"),
    "account_number": ("account number", "account #", "acct", "iban", "routing"),
    "currency_amount": (
        "total",
        "amount",
        "balance",
        "due",
        "fee",
        "charge",
        "price",
        "cost",
        "payment",
        "debit",
        "credit",
    ),
    "date": ("date", "dob", "date of birth", "issued", "expiry", "expires", "statement date"),
    # Small-vocabulary health/demographic fields: refilling their
    # shape_pattern with random characters (the generic_text fallback)
    # produces unreadable gibberish ("Hhexdv", "Yzfwn") instead of a
    # plausible value, because these fields only ever take one of a
    # handful of real-world values - real bug caught by inspecting an
    # actual generated synthetic twin, where every field in this category
    # rendered as scrambled nonsense. Each role below is generated from a
    # small fixed vocabulary instead (see src/documents/binding_engine.py
    # and src/documents/value_generator.py). Listed before reference_code
    # below: "Diet Preference" incidentally contains both "ref" and
    # "reference" as substrings (of "prefeREFerence"/"pREFerence"), which
    # would otherwise outscore/tie-break against diet_preference in the
    # wrong direction - the extra "diet preference" keyword brings this
    # role's own score up to the same total so insertion order (this role
    # first) settles the tie correctly, found by a real regression test.
    "biological_sex": ("sex", "gender"),
    "smoking_status": ("smoking",),
    "alcohol_consumption": ("alcohol",),
    "diet_preference": ("diet", "diet preference"),
    "exercise_habits": ("exercise",),
    # Table / form columns that must never fall through to shaped_text
    # gibberish or weak PERSON name generation (Description/Reason → names).
    "transaction_description": ("description", "merchant", "payee", "memo", "narrative"),
    "medical_reason": ("reason", "diagnosis", "chief complaint", "visit reason"),
    "medication_name": ("medication", "medicine", "drug", "rx"),
    "dosage": ("dosage", "dose", "strength"),
    "frequency": ("frequency", "freq", "schedule"),
    "narrative_text": ("notes", "summary", "comment", "comments", "narrative", "history"),
    # Deliberately no bare "number": it collides with far more specific
    # labels ("Account Number", "Phone Number") that must resolve to their
    # own role, not fall back to a generic reference code - caught by
    # testing against a realistic label ("Account Number" tied 2.0-2.0
    # between account_number and reference_code before this fix).
    # Bare "account" alone is intentionally weaker than "account number"
    # / "account holder" phrases above so those win first.
    "reference_code": (
        "reference",
        "invoice number",
        "order number",
        "hospital id",
        "code",
        "ref",
        "ssn",
        "social security",
    ),
    "organization": (
        "company",
        "employer",
        "organization",
        "organisation",
        "vendor",
        "supplier",
        "institute",
        "clinic",
        "hospital",
        "provider",
    ),
}


def infer_field_semantics(label: str | None, value_type: str) -> dict[str, Any]:
    scores: dict[str, float] = {}
    evidence: list[str] = []

    role_from_value_type = _VALUE_TYPE_TO_ROLE.get(value_type)
    if role_from_value_type:
        scores[role_from_value_type] = scores.get(role_from_value_type, 0.0) + _VALUE_TYPE_WEIGHT
        evidence.append(f"value_type={value_type}")

    if label:
        lowered = label.lower()
        for role, keywords in _LABEL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    scores[role] = scores.get(role, 0.0) + _LABEL_KEYWORD_WEIGHT
                    evidence.append(f"label_keyword({role})={keyword}")
                    break  # one hit per role is enough; avoid double-counting phrases

    if not scores:
        default_role = "generic_numeric" if value_type == "numeric" else "generic_text"
        return {
            "semantic_role": default_role,
            "status": STATUS_REVIEW_REQUIRED,
            "confidence": 0.0,
            "evidence": ["no_signal"],
        }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_role, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(scores.values())
    confidence = round(top_score / total, 6)
    is_ambiguous = top_score == second_score

    return {
        "semantic_role": top_role,
        "status": STATUS_REVIEW_REQUIRED if is_ambiguous else STATUS_PROPOSED,
        "confidence": confidence,
        "evidence": evidence,
    }
