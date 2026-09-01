"""Shared context-aware field-kind registry for Database Twin.

Resolves a column's generation/masking kind from approved semantic_type plus
fast column-name rules. Used by:

- DeterministicFakerMasker (pre-input de-identification)
- sample-time context generators

Numeric/category/datetime/boolean stay PASSTHROUGH so statistical learning
is unchanged.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Mapping, Optional

from faker import Faker

_DEFAULT_LOCALE = "en_US"

_GIVEN_NAME_COLUMN_RE = re.compile(r"(^|_)(first|given|middle)_?names?$", re.IGNORECASE)
_FAMILY_NAME_COLUMN_RE = re.compile(r"(^|_)(last|family|maiden|sur)_?names?$", re.IGNORECASE)
_FULL_NAME_COLUMN_RE = re.compile(
    r"(^name$|(^|_)(full_?name|customer_?name|client_?name|patient_?name|contact_?name)$)",
    re.IGNORECASE,
)
_ADDRESS_COLUMN_RE = re.compile(r"(^|_)(address|street|street_?address|addr)(_|$)", re.IGNORECASE)
_CITY_COLUMN_RE = re.compile(r"(^|_)(city|town)(_|$)", re.IGNORECASE)
_STATE_COLUMN_RE = re.compile(r"(^|_)(state|province|region)(_|$)", re.IGNORECASE)
_POSTAL_COLUMN_RE = re.compile(r"(^|_)(postal|zip|zipcode|postcode)(_?code)?s?$", re.IGNORECASE)
_COUNTRY_COLUMN_RE = re.compile(r"(^|_)(country|nation|country_?code)(_|$)", re.IGNORECASE)
_COMPANY_COLUMN_RE = re.compile(
    r"(^|_)(company|organization|organisation|employer|business|merchant|vendor|supplier|"
    r"payee|retailer|store|facility|provider|hospital|clinic)(_?name)?$",
    re.IGNORECASE,
)
_UUID_COLUMN_RE = re.compile(r"(^|_)(uuid|guid)(_|$)", re.IGNORECASE)
_PHONE_COLUMN_RE = re.compile(r"(^|_)(phone|mobile|telephone|fax|cell)(_?number|_?no)?s?$", re.IGNORECASE)
_EMAIL_COLUMN_RE = re.compile(r"email", re.IGNORECASE)
_CARD_ID_RE = re.compile(r"(^|_)(card_?id|credit_?card|pan|cc_?number)(_|$)", re.IGNORECASE)
_ACCOUNT_ID_RE = re.compile(r"(^|_)(account_?id|account_?number|acct(_?no|_?num)?)(_|$)", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(
    r"(^|_)(customer_?id|client_?id|user_?id|member_?id|patient_?id)(_|$)",
    re.IGNORECASE,
)
_TRANSACTION_ID_RE = re.compile(r"(^|_)(transaction_?id|txn_?id|payment_?id)(_|$)", re.IGNORECASE)
_LOAN_ID_RE = re.compile(r"(^|_)(loan_?id)(_|$)", re.IGNORECASE)
_BRANCH_ID_RE = re.compile(r"(^|_)(branch_?id)(_|$)", re.IGNORECASE)
_MERCHANT_ID_RE = re.compile(r"(^|_)(merchant_?id)(_|$)", re.IGNORECASE)
_ORDER_ID_RE = re.compile(r"(^|_)(order_?id)(_|$)", re.IGNORECASE)
_SSN_RE = re.compile(r"(^|_)(ssn|social_?security)(_|$)", re.IGNORECASE)
_IBAN_RE = re.compile(r"(^|_)(iban)(_|$)", re.IGNORECASE)
_ROUTING_RE = re.compile(r"(^|_)(routing(_?number)?|aba)(_|$)", re.IGNORECASE)


class ContextFieldKind(str, Enum):
    PERSON_NAME = "person_name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CITY = "city"
    STATE = "state"
    POSTAL_CODE = "postal_code"
    COUNTRY = "country"
    COMPANY = "company"
    IDENTIFIER = "identifier"
    UUID = "uuid"
    FREE_TEXT = "free_text"
    PASSTHROUGH = "passthrough"


# Kinds that must never enter ML as raw production values.
CONTEXT_AWARE_KINDS = frozenset(
    {
        ContextFieldKind.PERSON_NAME,
        ContextFieldKind.EMAIL,
        ContextFieldKind.PHONE,
        ContextFieldKind.ADDRESS,
        ContextFieldKind.CITY,
        ContextFieldKind.STATE,
        ContextFieldKind.POSTAL_CODE,
        ContextFieldKind.COUNTRY,
        ContextFieldKind.COMPANY,
        ContextFieldKind.IDENTIFIER,
        ContextFieldKind.UUID,
        ContextFieldKind.FREE_TEXT,
    }
)


def make_faker(seed: int | None = None, locale: str | None = None) -> Faker:
    faker = Faker(locale or _DEFAULT_LOCALE)
    if seed is not None:
        faker.seed_instance(int(seed))
    return faker


def _kind_from_column_name(column_name: str) -> ContextFieldKind | None:
    """Column-name PII/context cues — used fail-closed even when semantic_type is statistical."""
    if _EMAIL_COLUMN_RE.search(column_name):
        return ContextFieldKind.EMAIL
    if _PHONE_COLUMN_RE.search(column_name):
        return ContextFieldKind.PHONE
    if _GIVEN_NAME_COLUMN_RE.search(column_name) or _FAMILY_NAME_COLUMN_RE.search(column_name):
        return ContextFieldKind.PERSON_NAME
    if _FULL_NAME_COLUMN_RE.search(column_name) or column_name.lower() == "name":
        return ContextFieldKind.PERSON_NAME
    if _ADDRESS_COLUMN_RE.search(column_name):
        return ContextFieldKind.ADDRESS
    if _CITY_COLUMN_RE.search(column_name):
        return ContextFieldKind.CITY
    if _STATE_COLUMN_RE.search(column_name):
        return ContextFieldKind.STATE
    if _POSTAL_COLUMN_RE.search(column_name):
        return ContextFieldKind.POSTAL_CODE
    if _COUNTRY_COLUMN_RE.search(column_name):
        return ContextFieldKind.COUNTRY
    if _COMPANY_COLUMN_RE.search(column_name):
        return ContextFieldKind.COMPANY
    if _UUID_COLUMN_RE.search(column_name):
        return ContextFieldKind.UUID
    if (
        _CARD_ID_RE.search(column_name)
        or _ACCOUNT_ID_RE.search(column_name)
        or _CUSTOMER_ID_RE.search(column_name)
        or _TRANSACTION_ID_RE.search(column_name)
        or _LOAN_ID_RE.search(column_name)
        or _BRANCH_ID_RE.search(column_name)
        or _MERCHANT_ID_RE.search(column_name)
        or _ORDER_ID_RE.search(column_name)
        or _SSN_RE.search(column_name)
        or _IBAN_RE.search(column_name)
        or _ROUTING_RE.search(column_name)
    ):
        return ContextFieldKind.IDENTIFIER
    return None


def resolve_context_kind(column_name: str, semantic_type: str | None) -> ContextFieldKind:
    """Resolve generation/masking kind.

    Column-name PII cues win over an approved statistical semantic_type so a
    mis-labelled ``email``/``name`` column marked ``category`` still never
    enters model fit as raw production values.
    """
    semantic = (semantic_type or "").strip().lower()

    # Strong column-name context cues win over broad semantic labels.
    #
    # Near-unique text fields such as address_line1, city or postal_code
    # may be inferred as "identifier".  They are still contextual fields
    # and must use the appropriate Faker provider rather than generic
    # identifier placeholders such as ADDRESSL-123456.
    name_kind = _kind_from_column_name(column_name)
    if name_kind is not None:
        return name_kind

    # Explicit PII semantic types are used when the column name does not
    # provide a stronger contextual signal.
    if semantic == "person_name":
        return ContextFieldKind.PERSON_NAME
    if semantic == "email":
        return ContextFieldKind.EMAIL
    if semantic == "phone_number":
        return ContextFieldKind.PHONE
    if semantic == "identifier":
        return ContextFieldKind.IDENTIFIER

    if semantic in {"numerical", "category", "boolean", "datetime"}:
        return ContextFieldKind.PASSTHROUGH

    if semantic == "free_text":
        return ContextFieldKind.FREE_TEXT
    if semantic in {"", "none", "null"}:
        return ContextFieldKind.PASSTHROUGH
    # Unknown semantic: fail closed to free_text masking rather than passthrough of raw text.
    return ContextFieldKind.FREE_TEXT


def render_identifier_value(faker: Faker, column_name: str) -> str:
    """Render a realistic identifier via Faker, keyed off the column name.

    Never emits ``SYN-…`` placeholders — those leak synthetic-looking markers
    into demos and break the expectation that IDs look like domain values.
    """
    if _CARD_ID_RE.search(column_name):
        # Luhn-valid card numbers read as real card_ids without storing PANs from source.
        return str(faker.credit_card_number())
    if _ACCOUNT_ID_RE.search(column_name):
        return f"ACC-{faker.random_number(digits=5, fix_len=True)}-{faker.random_number(digits=1, fix_len=True)}"
    if _CUSTOMER_ID_RE.search(column_name):
        return f"CUST-{faker.random_number(digits=5, fix_len=True)}"
    if _TRANSACTION_ID_RE.search(column_name):
        return f"TXN-{faker.random_number(digits=8, fix_len=True)}"
    if _LOAN_ID_RE.search(column_name):
        return f"LOAN-{faker.random_number(digits=6, fix_len=True)}"
    if _BRANCH_ID_RE.search(column_name):
        return f"BR-{faker.random_number(digits=3, fix_len=True)}"
    if _MERCHANT_ID_RE.search(column_name):
        return f"MER-{faker.random_number(digits=5, fix_len=True)}"
    if _ORDER_ID_RE.search(column_name):
        return f"ORD-{faker.random_number(digits=7, fix_len=True)}"
    if _SSN_RE.search(column_name):
        return str(faker.ssn())
    if _IBAN_RE.search(column_name):
        return str(faker.iban())
    if _ROUTING_RE.search(column_name):
        # ABA routing numbers when available; otherwise a fixed-width digit string.
        aba = getattr(faker, "aba", None)
        if callable(aba):
            return str(aba())
        return str(faker.random_number(digits=9, fix_len=True))
    if _UUID_COLUMN_RE.search(column_name):
        return str(faker.uuid4())

    # Generic id-like columns: short alphabetic stem + digits (e.g. WIDGET-004821).
    stem = re.sub(r"[^A-Za-z0-9]+", "", column_name).upper()
    stem = re.sub(r"ID$", "", stem) or "ID"
    stem = stem[:8]
    return f"{stem}-{faker.random_number(digits=6, fix_len=True)}"


def render_context_value(
    kind: ContextFieldKind,
    faker: Faker,
    column_name: str,
    *,
    row_context: Mapping[str, str] | None = None,
) -> Any:
    """Render one synthetic value for a context-aware kind."""
    if kind is ContextFieldKind.PERSON_NAME:
        if _GIVEN_NAME_COLUMN_RE.search(column_name):
            return faker.first_name()
        if _FAMILY_NAME_COLUMN_RE.search(column_name):
            return faker.last_name()
        return faker.name()
    if kind is ContextFieldKind.EMAIL:
        ctx = row_context or {}
        first = re.sub(r"[^A-Za-z0-9]+", "", ctx.get("first_name", "") or "").lower() or "user"
        last = re.sub(r"[^A-Za-z0-9]+", "", ctx.get("last_name", "") or "").lower()
        domain = faker.free_email_domain()
        local = f"{first}.{last}" if last and last != "synthetic" else first
        return f"{local}@{domain}"
    if kind is ContextFieldKind.PHONE:
        return faker.phone_number()
    if kind is ContextFieldKind.ADDRESS:
        normalized = column_name.strip().lower()

        # Secondary-address fields should contain suite/unit/apartment
        # information rather than another street address.
        if re.search(
            r"(^|_)(address_?line_?2|line_?2|address2|addr2|unit|suite|apartment)(_|$)",
            normalized,
        ):
            return faker.secondary_address()

        return faker.street_address()
    if kind is ContextFieldKind.CITY:
        return faker.city()
    if kind is ContextFieldKind.STATE:
        return faker.state()
    if kind is ContextFieldKind.POSTAL_CODE:
        return faker.postcode()
    if kind is ContextFieldKind.COUNTRY:
        return faker.country()
    if kind is ContextFieldKind.COMPANY:
        return faker.company()
    if kind is ContextFieldKind.UUID:
        return str(faker.uuid4())
    if kind is ContextFieldKind.IDENTIFIER:
        return render_identifier_value(faker, column_name)
    if kind is ContextFieldKind.FREE_TEXT:
        return faker.sentence()
    return None


def is_person_name_column(column_name: str) -> bool:
    return bool(
        _GIVEN_NAME_COLUMN_RE.search(column_name)
        or _FAMILY_NAME_COLUMN_RE.search(column_name)
        or _FULL_NAME_COLUMN_RE.search(column_name)
        or column_name.lower() == "name"
    )


def person_name_part(column_name: str, value: str) -> dict[str, str]:
    """Extract first/last/full parts from a masked/generated person name column."""
    text = (value or "").strip()
    if _GIVEN_NAME_COLUMN_RE.search(column_name):
        return {"first_name": text}
    if _FAMILY_NAME_COLUMN_RE.search(column_name):
        return {"last_name": text}
    tokens = [t for t in re.split(r"\s+", text) if t]
    if not tokens:
        return {"full_name": "", "first_name": "user", "last_name": "synthetic"}
    if len(tokens) == 1:
        return {"full_name": tokens[0], "first_name": tokens[0], "last_name": "synthetic"}
    return {
        "full_name": text,
        "first_name": tokens[0],
        "last_name": tokens[-1],
    }
