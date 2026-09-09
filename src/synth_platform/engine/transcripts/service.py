"""Transcript input adapter for the unified SSOT pipeline."""
from __future__ import annotations

import hashlib
import importlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from faker import Faker

from synth_platform.engine.generation.schema.locales import DEFAULT_LOCALE, LOCALE_ALIASES, get_locale_pack
from synth_platform.domain.contracts.models import (
    CanonicalContract,
    CanonicalEntity,
    CanonicalField,
    CanonicalProvenance,
    SourceDescriptor,
)
from synth_platform.domain.ssot.preview import SSOTPreview, SSOTPreviewItem
from synth_platform.engine.transcripts.ssot_designer import build_transcript_ssot_with_data_designer


_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_DOB = re.compile(
    r"\b(?:date of birth|dob|birth date)\s+(?:is|:)?\s*"
    r"(?:[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.IGNORECASE,
)
_ACCOUNT_ID = re.compile(
    r"(?:#\s*)?\b(?:policy|claim|account|case|ticket|tracking|member|customer|registration|license)\b"
    r"(?:\s+(?:number|id|tracking|plate))?\s*(?:#|id|number|is|:)?\s*"
    r"(?=[A-Z0-9 -]*\d)[A-Z]{1,6}[- ]?[A-Z0-9]{2,}(?:[- ][A-Z0-9]{2,})*\b",
    re.IGNORECASE,
)
_HASHED_ID = re.compile(r"#\s*[A-Z]{2,8}[- ]?[A-Z0-9]{2,}(?:[- ][A-Z0-9]{2,})*\b")
_SYNTHETIC_POLICY_ID = re.compile(r"\bPOL-[A-Z]{3}-\d{4}\b")
_SYNTHETIC_CLAIM_ID = re.compile(r"\bCLM-\d{8}\b")
_SYNTHETIC_GENERIC_ID = re.compile(r"\bID-\d{8}\b")
_BRACKET_TIMESTAMP = re.compile(r"\s*\[\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\]")
_ADDRESS_LABEL = re.compile(
    r"\b(?P<prefix>(?:my\s+)?(?:home\s+|mailing\s+|billing\s+|street\s+)?address\s+(?:is|:)\s*)"
    r"(?P<value>[^.\n;]+?)(?=\s+(?:and\s+)?(?:my\s+)?(?:phone|email|first|last|state|country|dob|date\s+of\s+birth)\b|[.\n;]|$)",
    re.IGNORECASE,
)
_STATE_LABEL = re.compile(
    r"\b(?P<prefix>(?:my\s+)?state\s+(?:is|:)\s*)"
    r"(?P<value>[A-Z]{2}|[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)?)\b",
    re.IGNORECASE,
)
_COUNTRY_LABEL = re.compile(
    r"\b(?P<prefix>(?:my\s+)?country\s+(?:is|:)\s*)"
    r"(?P<value>[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2})\b",
    re.IGNORECASE,
)
_FIRST_NAME_LABEL = re.compile(
    r"\b(?P<prefix>(?:my\s+)?first\s+name\s+(?:is|:)\s*)"
    r"(?P<value>[A-Z][A-Za-z'-]{1,40})\b",
    re.IGNORECASE,
)
_LAST_NAME_LABEL = re.compile(
    r"\b(?P<prefix>(?:my\s+)?last\s+name\s+(?:is|:)\s*)"
    r"(?P<value>[A-Z][A-Za-z'-]{1,40})\b",
    re.IGNORECASE,
)
_PROPER_NAME_AFTER_INTRO = re.compile(
    r"\b(?:my name is|my name|this is|i am|i'm|mr\.|mrs\.|ms\.|dr\.)\s+"
    r"(?!(?:in|from|located|based)\b)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
    re.IGNORECASE,
)
_ORG_PLACE = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}\s+"
    r"(?:Hospital|Clinic|Medical Center|Bank|Insurance|Claims Support|Support|Mutual|Street|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd))\b"
)
_SPEAKER_LINE = re.compile(r"^(?:(?P<timestamp>\[?\d{1,2}:\d{2}(?::\d{2})?\]?)\s+)?(?P<speaker>[^:\n]{1,80}):\s*(?P<text>.*)$")
_INLINE_SPEAKER = re.compile(
    r"(?P<speaker>Agent|Caller|Customer|Client|User|Assistant|Support|Representative|Rep|Patient|Doctor|Nurse|"
    r"Speaker\s*\d+|[A-Z][A-Za-z][A-Za-z _-]{0,38})\s*:",
)
_LEADING_DATESTAMP = re.compile(
    r"^\s*(?:[A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\s*",
)
_WORD = re.compile(r"\b[A-Za-z][A-Za-z\-]{2,}\b")
_STOPWORDS = {
    "about", "after", "again", "also", "and", "because", "before", "between", "could",
    "from", "have", "into", "like", "more", "must", "that", "the", "their",
    "them", "then", "there", "these", "this", "through", "using", "what",
    "when", "where", "which", "with", "without", "would", "your", "you", "are",
    "address", "birth", "cell", "date", "dob", "email",
    "active", "actually", "alright", "bear", "but", "bye", "calling", "can",
    "cause", "come", "correct", "day", "exactly", "external", "fine", "for",
    "give", "go", "going", "good", "got", "great", "guess", "hello", "help",
    "her", "hers", "here", "him", "his", "hold", "huh", "internal", "just",
    "know", "let", "look", "looking", "matter", "maybe", "mean", "moment",
    "morning", "need", "needs", "not", "okay", "out", "please", "problem",
    "provide", "pull", "really", "request", "right", "said", "say", "second",
    "see", "she", "sure", "take", "thank", "thanks", "they", "thing",
    "things", "today", "try", "trying", "used", "was", "welcome", "well",
    "yeah", "yes", "yep",
    "identifier", "name", "number", "phone", "system", "ssn",
}


def _run_curator_pii_redaction(
    text: str,
    *,
    enabled: bool,
    base_url: str = "",
    api_key: str | None = None,
    model: str = "meta/llama-3.1-70b-instruct",
    language: str = "en",
) -> dict[str, Any]:
    nemo = importlib.import_module("synth_platform.infrastructure.integrations.nvidia_nemo")
    return nemo.run_curator_pii_redaction(
        text,
        enabled=enabled,
        base_url=base_url,
        api_key=api_key,
        model=model,
        language=language,
    )


def _run_guardrails_transcript_check(
    synthetic_turns: list[dict[str, Any]],
    *,
    enabled: bool,
    config_path: str = "",
    model: str = "",
) -> dict[str, Any]:
    nemo = importlib.import_module("synth_platform.infrastructure.integrations.nvidia_nemo")
    return nemo.run_guardrails_transcript_check(
        synthetic_turns,
        enabled=enabled,
        config_path=config_path,
        model=model,
    )


@dataclass(frozen=True)
class TranscriptTurn:
    index: int
    speaker: str
    text: str
    timestamp: str | None = None


@dataclass
class FakerLocaleContext:
    locale: str
    faker_locale: str
    profile_seed: int = 0
    profile_first_name: str = ""
    profile_last_name: str = ""
    profile_address: str = ""
    profile_state: str = ""
    profile_country: str = ""
    profile_email: str = ""
    replacement_terms: set[str] | None = None


def parse_transcript_text(text: str) -> list[TranscriptTurn]:
    """Parse common speaker-prefixed, paragraph, or JSON transcript text."""
    json_turns = _parse_json_transcript(text)
    if json_turns is not None:
        return json_turns

    turns: list[TranscriptTurn] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        segments = _split_inline_speaker_segments(line)
        if segments:
            for segment in segments:
                match = _SPEAKER_LINE.match(segment)
                if match and match.group("text").strip():
                    turns.append(
                        TranscriptTurn(
                            index=len(turns),
                            speaker=match.group("speaker").strip(),
                            text=match.group("text").strip(),
                            timestamp=(match.group("timestamp") or "").strip() or None,
                        )
                    )
            continue
        match = _SPEAKER_LINE.match(line)
        if match:
            turns.append(
                TranscriptTurn(
                    index=len(turns),
                    speaker=match.group("speaker").strip(),
                    text=match.group("text").strip(),
                    timestamp=(match.group("timestamp") or "").strip() or None,
                )
            )
        else:
            turns.append(TranscriptTurn(index=len(turns), speaker="speaker_unknown", text=line))
    return turns


def _split_inline_speaker_segments(line: str) -> list[str]:
    cleaned = _LEADING_DATESTAMP.sub("", line).strip()
    matches = list(_INLINE_SPEAKER.finditer(cleaned))
    if not matches:
        return []
    segments: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        segment = cleaned[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def _parse_json_transcript(text: str) -> list[TranscriptTurn] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        rows = payload.get("turns") or payload.get("messages") or payload.get("transcript")
    else:
        rows = payload
    if not isinstance(rows, list):
        return None
    turns: list[TranscriptTurn] = []
    for row in rows:
        if isinstance(row, str):
            turns.append(TranscriptTurn(index=len(turns), speaker="speaker_unknown", text=row))
            continue
        if not isinstance(row, dict):
            continue
        speaker = row.get("speaker") or row.get("role") or row.get("author") or "speaker_unknown"
        body = row.get("text") or row.get("content") or row.get("message") or ""
        timestamp = row.get("timestamp") or row.get("time") or row.get("created_at")
        if str(body).strip():
            turns.append(
                TranscriptTurn(
                    index=len(turns),
                    speaker=str(speaker).strip() or "speaker_unknown",
                    text=str(body).strip(),
                    timestamp=str(timestamp).strip() if timestamp else None,
                )
            )
    return turns


def _redact_text(text: str, *, locale_context: FakerLocaleContext | None = None) -> str:
    locale_context = locale_context or _locale_context_from_text(text)
    redacted = _BRACKET_TIMESTAMP.sub("", text)
    redacted = _ADDRESS_LABEL.sub(
        lambda match: match.group("prefix") + _fake_for("address", match.group("value"), locale_context),
        redacted,
    )
    redacted = _STATE_LABEL.sub(
        lambda match: match.group("prefix") + _fake_for("state", match.group("value"), locale_context),
        redacted,
    )
    redacted = _COUNTRY_LABEL.sub(
        lambda match: match.group("prefix") + _fake_for("country", match.group("value"), locale_context),
        redacted,
    )
    redacted = _FIRST_NAME_LABEL.sub(
        lambda match: match.group("prefix") + _fake_for("first_name", match.group("value"), locale_context),
        redacted,
    )
    redacted = _LAST_NAME_LABEL.sub(
        lambda match: match.group("prefix") + _fake_for("last_name", match.group("value"), locale_context),
        redacted,
    )
    redacted = _DOB.sub(lambda match: _replace_dob_phrase(match.group(0), locale_context), redacted)
    redacted = _URL.sub(lambda match: _fake_for("url", match.group(0), locale_context), redacted)
    redacted = _EMAIL.sub(lambda match: _fake_for("email", match.group(0), locale_context), redacted)
    redacted = _SSN.sub(lambda match: _fake_for("ssn", match.group(0), locale_context), redacted)
    redacted = _CARD.sub(lambda match: _fake_for("card", match.group(0), locale_context), redacted)
    redacted = _PHONE.sub(
        lambda match: match.group(0) if _SSN.fullmatch(match.group(0)) else _fake_for("phone", match.group(0), locale_context),
        redacted,
    )
    redacted = _PROPER_NAME_AFTER_INTRO.sub(
        lambda match: _replace_intro_name(match, locale_context),
        redacted,
    )
    redacted = _ORG_PLACE.sub(lambda match: _fake_for("org_or_place", match.group(0), locale_context), redacted)
    redacted = _ACCOUNT_ID.sub(lambda match: _replace_identifier_phrase(match.group(0), locale_context), redacted)
    redacted = _HASHED_ID.sub(lambda match: _fake_for("generic_id", match.group(0), locale_context), redacted)
    return redacted


def _replace_intro_name(match: re.Match[str], locale_context: FakerLocaleContext) -> str:
    full_match = match.group(0)
    raw_name = match.group(1)
    lowered = full_match.lower()
    if lowered.startswith("my name"):
        replacement = _fake_for("name", raw_name, locale_context)
    elif lowered.startswith(("mr.", "mrs.", "ms.", "dr.")):
        replacement = _fake_for("last_name", raw_name, locale_context)
    else:
        replacement = _fake_for("independent_name", raw_name, locale_context)
    return full_match.replace(raw_name, replacement)


def _replace_dob_phrase(value: str, locale_context: FakerLocaleContext) -> str:
    prefix = re.match(r"(?P<prefix>.*?(?:is|:)\s*)", value, flags=re.IGNORECASE)
    fake_date = _fake_for("date_of_birth", value, locale_context)
    if prefix:
        return prefix.group("prefix") + fake_date
    return fake_date


def _replace_identifier_phrase(value: str, locale_context: FakerLocaleContext) -> str:
    lowered = value.lower()
    if "claim" in lowered:
        return "claim id " + _fake_for("claim_id", value, locale_context)
    if "policy" in lowered:
        return "policy number " + _fake_for("policy_id", value, locale_context)
    if "account" in lowered:
        return "account id " + _fake_for("account_id", value, locale_context)
    if "license" in lowered or "registration" in lowered:
        return "vehicle identifier " + _fake_for("vehicle_id", value, locale_context)
    return "identifier " + _fake_for("generic_id", value, locale_context)


def _fake_for(kind: str, raw_value: str, locale_context: FakerLocaleContext) -> str:
    if kind == "first_name":
        return locale_context.profile_first_name
    if kind == "last_name":
        return locale_context.profile_last_name
    if kind == "name":
        return f"{locale_context.profile_first_name} {locale_context.profile_last_name}".strip()
    if kind == "email":
        return locale_context.profile_email
    if kind == "address":
        return locale_context.profile_address
    if kind == "state":
        return locale_context.profile_state
    if kind == "country":
        return locale_context.profile_country
    faker = _seeded_faker(kind, raw_value, locale_context)
    if kind == "url":
        return faker.url().rstrip("/")
    if kind == "ssn":
        return faker.ssn()
    if kind == "card":
        return faker.credit_card_number(card_type=None)
    if kind == "phone":
        return faker.phone_number()
    if kind == "independent_name":
        return faker.name()
    if kind == "date_of_birth":
        date_value = faker.date_of_birth(minimum_age=18, maximum_age=90)
        return f"{date_value:%B} {date_value.day}, {date_value.year}"
    if kind == "org_or_place":
        return faker.company()
    if kind == "claim_id":
        return "CLM-" + faker.bothify(text="########")
    if kind == "policy_id":
        return "POL-" + faker.bothify(text="???-####").upper()
    if kind == "account_id":
        return "ACCT-" + faker.bothify(text="########")
    if kind == "vehicle_id":
        return _safe_fake(faker, "license_plate", fallback=faker.bothify(text="???-####").upper())
    return faker.bothify(text="ID-########").upper()


def _seeded_faker(kind: str, raw_value: str, locale_context: FakerLocaleContext) -> Faker:
    seed = _faker_seed(kind, raw_value, locale_context)
    return _seeded_faker_from_seed(locale_context.faker_locale, seed)


def _seeded_faker_from_seed(faker_locale: str, seed: int) -> Faker:
    try:
        faker = Faker(faker_locale)
    except Exception:
        faker = Faker("en_US")
    faker.seed_instance(seed)
    return faker


def _faker_seed(kind: str, raw_value: str, locale_context: FakerLocaleContext) -> int:
    return _faker_seed_from_text(kind, raw_value, locale_context.faker_locale)


def _faker_seed_from_text(kind: str, value: str, faker_locale: str) -> int:
    digest = hashlib.sha256(f"{faker_locale}:{kind}:{value.strip().lower()}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _locale_context_from_text(text: str) -> FakerLocaleContext:
    locale = _detect_locale_from_transcript_text(text)
    pack = get_locale_pack(locale)
    context = FakerLocaleContext(
        locale=locale,
        faker_locale=pack.faker_locale,
        profile_seed=_faker_seed_from_text("profile", text, pack.faker_locale),
    )
    _ensure_profile(context)
    return context


def _ensure_profile(context: FakerLocaleContext) -> None:
    if context.profile_first_name and context.profile_last_name:
        return
    faker = _seeded_faker_from_seed(context.faker_locale, context.profile_seed)
    context.profile_first_name = faker.first_name()
    context.profile_last_name = faker.last_name()
    context.profile_address = _single_line(faker.address())
    context.profile_state = _fake_state(faker, context)
    context.profile_country = _fake_country(context)
    context.profile_email = _profile_email(context, faker)
    context.replacement_terms = _terms_from_text(
        " ".join(
            [
                context.profile_first_name,
                context.profile_last_name,
                context.profile_address,
                context.profile_state,
                context.profile_country,
                context.profile_email,
            ]
        )
    )


def _profile_email(context: FakerLocaleContext, faker: Faker) -> str:
    first = _email_token(context.profile_first_name)
    last = _email_token(context.profile_last_name)
    domain = _safe_fake(faker, "free_email_domain", fallback="example.com")
    if not first or not last:
        return faker.ascii_email()
    return f"{first}.{last}@{domain}".lower()


def _terms_from_text(value: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(value) if len(word) > 2}


def _email_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", value.lower())
    return token or "customer"


def _detect_locale_from_transcript_text(text: str) -> str:
    if not text:
        return DEFAULT_LOCALE
    normalized = text.lower()
    scores: dict[str, int] = {}
    for alias, locale in LOCALE_ALIASES.items():
        if _contains_locale_alias(normalized, alias):
            scores[locale] = scores.get(locale, 0) + 1
    if not scores:
        return DEFAULT_LOCALE
    return max(scores, key=lambda locale: scores[locale])


def _contains_locale_alias(text: str, alias: str) -> bool:
    escaped = re.escape(alias.lower())
    if len(alias) <= 3 and alias.isascii():
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
    return escaped in text


def _single_line(value: str) -> str:
    return " ".join(str(value).replace("\r", "\n").split())


def _fake_state(faker: Faker, locale_context: FakerLocaleContext) -> str:
    if locale_context.locale == "en_US":
        return _safe_fake(faker, "state_abbr", fallback=faker.state())
    return _safe_fake(faker, "state", fallback=faker.city())


def _fake_country(locale_context: FakerLocaleContext) -> str:
    return get_locale_pack(locale_context.locale).country_name


def _safe_fake(faker: Faker, provider: str, *, fallback: str) -> str:
    method = getattr(faker, provider, None)
    if method is None:
        return fallback
    try:
        return str(method())
    except Exception:
        return fallback


def sanitize_transcript_turns(
    turns: list[TranscriptTurn],
    *,
    locale_context: FakerLocaleContext | None = None,
) -> list[TranscriptTurn]:
    """Remove direct identifiers from preview/contract text."""
    sanitized: list[TranscriptTurn] = []
    speaker_map: dict[str, str] = {}
    locale_context = locale_context or _locale_context_from_text(" ".join(turn.text for turn in turns))
    for turn in turns:
        speaker_key = turn.speaker.strip() or "speaker_unknown"
        speaker_map.setdefault(speaker_key, _safe_speaker_label(speaker_key, len(speaker_map) + 1))
        text = _redact_text(turn.text, locale_context=locale_context)
        sanitized.append(
            TranscriptTurn(
                index=turn.index,
                speaker=speaker_map[speaker_key],
                text=text,
                timestamp=turn.timestamp,
            )
        )
    return sanitized


def _safe_speaker_label(value: str, index: int) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip(" '\"\t,"))
    lower = cleaned.lower()
    if lower in {"internal", "agent_internal", "support_internal", "representative", "rep"}:
        return "Agent"
    if lower in {"external", "caller_external", "customer_external"}:
        return "Customer"
    if lower in {"agent", "customer", "caller", "user", "client", "member", "patient", "supervisor", "specialist"}:
        return cleaned[:1].upper() + cleaned[1:]
    if re.fullmatch(r"person\s+\d{1,3}", lower):
        return "Person " + re.search(r"\d{1,3}", lower).group(0)
    if re.fullmatch(r"speaker[_ -]?\d{1,3}", lower):
        return f"speaker_{index}"
    return f"speaker_{index}"


def _top_terms(
    turns: list[TranscriptTurn],
    *,
    limit: int = 8,
    privacy_terms: set[str] | None = None,
) -> list[str]:
    privacy_terms = privacy_terms or set()
    counts: dict[str, int] = {}
    priority_terms = [
        "charge",
        "charged",
        "payment",
        "refund",
        "card",
        "order",
        "login",
        "password",
        "disabled",
        "account",
        "claim",
        "activate",
        "activation",
        "benefit",
        "plan",
    ]
    for turn in turns:
        for word in _WORD.findall(turn.text.lower()):
            word = word.strip("-")
            if (
                not word
                or word in _STOPWORDS
                or word in privacy_terms
                or word.startswith("speaker")
                or word in {"clm", "pol", "acct", "id"}
            ):
                continue
            counts[word] = counts.get(word, 0) + 1
    ordered = [term for term in priority_terms if counts.get(term)]
    ordered.extend(word for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if word not in ordered)
    return ordered[:limit]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def _synthetic_context_from_turns(
    turns: list[TranscriptTurn],
    *,
    topic_terms: list[str],
) -> dict[str, Any]:
    text = " ".join(turn.text for turn in turns)
    context: dict[str, Any] = {
        "topic_terms": topic_terms,
        "domain": _context_domain(topic_terms, text),
    }
    name = _first_match(
        [
            r"\bMy name is ([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,2})\b",
            r"\bMr\. ([A-Z][A-Za-z'-]+)\b",
            r"\bMs\. ([A-Z][A-Za-z'-]+)\b",
            r"\bMrs\. ([A-Z][A-Za-z'-]+)\b",
        ],
        text,
    )
    if name:
        context["customer_name"] = name
    email = _first_match([_EMAIL], text)
    if email:
        context["email"] = email
    phone = _first_match([r"\bPhone number is ([^,.;]+(?:x\d+)?)", _PHONE], text)
    if phone:
        context["phone"] = phone
    dob = _first_match([r"\bdate of birth is ([A-Z][a-z]+ \d{1,2}, \d{4})\b"], text)
    if dob:
        context["date_of_birth"] = dob
    policy_id = _first_match([_SYNTHETIC_POLICY_ID], text)
    if policy_id:
        context["policy_id"] = policy_id
    claim_id = _first_match([_SYNTHETIC_CLAIM_ID], text)
    if claim_id:
        context["claim_id"] = claim_id
    generic_id = _first_match([_SYNTHETIC_GENERIC_ID], text)
    if generic_id:
        context["tracking_id"] = generic_id
    vehicle_id = _first_match([r"\bvehicle identifier ([A-Z0-9 -]{3,20})\b"], text)
    if vehicle_id:
        context["vehicle_id"] = vehicle_id.strip()
    organization = _first_match(
        [
            r"\b(?:left|at|from) ([A-Z][A-Za-z,&' -]+(?:Inc|LLC|Group|Hospital|Clinic|Center|Rodgers|Partners))\b",
        ],
        text,
    )
    if organization:
        context["care_location"] = organization
    if "injury" in text.lower() or "concussion" in text.lower() or "medical" in text.lower():
        context["has_medical_context"] = True
    return context


def _context_domain(topic_terms: list[str], text: str) -> str:
    lower_text = text.lower()
    terms = set(topic_terms)
    if "claim" in terms and ("auto" in terms or "vehicle" in lower_text or "rear-ended" in lower_text):
        return "auto_claim"
    if "claim" in terms:
        return "claim"
    return "general"


def _first_match(patterns: list[str | re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text) if isinstance(pattern, str) else pattern.search(text)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return None


def _ngram_hashes(turns: list[TranscriptTurn], *, n: int = 5) -> list[str]:
    hashes: set[str] = set()
    for turn in turns:
        words = _WORD.findall(turn.text.lower())
        for idx in range(0, max(0, len(words) - n + 1)):
            hashes.add(_hash_text(" ".join(words[idx : idx + n])))
    return sorted(hashes)


def summarize_transcript_preview(text: str, *, source_name: str = "") -> SSOTPreview:
    turns = sanitize_transcript_turns(parse_transcript_text(text))
    speakers = sorted({turn.speaker for turn in turns})
    sample = [
        {
            "turn": turn.index + 1,
            "speaker": turn.speaker,
            "timestamp": turn.timestamp or "",
            "text": turn.text,
        }
        for turn in turns[:10]
    ]
    return SSOTPreview(
        source_type="transcript",
        source_name=source_name,
        record_count=len(turns),
        field_count=4,
        entity_count=len(speakers),
        relationship_count=0,
        items=[
            SSOTPreviewItem(
                name="turns",
                kind="transcript_turns",
                record_count=len(turns),
                field_count=4,
                sample=sample,
                metadata={"speaker_count": len(speakers), "speakers": speakers},
            )
        ],
        privacy_warnings=[],
    )


def build_transcript_contract(
    text: str,
    *,
    source_name: str = "",
    nvidia_options: dict[str, Any] | None = None,
) -> CanonicalContract:
    nvidia_options = nvidia_options or {}
    curator_result = _run_curator_pii_redaction(
        text,
        enabled=bool(nvidia_options.get("curator_enabled", nvidia_options.get("enabled", False))),
        base_url=str(nvidia_options.get("curator_base_url") or ""),
        api_key=str(nvidia_options.get("curator_api_key") or "") or None,
        model=str(nvidia_options.get("curator_model") or "meta/llama-3.1-70b-instruct"),
    )
    curated_text = str(curator_result.get("text") or text)
    locale_context = _locale_context_from_text(curated_text)
    sanitized_turns = sanitize_transcript_turns(
        parse_transcript_text(curated_text),
        locale_context=locale_context,
    )
    preview = summarize_transcript_preview(curated_text, source_name=source_name)
    speakers = sorted({turn.speaker for turn in sanitized_turns})
    top_terms = _top_terms(
        sanitized_turns,
        privacy_terms=locale_context.replacement_terms or set(),
    )
    synthetic_context = _synthetic_context_from_turns(sanitized_turns, topic_terms=top_terms)
    synthetic_context["source_locale"] = locale_context.locale
    synthetic_context["target_locale"] = str(nvidia_options.get("target_locale") or "").strip() or locale_context.locale
    synthetic_context["target_persona"] = str(nvidia_options.get("target_persona") or "").strip() or "privacy-safe support participant"
    build_structured_ssot = bool(nvidia_options.get("build_structured_ssot", nvidia_options.get("enabled", False)))
    if build_structured_ssot:
        structured_ssot = build_transcript_ssot_with_data_designer(
            sanitized_turns,
            source_name=source_name,
            enabled=bool(nvidia_options.get("enabled", False)),
        )
    else:
        structured_ssot = {"status": "pending", "reason": "deferred_until_generation"}
    provenance = CanonicalProvenance(
        produced_by="build_transcript_contract",
        source=SourceDescriptor(source_type="transcript", source_id=source_name),
    )
    return CanonicalContract(
        contract_id=source_name,
        source_type="transcript",
        entities=[
            CanonicalEntity(
                entity_id="entity:transcript_turn",
                name="transcript_turn",
                entity_type="transcript_turn",
                fields=[
                    "field:transcript_turn.turn_index",
                    "field:transcript_turn.speaker",
                    "field:transcript_turn.timestamp",
                    "field:transcript_turn.text",
                ],
                metadata={
                    "turn_count": preview.record_count,
                    "speaker_count": preview.entity_count,
                    "speakers": speakers,
                    "speaker_sequence": [turn.speaker for turn in sanitized_turns],
                    "turn_plan": _turn_plan_from_turns(sanitized_turns),
                    "speaker_roles": _speaker_roles_from_turns(sanitized_turns),
                    "topic_terms": top_terms,
                    "synthetic_context": synthetic_context,
                    "structured_ssot": structured_ssot,
                    "source_turn_hashes": [_hash_text(turn.text) for turn in sanitized_turns],
                    "source_ngram_hashes": _ngram_hashes(sanitized_turns),
                },
            )
        ],
        fields=[
            CanonicalField(field_id="field:transcript_turn.turn_index", name="turn_index", physical_type="int", semantic_type="identifier", nullable=False),
            CanonicalField(field_id="field:transcript_turn.speaker", name="speaker", physical_type="text", semantic_type="category", nullable=False),
            CanonicalField(field_id="field:transcript_turn.timestamp", name="timestamp", physical_type="text", semantic_type="datetime", nullable=True),
            CanonicalField(field_id="field:transcript_turn.text", name="text", physical_type="text", semantic_type="free_text", nullable=False),
        ],
        provenance=provenance,
        privacy_policy={
            "raw_text_persisted": False,
            "preview_sanitized": True,
            "direct_identifier_redaction": ["email", "phone", "ssn", "card", "url"],
            "nvidia_nemo_curator": {
                key: value
                for key, value in curator_result.items()
                if key != "text"
            },
            "nvidia_nemo_guardrails": "pending_validation",
            "nvidia_nemo_evaluator": "optional_not_run",
        },
    )


def generate_synthetic_transcript(
    contract: CanonicalContract,
    *,
    turn_count: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate source-free synthetic interaction turns from contract metadata only."""
    entity = contract.entities[0] if contract.entities else None
    metadata = entity.metadata if entity else {}
    total = int(turn_count or metadata.get("turn_count") or 6)
    rng = random.Random(seed)
    topic_terms = [str(term) for term in metadata.get("topic_terms") or []]
    topic = _conversation_topic(topic_terms)
    synthetic_context = dict(metadata.get("synthetic_context") or {})
    turn_plan = [row for row in metadata.get("turn_plan") or [] if isinstance(row, dict)]
    speaker_roles = {str(key): str(value) for key, value in dict(metadata.get("speaker_roles") or {}).items()}
    speakers = _speaker_sequence_for_generation(metadata, total)
    customer_templates, agent_templates = _conversation_templates(topic, synthetic_context)
    rows: list[dict[str, Any]] = []
    speaker_occurrences: dict[str, int] = {}
    used_texts: set[str] = set()
    for idx in range(max(1, total)):
        speaker = speakers[idx]
        role = speaker_roles.get(speaker) or ("customer" if _is_customer_speaker(speaker) else "agent")
        pool = customer_templates if role == "customer" else agent_templates
        sequence_index = speaker_occurrences.get(speaker, 0)
        speaker_occurrences[speaker] = sequence_index + 1
        text = _select_turn_text(pool, sequence_index, rng)
        if idx < len(turn_plan):
            text = _render_turn_from_plan(
                base_text=text,
                intent=str(turn_plan[idx].get("intent") or ""),
                topic=topic,
                role=role,
                context=synthetic_context,
                occurrence=sequence_index,
                turn_index=idx,
                total_turns=total,
            )
        if text in used_texts and idx < len(turn_plan):
            variants = _intent_variants(
                intent=str(turn_plan[idx].get("intent") or ""),
                role=role,
                topic=topic,
                turn_index=idx,
                total_turns=total,
            )
            alternatives = [variant for variant in variants if variant not in used_texts]
            if alternatives:
                text = alternatives[0]
        used_texts.add(text)
        rows.append({"turn": idx + 1, "speaker": speaker, "timestamp": "", "text": text})
    return rows


def _is_customer_speaker(speaker: str) -> bool:
    return speaker.strip().lower() in {"customer", "caller", "user", "client", "member", "patient"}


def _turn_plan_from_turns(turns: list[TranscriptTurn]) -> list[dict[str, str | int]]:
    return [
        {
            "turn": idx + 1,
            "speaker": turn.speaker,
            "intent": _turn_intent(turn.text),
            "role": _turn_role(turn.text, turn.speaker),
        }
        for idx, turn in enumerate(turns)
    ]


def _speaker_roles_from_turns(turns: list[TranscriptTurn]) -> dict[str, str]:
    scores: dict[str, dict[str, int]] = {}
    for turn in turns:
        role = _turn_role(turn.text, turn.speaker)
        bucket = scores.setdefault(turn.speaker, {"customer": 0, "agent": 0})
        bucket[role] += 1
    return {
        speaker: "customer" if values["customer"] > values["agent"] else "agent"
        for speaker, values in scores.items()
    }


def _turn_role(text: str, speaker: str) -> str:
    speaker_lower = speaker.strip().lower()
    if _is_customer_speaker(speaker):
        return "customer"
    if speaker_lower in {"agent", "supervisor", "specialist"}:
        return "agent"
    lower = text.lower()
    customer_markers = (
        "i'm calling",
        "i am calling",
        "i'm ",
        "i am ",
        "my name",
        "my account",
        "my request",
        "my plan",
        "i need",
        "i have",
        "i didn't",
        "i never",
        "i've never",
        "unable to process",
        "not giving me",
        "it's ",
        "with us",
    )
    agent_markers = (
        "how may i help",
        "how can i help",
        "what is your",
        "do you have",
        "can i get",
        "let me",
        "one moment",
        "pull up",
        "provide me",
        "i can help",
        "i'll",
        "we can",
    )
    customer_score = sum(marker in lower for marker in customer_markers)
    agent_score = sum(marker in lower for marker in agent_markers)
    if customer_score == agent_score == 0:
        return "agent" if "?" in text else "customer"
    return "customer" if customer_score > agent_score else "agent"


def _turn_intent(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("thank you", "thanks", "good morning", "hello", "hi ")):
        return "greeting or acknowledgment"
    if any(term in lower for term in ("member id", "policy", "claim", "account", "reference", "number")):
        return "identity or case lookup"
    if any(term in lower for term in ("unable", "can't", "cannot", "issue", "problem", "not giving", "not working")):
        return "problem description"
    if any(term in lower for term in ("plan", "benefit", "coverage", "insurance", "provider")):
        return "plan or benefit detail"
    if any(term in lower for term in ("one moment", "pull up", "check", "look up", "review")):
        return "agent investigation"
    if any(term in lower for term in ("resolve", "fixed", "confirmed", "complete", "next step")):
        return "resolution or next step"
    return "conversation progress"


def _speaker_sequence_for_generation(metadata: dict[str, Any], total: int) -> list[str]:
    source_sequence = [str(value) for value in metadata.get("speaker_sequence") or [] if str(value).strip()]
    if source_sequence:
        labels = set(source_sequence)
        if labels and labels <= {"Customer", "Agent"}:
            first = source_sequence[0] if source_sequence[0] in {"Customer", "Agent"} else "Customer"
            second = "Agent" if first == "Customer" else "Customer"
            return [first if idx % 2 == 0 else second for idx in range(max(1, total))]
        return [source_sequence[idx % len(source_sequence)] for idx in range(max(1, total))]
    source_speakers = list(metadata.get("speakers") or ["speaker_1", "speaker_2"])
    speakers = _interaction_speaker_names(len(source_speakers))
    return [speakers[idx % len(speakers)] for idx in range(max(1, total))]


def _render_turn_from_plan(
    *,
    base_text: str,
    intent: str,
    topic: str,
    role: str,
    context: dict[str, Any],
    occurrence: int,
    turn_index: int,
    total_turns: int,
) -> str:
    if _text_has_synthetic_context(base_text, context):
        return base_text
    variants = _intent_variants(intent=intent, role=role, topic=topic, turn_index=turn_index, total_turns=total_turns)
    if not variants:
        return base_text
    return variants[(occurrence + turn_index) % len(variants)]


def _text_has_synthetic_context(text: str, context: dict[str, Any]) -> bool:
    if "[MOCK_" in text:
        return True
    return any(
        value and str(value) in text
        for value in (
            context.get("customer_name"),
            context.get("policy_id"),
            context.get("claim_id"),
            context.get("tracking_id"),
            context.get("email"),
            context.get("phone"),
            context.get("date_of_birth"),
            context.get("vehicle_id"),
        )
    )


def _intent_variants(
    *,
    intent: str,
    role: str,
    topic: str,
    turn_index: int,
    total_turns: int,
) -> list[str]:
    late_turn = turn_index >= max(0, total_turns - 2)
    if late_turn and intent in {"conversation progress", "greeting or acknowledgment"}:
        intent = "resolution or next step"
    if role == "customer":
        return {
            "greeting or acknowledgment": [
                f"Hi, I am calling about {topic} and need help with the next step.",
                f"Thanks for taking the call. I want to sort out {topic}.",
                f"Hello, I am trying to understand what to do next for {topic}.",
            ],
            "problem description": [
                f"I am still seeing an issue with {topic}, and I need help figuring out the next step.",
                f"The issue around {topic} is still not clear from my side.",
                f"I tried to move forward with {topic}, but I still need support.",
            ],
            "plan or benefit detail": [
                f"I want to understand the plan and benefit details for {topic}.",
                f"Can you explain what applies to {topic} before I continue?",
                f"I need to confirm which benefit details matter for {topic}.",
            ],
            "resolution or next step": [
                "That answers my question. I understand the next step.",
                "Thanks, that gives me what I need to continue.",
                "Great, I am clear on what happens next.",
            ],
        }.get(intent, [])
    return {
        "greeting or acknowledgment": [
            f"I can help with {topic}.",
            f"Thanks for reaching out. I will review {topic} with you.",
            f"Let me help you work through {topic}.",
        ],
        "problem description": [
            f"I understand there is still an issue with {topic}. I will check what is blocking it.",
            f"Let me review why {topic} is not moving forward as expected.",
            f"I will look into the issue and keep the next step focused on {topic}.",
        ],
        "plan or benefit detail": [
            f"Let me review the plan and benefit details for {topic}.",
            f"I will check the applicable benefit information for {topic}.",
            f"We can walk through the relevant plan details for {topic}.",
        ],
        "agent investigation": [
            "I am checking the case details now and will confirm what I find.",
            "Give me a moment while I review the available details.",
            "I am looking through the record so I can give you the right next step.",
        ],
        "identity or case lookup": [
            "I can use the verified details already on file to review the case.",
            "I have enough verified information to continue reviewing the request.",
            "I will use the verified case details to move this forward.",
        ],
        "resolution or next step": [
            "The next step is confirmed, and the case is ready for review.",
            "I have recorded the update and confirmed the follow-up path.",
            "Everything needed for now is captured, and the next action is clear.",
        ],
    }.get(intent, [])


def _interaction_speaker_names(count: int) -> list[str]:
    names = ["Customer", "Agent", "Supervisor", "Specialist"]
    return names[: max(2, min(max(count, 1), len(names)))]


def _conversation_topic(topic_terms: list[str]) -> str:
    useful_terms = [
        term.replace("_", " ").strip().lower()
        for term in topic_terms
        if term and not term.startswith("[") and len(term) > 2
    ]
    useful_set = set(useful_terms)
    if {"activate", "activation"} & useful_set and "account" in useful_set:
        return "the account activation"
    if {"login", "logging", "password", "disabled"} & useful_set:
        return "the login issue"
    if {"charge", "charged", "payment", "card"} & useful_set:
        return "the payment question"
    if "benefit" in useful_set and ("order" in useful_set or "otc" in useful_set):
        return "the benefit order"
    if "claim" in useful_set and "auto" in useful_set:
        return "the auto claim"
    if "claim" in useful_set and {"injury", "medical", "concussion"} & useful_set:
        return "the injury claim"
    if "claim" in useful_set:
        return "the claim"
    if not useful_terms:
        return "the request"
    return "the " + " ".join(useful_terms[:2]) + " request"


def _conversation_templates(topic: str, context: dict[str, Any]) -> tuple[list[str], list[str]]:
    customer_templates = _context_customer_templates(topic, context)
    agent_templates = _context_agent_templates(topic, context)
    return customer_templates, agent_templates


def _context_customer_templates(topic: str, context: dict[str, Any]) -> list[str]:
    customer_name = context.get("customer_name")
    policy_id = context.get("policy_id")
    claim_id = context.get("claim_id") or context.get("tracking_id")
    vehicle_id = context.get("vehicle_id")
    templates = [
        _join_sentence_parts(
            "Hi, I need help understanding " + topic,
            "my profile name is [MOCK_CUSTOMER_NAME]" if customer_name else "",
            "my member or policy reference is [MOCK_POLICY_ID]" if policy_id else "",
        ),
        _join_sentence_parts(
            "Thanks",
            "please use the verified contact details already on file"
            if context.get("email") or context.get("phone")
            else "",
        ),
        _join_sentence_parts(
            "That helps",
            "I also want to make sure case [MOCK_CASE_ID] stays updated" if claim_id else "",
        ),
        _join_sentence_parts(
            "I appreciate the update",
            "the related item reference is [MOCK_REFERENCE_ID]" if vehicle_id else "",
        ),
        "Great, that answers my question.",
    ]
    return [template for template in templates if template]


def _context_agent_templates(topic: str, context: dict[str, Any]) -> list[str]:
    customer_name = context.get("customer_name")
    policy_id = context.get("policy_id")
    claim_id = context.get("claim_id") or context.get("tracking_id")
    dob = context.get("date_of_birth")
    care_location = context.get("care_location")
    medical = bool(context.get("has_medical_context"))
    templates = [
        _join_sentence_parts(
            "I can help with " + topic,
            "I have the mock customer profile open" if customer_name else "",
            "under reference [MOCK_POLICY_ID]" if policy_id else "",
        ),
        _join_sentence_parts(
            "I confirmed the contact and identity details",
            "including [MOCK_DATE_OF_BIRTH]" if dob else "",
            "I will route the update to the right team",
        ),
        _join_sentence_parts(
            "Everything needed for now has been captured in the synthetic case notes",
            "with case reference [MOCK_CASE_ID]" if claim_id else "",
        ),
        _join_sentence_parts(
            "You do not need to provide anything else at this point",
            f"I noted the medical update from {care_location}" if medical and care_location else "",
        ),
        "You are welcome. I will share the update with the right team.",
    ]
    return [template for template in templates if template]


def _join_sentence_parts(*parts: str) -> str:
    cleaned = [part.strip(" .") for part in parts if part and part.strip(" .")]
    if not cleaned:
        return ""
    first, *rest = cleaned
    if not rest:
        return first + "."
    return first + ". " + " ".join(part[:1].upper() + part[1:] + "." for part in rest)


def _select_turn_text(pool: list[str], sequence_index: int, rng: random.Random) -> str:
    if sequence_index < len(pool):
        return pool[sequence_index]
    alternatives = list(pool[1:-1] or pool)
    rng.shuffle(alternatives)
    return alternatives[sequence_index % len(alternatives)]


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def validate_transcript_non_replay(
    contract: CanonicalContract,
    synthetic_turns: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.85,
    nvidia_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate synthetic interaction text against hashed source evidence."""
    entity = contract.entities[0] if contract.entities else None
    metadata = entity.metadata if entity else {}
    source_hashes = set(metadata.get("source_turn_hashes") or [])
    source_ngram_hashes = set(metadata.get("source_ngram_hashes") or [])
    exact_replays = []
    ngram_replays = []
    max_self_similarity = 0.0

    synthetic_texts = [str(row.get("text") or "").strip() for row in synthetic_turns]
    for text in synthetic_texts:
        if _hash_text(text) in source_hashes:
            exact_replays.append(text)
        words = _WORD.findall(text.lower())
        for idx in range(0, max(0, len(words) - 5 + 1)):
            if _hash_text(" ".join(words[idx : idx + 5])) in source_ngram_hashes:
                ngram_replays.append(text)
                break

    # No raw source text is retained, so edit distance is only a synthetic
    # variety signal. Source replay validation uses exact/ngram hashes above.
    for i, left in enumerate(synthetic_texts):
        for right in synthetic_texts[i + 1 :]:
            longer = max(len(left), len(right), 1)
            similarity = 1.0 - (_levenshtein_distance(left, right) / longer)
            max_self_similarity = max(max_self_similarity, similarity)

    repeated_phrase_warning = max_self_similarity >= similarity_threshold
    privacy_findings = _synthetic_privacy_findings(synthetic_texts)
    base_passed = not exact_replays and not ngram_replays
    nvidia_options = nvidia_options or {}
    guardrails_result = _run_guardrails_transcript_check(
        synthetic_turns,
        enabled=bool(nvidia_options.get("enabled", False)),
        config_path=str(nvidia_options.get("guardrails_config_path") or ""),
        model=str(nvidia_options.get("guardrails_model") or ""),
    )
    passed = (
        base_passed
        and not repeated_phrase_warning
        and not privacy_findings
        and guardrails_result.get("passed") is not False
    )
    return {
        "passed": passed,
        "exact_replay_count": len(exact_replays),
        "ngram_replay_count": len(ngram_replays),
        "max_self_similarity": round(max_self_similarity, 6),
        "variety_warning_threshold": similarity_threshold,
        "repeated_phrase_warning": repeated_phrase_warning,
        "privacy_findings": privacy_findings,
        "raw_source_text_used": False,
        "nvidia_nemo_guardrails": guardrails_result,
    }


def _synthetic_privacy_findings(texts: list[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for turn_index, text in enumerate(texts, start=1):
        cleaned = re.sub(r"\[MOCK_[A-Z0-9_]+\]", "", text)
        labels = []
        if _EMAIL.search(cleaned):
            labels.append("email")
        if _PHONE.search(cleaned):
            labels.append("phone")
        if _SSN.search(cleaned):
            labels.append("ssn")
        if _CARD.search(cleaned):
            labels.append("card")
        if re.search(
            r"\b(?:acct|account|policy|claim|case|member)(?:\s+(?:id|number|reference))?\s*[:#-]\s*[A-Z0-9-]*\d[A-Z0-9-]{3,}\b",
            cleaned,
            flags=re.IGNORECASE,
        ):
            labels.append("identifier")
        if labels:
            findings.append({"turn": turn_index, "types": sorted(set(labels))})
    return findings
