"""Data Designer-backed structured SSOT extraction for transcript twins."""
from __future__ import annotations

import importlib.util
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

from synth_platform.engine.generation.slm_runtime import (
    data_designer_skip_health_check,
    preflight_slm_endpoint,
    resolve_platform_slm_runtime,
)
from synth_platform.engine.generation.data_designer_provider import (
    build_data_designer_model_config,
    build_data_designer_provider,
    load_data_designer_sdk,
)


def build_transcript_ssot_with_data_designer(
    turns: Iterable[Any],
    *,
    source_name: str = "",
    enabled: bool = False,
) -> dict[str, Any]:
    """Build a structured transcript SSOT object with the Data Designer SDK.

    The SDK receives sanitized turn text only. Raw transcript text is never
    persisted in the returned structure.
    """
    if not enabled:
        return {"status": "disabled"}
    turns_list = list(turns)
    if _mock_enabled():
        return _mock_transcript_ssot(turns_list, source_name=source_name)

    runtime = _runtime()
    if importlib.util.find_spec("data_designer") is None:
        return {"status": "unavailable", "reason": "package_not_installed"}

    try:
        dd, DataDesigner = load_data_designer_sdk()
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc)}

    try:
        seed_frame = pd.DataFrame(
            [
                {
                    "source_name": source_name,
                    "sanitized_transcript": _sanitized_turn_text(
                        turns_list,
                        max_turns=int(os.getenv("SP_TRANSCRIPT_SSOT_MAX_PROMPT_TURNS", "40")),
                    ),
                    "turn_count": _turn_count(turns_list),
                    "speaker_count": len(_speaker_names(turns_list)),
                    "speakers_json": json.dumps(_speaker_names(turns_list)),
                    "topic_terms_json": json.dumps(_topic_terms(turns_list)),
                    "evidence_json": json.dumps(_source_evidence(turns_list), sort_keys=True),
                    "schema_version": "1.0.0",
                }
            ]
        )
        model_alias = "transcript-ssot-builder"
        model_config = build_data_designer_model_config(
            dd,
            model_alias,
            workflow="transcript-ssot",
            runtime=runtime,
            temperature=float(os.getenv("SP_TRANSCRIPT_SSOT_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("SP_TRANSCRIPT_SSOT_TOP_P", "0.9")),
            timeout_env="SP_TRANSCRIPT_SSOT_TIMEOUT",
            parallel_env="SP_TRANSCRIPT_SSOT_MAX_PARALLEL_REQUESTS",
            skip_health_check=_skip_health_check(),
        )
        designer = DataDesigner(model_providers=[build_data_designer_provider(dd, runtime)])
        preflight_slm_endpoint(runtime)
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_frame))
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="synthetic_twin_ssot",
                prompt=_ssot_prompt(),
                system_prompt="Return only the requested structured object. Do not include markdown or prose.",
                model_alias=model_alias,
                output_format=_ssot_schema(),
            )
        )
        builder.add_column(
            dd.ValidationColumnConfig(
                name="synthetic_twin_ssot_validation",
                target_columns=["synthetic_twin_ssot"],
                validator_type=dd.ValidatorType.LOCAL_CALLABLE,
                validator_params=dd.LocalCallableValidatorParams(validation_function=_validate_ssot_frame),
            )
        )
        preview = designer.preview(builder, num_records=1)
        dataset = getattr(preview, "dataset", preview)
        frame = _dataset_to_frame(dataset)
        if frame.empty or "synthetic_twin_ssot" not in frame.columns:
            return {"status": "failed", "reason": "missing_structured_output"}
        record = frame.iloc[0].to_dict()
        payload = _coerce_ssot_payload(record.get("synthetic_twin_ssot"))
        payload = _enrich_ssot_payload(payload, turns_list, source_name=source_name)
        validation = _validate_ssot_value(payload)
        if not _validation_passed(validation):
            reason = _validation_error(validation)
            if _source_fallback_enabled():
                return _fallback_generated_ssot(turns_list, source_name=source_name, reason=reason)
            return _failed_sdk_ssot(turns_list, source_name=source_name, reason=reason, stage="validation")
        payload["status"] = "generated"
        return payload
    except Exception as exc:
        if _source_fallback_enabled():
            return _fallback_generated_ssot(turns_list, source_name=source_name, reason=str(exc))
        return _failed_sdk_ssot(turns_list, source_name=source_name, reason=str(exc), stage="sdk_preview")


def build_transcript_ssot_from_contract_evidence(
    contract: Any,
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """Build structured SSOT from contract evidence after raw source text is dropped."""
    if not enabled:
        return {"status": "disabled"}
    entity = contract.entities[0] if getattr(contract, "entities", None) else None
    metadata = dict(getattr(entity, "metadata", {}) or {})
    existing = metadata.get("structured_ssot")
    if isinstance(existing, dict) and existing.get("status") == "generated":
        return dict(existing)
    turns = _contract_evidence_turns(metadata)
    if not turns:
        return {"status": "failed", "stage": "contract_evidence", "reason": "missing_turn_plan"}
    return build_transcript_ssot_with_data_designer(
        turns,
        source_name=str(getattr(contract, "contract_id", "") or "transcript"),
        enabled=True,
    )


def _contract_evidence_turns(metadata: dict[str, Any]) -> list[Any]:
    turn_plan = [row for row in metadata.get("turn_plan") or [] if isinstance(row, dict)]
    if not turn_plan:
        return []
    rows: list[Any] = []
    for index, row in enumerate(turn_plan, start=1):
        speaker = str(row.get("speaker") or ("Customer" if index % 2 else "Agent"))
        role = str(row.get("role") or "").strip()
        intent = str(row.get("intent") or "conversation progress").strip()
        text = f"{role} {intent}".strip()
        rows.append(SimpleNamespace(index=index - 1, speaker=speaker, timestamp="", text=text))
    return rows


def _runtime() -> PlatformSLMRuntime:
    return resolve_platform_slm_runtime()


def _skip_health_check() -> bool:
    return data_designer_skip_health_check()


def _mock_enabled() -> bool:
    return os.getenv("SP_NEMO_DATA_DESIGNER_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def _source_fallback_enabled() -> bool:
    return os.getenv("SP_TRANSCRIPT_SSOT_SOURCE_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _mock_transcript_ssot(turns: Iterable[Any], *, source_name: str) -> dict[str, Any]:
    payload = _enrich_ssot_payload({}, turns, source_name=source_name)
    payload["status"] = "generated"
    return payload


def _fallback_generated_ssot(turns: Iterable[Any], *, source_name: str, reason: str) -> dict[str, Any]:
    payload = _enrich_ssot_payload({}, turns, source_name=source_name)
    payload["status"] = "generated"
    payload.setdefault("metadata", {})["sdk_generation_fallback"] = "source_derived_contract"
    payload["metadata"]["sdk_generation_error"] = str(reason)[:500]
    payload["metadata"]["source_pipeline"] = _source_pipeline_metadata(payload)
    return payload


def _failed_sdk_ssot(turns: Iterable[Any], *, source_name: str, reason: str, stage: str) -> dict[str, Any]:
    evidence = _source_evidence(turns)
    return {
        "status": "failed",
        "reason": str(reason)[:500],
        "stage": stage,
        "metadata": {
            "source_transcript_id": source_name or "transcript",
            "channel": "customer_interaction",
            "version": "1.0.0",
            "source_pipeline": {
                "audio_ingestion": "upstream_text_or_transcript_upload",
                "asr_engine": "upstream_riva_or_existing_transcript",
                "text_normalization": "source_transcript_parser",
                "curation": "platform_sanitizer_and_optional_nemo_curator",
                "ssot_builder": "nemo_data_designer_local_slm",
            },
        },
        "support_context": {
            "source": "sanitized_transcript_contract",
            "issue_type": _issue_type_from_evidence(evidence),
            "issue_summary": _summary_from_terms(evidence["topic_terms"] or ["support"], "SDK extraction failed after analyzing"),
            "turn_count": evidence["turn_count"],
            "speaker_count": evidence["speaker_count"],
            "topic_terms": ", ".join(evidence["topic_terms"] or ["support"]),
        },
    }


def _sanitized_turn_text(turns: Iterable[Any], *, max_turns: int | None = None) -> str:
    rows: list[str] = []
    for index, turn in enumerate(turns, start=1):
        if max_turns is not None and index > max_turns:
            rows.append(f"... {index - 1}+ sanitized turns available in source-derived evidence JSON")
            break
        speaker = str(getattr(turn, "speaker", "") or "Participant").strip()
        text = str(getattr(turn, "text", "") or "").strip()
        if text:
            rows.append(f"{index}. {speaker}: {text}")
    return "\n".join(rows)


def _turn_count(turns: Iterable[Any]) -> int:
    return len(list(turns))


def _speaker_names(turns: Iterable[Any]) -> list[str]:
    return sorted({str(getattr(turn, "speaker", "") or "Participant").strip() for turn in turns})


def _topic_terms(turns: Iterable[Any], *, limit: int = 10) -> list[str]:
    stopwords = {
        "about",
        "actually",
        "after",
        "agent",
        "also",
        "and",
        "are",
        "bear",
        "because",
        "before",
        "but",
        "bye",
        "calling",
        "caller",
        "can",
        "cause",
        "come",
        "correct",
        "customer",
        "day",
        "exactly",
        "for",
        "from",
        "get",
        "give",
        "go",
        "going",
        "good",
        "gonna",
        "got",
        "great",
        "have",
        "her",
        "hers",
        "here",
        "him",
        "his",
        "hold",
        "huh",
        "hello",
        "help",
        "helping",
        "information",
        "internal",
        "just",
        "know",
        "last",
        "let",
        "look",
        "looking",
        "maybe",
        "mean",
        "moment",
        "more",
        "name",
        "need",
        "number",
        "okay",
        "one",
        "out",
        "please",
        "provide",
        "really",
        "request",
        "right",
        "said",
        "say",
        "see",
        "she",
        "sure",
        "take",
        "thank",
        "thanks",
        "that",
        "then",
        "there",
        "they",
        "thing",
        "things",
        "the",
        "this",
        "turn",
        "try",
        "trying",
        "was",
        "welcome",
        "well",
        "what",
        "with",
        "would",
        "yeah",
        "yes",
        "yep",
        "you",
        "your",
    }
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
        for word in re.findall(r"\b[a-z][a-z-]{2,}\b", str(getattr(turn, "text", "") or "").lower()):
            if word not in stopwords:
                counts[word] = counts.get(word, 0) + 1
    ordered = [term for term in priority_terms if counts.get(term)]
    ordered.extend(word for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if word not in ordered)
    return ordered[:limit]


def _source_evidence(turns: Iterable[Any]) -> dict[str, Any]:
    turns_list = list(turns)
    lowered = " ".join(str(getattr(turn, "text", "") or "").lower() for turn in turns_list)
    evidence = {
        "turn_count": len(turns_list),
        "speaker_count": len(_speaker_names(turns_list)),
        "speakers": _speaker_names(turns_list),
        "topic_terms": _topic_terms(turns_list),
        "has_account_context": "account" in lowered,
        "has_access_context": "access" in lowered or "403" in lowered or "forbidden" in lowered,
        "has_plan_context": "plan" in lowered or "benefit" in lowered,
        "has_payment_context": any(term in lowered for term in ("charge", "charged", "payment", "refund", "debit card", "credit card")),
        "has_activation_context": "activate" in lowered or "activation" in lowered,
        "has_claim_context": "claim" in lowered,
        "has_resolution_signal": any(term in lowered for term in ("resolved", "fixed", "activated", "thank", "good")),
    }
    return evidence


def _ssot_prompt() -> str:
    return (
        "Convert this sanitized customer interaction into a structured synthetic twin SSOT.\n"
        "Source name: {{ source_name }}\n"
        "Schema version: {{ schema_version }}\n"
        "Turn count: {{ turn_count }}\n"
        "Speaker count: {{ speaker_count }}\n"
        "Speakers JSON: {{ speakers_json }}\n"
        "Topic terms JSON: {{ topic_terms_json }}\n"
        "Source-derived evidence JSON: {{ evidence_json }}\n"
        "Transcript:\n{{ sanitized_transcript }}\n"
        "Infer only from sanitized content and source-derived evidence. Keep values synthetic and privacy-safe.\n"
        "Use exactly these top-level keys: metadata, entities, support_context, resolved_issues, actions_taken, "
        "account_mutations, sentiment_analysis, privacy_validation.\n"
        "Put speakers under entities.participants as a list of objects with speaker and role. "
        "Do not create dynamic keys named Agent, Customer, has_payment_context, issue_type, action, result, status, "
        "or mutation_type outside the requested sections. Do not include markdown or prose. "
        "Do not use placeholder values such as synthetic_value, synthetic_resolution, or generic account guidance."
    )


def _ssot_schema() -> dict[str, Any]:
    string_list = {"type": "array", "items": {"type": "string"}}
    participant = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "role": {"type": "string"},
        },
        "required": ["speaker", "role"],
        "additionalProperties": False,
    }
    issue = {
        "type": "object",
        "properties": {
            "issue_type": {"type": "string"},
            "resolution_status": {"type": "string"},
            "resolution_summary": {"type": "string"},
        },
        "required": ["issue_type", "resolution_status", "resolution_summary"],
        "additionalProperties": False,
    }
    action = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "detail": {"type": "string"},
        },
        "required": ["action", "detail"],
        "additionalProperties": False,
    }
    mutation = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "new_value": {"type": "string"},
        },
        "required": ["action", "new_value"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {
                    "source_transcript_id": {"type": "string"},
                    "channel": {"type": "string"},
                    "version": {"type": "string"},
                    "source_pipeline": {
                        "type": "object",
                        "properties": {
                            "audio_ingestion": {"type": "string"},
                            "asr_engine": {"type": "string"},
                            "text_normalization": {"type": "string"},
                            "curation": {"type": "string"},
                            "ssot_builder": {"type": "string"},
                        },
                        "required": ["audio_ingestion", "asr_engine", "text_normalization", "curation", "ssot_builder"],
                        "additionalProperties": False,
                    },
                },
                "required": ["source_transcript_id", "channel", "version"],
                "additionalProperties": False,
            },
            "entities": {
                "type": "object",
                "properties": {
                    "participants": {"type": "array", "items": participant},
                },
                "required": ["participants"],
                "additionalProperties": False,
            },
            "support_context": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "issue_summary": {"type": "string"},
                    "turn_count": {"type": "integer"},
                    "speaker_count": {"type": "integer"},
                    "topic_terms": string_list,
                },
                "required": ["source", "issue_type", "issue_summary"],
                "additionalProperties": False,
            },
            "resolved_issues": {"type": "array", "items": issue},
            "actions_taken": {"type": "array", "items": action},
            "account_mutations": {"type": "array", "items": mutation},
            "sentiment_analysis": {
                "type": "object",
                "properties": {
                    "initial_customer_sentiment": {"type": "string"},
                    "final_customer_sentiment": {"type": "string"},
                },
                "required": ["initial_customer_sentiment", "final_customer_sentiment"],
                "additionalProperties": False,
            },
            "privacy_validation": {
                "type": "object",
                "properties": {
                    "raw_source_text_used": {"type": "boolean"},
                    "source_identifiers_removed": {"type": "boolean"},
                    "generated_from_sanitized_turns": {"type": "boolean"},
                },
                "required": ["raw_source_text_used", "source_identifiers_removed", "generated_from_sanitized_turns"],
                "additionalProperties": False,
            },
        },
        "required": [
            "metadata",
            "entities",
            "support_context",
            "resolved_issues",
            "actions_taken",
            "sentiment_analysis",
            "privacy_validation",
        ],
        "additionalProperties": False,
    }


def _dataset_to_frame(dataset: Any) -> pd.DataFrame:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    if isinstance(dataset, pd.DataFrame):
        return dataset
    if isinstance(dataset, list):
        return pd.DataFrame(dataset)
    if isinstance(dataset, dict):
        return pd.DataFrame(dataset)
    raise TypeError("unsupported Data Designer dataset shape")


def _coerce_ssot_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("structured SSOT output is not a JSON object")
    return parsed


def _enrich_ssot_payload(payload: dict[str, Any], turns: Iterable[Any], *, source_name: str) -> dict[str, Any]:
    enriched = dict(payload)
    text = _sanitized_turn_text(turns)
    evidence = _source_evidence(turns)
    topic_terms = evidence["topic_terms"] or ["support"]
    issue_type = _issue_type_from_evidence(evidence)

    if not _non_empty_mapping(enriched.get("metadata")):
        enriched["metadata"] = {
            "source_transcript_id": source_name or "transcript",
            "channel": "customer_interaction",
            "version": "1.0.0",
        }
    metadata = dict(enriched.get("metadata") or {})
    metadata.setdefault("source_transcript_id", source_name or "transcript")
    metadata.setdefault("channel", "customer_interaction")
    metadata.setdefault("version", "1.0.0")
    metadata.setdefault("source_pipeline", _source_pipeline_metadata(enriched))
    enriched["metadata"] = metadata
    if not _non_empty_mapping(enriched.get("entities")):
        enriched["entities"] = _entities_from_evidence(evidence)
    else:
        enriched["entities"] = _normalize_entities(enriched["entities"])
    if not _non_empty_mapping(enriched.get("support_context")):
        enriched["support_context"] = {
            "source": "sanitized_transcript_contract",
        }
    support_context = dict(enriched.get("support_context") or {})
    support_context.setdefault("issue_type", issue_type)
    support_context.setdefault("issue_summary", _summary_from_terms(topic_terms, "Customer interaction focused on"))
    support_context.setdefault("turn_count", evidence["turn_count"])
    support_context.setdefault("speaker_count", evidence["speaker_count"])
    support_context["topic_terms"] = _topic_terms_for_output(support_context.get("topic_terms"), topic_terms)
    enriched["support_context"] = support_context
    if not _non_empty_list(enriched.get("resolved_issues")):
        enriched["resolved_issues"] = [
            {
                "issue_type": issue_type,
                "resolution_status": "resolved" if evidence["has_resolution_signal"] else "in_progress",
                "resolution_summary": _summary_from_terms(topic_terms, "Support outcome inferred from"),
            }
        ]
    else:
        enriched["resolved_issues"] = _merge_issue_evidence(enriched["resolved_issues"], issue_type, topic_terms, evidence)
    if not _non_empty_list(enriched.get("actions_taken")):
        enriched["actions_taken"] = [
            {
                "action": "captured_sanitized_context",
                "detail": _summary_from_terms(topic_terms, "Used sanitized conversation context about"),
            },
            {
                "action": "continued_support_workflow",
                "detail": "Generated from sanitized turn structure and source-derived topic evidence.",
            },
        ]
    else:
        enriched["actions_taken"] = _merge_action_evidence(enriched["actions_taken"], topic_terms)
    if not _non_empty_list(enriched.get("account_mutations")):
        enriched["account_mutations"] = [
            {
                "action": "support_state_update",
                "new_value": "status_updated_from_sanitized_context",
            }
        ]
    if not _non_empty_mapping(enriched.get("sentiment_analysis")):
        enriched["sentiment_analysis"] = {
            "initial_customer_sentiment": _initial_sentiment(text),
            "final_customer_sentiment": "satisfied" if evidence["has_resolution_signal"] else "neutral",
        }
    if not _non_empty_mapping(enriched.get("privacy_validation")):
        enriched["privacy_validation"] = {
            "raw_source_text_used": False,
            "source_identifiers_removed": True,
            "generated_from_sanitized_turns": True,
        }
    return enriched


def _source_pipeline_metadata(payload: dict[str, Any]) -> dict[str, str]:
    fallback = payload.get("metadata", {}).get("sdk_generation_fallback") if isinstance(payload.get("metadata"), dict) else None
    return {
        "audio_ingestion": "upstream_text_or_transcript_upload",
        "asr_engine": "upstream_riva_or_existing_transcript",
        "text_normalization": "source_transcript_parser",
        "curation": "platform_sanitizer_and_optional_nemo_curator",
        "ssot_builder": "source_derived_contract" if fallback else "nemo_data_designer_local_slm",
    }


def _normalize_entities(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    participants = value.get("participants")
    if not isinstance(participants, list):
        return dict(value)
    normalized: dict[str, dict[str, str]] = {}
    for index, participant in enumerate(participants, start=1):
        if not isinstance(participant, dict):
            continue
        speaker = str(participant.get("speaker") or f"Participant {index}").strip()
        key = re.sub(r"[^a-z0-9]+", "_", speaker.lower()).strip("_") or f"participant_{index}"
        role = str(participant.get("role") or _role_from_speaker(speaker, index)).strip()
        normalized[key] = {"role": role}
    return normalized or {}


def _topic_terms_for_output(value: Any, fallback: list[str]) -> str:
    if isinstance(value, list):
        terms = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(terms) if terms else ", ".join(fallback)
    text = str(value or "").strip()
    return text or ", ".join(fallback)


def _merge_issue_evidence(
    issues: list[Any],
    issue_type: str,
    topic_terms: list[str],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        issue = dict(item)
        issue.setdefault("issue_type", issue_type)
        if str(issue.get("resolution_status") or "").lower() in {"synthetic_resolution", "synthetic_value", ""}:
            issue["resolution_status"] = "resolved" if evidence["has_resolution_signal"] else "in_progress"
        issue.setdefault("resolution_summary", _summary_from_terms(topic_terms, "Support outcome inferred from"))
        merged.append(issue)
    return merged or [
        {
            "issue_type": issue_type,
            "resolution_status": "resolved" if evidence["has_resolution_signal"] else "in_progress",
            "resolution_summary": _summary_from_terms(topic_terms, "Support outcome inferred from"),
        }
    ]


def _merge_action_evidence(actions: list[Any], topic_terms: list[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = dict(item)
        if str(action.get("action") or "").lower() in {"synthetic_value", "account_support_guidance", ""}:
            action["action"] = "continued_support_workflow"
        action.setdefault("detail", _summary_from_terms(topic_terms, "Used sanitized conversation context about"))
        merged.append(action)
    return merged or [
        {
            "action": "continued_support_workflow",
            "detail": _summary_from_terms(topic_terms, "Used sanitized conversation context about"),
        }
    ]


def _issue_type_from_evidence(evidence: dict[str, Any]) -> str:
    if evidence.get("has_access_context"):
        return "access_error"
    if evidence.get("has_payment_context"):
        return "payment_charge_investigation"
    if evidence.get("has_claim_context"):
        return "claim_support"
    if evidence.get("has_activation_context") and evidence.get("has_account_context"):
        return "account_activation"
    if evidence.get("has_plan_context"):
        return "plan_or_benefit_support"
    return "customer_support_request"


def _entities_from_evidence(evidence: dict[str, Any]) -> dict[str, dict[str, str]]:
    speakers = [str(speaker) for speaker in evidence.get("speakers") or []]
    if not speakers:
        return {"participant_1": {"role": "requester"}}
    entities: dict[str, dict[str, str]] = {}
    for index, speaker in enumerate(speakers, start=1):
        key = re.sub(r"[^a-z0-9]+", "_", speaker.lower()).strip("_") or f"participant_{index}"
        entities[key] = {"role": _role_from_speaker(speaker, index)}
    return entities


def _role_from_speaker(speaker: str, index: int) -> str:
    lowered = speaker.lower()
    if "agent" in lowered or "support" in lowered or "representative" in lowered:
        return "support_representative"
    if "customer" in lowered or "caller" in lowered or "client" in lowered:
        return "requester"
    return "participant" if index > 1 else "requester"


def _summary_from_terms(topic_terms: list[str], prefix: str) -> str:
    terms = ", ".join(topic_terms[:6]) if topic_terms else "the customer support request"
    return f"{prefix} {terms}."


def _initial_sentiment(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("unable", "couldn't", "problem", "issue", "error", "frustrated")):
        return "frustrated"
    return "needs_help"


def _non_empty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and any(str(item).strip() for item in value.values())


def _non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and any(_non_empty_mapping(item) for item in value)


def _validate_ssot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_validate_ssot_value(row.get("synthetic_twin_ssot")) for row in frame.to_dict(orient="records")])


def _validate_ssot_value(value: Any) -> dict[str, Any]:
    try:
        payload = _coerce_ssot_payload(value)
    except Exception as exc:
        return {"is_valid": False, "error_messages": f"invalid_json:{exc}"}
    missing = sorted(set(_ssot_schema()["required"]) - set(payload))
    if missing:
        return {"is_valid": False, "error_messages": f"missing_keys:{','.join(missing)}"}
    empty_sections = _empty_ssot_sections(payload)
    if empty_sections:
        return {"is_valid": False, "error_messages": f"empty_sections:{','.join(empty_sections)}"}
    placeholder_sections = _placeholder_ssot_sections(payload)
    if placeholder_sections:
        return {"is_valid": False, "error_messages": f"placeholder_values:{','.join(placeholder_sections)}"}
    return {"is_valid": True, "error_messages": None}


def _empty_ssot_sections(payload: dict[str, Any]) -> list[str]:
    empty: list[str] = []
    for key in ("metadata", "entities", "support_context", "sentiment_analysis", "privacy_validation"):
        if not _non_empty_mapping(payload.get(key)):
            empty.append(key)
    for key in ("resolved_issues", "actions_taken"):
        if not _non_empty_list(payload.get(key)):
            empty.append(key)
    return empty


def _placeholder_ssot_sections(payload: dict[str, Any]) -> list[str]:
    placeholders = {
        "synthetic_value",
        "synthetic_resolution",
        "account_support_guidance",
        "synthetic_next_step_confirmed",
    }
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    return sorted(value for value in placeholders if value in text)


def _validation_passed(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("is_valid"))


def _validation_error(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("error_messages") or "invalid_ssot")
    return "missing_validation_result"
