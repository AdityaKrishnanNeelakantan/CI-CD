from __future__ import annotations

import importlib.machinery
import json
import re
import sys
import types

import pandas as pd

from synth_platform.engine.transcripts import ssot_designer
from synth_platform.engine.transcripts import (
    build_transcript_contract,
    generate_synthetic_transcript,
    parse_transcript_text,
    summarize_transcript_preview,
    validate_transcript_non_replay,
)


def _install_fake_transcript_ssot_data_designer(monkeypatch):
    monkeypatch.setenv("SP_PLATFORM_SLM_PREFLIGHT", "0")
    captured: dict[str, object] = {}

    class FakeModelProvider:
        def __init__(self, **kwargs):
            captured["model_provider"] = kwargs

    class FakeModelConfig:
        def __init__(self, **kwargs):
            captured["model_config"] = kwargs

    class FakeChatCompletionInferenceParams:
        def __init__(self, **kwargs):
            captured["inference_parameters"] = kwargs

    class FakeDataFrameSeedSource:
        def __init__(self, df):
            captured["seed_frame"] = df

    class FakeLLMStructuredColumnConfig:
        def __init__(self, **kwargs):
            captured["structured_column"] = kwargs
            self.kwargs = kwargs

    class FakeLocalCallableValidatorParams:
        def __init__(self, **kwargs):
            captured["validator_params"] = kwargs
            self.validation_function = kwargs["validation_function"]

    class FakeValidationColumnConfig:
        def __init__(self, **kwargs):
            captured["validation_column"] = kwargs
            self.kwargs = kwargs

    class FakeDataDesignerConfigBuilder:
        def __init__(self, model_configs=None):
            captured["builder_model_configs"] = model_configs

        def with_seed_dataset(self, source):
            captured["seed_source"] = source
            return self

        def add_column(self, column):
            captured.setdefault("columns", []).append(column)
            return self

    class FakeDataDesigner:
        def __init__(self, **kwargs):
            captured["data_designer"] = kwargs

        def preview(self, _builder, *, num_records):
            captured["preview_num_records"] = num_records
            if captured.get("preview_exception"):
                raise RuntimeError(str(captured["preview_exception"]))
            payload = {
                "metadata": {
                    "source_transcript_id": "chat.txt",
                    "channel": "chat",
                    "version": "1.0.0",
                },
                "entities": {
                    "customer": {"role": "requester"},
                    "agent": {"role": "support"},
                },
                "support_context": {"source": "sanitized_transcript_contract"},
                "resolved_issues": [{"issue_type": "access_error", "resolution_status": "resolved"}],
                "actions_taken": [{"action": "plan_change", "new_value": "synthetic_enterprise"}],
                "account_mutations": [{"action": "plan_change", "new_value": "synthetic_enterprise"}],
                "sentiment_analysis": {"initial_customer_sentiment": "frustrated", "final_customer_sentiment": "satisfied"},
                "privacy_validation": {"raw_source_text_used": False},
            }
            validator = captured["validation_column"]["validator_params"].validation_function
            records = [{"synthetic_twin_ssot": payload}]
            validation = validator(pd.DataFrame(records)).to_dict(orient="records")
            records[0]["synthetic_twin_ssot_validation"] = validation[0]
            return types.SimpleNamespace(dataset=pd.DataFrame(records))

    fake_config = types.SimpleNamespace(
        ModelProvider=FakeModelProvider,
        ModelConfig=FakeModelConfig,
        ChatCompletionInferenceParams=FakeChatCompletionInferenceParams,
        DataFrameSeedSource=FakeDataFrameSeedSource,
        LLMStructuredColumnConfig=FakeLLMStructuredColumnConfig,
        LocalCallableValidatorParams=FakeLocalCallableValidatorParams,
        ValidationColumnConfig=FakeValidationColumnConfig,
        ValidatorType=types.SimpleNamespace(LOCAL_CALLABLE="local_callable"),
        DataDesignerConfigBuilder=FakeDataDesignerConfigBuilder,
    )
    fake_interface = types.SimpleNamespace(DataDesigner=FakeDataDesigner)
    fake_package = types.ModuleType("data_designer")
    fake_package.__path__ = []
    fake_package.__spec__ = importlib.machinery.ModuleSpec("data_designer", loader=None, is_package=True)
    fake_package.config = fake_config
    fake_package.interface = fake_interface
    monkeypatch.setitem(sys.modules, "data_designer", fake_package)
    monkeypatch.setitem(sys.modules, "data_designer.config", fake_config)
    monkeypatch.setitem(sys.modules, "data_designer.interface", fake_interface)
    return captured


def test_transcript_preview_is_sanitized_and_counts_turns_speakers():
    text = (
        "Agent: Email me at alex@example.com and visit https://example.com\n"
        "Customer: Call +1 555 123 4567 tomorrow. SSN 111-22-3333"
    )

    preview = summarize_transcript_preview(text, source_name="chat.txt")

    assert preview.source_type == "transcript"
    assert preview.record_count == 2
    assert preview.entity_count == 2
    sample_text = " ".join(row["text"] for row in preview.items[0].sample)
    assert "alex@example.com" not in sample_text
    assert re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", sample_text)
    assert re.search(r"\b\d{3}-\d{2}-\d{4}\b", sample_text)
    assert "[" not in sample_text
    assert "]" not in sample_text


def test_transcript_preview_masks_common_sensitive_customer_context():
    text = (
        "Agent: OmniMutual Claims Support, this is David.\n"
        "Customer: My policy number is INS-441-B83. My name is Timothy Brooks.\n"
        "Agent: What is your date of birth?\n"
        "Customer: My date of birth is February 14, 1988. I left Saratoga Hospital. "
        "Claim ID CLM-3349102."
    )

    preview = summarize_transcript_preview(text, source_name="claim.txt")
    sample_text = " ".join(row["text"] for row in preview.items[0].sample)

    assert "Timothy Brooks" not in sample_text
    assert "INS-441-B83" not in sample_text
    assert "February 14, 1988" not in sample_text
    assert "Saratoga Hospital" not in sample_text
    assert "CLM-3349102" not in sample_text
    assert re.search(r"My name is [A-Z][a-z]+ [A-Z][a-z]+", sample_text)
    assert re.search(r"date of birth is [A-Z][a-z]+ \d{1,2}, \d{4}", sample_text)
    assert re.search(r"claim id CLM-\d{8}", sample_text)
    assert "[" not in sample_text
    assert "]" not in sample_text


def test_transcript_preview_uses_locale_aware_faker_replacements():
    india_text = (
        "Customer: I am in India. My first name is Arjun. My last name is Mehta. "
        "My address is 221 MG Road, Bengaluru, Karnataka 560001. "
        "My country is India. Email me at arjun.mehta@example.in."
    )
    us_text = (
        "Customer: I am in the United States. My first name is Arjun. My last name is Mehta. "
        "My address is 221 MG Road, Bengaluru, Karnataka 560001. "
        "My state is California. My country is United States. Email me at arjun.mehta@example.in."
    )

    india_preview = summarize_transcript_preview(india_text, source_name="india.txt")
    us_preview = summarize_transcript_preview(us_text, source_name="us.txt")
    india_sample = " ".join(row["text"] for row in india_preview.items[0].sample)
    us_sample = " ".join(row["text"] for row in us_preview.items[0].sample)

    for raw_value in ["Arjun", "Mehta", "221 MG Road", "Bengaluru", "arjun.mehta@example.in"]:
        assert raw_value not in india_sample
        assert raw_value not in us_sample

    assert "country is India" in india_sample
    assert "country is United States" in us_sample
    assert "I am in India" in india_sample
    assert re.search(r"first name is [A-Z][A-Za-z'-]+", india_sample)
    assert re.search(r"last name is [A-Z][A-Za-z'-]+", india_sample)
    assert re.search(r"state is [A-Z]{2}", us_sample)
    assert re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", india_sample)
    assert india_sample != us_sample


def test_transcript_preview_uses_context_aware_identity_replacements():
    text = (
        "Customer: I am in the United States. My first name is Ada. "
        "My last name is Lovelace. My address is 1 Main St, Boston, MA 02108. "
        "My state is Massachusetts. My country is United States. "
        "My email is ada.lovelace@example.com."
    )

    preview = summarize_transcript_preview(text, source_name="identity.txt")
    sample_text = " ".join(row["text"] for row in preview.items[0].sample)
    first = re.search(r"first name is (?P<value>[A-Z][A-Za-z'-]+)", sample_text)
    last = re.search(r"last name is (?P<value>[A-Z][A-Za-z'-]+)", sample_text)
    email = re.search(r"\b(?P<value>[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})\b", sample_text)

    assert first is not None
    assert last is not None
    assert email is not None
    assert "Ada" not in sample_text
    assert "Lovelace" not in sample_text
    assert "1 Main St" not in sample_text
    assert "ada.lovelace@example.com" not in sample_text
    assert first.group("value").lower() in email.group("value")
    assert last.group("value").lower() in email.group("value")
    assert ". My country is United States" in sample_text
    assert "country is United States" in sample_text
    assert re.search(r"state is [A-Z]{2}", sample_text)


def test_customer_claim_interaction_preview_and_topic_are_sanitized():
    text = (
        "[00:00:01.200] Agent: OmniMutual Claims Support, this is David. "
        "Can I get your auto policy number to bring up your details? "
        "[00:00:06.850] Caller: Hello, yes. My policy number is INS-441-B83. "
        "My name is Timothy Brooks. [00:00:13.100] Agent: Thank you, Mr. Brooks. "
        "To confirm identity, what is your date of birth, cell phone number, and the email address attached to the account? "
        "[00:00:20.450] Caller: My date of birth is February 14, 1988. Phone number is 555-012-9988, "
        "and my email is tim.brooks88@postbox.com. [00:00:30.900] Agent: Perfect, I have the system record open. "
        "Let's discuss the claim you started filing online. It looks like an incident involving a 2022 Honda Civic? "
        "[00:00:38.250] Caller: Yes. I was rear-ended at a stoplight on Fifth Ave near 42nd St, New York. "
        "The other driver's license plate was New York registration TX-R99-XPD. "
        "[00:00:49.700] Agent: Got it. Our system shows you checked a box regarding physical injury. Are you alright? "
        "[00:00:54.150] Caller: I have some severe concussion and a bruised ribs. I just left the emergency room at Saratoga Hospital, "
        "and they gave me a prescription for 400mg Ibuprofen. "
        "[00:01:04.600] Agent: Understood. I will log those medical updates under your active claim tracking ID #CLM-3349102."
    )

    preview = summarize_transcript_preview(text, source_name="claim.txt")
    preview_text = " ".join(row["text"] for row in preview.items[0].sample)
    contract = build_transcript_contract(text, source_name="claim.txt")
    topic_terms = contract.entities[0].metadata["topic_terms"]

    for raw_value in [
        "David",
        "Timothy Brooks",
        "Brooks",
        "INS-441-B83",
        "February 14, 1988",
        "555-012-9988",
        "tim.brooks88@postbox.com",
        "TX-R99-XPD",
        "Saratoga Hospital",
        "CLM-3349102",
    ]:
        assert raw_value not in preview_text
        assert raw_value not in json.dumps(contract.model_dump(mode="json"))
    assert "this is identifier" not in preview_text
    assert re.search(r"this is [A-Z][A-Za-z'-]+ [A-Z][A-Za-z'-]+", preview_text)
    assert "claim" in topic_terms
    assert not {"you", "clm", "id", "system"} & set(topic_terms)
    context = contract.entities[0].metadata["synthetic_context"]
    synthetic = generate_synthetic_transcript(contract, turn_count=6, seed=1)
    synthetic_text = " ".join(row["text"] for row in synthetic)
    assert "claim" in synthetic_text.lower() or "vehicle" in synthetic_text.lower()
    assert "[MOCK_CUSTOMER_NAME]" in synthetic_text
    assert "[MOCK_POLICY_ID]" in synthetic_text
    assert "[MOCK_DATE_OF_BIRTH]" in synthetic_text
    assert "[MOCK_CASE_ID]" in synthetic_text
    assert context["customer_name"] not in synthetic_text
    assert context["policy_id"] not in synthetic_text
    assert context["email"] not in synthetic_text
    assert context["phone"] not in synthetic_text
    assert context["date_of_birth"] not in synthetic_text
    assert (context.get("claim_id") or context.get("tracking_id")) not in synthetic_text
    assert "Timothy Brooks" not in synthetic_text
    assert "INS-441-B83" not in synthetic_text
    assert "555-012-9988" not in synthetic_text
    assert "tim.brooks88@postbox.com" not in synthetic_text


def test_transcript_contract_uses_same_canonical_contract_shape():
    contract = build_transcript_contract(
        "Agent: Hello alex@example.com\nCustomer: Hi",
        source_name="chat.txt",
        nvidia_options={"enabled": False},
    )

    assert contract.source_type == "transcript"
    assert contract.entities[0].entity_type == "transcript_turn"
    assert any(field.semantic_type == "free_text" for field in contract.fields)
    assert contract.privacy_policy["raw_text_persisted"] is False
    assert contract.privacy_policy["nvidia_nemo_curator"]["status"] == "disabled"
    serialized = json.dumps(contract.model_dump(mode="json"))
    assert "alex@example.com" not in serialized
    assert "source_turn_hashes" in serialized


def test_transcript_contract_can_use_data_designer_structured_ssot(monkeypatch):
    captured = _install_fake_transcript_ssot_data_designer(monkeypatch)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)

    contract = build_transcript_contract(
        "Agent: I can help with the dashboard access error.\n"
        "Customer: Please upgrade the plan after access is fixed.",
        source_name="chat.txt",
        nvidia_options={"enabled": True},
    )

    ssot = contract.entities[0].metadata["structured_ssot"]
    assert ssot["status"] == "generated", ssot
    assert ssot["resolved_issues"][0]["issue_type"] == "access_error"
    assert ssot["account_mutations"][0]["action"] == "plan_change"
    assert captured["structured_column"]["name"] == "synthetic_twin_ssot"
    assert captured["structured_column"]["model_alias"] == "transcript-ssot-builder"
    assert captured["validation_column"]["target_columns"] == ["synthetic_twin_ssot"]
    assert captured["model_provider"]["endpoint"] == "http://127.0.0.1:11434/v1"
    assert captured["model_provider"]["api_key"] is None
    assert "provider" not in ssot
    assert "model" not in ssot


def test_transcript_contract_can_defer_structured_ssot_for_fast_ui_build(monkeypatch):
    captured = _install_fake_transcript_ssot_data_designer(monkeypatch)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)

    contract = build_transcript_contract(
        "Agent: I can help with the dashboard access error.\n"
        "Customer: Please upgrade the plan after access is fixed.",
        source_name="chat.txt",
        nvidia_options={"enabled": True, "build_structured_ssot": False},
    )

    ssot = contract.entities[0].metadata["structured_ssot"]
    assert ssot == {"status": "pending", "reason": "deferred_until_generation"}
    assert "preview_num_records" not in captured


def test_transcript_ssot_fallback_is_generated_and_records_sdk_error():
    turns = parse_transcript_text(
        "Agent: I can review the card charge.\n"
        "Customer: I do not recognize the payment."
    )

    payload = ssot_designer._fallback_generated_ssot(turns, source_name="charge.txt", reason="bad structured output")

    assert payload["status"] == "generated"
    assert payload["metadata"]["sdk_generation_fallback"] == "source_derived_contract"
    assert payload["metadata"]["sdk_generation_error"] == "bad structured output"
    assert payload["metadata"]["source_pipeline"]["ssot_builder"] == "source_derived_contract"
    assert payload["support_context"]["issue_type"] == "payment_charge_investigation"


def test_transcript_ssot_schema_uses_explicit_entity_and_evidence_fields():
    schema = ssot_designer._ssot_schema()

    assert schema["properties"]["entities"]["properties"]["participants"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in schema["properties"]["entities"]["properties"]["participants"]["items"]["properties"]
    assert schema["properties"]["support_context"]["additionalProperties"] is False
    assert schema["properties"]["privacy_validation"]["additionalProperties"] is False
    serialized = json.dumps(schema)
    assert '"additionalProperties": {"type"' not in serialized


def test_transcript_ssot_enrichment_normalizes_sdk_participants_shape():
    turns = parse_transcript_text(
        "Agent: I can review the card charge.\n"
        "Customer: I do not recognize the payment."
    )
    payload = {
        "entities": {
            "participants": [
                {"speaker": "Agent", "role": "support_representative"},
                {"speaker": "Customer", "role": "requester"},
            ]
        },
        "support_context": {"source": "sanitized_transcript_contract", "topic_terms": ["charge", "payment"]},
    }

    enriched = ssot_designer._enrich_ssot_payload(payload, turns, source_name="charge.txt")

    assert enriched["entities"] == {
        "agent": {"role": "support_representative"},
        "customer": {"role": "requester"},
    }
    assert enriched["support_context"]["topic_terms"] == "charge, payment"
    assert enriched["metadata"]["source_pipeline"]["asr_engine"] == "upstream_riva_or_existing_transcript"


def test_transcript_ssot_uses_sdk_for_large_local_slm_inputs(monkeypatch):
    captured = _install_fake_transcript_ssot_data_designer(monkeypatch)
    monkeypatch.setenv("SP_TRANSCRIPT_SSOT_SDK_MAX_TURNS", "2")
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_TRANSCRIPT_SSOT_SOURCE_FALLBACK", raising=False)
    turns = parse_transcript_text(
        "Agent: I can review the card charge.\n"
        "Customer: I do not recognize the payment.\n"
        "Agent: I will check the payment history."
    )

    payload = ssot_designer.build_transcript_ssot_with_data_designer(
        turns,
        source_name="charge.txt",
        enabled=True,
    )

    assert payload["status"] == "generated"
    assert "sdk_generation_fallback" not in payload["metadata"]
    assert payload["support_context"]["issue_type"] == "payment_charge_investigation"
    assert captured["preview_num_records"] == 1


def test_transcript_ssot_reports_sdk_failure_without_source_fallback(monkeypatch):
    captured = _install_fake_transcript_ssot_data_designer(monkeypatch)
    monkeypatch.delenv("SP_TRANSCRIPT_SSOT_SOURCE_FALLBACK", raising=False)
    turns = parse_transcript_text(
        "Agent: I can review the card charge.\n"
        "Customer: I do not recognize the payment."
    )

    captured["preview_exception"] = "bad structured output"
    payload = ssot_designer.build_transcript_ssot_with_data_designer(
        turns,
        source_name="charge.txt",
        enabled=True,
    )

    assert payload["status"] == "failed"
    assert payload["stage"] == "sdk_preview"
    assert payload["metadata"]["source_pipeline"]["ssot_builder"] == "nemo_data_designer_local_slm"
    assert "sdk_generation_fallback" not in payload["metadata"]


def test_transcript_parser_accepts_json_turn_arrays():
    turns = parse_transcript_text(
        json.dumps(
            {
                "messages": [
                    {"role": "agent", "content": "Hello"},
                    {"speaker": "customer", "text": "Need help"},
                ]
            }
        )
    )

    assert [turn.speaker for turn in turns] == ["agent", "customer"]
    assert [turn.text for turn in turns] == ["Hello", "Need help"]


def test_transcript_parser_splits_inline_speaker_markers_with_datestamp():
    turns = parse_transcript_text(
        "October 14, 2026Agent: Sarah from support. Caller: John needs help. "
        "Agent: I can review the account."
    )

    assert [turn.speaker for turn in turns] == ["Agent", "Caller", "Agent"]
    assert [turn.text for turn in turns] == [
        "Sarah from support.",
        "John needs help.",
        "I can review the account.",
    ]


def test_transcript_generation_and_non_replay_validation_use_contract_only():
    contract = build_transcript_contract(
        "Agent: Semantic understanding uses vectors\nCustomer: Please explain attention",
        source_name="chat.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=4, seed=1)
    report = validate_transcript_non_replay(contract, synthetic, nvidia_options={"enabled": False})

    assert len(synthetic) == 4
    assert report["raw_source_text_used"] is False
    assert report["passed"] is True
    assert report["nvidia_nemo_guardrails"]["status"] == "disabled"


def test_transcript_generation_is_role_ordered_and_topic_aware():
    contract = build_transcript_contract(
        "Agent: I can help with your claim.\nCustomer: I need claim support.",
        source_name="chat.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=6, seed=1)

    assert [row["speaker"] for row in synthetic] == [
        "Agent",
        "Customer",
        "Agent",
        "Customer",
        "Agent",
        "Customer",
    ]
    assert "help with" in synthetic[0]["text"] or "verified details" in synthetic[0]["text"]
    assert "claim" in " ".join(row["text"] for row in synthetic).lower()
    assert "You're welcome" not in synthetic[0]["text"]


def test_transcript_generation_preserves_source_speaker_sequence_and_turn_intents():
    contract = build_transcript_contract(
        "Customer: Thank you for calling line.\n"
        "Agent: I see the request is not processing.\n"
        "Person 3: What health plan do you have?\n"
        "Person 4: Blue Cross Blue Shield plan.\n",
        source_name="multi.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=6, seed=1)

    assert contract.entities[0].metadata["speaker_sequence"] == [
        "Customer",
        "Agent",
        "Person 3",
        "Person 4",
    ]
    assert [row["speaker"] for row in synthetic] == [
        "Customer",
        "Agent",
        "Person 3",
        "Person 4",
        "Customer",
        "Agent",
    ]
    joined = " ".join(row["text"] for row in synthetic).lower()
    assert "plan" in joined or "benefit" in joined


def test_transcript_contract_maps_quoted_internal_external_to_roles():
    contract = build_transcript_contract(
        "'internal : Thank you for calling support. How can I help?',\n"
        "\"external : I have a credit card charge question.\",\n"
        "'internal : I can review the charge.'",
        source_name="quoted_call.txt",
    )
    metadata = contract.entities[0].metadata

    assert metadata["speakers"] == ["Agent", "Customer"]
    assert metadata["speaker_roles"] == {"Agent": "agent", "Customer": "customer"}
    assert "charge" in metadata["topic_terms"]


def test_current_transcript_generation_alternates_two_role_call_fragments():
    contract = build_transcript_contract(
        "internal : Thank you for calling support. How can I help?\n"
        "external : I have a card charge question.\n"
        "external : I do not recognize the amount.\n"
        "internal : Let me review that.",
        source_name="call.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=6, seed=1)

    assert [row["speaker"] for row in synthetic] == ["Agent", "Customer", "Agent", "Customer", "Agent", "Customer"]
    assert "payment" in " ".join(row["text"] for row in synthetic).lower()


def test_transcript_topic_ignores_call_filler_words():
    contract = build_transcript_contract(
        "Agent: Okay yeah, one moment while I look up the account.\n"
        "Customer: I am trying to activate my account for the OTC benefit order.\n"
        "Agent: Sure, yes, I can help with account activation.",
        source_name="chat.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=2, seed=1)
    joined = " ".join(row["text"] for row in synthetic).lower()

    assert "account activation" in joined
    assert "okay yeah request" not in joined


def test_transcript_default_generation_is_not_self_repetitive():
    contract = build_transcript_contract(
        "Agent: Semantic understanding uses vectors\nCustomer: Please explain attention",
        source_name="chat.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=5, seed=1)
    report = validate_transcript_non_replay(contract, synthetic)

    assert report["passed"] is True
    assert report["max_self_similarity"] < report["variety_warning_threshold"]


def test_repeated_synthetic_phrasing_is_quality_failure_not_replay():
    contract = build_transcript_contract(
        "Agent: Please explain your request\nCustomer: I need support",
        source_name="chat.txt",
    )
    synthetic = [
        {"turn": 1, "speaker": "Customer", "text": "Thanks, that answers my question."},
        {"turn": 2, "speaker": "Agent", "text": "Thanks, that answers my question."},
    ]

    report = validate_transcript_non_replay(contract, synthetic)

    assert report["passed"] is False
    assert report["exact_replay_count"] == 0
    assert report["ngram_replay_count"] == 0
    assert report["max_self_similarity"] == 1
    assert report["repeated_phrase_warning"] is True


def test_transcript_validation_flags_privacy_like_generated_values():
    contract = build_transcript_contract(
        "Agent: Please explain your request\nCustomer: I need support",
        source_name="chat.txt",
    )
    report = validate_transcript_non_replay(
        contract,
        [{"turn": 1, "speaker": "Customer", "text": "Please call me at 555-123-4567 about account: ABCD-12345."}],
    )

    assert report["passed"] is False
    assert report["privacy_findings"] == [{"turn": 1, "types": ["identifier", "phone"]}]


def test_transcript_validation_flags_exact_source_replay_by_hash():
    contract = build_transcript_contract("Agent: Please explain attention", source_name="chat.txt")
    report = validate_transcript_non_replay(
        contract,
        [{"turn": 1, "speaker": "speaker_1", "text": "Please explain attention"}],
    )

    assert report["passed"] is False
    assert report["exact_replay_count"] == 1
