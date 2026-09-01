from __future__ import annotations

import json
import re

from synth_platform.engine.transcripts import (
    build_transcript_contract,
    generate_synthetic_transcript,
    parse_transcript_text,
    summarize_transcript_preview,
    validate_transcript_non_replay,
)


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
    assert "auto claim" in synthetic[0]["text"]
    assert context["customer_name"] in synthetic_text
    assert context["policy_id"] in synthetic_text
    assert context["email"] in synthetic_text
    assert context["phone"] in synthetic_text
    assert context["date_of_birth"] in synthetic_text
    assert (context.get("claim_id") or context.get("tracking_id")) in synthetic_text
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
        "Customer",
        "Agent",
        "Customer",
        "Agent",
        "Customer",
        "Agent",
    ]
    assert synthetic[0]["text"].startswith("Hi, I need help")
    assert "help with" in synthetic[1]["text"]
    assert "claim" in " ".join(row["text"] for row in synthetic).lower()
    assert "You're welcome" not in synthetic[0]["text"]


def test_transcript_default_generation_is_not_self_repetitive():
    contract = build_transcript_contract(
        "Agent: Semantic understanding uses vectors\nCustomer: Please explain attention",
        source_name="chat.txt",
    )

    synthetic = generate_synthetic_transcript(contract, turn_count=5, seed=1)
    report = validate_transcript_non_replay(contract, synthetic)

    assert report["passed"] is True
    assert report["max_self_similarity"] < report["variety_warning_threshold"]


def test_repeated_synthetic_phrasing_is_variety_warning_not_replay_failure():
    contract = build_transcript_contract(
        "Agent: Please explain your request\nCustomer: I need support",
        source_name="chat.txt",
    )
    synthetic = [
        {"turn": 1, "speaker": "Customer", "text": "Thanks, that answers my question."},
        {"turn": 2, "speaker": "Agent", "text": "Thanks, that answers my question."},
    ]

    report = validate_transcript_non_replay(contract, synthetic)

    assert report["passed"] is True
    assert report["exact_replay_count"] == 0
    assert report["ngram_replay_count"] == 0
    assert report["max_self_similarity"] == 1
    assert report["repeated_phrase_warning"] is True


def test_transcript_validation_flags_exact_source_replay_by_hash():
    contract = build_transcript_contract("Agent: Please explain attention", source_name="chat.txt")
    report = validate_transcript_non_replay(
        contract,
        [{"turn": 1, "speaker": "speaker_1", "text": "Please explain attention"}],
    )

    assert report["passed"] is False
    assert report["exact_replay_count"] == 1
