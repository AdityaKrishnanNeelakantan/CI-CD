from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.pii import (
    PII_TYPE_CREDIT_CARD,
    PII_TYPE_EMAIL,
    PII_TYPE_IBAN,
    PII_TYPE_IP_ADDRESS,
    PII_TYPE_PHONE,
    PII_TYPE_ROUTING_NUMBER,
    PII_TYPE_SSN,
    PII_TYPE_STREET_ADDRESS,
    PII_TYPE_URL,
    detect_pii,
)

pytestmark = pytest.mark.unit


def test_email_is_found_with_exact_offsets():
    text = "Contact Ada at ada.lovelace@example.com for details."
    findings = detect_pii(text)
    email_findings = [f for f in findings if f["type"] == PII_TYPE_EMAIL]
    assert len(email_findings) == 1
    f = email_findings[0]
    assert text[f["start"] : f["end"]] == "ada.lovelace@example.com"


def test_url_is_found_with_exact_offsets():
    text = "Visit https://example.com/docs for more info."
    findings = detect_pii(text)
    url_findings = [f for f in findings if f["type"] == PII_TYPE_URL]
    assert len(url_findings) == 1
    f = url_findings[0]
    assert text[f["start"] : f["end"]] == "https://example.com/docs"


def test_ssn_is_found_with_exact_offsets():
    text = "SSN on file: 123-45-6789 for verification."
    findings = detect_pii(text)
    ssn_findings = [f for f in findings if f["type"] == PII_TYPE_SSN]
    assert len(ssn_findings) == 1
    assert text[ssn_findings[0]["start"] : ssn_findings[0]["end"]] == "123-45-6789"


def test_credit_card_is_found_with_exact_offsets():
    text = "Card number 4111-1111-1111-1111 was charged."
    findings = detect_pii(text)
    cc_findings = [f for f in findings if f["type"] == PII_TYPE_CREDIT_CARD]
    assert len(cc_findings) == 1
    assert text[cc_findings[0]["start"] : cc_findings[0]["end"]] == "4111-1111-1111-1111"


def test_ip_address_is_found_with_exact_offsets():
    text = "Server responded from 192.168.1.42 during the outage."
    findings = detect_pii(text)
    ip_findings = [f for f in findings if f["type"] == PII_TYPE_IP_ADDRESS]
    assert len(ip_findings) == 1
    assert text[ip_findings[0]["start"] : ip_findings[0]["end"]] == "192.168.1.42"


def test_phone_number_is_found_with_exact_offsets():
    text = "Call us at (555) 123-4567 during business hours."
    findings = detect_pii(text)
    phone_findings = [f for f in findings if f["type"] == PII_TYPE_PHONE]
    assert len(phone_findings) == 1
    assert text[phone_findings[0]["start"] : phone_findings[0]["end"]] == "(555) 123-4567"


def test_street_address_is_found_with_exact_offsets():
    text = "Please mail the form to 1765 Sheridan Drive by Friday."
    findings = detect_pii(text)
    address_findings = [f for f in findings if f["type"] == PII_TYPE_STREET_ADDRESS]
    assert len(address_findings) == 1
    assert text[address_findings[0]["start"] : address_findings[0]["end"]] == "1765 Sheridan Drive"


def test_all_caps_street_address_is_found():
    """Regression test: a real scanned bank statement's letterhead printed
    its address in all caps ("1765 SHERIDAN DRIVE") - caught via smoke
    testing against actual data, not a hand-written fixture.
    """
    text = "JAMES C. MORRISON 1765 SHERIDAN DRIVE YOUR CITY, USA 03087"
    findings = detect_pii(text)
    address_findings = [f for f in findings if f["type"] == PII_TYPE_STREET_ADDRESS]
    assert len(address_findings) == 1
    assert text[address_findings[0]["start"] : address_findings[0]["end"]] == "1765 SHERIDAN DRIVE"


def test_multiple_pii_types_in_one_document_all_found():
    text = (
        "Contact Ada Lovelace at ada@example.com or (555) 123-4567.\n"
        "SSN: 123-45-6789. Visit https://example.com for details."
    )
    findings = detect_pii(text)
    found_types = {f["type"] for f in findings}
    assert found_types == {
        PII_TYPE_EMAIL,
        PII_TYPE_PHONE,
        PII_TYPE_SSN,
        PII_TYPE_URL,
    }
    # Findings must be sorted by position in the document.
    starts = [f["start"] for f in findings]
    assert starts == sorted(starts)


def test_findings_never_contain_raw_matched_text():
    text = "Email me at secret.person@example.com right away."
    findings = detect_pii(text)
    for finding in findings:
        assert set(finding) == {"type", "start", "end", "confidence", "redaction_preview"}
    serialized = str(findings)
    assert "secret.person@example.com" not in serialized


def test_redaction_preview_is_masked_not_raw():
    text = "Reach me at ada@example.com."
    findings = detect_pii(text)
    email_finding = next(f for f in findings if f["type"] == PII_TYPE_EMAIL)
    assert email_finding["redaction_preview"] != "ada@example.com"
    assert "*" in email_finding["redaction_preview"]


def test_credit_card_with_valid_luhn_checksum_gets_boosted_confidence():
    """4111-1111-1111-1111 is the standard Visa test number and is Luhn-valid
    - the checksum should raise confidence above the pattern's 0.6 base."""
    text = "Card number 4111-1111-1111-1111 was charged."
    findings = detect_pii(text)
    cc_finding = next(f for f in findings if f["type"] == PII_TYPE_CREDIT_CARD)
    assert cc_finding["confidence"] > 0.6


def test_credit_card_with_invalid_luhn_checksum_is_still_detected_at_base_confidence():
    """A checksum failure must never reduce recall - a card-shaped number
    is exactly as sensitive to leave unmasked whether or not it passes
    Luhn, so it is still found and redacted, just without the boost."""
    text = "Card number 4111-1111-1111-1112 was charged."
    findings = detect_pii(text)
    cc_finding = next(f for f in findings if f["type"] == PII_TYPE_CREDIT_CARD)
    assert text[cc_finding["start"] : cc_finding["end"]] == "4111-1111-1111-1112"
    assert cc_finding["confidence"] == 0.6


def test_iban_is_found_with_exact_offsets_and_boosted_confidence():
    # GB29 NWBK 6016 1331 9268 19 is the well-known valid IBAN test vector.
    text = "Wire to IBAN GB29NWBK60161331926819 before Friday."
    findings = detect_pii(text)
    iban_findings = [f for f in findings if f["type"] == PII_TYPE_IBAN]
    assert len(iban_findings) == 1
    f = iban_findings[0]
    assert text[f["start"] : f["end"]] == "GB29NWBK60161331926819"
    assert f["confidence"] > 0.7


def test_iban_with_invalid_checksum_is_still_detected_at_base_confidence():
    text = "Wire to IBAN GB29NWBK60161331926810 before Friday."
    findings = detect_pii(text)
    iban_finding = next(f for f in findings if f["type"] == PII_TYPE_IBAN)
    assert iban_finding["confidence"] == 0.7


def test_routing_number_with_valid_aba_checksum_is_found():
    # 111000025 is a real, published Bank of America ABA routing number.
    text = "Routing number 111000025 for domestic wires."
    findings = detect_pii(text)
    routing_findings = [f for f in findings if f["type"] == PII_TYPE_ROUTING_NUMBER]
    assert len(routing_findings) == 1
    assert text[routing_findings[0]["start"] : routing_findings[0]["end"]] == "111000025"


def test_random_9_digit_number_without_valid_checksum_is_not_flagged_as_routing_number():
    """Unlike credit_card/iban, routing_number's checksum is a required
    gate: without it, a bare 9-digit regex would fire on almost any
    9-digit value (reference numbers, zip+4, etc)."""
    text = "Reference number 123456789 was logged."
    findings = detect_pii(text)
    assert [f["type"] for f in findings if f["type"] == PII_TYPE_ROUTING_NUMBER] == []


def test_no_false_positive_pii_in_plain_text():
    text = "The quick brown fox jumps over the lazy dog near the old oak tree."
    assert detect_pii(text) == []


def test_empty_text_returns_no_findings():
    assert detect_pii("") == []
