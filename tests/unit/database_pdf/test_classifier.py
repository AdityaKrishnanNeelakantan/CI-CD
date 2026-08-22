from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.classifier import STATUS_PROPOSED, STATUS_REVIEW_REQUIRED, classify_document

pytestmark = pytest.mark.unit


def test_invoice_is_confidently_classified():
    text = (
        "INVOICE\nInvoice Number: INV-1001\nBill To: Acme Corp\n"
        "Subtotal: $500.00\nTotal Due: $540.00\nPayment Terms: Net 30"
    )
    result = classify_document(text)
    assert result["document_type"] == "invoice"
    assert result["status"] == STATUS_PROPOSED


def test_contract_is_confidently_classified():
    text = (
        "This Agreement is entered into by and between the Parties. "
        "WHEREAS the Parties wish to define their rights hereby; "
        "NOW THEREFORE the Parties agree to the following Terms and Conditions, "
        "governed by the Governing Law of the State."
    )
    result = classify_document(text)
    assert result["document_type"] == "contract"
    assert result["status"] == STATUS_PROPOSED


def test_letter_is_confidently_classified():
    text = (
        "Dear Mr. Smith,\nI hope this letter finds you well. "
        "Thank you for your continued business.\nSincerely,\nJohn Doe"
    )
    result = classify_document(text)
    assert result["document_type"] == "letter"
    assert result["status"] == STATUS_PROPOSED


def test_report_is_confidently_classified():
    text = (
        "Executive Summary\nThis report presents our findings from the study. "
        "Introduction: background context. Methodology: survey based. "
        "Conclusion: our findings suggest improvement. Abstract: summary of results."
    )
    result = classify_document(text)
    assert result["document_type"] == "report"
    assert result["status"] == STATUS_PROPOSED


def test_resume_is_confidently_classified():
    text = (
        "Curriculum Vitae\nWork Experience: Software Engineer at TechCo\n"
        "Education: BS Computer Science\nSkills: Python, SQL\n"
        "References available upon request"
    )
    result = classify_document(text)
    assert result["document_type"] == "resume"
    assert result["status"] == STATUS_PROPOSED


def test_ambiguous_document_requires_review():
    text = "Dear Sir, please review the attached agreement."
    result = classify_document(text)
    assert result["status"] == STATUS_REVIEW_REQUIRED
    assert len(result["alternatives"]) >= 1


def test_no_signal_document_falls_back_to_other():
    text = "The quick brown fox jumps over the lazy dog near the river."
    result = classify_document(text)
    assert result["document_type"] == "other"
    assert result["status"] == STATUS_REVIEW_REQUIRED
    assert result["confidence"] == 0.0
    assert "no_signal" in result["evidence"]


def test_alternatives_are_reported():
    text = (
        "INVOICE\nInvoice Number: INV-1001\nBill To: Acme Corp\n"
        "Subtotal: $500.00\nTotal Due: $540.00\nPayment Terms: Net 30"
    )
    result = classify_document(text)
    for alt in result["alternatives"]:
        assert set(alt) == {"document_type", "confidence"}
