from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.text_stats import compute_text_statistics

pytestmark = pytest.mark.unit


def test_stats_against_hand_calculated_values():
    text = "Hello world. This is a test!"
    stats = compute_text_statistics(text)

    assert stats["char_count"] == len(text)
    assert stats["word_count"] == 6  # Hello, world, This, is, a, test
    assert stats["sentence_count"] == 2  # "Hello world" / "This is a test"
    assert stats["avg_word_length"] == pytest.approx((5 + 5 + 4 + 2 + 1 + 4) / 6)
    assert stats["avg_sentence_length_words"] == pytest.approx(3.0)
    assert stats["distinct_word_count"] == 6
    assert stats["distinct_word_ratio"] == pytest.approx(1.0)


def test_repeated_words_lower_distinct_ratio():
    text = "test test test test."
    stats = compute_text_statistics(text)
    assert stats["word_count"] == 4
    assert stats["distinct_word_count"] == 1
    assert stats["distinct_word_ratio"] == pytest.approx(0.25)


def test_line_count():
    text = "line one\nline two\nline three"
    stats = compute_text_statistics(text)
    assert stats["line_count"] == 3


def test_empty_text_produces_zeroed_stats():
    stats = compute_text_statistics("")
    assert stats["char_count"] == 0
    assert stats["word_count"] == 0
    assert stats["sentence_count"] == 0
    assert stats["line_count"] == 0
    assert stats["avg_word_length"] == 0.0
    assert stats["avg_sentence_length_words"] == 0.0
    assert stats["distinct_word_ratio"] == 0.0
