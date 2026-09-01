from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.shape import infer_shape_pattern

pytestmark = pytest.mark.unit


def test_digits_become_n():
    assert infer_shape_pattern("8823471") == "NNNNNNN"


def test_uppercase_letters_become_u():
    assert infer_shape_pattern("ABC") == "UUU"


def test_lowercase_letters_become_l():
    assert infer_shape_pattern("abc") == "lll"


def test_punctuation_and_whitespace_are_preserved_literally():
    assert infer_shape_pattern("RC-88291") == "UU-NNNNN"
    assert infer_shape_pattern("$1,845.20") == "$N,NNN.NN"


def test_mixed_case_name_preserves_case_shape():
    assert infer_shape_pattern("Grace Hopper") == "Ullll Ulllll"


def test_empty_string_produces_empty_pattern():
    assert infer_shape_pattern("") == ""


def test_ocr_noise_unicode_characters_are_replaced_with_a_space():
    """Regression test: OCR misread a smudge on a real scanned bank
    statement as U+2018 (a curly left-single-quote) in front of a table
    cell. That literal character survived into the shape pattern, and
    later crashed rendering entirely (fpdf2's default core font can't
    encode it) when the shape-filler replayed it verbatim as "structure".
    """
    pattern = infer_shape_pattern("‘TERMINAL 098765")
    assert "‘" not in pattern
    assert pattern[0] == " "


def test_pattern_never_contains_the_original_characters():
    """The whole point of a shape pattern: it must be safe to persist even
    though it's derived from the raw value - no digit or letter from the
    source can survive into the pattern.
    """
    value = "8823471"
    pattern = infer_shape_pattern(value)
    assert value not in pattern
    for digit in "0123456789":
        assert digit not in pattern
