"""Text statistics: the profiling equivalent for unstructured documents.

Purely descriptive - counts and ratios computed directly from the
normalised text, no interpretation of meaning. Mirrors the separation of
concerns in src/profiling/profiler.py: statistics are measurable evidence,
not semantic classification.
"""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


def compute_text_statistics(text: str) -> dict[str, Any]:
    char_count = len(text)
    words = _WORD_RE.findall(text)
    word_count = len(words)

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    sentence_count = len(sentences)

    line_count = text.count("\n") + 1 if text else 0

    distinct_words = {w.lower() for w in words}

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "line_count": line_count,
        "avg_word_length": round(sum(len(w) for w in words) / word_count, 4) if word_count else 0.0,
        "avg_sentence_length_words": round(word_count / sentence_count, 4) if sentence_count else 0.0,
        "distinct_word_count": len(distinct_words),
        "distinct_word_ratio": round(len(distinct_words) / word_count, 6) if word_count else 0.0,
    }
