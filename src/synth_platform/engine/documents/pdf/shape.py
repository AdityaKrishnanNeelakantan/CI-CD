"""Character-class shape patterns: a structural fingerprint of a value's
format (letter case, digit positions, punctuation) with zero characters
of the actual value in it - safe to persist in a portable template
artifact that must never carry source content, while still giving a
later generation stage enough to synthesize a same-shaped replacement
(e.g. "RC-88291" -> "UU-NNNNN" -> a new value can be composed back into
that exact shape without ever having seen the original).
"""

from __future__ import annotations

# Punctuation/whitespace is kept literally (it's genuine format structure:
# "-" in "RC-88291", ":" in a time, "." in a decimal). Anything outside
# this + plain ASCII space is replaced with a space instead - most
# commonly OCR misreads a smudge as a stray curly quote, em dash, or
# other non-ASCII glyph (e.g. "‘" in front of a real scanned bank
# statement's table cell) rather than a shape a synthetic replacement
# should ever try to reproduce, and some of those aren't even encodable
# by a standard PDF core font at render time.
_SAFE_LITERAL_CHARS = frozenset(" -_/.,:;()#&$%'\"@+*")


def infer_shape_pattern(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isupper():
            chars.append("U")
        elif ch.islower():
            chars.append("l")
        elif ch.isdigit():
            chars.append("N")
        elif ch in _SAFE_LITERAL_CHARS:
            chars.append(ch)
        else:
            chars.append(" ")
    return "".join(chars)
