"""Map a column profile to a privacy action (pure policy rules).

Corrects the legacy over-aggression: safe public business enums are PRESERVED
(not turned into CATEGORY_001); only genuine identifiers are transformed and
only genuinely risky rare values are bucketed/suppressed.
"""
from __future__ import annotations

from synth_platform.domain.privacy.models import ColumnPolicy, PrivacyAction
from synth_platform.domain.profiling.models import ColumnProfile, SensitivityClass


def classify(profile: ColumnProfile, rare_threshold: int = 10) -> ColumnPolicy:
    s = profile.sensitivity
    if s == SensitivityClass.DIRECT_IDENTIFIER:
        action = PrivacyAction.TRANSFORM
        reason = "direct identifier: synthesize, never store raw values"
    elif s == SensitivityClass.FREE_TEXT:
        action = PrivacyAction.EXCLUDE
        reason = "free text excluded pending a separately governed model"
    elif s == SensitivityClass.QUASI_IDENTIFIER:
        # A quasi-identifier is protected regardless of cardinality. Its raw
        # labels never survive: they are replaced with stable placeholders
        # (distribution kept, meaning removed). Higher cardinality = MORE
        # identifying, so "preserve when many distinct values" would be exactly
        # backwards. Very low cardinality is suppressed outright.
        if 0 < profile.cardinality <= rare_threshold:
            action = PrivacyAction.SUPPRESS
            reason = "rare quasi-identifier values suppressed (few distinct)"
        else:
            action = PrivacyAction.BUCKET
            reason = "quasi-identifier labels bucketed to placeholders"
    else:
        action = PrivacyAction.PRESERVE
        reason = "public business value preserved (sufficiently supported)"
    return ColumnPolicy(table=profile.table_name, column=profile.column_name,
                        action=action, reason=reason)
