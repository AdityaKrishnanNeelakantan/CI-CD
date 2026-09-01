"""Assemble a PrivacyPolicy from a DataProfile (pure)."""
from __future__ import annotations

from synth_platform.domain.privacy.classification import classify
from synth_platform.domain.privacy.models import PrivacyPolicy
from synth_platform.domain.profiling.models import DataProfile


def build_policy(profile: DataProfile, rare_threshold: int = 10) -> PrivacyPolicy:
    cols = []
    for tprof in profile.tables.values():
        for cprof in tprof.columns.values():
            cols.append(classify(cprof, rare_threshold))
    return PrivacyPolicy(rare_category_threshold=rare_threshold, columns=cols, formal_dp=False)
