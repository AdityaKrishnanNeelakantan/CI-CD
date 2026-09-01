from synth_platform.domain.privacy.classification import classify
from synth_platform.domain.privacy.models import PrivacyAction
from synth_platform.domain.profiling.models import (
    CategoricalStatistics, ColumnProfile, SensitivityClass,
)


def _col(sem, sens, card=3):
    return ColumnProfile(table_name="accounts", column_name="status", physical_type="categorical",
                         semantic_type=sem, nullable=False, cardinality=card, missing_rate=0.0,
                         sensitivity=sens,
                         categorical=CategoricalStatistics(values=["active", "closed", "premium"],
                                                           probabilities=[0.6, 0.3, 0.1]))


def test_public_business_enum_preserved():
    assert classify(_col("category", SensitivityClass.PUBLIC)).action == PrivacyAction.PRESERVE


def test_direct_identifier_transformed():
    c = _col("email", SensitivityClass.DIRECT_IDENTIFIER)
    assert classify(c).action == PrivacyAction.TRANSFORM
