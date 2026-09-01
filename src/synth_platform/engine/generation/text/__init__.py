"""Generic LLM-assisted text generation for text-heavy schema columns."""

from synth_platform.engine.generation.text.eligibility import (
    EligibilityResult,
    assess_column_eligibility,
    list_eligible_columns,
)
from synth_platform.engine.generation.text.evidence import TextGenerationEvidence, merge_evidence
from synth_platform.engine.generation.text.generator import (
    TextGenerationConfig,
    TextGenerationEngine,
    TextGenerationResult,
    generate_text_column,
)
from synth_platform.engine.generation.text.vocabulary import (
    DOMAIN_VOCABULARIES,
    get_domain_vocabulary,
    register_domain_vocabulary,
)

__all__ = [
    "DOMAIN_VOCABULARIES",
    "EligibilityResult",
    "TextGenerationConfig",
    "TextGenerationEngine",
    "TextGenerationEvidence",
    "TextGenerationResult",
    "assess_column_eligibility",
    "generate_text_column",
    "get_domain_vocabulary",
    "list_eligible_columns",
    "merge_evidence",
    "register_domain_vocabulary",
]
