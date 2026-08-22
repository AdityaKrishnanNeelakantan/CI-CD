"""
Generators package for MVP.

Provides type-safe data generators for all supported column types.
"""

from synth_platform.engine.generation.schema.generators.base import (
    BaseGenerator,
    BooleanGenerator,
    CategoricalGenerator,
    DateGenerator,
    FloatGenerator,
    ForeignKeyGenerator,
    GeneratorFactory,
    IntegerGenerator,
    TextGenerator,
)

# Optional SDV-based generators (require: pip install sdv)
try:
    from synth_platform.engine.generation.schema.generators.copula import (
        CopulaGenerator,
        ConstraintAwareCopulaGenerator,
        create_copula_generator,
    )
    COPULA_AVAILABLE = True
except ImportError:
    COPULA_AVAILABLE = False
    CopulaGenerator = None
    ConstraintAwareCopulaGenerator = None
    create_copula_generator = None

__all__ = [
    "BaseGenerator",
    "GeneratorFactory",
    "IntegerGenerator",
    "FloatGenerator",
    "BooleanGenerator",
    "CategoricalGenerator",
    "DateGenerator",
    "TextGenerator",
    "ForeignKeyGenerator",
    # Optional SDV
    "CopulaGenerator",
    "ConstraintAwareCopulaGenerator",
    "create_copula_generator",
    "COPULA_AVAILABLE",
]
