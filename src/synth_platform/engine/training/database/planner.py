"""Advisory rules-based recommendation of a registered synthesizer for a
given dataset contract - never a hard gate. The training service and
synthesis registry (src/synthesis/registry.py) are completely unaware
this module exists; a caller (the UI, or a future automated runtime) may
take the recommendation, ignore it, or pass a different model_type to
create_synthesizer_adapter() entirely.

Rule (evidence-based, not learned): reuses SENSITIVE_SEMANTIC_TYPES, the
same set src/inference/contract.py already uses to compute each approved
column's "sensitive" flag. If any approved column in any table has a
sensitive semantic_type, the differentially private adapter is
recommended over the plain Gaussian-copula one - otherwise the plain
(higher-fidelity, JSON-safe) adapter is recommended.

The cloudpickle-based SDV adapter ("sdv_gaussian_copula") is deliberately
never auto-recommended: recommending a change in serialization/security
posture (src/artifact/manifest.py's allow_cloudpickle_models gate) is a
decision this planner leaves to an explicit, informed user choice.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.inference.database.contract import SENSITIVE_SEMANTIC_TYPES

RECOMMENDED_MODEL_TYPE_SENSITIVE = "dp_gaussian_copula"
RECOMMENDED_MODEL_TYPE_DEFAULT = "safe_gaussian_copula"


def recommend_synthesizer(contract: dict[str, Any]) -> dict[str, Any]:
    """Inspect an approved dataset contract's columns and recommend a
    model_type key registered in src/synthesis/registry.py.

    Returns {"recommended_model_type": str, "reasons": list[str],
    "sensitive_columns": list[str]} - "table.column" entries for every
    approved column whose semantic_type is in SENSITIVE_SEMANTIC_TYPES.
    """
    sensitive_columns: list[str] = []
    for table_name, table_contract in contract.get("tables", {}).items():
        for column_name, column in table_contract.get("columns", {}).items():
            if column.get("inference_status") != "approved":
                continue
            if column.get("semantic_type") in SENSITIVE_SEMANTIC_TYPES:
                sensitive_columns.append(f"{table_name}.{column_name}")

    if sensitive_columns:
        return {
            "recommended_model_type": RECOMMENDED_MODEL_TYPE_SENSITIVE,
            "reasons": [
                f"{len(sensitive_columns)} approved column(s) are flagged sensitive "
                "(identifier/email/person_name/phone_number) - recommending the "
                "differentially private adapter to bound re-identification risk."
            ],
            "sensitive_columns": sensitive_columns,
        }

    return {
        "recommended_model_type": RECOMMENDED_MODEL_TYPE_DEFAULT,
        "reasons": [
            "No approved column is flagged sensitive - recommending the "
            "higher-fidelity, JSON-safe Gaussian-copula adapter."
        ],
        "sensitive_columns": [],
    }
