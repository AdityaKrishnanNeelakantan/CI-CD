"""Runtime self-test for a loaded artifact.

Proves generation actually works from the artifact alone, immediately
after loading, rather than only claiming portability. See the loading
sequence's own self-test step: a "portable" artifact that has never
generated anything outside the training process is not evidence of
portability.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.training.database.artifact.loader import LoadedArtifact
from synth_platform.engine.training.database.base import SynthesisError


def run_self_test(loaded_artifact: LoadedArtifact) -> dict[str, Any]:
    table_results: dict[str, Any] = {}
    all_passed = True

    for table_name, spec in loaded_artifact.self_test_spec["tables"].items():
        try:
            # Trusted by construction: this runs in the same process right
            # after build_artifact(), before the artifact has left this
            # machine - see run_artifact_export() in src/artifact/service.py.
            model = loaded_artifact.get_model(table_name, allow_cloudpickle_models=True)
            generated = model.sample(spec["min_generate_rows"], seed=0)
            columns_match = set(generated.columns) == set(spec["expected_columns"])
            row_count_ok = len(generated) == spec["min_generate_rows"]
            passed = columns_match and row_count_ok
            table_results[table_name] = {
                "passed": passed,
                "generated_row_count": len(generated),
                "columns_match": columns_match,
            }
        except (SynthesisError, OSError) as exc:
            passed = False
            table_results[table_name] = {"passed": False, "error": str(exc)}
        all_passed = all_passed and passed

    return {"passed": all_passed, "tables": table_results}
