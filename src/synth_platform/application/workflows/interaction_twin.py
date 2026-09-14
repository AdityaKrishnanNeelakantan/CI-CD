"""Customer Interaction Twin application workflow facade."""

from __future__ import annotations

from pathlib import Path

from synth_platform.engine.generation.interaction import (
    InteractionGenerationConfig,
    InteractionTwinResult,
    Transcript,
    TranscriptTurn,
    generate_interaction_twin,
    parse_transcript,
    redact_sensitive_text,
)
from synth_platform.engine.generation.interaction.service import new_run_id
from synth_platform.domain.product_settings import (
    ProductSettingsReader,
    privacy_level_removes_sensitive_information,
    read_generation_defaults,
)

WORKFLOW_STAGES = (
    "upload",
    "configure",
    "generation",
    "validation",
    "export",
)


def run_interaction_twin(
    transcript_text: str,
    *,
    interaction_type: str = "Customer Support",
    output_format: str | None = None,
    remove_sensitive_information: bool | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    project_name: str | None = None,
    history=None,
    product_settings: ProductSettingsReader | None = None,
) -> InteractionTwinResult:
    defaults = read_generation_defaults(product_settings)
    resolved_output_format = output_format or "Structured JSON + Synthetic Logs"
    if defaults["default_output_format"] in {"json", "log"}:
        resolved_output_format = "Structured JSON + Synthetic Logs"
    resolved_remove_sensitive = (
        bool(remove_sensitive_information)
        if remove_sensitive_information is not None
        else privacy_level_removes_sensitive_information(str(defaults["privacy_level"]))
    )
    config = InteractionGenerationConfig(
        interaction_type=interaction_type,
        output_format=resolved_output_format,
        remove_sensitive_information=resolved_remove_sensitive,
        seed=seed,
    )
    result = generate_interaction_twin(transcript_text, config, output_dir=output_dir)
    if history is not None:
        try:
            output_id = "interaction_twin.zip"
            passed = result.hard_checks_passed
            history.record_run(
                workflow_type="interaction",
                project_name=project_name or "Customer Interaction Twin",
                status="completed" if passed else "failed",
                validation_status="PASS" if passed else "FAIL",
                validation_passed=passed,
                output_id=output_id,
                run_id=new_run_id(),
                metadata={
                    "interaction_type": interaction_type,
                    "output_format": resolved_output_format,
                    "remove_sensitive_information": resolved_remove_sensitive,
                    "turn_count": len(result.synthetic.turns),
                    "redaction_findings": result.redaction_report.get("finding_count", 0),
                },
            )
        except Exception:
            pass
    return result


__all__ = [
    "InteractionGenerationConfig",
    "InteractionTwinResult",
    "Transcript",
    "TranscriptTurn",
    "WORKFLOW_STAGES",
    "parse_transcript",
    "redact_sensitive_text",
    "run_interaction_twin",
]
