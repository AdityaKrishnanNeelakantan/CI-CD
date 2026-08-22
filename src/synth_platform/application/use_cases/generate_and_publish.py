"""Generate, validate, resolve output format, and publish through one gate."""
from __future__ import annotations

from synth_platform.application.use_cases.publish_dataset import publish_dataset


def generate_validate_publish(
    artifact,
    request,
    *,
    generator,
    validator,
    sink_factory,
    destination: str,
    dataset_id: str,
    sink_options: dict | None = None,
):
    tables = generator.run(artifact, request)
    report = validator(artifact, tables)
    sink = sink_factory(request.output_format, destination, **(sink_options or {}))
    receipt = publish_dataset(artifact, tables, report, sink, dataset_id)
    return {"tables": tables, "report": report, "receipt": receipt}
