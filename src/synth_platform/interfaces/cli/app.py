"""CLI entry point: train and release-gated source-free generation."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from synth_platform.infrastructure.sinks.registry import build_output_sink
from synth_platform.application.dto.commands import GenerationRequest, TrainingRequest
from synth_platform.application.use_cases.publish_dataset import publish_dataset
from synth_platform.interfaces.sdk.client import SyntheticDataPlatform


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="synth-platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--source", required=True)
    train_parser.add_argument("--artifact", required=True)
    train_parser.add_argument("--sample-size", type=int, default=50_000)
    train_parser.add_argument("--seed", type=int, default=42)

    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--artifact", required=True)
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--rows", default="{}", help='JSON: {"table": n}')
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument(
        "--format", choices=["csv", "sqlite", "parquet"], default="csv")

    args = parser.parse_args(argv)
    platform = SyntheticDataPlatform.from_settings()

    if args.cmd == "train":
        artifact = platform.train(TrainingRequest(
            source=args.source, sample_size=args.sample_size, seed=args.seed))
        platform.export(artifact, args.artifact)
        print(f"trained artifact_id={artifact.manifest.artifact_id[:12]} -> {args.artifact}")
        return 0

    if args.cmd == "generate":
        artifact = platform.load(args.artifact)
        request = GenerationRequest(
            root_table_rows=json.loads(args.rows), seed=args.seed,
            output_format=args.format)
        tables = platform.generate(artifact, request)
        report = platform.validate(artifact, tables)
        sink = build_output_sink(args.format, args.out)
        receipt = publish_dataset(
            artifact, tables, report, sink, dataset_id=uuid.uuid4().hex)
        report_path = (
            Path(args.out) / "validation_report.json"
            if args.format in {"csv", "parquet"}
            else Path(str(args.out) + ".validation_report.json")
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(indent=2))
        print(
            f"generated and release-gated {len(tables)} tables -> {args.out} "
            f"(overall={report.overall.value}, dataset_id={receipt['dataset_id']})"
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
