# Synth Platform

Synth Platform is a Python implementation of a Synthetic Data Twin platform for generating privacy-aware synthetic datasets and documents. Current working surfaces are the Schema Twin, SQLite-centered Database Twin, and PDF Twin workflows through Streamlit plus SDK/CLI paths; validation, packaging, read-only source access, privacy masking/redaction, and local artifact generation are real code. The major unfinished areas are controlled transfer/download gating across every UI path, strict air-gapped LLM policy, Customer Interaction Twin, the shared Governance Hub, Section B multi-agent service platform, deployable security/authorization plane, and several designed UI pages such as Settings and My Projects.

For the prioritized architecture gap analysis and starting plan, see [`docs/gap-analysis.md`](docs/gap-analysis.md). For architecture terms that map to existing implementation names, see [`docs/terminology-map.md`](docs/terminology-map.md).

## Architecture Summary

The code is organized as a layered package under `src/synth_platform`: `interfaces/` for Streamlit, CLI, SDK, and scaffolded API entry points; `application/` for workflow facades and use cases; `engine/` for discovery, profiling, inference, training, generation, validation, and document processing; `domain/` for privacy, validation, schema, artifact, relational, and policy concepts; and `infrastructure/` for sources, sinks, artifacts, local LLM adapter scaffolding, storage, observability, and persistence. See [`docs/architecture.md`](docs/architecture.md) and [`docs/code-map.md`](docs/code-map.md) for the internal architecture notes.

## Setup / Run Instructions

The project metadata declares Python 3.11+, setuptools, and the `synth-platform` / `synth-platform-ui` console scripts in `pyproject.toml`.

Install from source for local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[ui,database,pdf,schema,parquet]'
synth-platform-ui --server.port 8502
```

Then open:

```text
http://localhost:8502
```

Install the released wheel:

```bash
python -m venv .venv-release-ui
source .venv-release-ui/bin/activate
python -m pip install --upgrade pip
pip install 'synth-platform[ui,database,pdf] @ https://github.com/AdityaKrishnanNeelakantan/CI-CD/releases/download/v3.0.2/synth_platform-3.0.2-py3-none-any.whl'
synth-platform-ui --server.port 8502
```

CLI entry points visible from `interfaces/cli/app.py`:

```bash
synth-platform train --source SOURCE --artifact ARTIFACT_PATH
synth-platform generate --artifact ARTIFACT_PATH --out OUTPUT_PATH --rows '{"table": 100}' --format csv
```

Run tests:

```bash
pytest
```
