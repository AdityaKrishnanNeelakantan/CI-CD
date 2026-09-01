# Synthetic Data Twin Platform - Implemented Project Guide

## Overview

`synth_platform` is a unified synthetic-data platform with multiple SSOT input workflows. The product lifecycle is:

```text
Connect -> Understand -> Build twin -> Generate -> Validate -> Package
```

The implemented app supports schema-driven generation, database-trained twins, PDF/document twins, and customer interaction transcript twins. The Streamlit UI now reads user-facing page titles, navigation labels, icons, intent choices, extraction choices, model labels, and step guidance from `configs/streamlit_ui.yaml`.

## Implemented User Workflows

| Workflow | Source Input | Main Output | Implemented Path |
|---|---|---|---|
| Schema Mode | SQL DDL schema files | Synthetic relational tables plus validation package | `interfaces/streamlit/pages/schema_twin.py` -> `application/workflows/schema_twin.py` |
| Database Twin | SQLite database or generated sample database | Portable trained twin artifact, generated database, QA report | `interfaces/streamlit/pages/database_twin.py` -> `application/workflows/database_twin.py` |
| PDF Twin | PDF document | Synthetic twin PDF, validation report, optional redacted PDF | `interfaces/streamlit/pages/pdf_twin.py` -> `application/workflows/pdf_twin.py` |
| Customer Interactions Twin | Pasted or uploaded TXT/LOG conversation | Synthetic interaction ZIP with contract and non-replay validation | `interfaces/streamlit/pages/transcript_twin.py` -> `engine/transcripts/service.py` |

## Streamlit UI Configuration

User-facing UI configuration lives in:

```text
configs/streamlit_ui.yaml
```

The loader is implemented in:

```text
src/synth_platform/interfaces/streamlit/ui_config.py
```

You can override the config path with:

```bash
SP_STREAMLIT_UI_CONFIG=/path/to/streamlit_ui.yaml
```

Configurable UI areas include:

| Area | Examples |
|---|---|
| App shell | Browser title, page icon, layout, top navigation groups |
| Home page | Workflow cards, shared controls, technical details text |
| Schema Mode | Page title, caption, default intent, intent options, locales, export formats, step guidance |
| Database Twin | Page title, caption, default intent, source options, model labels, step guidance |
| PDF Twin | Page title, caption, default intent, extraction engines, step guidance |
| Transcript Twin | Page title, caption, step guidance |

The code has fallback defaults, so a partial YAML override keeps unspecified values from the built-in default config.

## Architecture

The repository follows a layered architecture:

```text
interfaces -> application -> engine -> domain
```

`infrastructure` provides external implementations such as source connectors, sinks, artifacts, PDF/OCR adapters, persistence, storage, jobs, LLM clients, and observability.

| Layer | Responsibility |
|---|---|
| `interfaces` | Streamlit UI, CLI, SDK, API entry points |
| `application` | Workflow facades, use cases, orchestration, DTOs, ports |
| `engine` | Discovery, profiling, inference, training, generation, validation, document processing |
| `domain` | Business models, contracts, privacy policy, constraints, relational concepts |
| `infrastructure` | Databases, file storage, sinks, PDF/OCR libraries, artifacts, integrations |

## Workflow Details

### Schema Mode

Schema Mode is source-free. It parses SQL DDL, reviews tables/columns/PK/FK relationships, configures row counts and seed, generates synthetic tables, validates structural integrity, and packages CSV or Parquet outputs.

Important implementation areas:

```text
src/synth_platform/application/workflows/schema_twin.py
src/synth_platform/application/orchestration/schema/
src/synth_platform/engine/inference/schema/
src/synth_platform/engine/generation/schema/
src/synth_platform/engine/validation/schema/
```

### Database Twin

Database Twin learns from an existing SQLite database or generated sample source. The flow discovers schema, profiles values, infers semantics, creates an approved dataset contract, trains per-table Gaussian-copula family model artifacts, disconnects the source, generates from the portable twin, validates QA/integrity, and writes a target SQLite database.

Important implementation areas:

```text
src/synth_platform/application/workflows/database_twin.py
src/synth_platform/engine/discovery/database/
src/synth_platform/engine/profiling/database/
src/synth_platform/engine/inference/database/
src/synth_platform/engine/training/database/
src/synth_platform/engine/generation/database/
src/synth_platform/engine/validation/database/
```

### PDF Twin

PDF Twin uploads a document, selects automatic native/OCR extraction or Docling, profiles the document, compiles a template, binds semantic fields, generates synthetic values, renders a twin PDF, and validates field/layout fidelity. Redaction is an optional sibling output.

Important implementation areas:

```text
src/synth_platform/application/workflows/pdf_twin.py
src/synth_platform/engine/documents/pdf/
src/synth_platform/infrastructure/documents/
```

### Customer Interactions Twin

Transcript Twin parses speaker-prefixed, paragraph, or JSON-like conversations. It sanitizes previews, builds a canonical contract without retaining raw source text, generates source-free synthetic turns, and validates against hashed source turn and n-gram evidence.

Important implementation areas:

```text
src/synth_platform/interfaces/streamlit/pages/transcript_twin.py
src/synth_platform/engine/transcripts/service.py
src/synth_platform/infrastructure/integrations/nvidia_nemo.py
```

## SDK Reference Guide

| Use Case Strategy | Best Open Source SDK | Core Class / Command |
|---|---|---|
| Conversational Transcripts | `nemo_curator` | `nc.synthetic.NemotronGenerator` |
| Document Schema / Recipes | `data-designer` | `dd.LLMTextColumnConfig` / `Judge` |
| Chain-of-Thought Data Twin | `synthetic-data-kit` | `sdg ingest` / `sdg create` |
| Visual / Structural Templates | `ydata-sdk` | `ydata.sdk.synthetic.DocumentGenerator` |

## Public Interfaces

| Interface | Status | Entry Point |
|---|---|---|
| Streamlit UI | Implemented for all four workflows | `streamlit_app.py`, `synth-platform-ui` |
| SDK | Implemented for database/PDF source tracks and shared runtime | `src/synth_platform/interfaces/sdk/client.py` |
| CLI | Implemented for train/generate flows | `synth-platform` |
| API | Scaffold only | `src/synth_platform/interfaces/api/app.py` |

## Run From Source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[ui,database,pdf,schema,parquet]'
python -m streamlit run streamlit_app.py --server.port 8502
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502
```

## Verification

Focused checks run after the UI config implementation:

```text
python -m compileall src\synth_platform\interfaces\streamlit
.\.venv\Scripts\python.exe -m pytest tests/unit/test_streamlit_ui_config.py tests/unit/database_pdf/test_streamlit_ux_helpers.py
.\.venv\Scripts\python.exe -m pytest tests/e2e/workflows/test_home_ui.py
.\.venv\Scripts\python.exe -m pytest tests/e2e/workflows/test_schema_twin_ui.py tests/e2e/workflows/test_pdf_twin_ui.py tests/e2e/workflows/test_transcript_twin_ui.py tests/e2e/workflows/test_database_twin_ui.py
```

Result:

```text
compileall passed
6 unit tests passed
home UI test passed
7 workflow UI tests passed, 1 skipped
```

## Notes For Future Work

- Add a dedicated `docs/workflows/transcript-twin.md` to match the Schema, Database, and PDF workflow docs.
- Expand `configs/streamlit_ui.yaml` if button labels, metric names, expander titles, or helper text also need to become no-code UI configuration.
- Keep business logic out of Streamlit pages; UI files should collect inputs, render state, and call application or engine facades.
