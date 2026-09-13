# Synthetic Data Twin — System Architecture

This document is the grounded architecture of **Synthetic Data Twin** (`synth-platform` / package `synth_platform`). It describes how the repository is structured, how each product workflow runs, and how synthetic data is actually generated. Claims below are taken from the running code, not from a desired future design.

Companion workflow notes:

- [`docs/workflows/schema-twin.md`](workflows/schema-twin.md)
- [`docs/workflows/database-twin.md`](workflows/database-twin.md)
- [`docs/workflows/pdf-twin.md`](workflows/pdf-twin.md)
- [`docs/privacy-model.md`](privacy-model.md)
- [`docs/validation-metrics.md`](validation-metrics.md)
- [`docs/docling-integration.md`](docling-integration.md)
- [`docs/ci.md`](ci.md)

Package version: `3.0.2` (`pyproject.toml`). Python `>=3.11`.

---

## 1. What the product is

Synthetic Data Twin is a **local-first** platform that produces privacy-aware synthetic **tables**, **relational databases**, **PDFs**, and **customer-interaction transcripts**.

It is not a hosted multi-tenant SaaS. There is no production authentication, RBAC, external queue, cloud object store, or deployment hardening in the current tree. Jobs execute in-process. Runtime state lives on the local filesystem and a local SQLite product database.

### 1.1 Four product workflows

| Product name | Workflow type | Primary input | Generation style | Training stage |
|---|---|---|---|---|
| **Schema Twin** | `schema_twin` | JSON / YAML / SQL schema, or a built-in template | Source-free, schema-driven simulation | None. There is no source dataset to fit. |
| **Database Twin** | `database_twin` | SQLite `.db` / `.sqlite` / `.sqlite3` | Source-driven: profile, infer, train, generate, QA, write Database B | Yes. Per-table synthesizers. |
| **Document Twin** (PDF Twin) | `pdf_twin` | PDF | Template + semantic binding + synthetic values + render | No statistical model fit. Optional de-identification is a sibling path. |
| **Customer Interaction Twin** | `interaction_twin` | `.txt` / `.log` transcript | Parse turns, redact, rewrite synthetic conversation | No statistical model fit. |

### 1.2 Two generation lineages in the same package

The package contains **two** generation lineages that share domain/privacy ideas but are not the same runtime path:

1. **Product / UI lineage (what FastAPI + React + Streamlit Database/PDF pages use)**  
   Stage services under `engine/*/database` and `engine/documents/pdf`, coordinated by `application/workflows/*` and `WorkflowOrchestrator`. Database Twin trains **Gaussian copula** adapters (`safe_gaussian_copula` by default). Schema Twin uses `DataSimulator`.

2. **SDK / CLI lineage (`synth-platform train|generate`)**  
   `interfaces/sdk/client.py` → `SyntheticDataPlatform` → use cases such as `train_model`, `generate_dataset`, `publish_dataset`. This path materializes a `RelationalDataset`, compiles a learning plan, builds a signed `SynthArtifact` (`.synthpkg`), and generates through `GenerationService`. It can open SQLite **and** PostgreSQL connectors. The React/FastAPI Database Twin **does not** currently use this path for user uploads.

When reading the code, treat these as parallel implementations. The UI Database Twin is the checkpoint pipeline (`discovery.json` → `target_write_report.json`). The CLI is the artifact-bundle compiler.

---

## 2. Layered architecture

The canonical package is `src/synth_platform/`. Layers are physical packages. Dependency direction is enforced by an AST scan in `tests/contract/architecture/test_layering.py`.

```text
                         +------------------+
                         |    interfaces    |   FastAPI, React is separate (web/),
                         |                  |   Streamlit, CLI, SDK
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |   application    |   workflows, use cases, orchestration, ports, DTO
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |      engine      |   discovery, profiling, inference, training,
                         |                  |   generation, validation, documents
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |      domain      |   schemas, privacy, constraints, artifacts,
                         |                  |   relational DAG, validation policy
                         +------------------+

infrastructure implements ports / external mechanisms and may depend inward.
domain, engine, and application must not import infrastructure or interfaces.
domain must not import pandas, numpy, sqlalchemy, streamlit, fastapi, sqlite3,
pdfplumber, safetensors, or celery.
```

### 2.1 `interfaces/`

Entry points only. They translate user input into application calls and render results. They must not own statistical, training, generation, or validation algorithms.

| Surface | Location | Role today |
|---|---|---|
| FastAPI | `interfaces/api/` | Primary product API for the React app |
| React SPA | `web/` | Primary UI |
| Streamlit | `interfaces/streamlit/` | Legacy/local UI; Database Twin and PDF Twin pages still call the same workflow facades |
| CLI | `interfaces/cli/app.py` | `synth-platform train` / `generate` on the SDK lineage |
| SDK | `interfaces/sdk/client.py` | `SyntheticDataPlatform` |

Console scripts (`pyproject.toml`):

- `synth-platform` → CLI
- `synth-platform-api` → FastAPI launcher
- `synth-platform-ui` → Streamlit launcher

### 2.2 `application/`

Coordinates product behavior.

- `workflows/` — one stable facade per product workflow (`schema_twin`, `database_twin`, `pdf_twin`, `interaction_twin`). UI and API import these, not individual engine internals when the facade exists.
- `use_cases/` — reusable operations for the SDK lineage (`train_from_database`, `generate_from_artifact`, `validate_dataset`, `publish_dataset`, PDF train/render, and others).
- `orchestration/` — schema pipeline sequencing (`run_schema_pipeline`), checkpointing, state machine.
- `ports/` — abstractions infrastructure can implement (`SourceConnector`, `JobQueue`, `ArtifactStore`, `ChatModel`, `IntentClassifier`, `OutputSink`, `ReleaseGate`, and others).
- `dto/` — boundary objects (`GenerationRequest`, `TrainingRequest`, results).
- `services/transfer_service.py` — **validation-gated download**. Bytes are not handed to the user unless the validation report carries an explicit PASS signal.

### 2.3 `engine/`

Reusable executable capabilities, grouped by **stage**, then by **track**:

```text
engine/
  discovery/database/     SQLite SourceAdapter, discovery.json
  profiling/database/     StructuredProfiler, profile.json, cleaning
  inference/
    schema/               SchemaConfig, YAML/JSON/SQL parse, GenerationPlanner
    database/             semantic candidates + dataset contract
    platform/             SDK-lineage inference (cross-table conditionals)
  training/database/      synthesizer adapters, training_report.json, artifact zip
  generation/
    schema/               DataSimulator, copula/text/locales/templates
    database/             relational generator, target writer
    text/                 LLM + Faker fallback
    interaction/          transcript twin
  validation/
    schema/               schema/type/rule/privacy/export validation
    database/             QA report + release manager
  documents/pdf/          extraction, template, bind, generate, render, validate
  common/database/        RunManifest, WorkflowOrchestrator, privacy primitives
```

The Database/PDF lineage keeps `engine/*/database` and `engine/documents/pdf` so algorithms are not copied into each UI page.

### 2.4 `domain/`

Business concepts and policies, kept free of dataframe, database, UI, PDF-library, and queue dependencies:

- schema and compatibility
- constraints (compile, evaluate, repair)
- privacy (classification, policies, redaction, LLM egress policy, verifier)
- relational structure (DAG, cycles, cardinality, key allocation)
- profiling / generation / training / validation models
- artifacts (manifest, checksums, bundle, version guards)
- document IR and extraction quality
- product settings (`generation_mode`, `default_record_count`, `privacy_level`, `default_output_format`)

`PrivacyPolicy.formal_dp` defaults to `False`. A DP claim is only valid when the DP adapter and accountant actually ran.

### 2.5 `infrastructure/`

External implementations:

| Area | Examples | Wired into product UI? |
|---|---|---|
| Persistence | `PlatformDB` (projects/runs/settings), metadata repository | Yes, FastAPI projects/settings |
| Jobs | `LocalJobRunner` (daemon thread); optional Celery module exists | FastAPI uses **inline/threaded** runner only |
| API state | `ApiStateStore` JSON/files under `.staging/api` | Yes |
| Sources | SQLite, Postgres, files, SQLAlchemy base | UI Database Twin: **SQLite only**. SDK CLI: SQLite + Postgres |
| Sinks | CSV, Parquet, SQLite/database, Postgres, staging | CLI publish path; Schema Twin CSV/Parquet export |
| Model backends | statistical, random forest, neural | SDK/learning-plan lineage |
| Documents | PDF loader, OCR, ReportLab renderer | SDK PDF path; product PDF Twin uses `engine/documents/pdf` |
| LLM | Ollama chat client | Schema/PDF text generation via `engine/generation/text` |
| Intent | rule-based + LLM classifiers | `/api` intent router |
| Artifacts | signing, verifier, package writer/reader, tensor store | SDK `.synthpkg` path |
| Storage | local, S3, MinIO stubs/adapters | local is the default |

---

## 3. Runtime topology

### 3.1 Local development

```text
Browser  http://127.0.0.1:5173
   │
   │  Vite proxies /api/* → http://127.0.0.1:8000
   v
React SPA (web/)
   │
   v
FastAPI  synth_platform.interfaces.api.app:app
   │
   ├── ApiStateStore     SP_API_STATE_DIR or .staging/api
   ├── PlatformDB        SP_PLATFORM_DB_PATH or .staging/platform.db
   ├── LocalJobRunner    daemon thread per job
   └── application/workflows + engine
            │
            ├── optional local Ollama  http://localhost:11434  model qwen3.5:9b
            └── optional Docling (pdf extra) for Document Twin
```

CORS allows localhost/127.0.0.1 on ports `3000` and `5173`.

### 3.2 Backend Docker

`Dockerfile` builds a Python 3.11 image, `uv sync --locked --extra api --extra schema --extra parquet`, and runs uvicorn on `:8000`. Frontend is **not** in the image.

Default container paths:

- `SP_API_STATE_DIR=/app/.data/api`
- `SP_PLATFORM_DB_PATH=/app/.data/platform.db`
- `SP_STAGING_ROOT=/app/.data/staging`

`docker-compose.yml` exposes the API and bind-mounts `./.data`.

### 3.3 Job execution model

`LocalJobRunner` creates a job record, then starts a **daemon thread** with the handler (`threaded=True` by default). The React UI polls `GET /api/jobs/{job_id}`. Statuses: `queued | running | succeeded | failed`.

There is no Redis, Celery worker, or durable queue in the FastAPI path. A process restart loses in-flight threads; job JSON on disk may remain `running`.

Typical job kinds:

- `schema.generate`
- `database.generate`
- `document.generate`
- `interaction.generate`

Progress stages the UI timeline expects: analyzing input → learning patterns → generating → validating → privacy checks (where applicable) → preparing files.

---

## 4. Persistence, sessions, and transfer

### 4.1 API state store

`interfaces/api/store.py` persists:

```text
<SP_API_STATE_DIR>/
  sessions/          workflow session JSON
  jobs/              job status JSON
  downloads/         download records (path + validation_report)
  blobs/             uploaded files
  results/           per-session result payloads
  result_bundles/    ResultBundle JSON (preview, quality_report, artifacts)
  schema_outputs/, document_outputs/, interaction_outputs/, ...
```

A **session** is the unit of a user run (intent, uploaded source, config, stage). A **job** is an async generation. A **result bundle** is the inspectable/downloadable outcome. Saving to **My Projects** copies metadata into `PlatformDB`.

### 4.2 Platform database

`infrastructure/persistence/platform_db.py` is a local SQLite catalog of:

- projects (`workflow_type` ∈ `schema | database | pdf | interaction`)
- runs (status, validation, transfer)
- product settings

It is **not** the synthetic target database. Database Twin writes synthetic Database B as a separate `.db` file under the job workdir.

### 4.3 Transfer gate

`TransferService` is the single gate for user-facing downloads on `/api/downloads/{id}`. It refuses to infer success from a missing or vaguely truthy report. Failed required checks must not be presented as a successful release. Blocked transfers raise `TransferBlockedError` → HTTP 403 `transfer_blocked`.

Result-artifact downloads via `/api/results/{id}/artifacts/{artifact_id}/download` stream files that the job already wrote; Schema Twin zip creation still records a validation report on the download record.

---

## 5. How a request becomes synthetic data

Shared API pattern for every workflow:

```text
POST /api/<workflow>/sessions
        create session
POST .../upload or /schema-file or /source
        persist blob, parse/validate input, store summary in session.state
POST .../configure
        persist generation controls
POST .../generate          → 202 { job }
        LocalJobRunner runs engine
GET  /api/jobs/{job_id}    poll until succeeded/failed
GET  /api/results/{id}     preview, quality_report, artifacts
POST /api/results/{id}/save
        persist to PlatformDB (My Projects)
```

Health: `GET /api/health` reports version and flags (`docling_available`, `auth_enabled: false`, `job_progress: polling`).

---

## 6. Schema Twin — source-free generation

### 6.1 User flow

```text
Intent → Provide schema (file or template) → Review tables/columns
      → Configure (rows, seed, locale, export, optional LLM text)
      → Generate → Preview → Validate → Download zip
```

This workflow is **source-free**. Calling the generate step “training” would misrepresent the implementation: nothing is fitted to production rows.

### 6.2 Runtime path

```text
web/src/pages/SchemaTwinPage.tsx
  → POST /api/schema/sessions
  → POST /api/schema/sessions/{id}/schema-file   or  /template
  → POST /api/schema/sessions/{id}/generate
       LocalJobRunner kind=schema.generate
         load_schema → generate_from_schema
           prepare_schema_for_generation
           run_schema_pipeline
             DataSimulator.generate_with_reports / generate_all
             distribution rules, duplicate repair, linked-field realism
             validate_data + build_validation_report
             streaming CSV/Parquet export
```

Streamlit equivalent: `interfaces/streamlit/pages/schema_twin.py` → `application/workflows/schema_twin.py`.

### 6.3 Schema ingestion

`load_schema` / `load_schema_bytes` accept JSON, YAML, SQL-ish, and dict payloads and normalize them into `SchemaConfig` (`engine/inference/schema/schema.py`): tables, columns, types, uniqueness, nullability, relationships, distribution params, realism config.

SQLite files uploaded here are rejected (`wrong_workflow`); they belong in Database Twin.

Built-in templates that **can generate**: `ecommerce`, `saas`, `healthcare`, `fintech` (`engine/generation/schema/templates/library.py`). Other catalog entries (`sqlite-customer-360`, `document-bank-statement`, `support-transcript`) are metadata-only (`can_generate: false`).

### 6.4 Planning before generation

`prepare_schema_for_generation`:

1. Semantic enrichment (`enrich_schema_semantics`).
2. Optional fintech domain hint from table names (account + transaction + customer/branch/card/loan/merchant).
3. Seed and locale on `RealismConfig`.
4. If relationships exist, `GenerationPlanner` computes **FK-aware cardinalities**: root tables get the requested base row count; children get heuristic ratios (e.g. customer→order 3–8, account→transaction 10–30). Reference-like tables (status, currency, country, …) stay small; activity/log tables grow.
5. Unique integer ranges are expanded so uniqueness is feasible at the planned row count.
6. If LLM text is enabled, eligible narrative columns get `llm_enabled` / `llm_text` flags.

Product defaults (`PlatformDB` / `read_generation_defaults`): `default_record_count` (100), `default_output_format` (`csv` or `parquet`; sqlite/pdf fall back to csv for this workflow).

### 6.5 `DataSimulator` — how rows are produced

`engine/generation/schema/simulator.py` is the schema-driven generator.

Mechanics:

- **Topological table order** from foreign keys (parents before children).
- **Vectorized column generation** (NumPy/pandas; no per-row Python loops for numeric/categorical columns).
- **Referential integrity**: child FK columns sample from already-generated parent keys.
- **Type/semantic generators**: ints, floats, booleans, dates, categoricals, Faker-backed PII-shaped fields, checksummed IDs (Luhn, IBAN, IFSC, SSN, Aadhaar, SWIFT, routing numbers) via `smart_values`.
- **Text**: `SemanticVocabularyGenerator`, `RealisticTextGenerator`, optional `TextGenerationEngine` (Ollama by default) with Faker/vocabulary fallback, sanitization, and row caps (`max_llm_rows` default 50; full-run cap 200).
- **Realism**: `EntityCoherenceEngine`, linked-field rules, domain priors, optional scenario/workflow engines.
- **Streaming**: when not preview-only and total rows exceed `chunk_size`, tables are generated and exported in batches (`StreamingCsvWriter` / `StreamingParquetWriter`) with reservoir sampling for validation.

Optional Gaussian copula helpers exist under `engine/generation/schema/generators/copula.py` for joint numeric structure inside schema mode; the primary orchestrator remains `DataSimulator`.

### 6.6 Schema validation and export

`run_schema_pipeline` then:

- applies distribution rules from `rules_text` if present
- enforces duplicate policy
- validates schema/type/rule/privacy/export
- builds a structured validation report used by the UI before download
- packages CSV/Parquet plus a zip via `package_download`

Hard-check semantics: `hard_checks_passed` / `passed` / status in `{pass, passed, ok, success}`.

LLM policy: external providers (`openai`, `groq`) require `SYNTH_ALLOW_EXTERNAL_LLM=true`. Default is local Ollama or off. `LlmPolicyError` fails the job with `llm_policy_error`.

---

## 7. Database Twin — source-driven relational twin

### 7.1 User flow (product)

```text
Intent → Connect SQLite (upload or sample DB)
      → Discover schema
      → Profile values (statistics only; no raw rows persisted)
      → Infer semantics + approve dataset contract
      → Train per-table synthesizers
      → Export portable artifact
      → Generate synthetic tables with FK repair
      → QA
      → Write target SQLite (Database B)
```

FastAPI currently **auto-runs this entire pipeline** in one `database.generate` job after configure. Unattended contract approval accepts the engine’s top-ranked semantic type for every column, including `REVIEW_REQUIRED`. Interactive Streamlit can still step through stages.

### 7.2 Runtime path

```text
web/src/pages/DatabaseTwinPage.tsx
  → POST /api/database/sessions
  → POST /api/database/sessions/{id}/source          (SQLite only)
     or  /sample-source
  → POST /api/database/sessions/{id}/configure       row counts / scale / seed
  → POST /api/database/sessions/{id}/generate
       run_database_twin_pipeline(SQLiteSourceAdapter, RunManifest, config)
         WorkflowOrchestrator stages (see below)
```

Source registry (`engine/discovery/database/adapters/registry.py`) currently registers **only** `"sqlite"`. Infrastructure also contains Postgres connectors; they are not wired into this workflow’s adapter registry. CSV/Parquet/Postgres appear in the UI as deferred modes.

### 7.3 Stage graph (hard dependency order)

From `build_database_workflow()`:

```text
discovery
  → profiling
    → inference
      → contract_approval
        → cleaning
          → training
            ├→ artifact_export
            └→ relational_generation
                 → qa_validation
                   → target_write
```

`WorkflowOrchestrator` records each `StageResult` on `RunManifest`, never overwrites a stage file, and stops at the first failure. Resume is supported from the last successful stage when reusing a manifest.

### 7.4 Run directory and lineage

Each run is `runs/<run_id>/` with `run_manifest.json`. Stage outputs are named files in that directory. A run directory is never reused.

| Stage | Output | Consumed by |
|---|---|---|
| Discovery | `discovery.json` | Profiling, inference, artifact |
| Profiling | `profile.json` | Inference, QA fidelity (sanitized companion fields only) |
| Inference | `semantic_candidates.json` | Contract approval |
| Contract approval | `metadata/dataset_contract.json` (+ history snapshots) | Cleaning, training, generation, QA, target write |
| Cleaning | cleaning evidence | Training uses the same cleaner |
| Training | `training_report.json`, `models/`, `generated_samples/` | Artifact export, relational generation |
| Artifact | portable zip: `manifest.json`, checksums, sanitized profile, model JSON, `self_test.json` | Source-free reload (`load_artifact`) |
| Relational generation | table CSVs + `relational_generation_report.json` | QA, target write |
| QA | `qa_report.json` | Release / target write gate |
| Target write | target `.db` + `target_write_report.json` | Final Database B |

Optional `domain_mapping.json` maps tables onto a canonical domain model for PDF-track parity checks.

### 7.5 Discovery

`SQLiteSourceAdapter` validates the file, `test_connection()`, then `discover()`: tables, columns, PKs, FKs, indexes, estimated row counts, `source_fingerprint`. `run_discovery` is the **only** writer of `discovery.json`. Later stages must not re-introspect the live source for schema truth.

### 7.6 Profiling

`run_profiling` samples through the same adapter (`sample_limit`, optional `chunk_size`). `StructuredProfiler` writes **derived statistics only** — not raw sampled rows.

The in-pipeline profile may hold exact min/max and category frequencies for operators. Anything that **leaves** the protected environment is passed through `sanitize_profile_for_export` (artifact builder). QA fidelity uses the already-sanitized companion fields (`generation_lower_bound` / `upper_bound`, `safe_category_frequencies`), never exact extrema.

### 7.7 Inference and the dataset contract

`SemanticInferenceEngine` scores columns using discovery types, profile stats, and sample values. It writes **proposals only**.

`run_contract_approval` is the only stage that can mark `inference_status: approved`. The contract is the durable semantic source of truth for every later stage (`physical_type`, `semantic_type`, `nullable`, `sensitive`, PK/FK, business rules).

Sensitive semantic types: `identifier`, `email`, `person_name`, `phone_number`.

Unattended API runs auto-approve top-ranked types. Streamlit can present review.

### 7.8 Cleaning and fit-time privacy

Before `fit()`:

1. `DataCleaner` applies the contract (types, nulls, warnings).
2. Identifier **formats** are learned from cleaned values (pattern, length) **before** masking, so hash-seeded stand-ins do not destroy format metadata.
3. `DeterministicFakerMasker` replaces context-aware / PII-like fields so raw production values never enter the copula.
4. `assert_no_raw_context_in_fit_frame` fails closed if a context field is still raw.

Training writes `training_report.json` and model files — **never the real training rows**.

### 7.9 Synthesizer adapters (the statistical heart)

Registry (`engine/training/database/registry.py`):

| `model_type` | Adapter | Serialization | When used |
|---|---|---|---|
| `safe_gaussian_copula` | `SafeCopulaSynthesizerAdapter` | JSON (no pickle) | Default unattended pipeline (`pipeline_runner` `model_type`) and planner default when no sensitive columns |
| `dp_gaussian_copula` | `DPCopulaSynthesizerAdapter` | JSON + privacy accountant | Planner recommendation when any approved column is sensitive. Formal (ε, δ) DP on fitted statistics **only if this adapter actually ran** with bounds and budget |
| `sdv_gaussian_copula` | `SDVSynthesizerAdapter` | SDV/cloudpickle | Never auto-recommended. Explicit user choice because of pickle/security posture |

`recommend_synthesizer` is **advisory**, not a hard gate.

**Safe copula behavior:**

- Fits `copulas.multivariate.GaussianMultivariate` on **DNA columns** only (numeric, categorical, boolean, datetime after custom JSON-safe encoders).
- PII / identifier / email / person_name / phone_number / free_text / context-aware columns are **excluded from the joint distribution** and generated independently (Faker or deterministic identifier formats) at sample time.
- Rare categories can be rebalanced (`category_minimum_support`).
- `save()`/`load()` are plain JSON.

**DP copula behavior:**

- Every fitted statistic (marginal mean/variance or histogram, correlation matrix) goes through `dp_primitives` / `dp_statistics` and `PrivacyAccountant`.
- Exceeding `(epsilon, delta)` raises rather than silently overspending.
- Numeric/datetime **bounds must be supplied as public knowledge**. Deriving min/max from the private sample is refused (that would leak via the bound).
- Univariate model is Gaussian for DP estimability, trading some fidelity for an auditable mechanism.
- PII columns still spend **zero** privacy budget (same independent generators).

Default API/pipeline `model_type` is `safe_gaussian_copula`, not the DP adapter. Do not describe a completed Database Twin API run as differentially private unless `dp_gaussian_copula` was selected and the accountant report is present.

### 7.10 Portable artifact

`build_artifact` / `run_artifact_export` produces a zip that is intended to generate **without the source database**:

- sanitized `reference_profile.json`
- dataset contract and discovery (no credentials, no raw rows)
- per-table model files
- `manifest.json`, `checksums.sha256`, `self_test.json`

Negative tests scan the archive for connection strings, credentials, and absolute training-machine paths.

`load_artifact` reconstitutes adapters for source-free sampling.

### 7.11 Relational generation

Per-table adapters sample independently. `generate_relational_dataset` then:

1. Builds a schema graph (nodes, FK edges, `generation_order`, `condensed_order` for SCCs).
2. Generates parents first, then children.
3. **Reassigns cross-table FKs** by uniform random choice among generated parent keys (not learned per-parent cardinality).
4. Self-FKs stay as the table adapter sampled them (batch generation cannot incrementally grow a self-hierarchy).
5. Cyclic FK components: pass 1 sample without cross-SCC assignment; pass 2 resolve intra-SCC FKs.

Row counts come from configure:

- preserve source counts
- scale factor
- explicit per-table map
- single target count applied to every table
- else product `default_record_count`

### 7.12 QA and target write

QA combines **separate** evidence (not one blended score):

- PK uniqueness (exact; streaming path uses temp SQLite UNIQUE)
- FK validity / orphans
- contract/schema adherence
- business-rule constraint reports
- fidelity vs sanitized profile companions
- optional DP report if provided
- `release_mode` (default `MODE_LEARNED_RESTRICTED`)

`hard_checks_passed` is the write gate. `run_target_write` **refuses** to write Database B if that flag is false. After write, `validate_write` checks the target file against generated tables.

---

## 8. Document / PDF Twin

### 8.1 User flow

```text
Intent → Upload PDF → Choose extraction (Docling or automatic native/OCR)
      → Profile document → Compile twin template
      → Optional: de-identify source PDF (sibling, not a prerequisite)
      → Semantic binding → Synthetic values → Render twin PDF → Validate
```

### 8.2 Runtime path

```text
WorkflowInputPage (document) or Streamlit pdf_twin.py
  → application/workflows/pdf_twin.py
  → engine/documents/pdf/
       PDFDocumentAdapter (validate, encryption, size, metadata/AcroForm PII)
       extraction_router
         ├ docling_engine          extraction_method="docling"  (no silent fallback)
         ├ native pypdf            extraction_method="native"
         └ OCR pytesseract         extraction_method="ocr"
       profiling → document_profile.json
       template compilation (layout backend matches extraction_method)
       semantic binding → document_binding_map.json
       value generation → document_synthetic_values.json
       rendering → synthetic PDF + document_ground_truth.json
       validation → document_validation_report.json
```

`build_pdf_workflow()` stage order:

```text
document_profiling → template_compilation → semantic_binding
  → value_generation → render → validation
```

De-identification is **not** in that chain. `run_document_deidentification` is optional and produces its own redacted PDF + report.

### 8.3 Extraction contract

Docling (pinned `docling==2.119.0`, `pdf` extra) converts the PDF and normalizes spans to:

```text
{text, x0, y0, x1, y1, page_number, confidence}
```

Coordinates are fractional `[0, 1]`, top-left origin. Table cells become positioned spans so region analysis stays backend-independent.

Automatic mode: native text first; OCR only if classified scanned/empty. Explicit Docling never silently falls back — failure is surfaced so the run cannot claim a Docling lineage it did not use.

AcroForm values and PDF Info are still merged for PII scanning even when Docling supplies body text.

Offline: `DOCLING_SERVE_ARTIFACTS_PATH` for pre-fetched model artifacts.

### 8.4 How synthetic PDF values are generated

`run_value_generation` reads **only** the already-redacted template and binding map. It does not open the source PDF. Same seed reproduces output.

`generate_document_values`:

- **Faker / vocabulary strategies** for names, emails, phones, addresses, orgs, medications, narrative.
- **Shape-filling** for IDs, currency, dates: refill the field’s persisted `shape_pattern` without seeing the source value.
- Optional LLM text flag from Document Twin configure (`llm_text_enabled`).

Rendering places those values into the compiled layout. Validation compares rendered output to ground truth / template expectations (`hard_checks_passed`).

---

## 9. Customer Interaction Twin

### 9.1 User flow

```text
Upload .txt/.log → parse turns → configure type/format/privacy/seed
  → generate synthetic transcript → validate → export JSON + logs (+ zip)
```

### 9.2 How generation works

`parse_transcript` accepts `Speaker: text` lines (optional timestamps) or JSON-like internal/external speaker blocks.

`generate_interaction_twin`:

1. Optionally redact PII (`detect_pii` / `redact_sensitive_text`) when `remove_sensitive_information` is true. Product privacy levels `standard | restricted | strict` all map to redaction-on for this workflow unless the request overrides.
2. Rebuilds a synthetic conversation with the same turn structure/speakers, deterministic from `seed`.
3. Writes structured JSON and synthetic log files.
4. Emits `validation_report` and `redaction_report`. API job fails if `hard_checks_passed` is false.

This is **not** an LLM conversation model by default; it is a deterministic structural twin with redaction and synthetic phrasing.

---

## 10. SDK / CLI lineage (signed artifacts)

For headless train-then-generate **without** the FastAPI session model:

```text
synth-platform train --source <sqlite path or postgres URL> --artifact out.synthpkg
synth-platform generate --artifact out.synthpkg --out ./out --rows '{"customers":100}' --format csv
```

`SyntheticDataPlatform.train`:

1. Open connector (`sqlite:///` or filesystem path; `postgresql://` via `PostgresSource`).
2. Health check; assert read-only when the connector supports it.
3. `train_model` / `train_relational_dataset`:
   - profile tables
   - `build_policy` + `redact_profile` + `verify_source_free`
   - `compile_learning_plan`
   - compile constraints and cross-table conditionals
   - relational DAG / cycle policy
   - fit configured backends (`Settings.allowed_backends`, default `statistical`)
   - emit `SynthArtifact` (schema fingerprint only — no URLs, credentials, or rows)
4. Export/sign via artifact store.

`generate` loads and verifies the artifact, `GenerationService().run`, `validate_dataset`, `publish_dataset` through a sink (`csv` / `sqlite` / `parquet`), writing `validation_report.json`. Release-gated: overall status is recorded; transfer/publish respects the gate.

This lineage is the right mental model for “train once, disconnect source, generate elsewhere.” Database Twin’s zip artifact is the **checkpoint** analogue of the same idea, with different on-disk format (JSON copula models vs `.synthpkg`).

---

## 11. Privacy model (how it is actually applied)

Privacy is an **explicit policy and evaluation concern**, not a blanket anonymity guarantee.

| Mechanism | Where | What it does |
|---|---|---|
| Semantic `sensitive` flag | Dataset contract | Drives planner recommendation |
| Deterministic masker + fit guard | Database training | Raw PII/context never enters `fit()` |
| PII excluded from copula | Safe/DP/SDV adapters | Independent Faker/format generators |
| Profile sanitizer | Artifact export | Strips exact min/max/examples/frequencies |
| QA uses sanitized companions | Database QA | Fidelity without reusing raw extrema |
| DP accountant | `dp_gaussian_copula` only | Formal (ε, δ) if that adapter ran |
| LLM egress policy | Schema/PDF text | Local Ollama default; external opt-in |
| Transcript redaction | Interaction Twin | PII detection before rewrite |
| PDF de-identification | Optional sibling | Redacted source ≠ synthetic twin |
| Transfer gate | Downloads | No PASS → no file |

`domain/privacy/policies.py` sets `formal_dp=False` unless a real DP mechanism is in the artifact. Do not claim differential privacy because noise, masking, or DCR-style metrics exist.

---

## 12. Validation and release semantics

Common rule: **structural correctness first**, then fidelity / privacy / utility. A failed required check is a failed release. A required capability that did not run is `NOT_RUN` / `SKIPPED` / warning — never a silent pass.

Schema Twin: type, uniqueness, FK, rules, privacy/readiness, export path checks.

Database Twin: PK, FK, constraints, sanitized fidelity, release manager; target write is downstream of QA.

PDF Twin: rendered document vs ground truth/template.

Interaction Twin: turn structure and redaction evidence.

SDK generate: `ValidationReport.overall` plus publish/transfer.

---

## 13. Frontend architecture

Vite + React Router (`web/src/App.tsx`).

| Route | Page |
|---|---|
| `/` | Home — four workflow cards |
| `/schema` | Schema Twin |
| `/database` | Database Twin |
| `/document` | Document Twin (`WorkflowInputPage`) |
| `/interaction` | Interaction Twin (`WorkflowInputPage`) |
| `/progress/:jobId` | Poll job |
| `/results/:resultId` | Preview, quality, downloads |
| `/projects`, `/projects/:id`, `.../runs/:runId` | My Projects |
| `/settings` | Generation defaults |
| `/templates` | Backend template catalog |

`web/src/api.ts` is the typed client. `web/src/lib/workflows.ts` holds labels, steps, and generation-stage copy. Shared UI: stepper, upload panel, progress timeline, quality summary, result tabs, download list.

The React app does not run engines. It only drives the FastAPI session/job/result protocol.

---

## 14. Configuration

### 14.1 Process settings (`synth_platform.settings.Settings`)

| Env | Default | Meaning |
|---|---|---|
| `SP_SEED` | 42 | Default seed |
| `SP_MAX_ROWS` | 50000 | Max rows per table (SDK/settings) |
| `SP_HOLDOUT` | 0.30 | Holdout fraction |
| `SP_APPROVED_OUTPUT_ROOT` | `output` | Approved output root |
| `SP_STAGING_ROOT` | `.staging` | Staging root |
| `SP_API_STATE_DIR` | `.staging/api` | API sessions/jobs |
| `SP_PLATFORM_DB_PATH` | `.staging/platform.db` | Projects/settings DB |
| `OLLAMA_HOST` | `http://localhost:11434` | Local LLM |
| `OLLAMA_MODEL` (or `OLLAMA_LLM_TEXT_MODEL`, `OLLAMA_SMART_VALUE_MODEL`, `MVP_LLM_TEXT_MODEL`) | `qwen3.5:9b` | Local model |
| `SYNTH_ALLOW_EXTERNAL_LLM` | unset | Opt in to OpenAI/Groq |
| `DOCLING_SERVE_ARTIFACTS_PATH` | unset | Offline Docling models |
| `VITE_API_BASE_URL` | empty | Frontend API base; empty uses Vite proxy |

### 14.2 Product settings (persisted)

`generation_mode`: `schema_driven` \| `source_driven`  
`default_record_count`: integer ≥ 1  
`privacy_level`: `standard` \| `restricted` \| `strict`  
`default_output_format`: `csv` \| `parquet` \| `sqlite` \| `pdf`

Workflows read these as defaults; an explicit request overrides them.

### 14.3 Optional extras (`pyproject.toml`)

`api`, `schema` (duckdb, scipy, simpleeval), `database` (sdv, copulas), `parquet`, `pdf`, `docling`, `ocr`, `postgresql`, `llm`, `jobs` (celery), `ui` (streamlit), `test`.

---

## 15. Tests, CI, and delivery

Tests live under `tests/` with markers: `contract`, `unit`, `integration`, `e2e`, `security`, `negative`, `smoke`.

Architecture contract (`tests/contract/architecture/test_layering.py`):

- imports only point inward
- application/engine/domain never import `infrastructure` or `interfaces`
- domain purity (no pandas/IO/UI libraries)

GitHub Actions (`.github/workflows/ci.yml`) on PR and `main`/`master`:

1. `backend-tests` — `uv sync` + pytest with `test,api,schema,parquet`
2. `frontend-tests-build` — npm test + production build
3. `docker-build` — image `synthetic-data-twin-api:ci` / `:${{ github.sha }}`, smoke `/api/health`  
   Images are **not** pushed to a registry.

`docs/ci.md` also describes a fuller matrix (package-smoke, architecture, security, integration-light/heavy, artifact-delivery that trains a twin, **deletes the source DB**, generates from the artifact only, and checks source canaries). Tag `v*` release workflow publishes package/release assets.

---

## 16. Current product boundaries (do not over-claim)

Grounded limitations from README and code:

- FastAPI/React Database Twin is **SQLite-only**. Postgres/CSV/Parquet connectors exist in infrastructure but are not registered on the Database Twin source adapter used by the UI.
- Natural-language schema generation is not implemented.
- Template-backed generation in the UI is Schema Twin built-ins only.
- No production auth. CORS is localhost-only.
- Jobs are in-process threads, polled by the frontend.
- Docker image is backend-only.
- Formal DP applies only when `dp_gaussian_copula` ran with explicit bounds and a recorded accountant.
- Unattended Database Twin auto-approves inferred semantics; it is not a human review substitute.
- Relational child counts are requested, not learned as “N children per parent.”
- SDK/CLI and UI Database Twin artifacts are different formats; they are not drop-in interchangeable.

---

## 17. How to add a feature (placement rule)

1. Decide: domain concept, engine capability, application orchestration, infrastructure integration, or UI.
2. Put reusable stage algorithms under the **stage engine**, not a page or route handler.
3. Expose cross-stage product behavior through an **application workflow or use case**.
4. Put vendor / database / filesystem / network details under **infrastructure** (or a SourceAdapter in `engine/discovery` if it is the Database Twin connector contract).
5. Add unit tests next to the capability; add integration/e2e under the product workflow.
6. Run the architecture contract before merge.

---

## 18. End-to-end map (all workflows)

```text
                    +---------------------------+
                    | React (web/) / Streamlit  |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | FastAPI interfaces/api    |
                    | sessions, jobs, results   |
                    +-------------+-------------+
                                  |
          +-----------+-----------+-----------+-----------+
          |           |           |           |           |
          v           v           v           v           v
     Schema Twin  Database Twin  PDF Twin  Interaction   Projects/
     generate_    pipeline_      document  generate_     Settings
     from_schema  runner         pdf/*     interaction   PlatformDB
          |           |           |           |
          v           v           v           v
     DataSimulator  Copula fit   Template +  Transcript
     + FK planner   + relational bind/fill   rewrite
     + validators   + QA + DB B  + render    + redact
          |           |           |           |
          +-----------+-----------+-----------+
                                  |
                                  v
                    ResultBundle + optional TransferService download
```

That is the architecture as implemented: layered, stage-oriented, local-first, with four product twins and two generation lineages, and with privacy and release treated as explicit gates rather than implied properties of “synthetic.”
