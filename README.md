# Synthetic Data Twin

Synthetic Data Twin is a local-first Python and React platform for generating privacy-aware synthetic datasets, documents, and interaction logs. The backend is a FastAPI service over the existing domain/application/engine layers, and the frontend is a Vite React app under `web/`.

## Current Capabilities

- **Schema Twin**: upload JSON/YAML/SQL schema files, review tables and columns, generate synthetic tables, validate results, and download artifacts.
- **Database Twin**: upload SQLite sources, configure supported generation controls, run the backend pipeline, inspect generated artifacts, and save results.
- **Document/PDF Twin**: upload PDFs, configure supported extraction/generation options, generate synthetic PDF artifacts and reports.
- **Customer Interaction Twin**: upload TXT/LOG transcripts, configure interaction type/output/privacy settings, generate synthetic conversations/logs.
- **Projects**: save result bundles to My Projects.
- **Project/run details**: inspect saved project metadata, run metadata, result links, artifacts, quality reports, and safe input/config metadata.
- **Settings**: update local generation defaults.
- **Templates**: browse real backend template metadata; built-in schema templates can start Schema Twin generation.
- **Results and downloads**: view workflow-aware previews, quality summaries, generated files, and artifact download links.

## Current Limitations

- Database input is currently SQLite-only in the React/FastAPI flow. CSV, Parquet, and PostgreSQL are visible as deferred/coming-soon modes.
- Natural-language schema generation is not implemented.
- Template-backed generation is currently supported for built-in Schema Twin templates only.
- Runtime state and artifact metadata are local filesystem/SQLite files by default.
- The backend Docker image is the production-oriented artifact; the React frontend is built separately with Vite.
- No production authentication, RBAC, external queue, cloud storage, or deployment hardening is included yet.
- Jobs run through the local inline job runner and are polled by the frontend.

## Repository Structure

```text
src/synth_platform/
  application/      Workflow facades, use cases, orchestration, ports
  domain/           Core schemas, validation, privacy, artifacts, planning
  engine/           Generation, discovery, profiling, inference, documents
  infrastructure/   Persistence, jobs, sources/sinks, storage, observability
  interfaces/
    api/            FastAPI app, routes, contracts, local API state store
    cli/            Command-line entry points
    sdk/            SDK models/client
    streamlit/      Legacy/local Streamlit surface
web/
  src/components/   React shared UI components
  src/pages/        React routed pages
  src/lib/          Frontend workflow metadata/helpers
tests/              Unit, integration, contract, security, and e2e tests
docs/               Architecture, workflow, privacy, validation, and CI/CD notes
.github/workflows/ GitHub Actions CI and release workflows
```

## Prerequisites

- Python 3.11 or newer
- `uv`
- Node.js 22 or newer and npm
- Docker, for backend image build/run verification

Install `uv` if needed:

```bash
python -m pip install --upgrade uv
```

## Environment

Copy the example when you want local overrides:

```bash
cp .env.example .env
```

Common backend variables:

- `SP_API_STATE_DIR`: local API sessions/jobs/downloads/results directory. Default: `.staging/api`.
- `SP_PLATFORM_DB_PATH`: local project/settings SQLite database path. Default: `.staging/platform.db`.
- `SP_STAGING_ROOT`: default local staging root. Default: `.staging`.
- `SP_APPROVED_OUTPUT_ROOT`: default approved output root. Default: `output`.
- `SP_SEED`, `SP_MAX_ROWS`, `SP_HOLDOUT`: generation defaults used by backend settings.

Local demo defaults, including the Ollama host/model, live in `src/synth_platform/settings.py`.

Ollama defaults for local LLM-backed text generation:

- Host: `http://localhost:11434`
- Model: `qwen3.5:9b`

To use a different local model without editing code, set `OLLAMA_MODEL`.

Frontend:

- `VITE_API_BASE_URL`: optional explicit API base URL. Leave empty for Vite proxy-based local development.

## Backend Local Setup

From the repository root:

```bash
uv sync --extra api --extra schema --extra parquet --extra test
```

Run the backend:

```bash
uv run uvicorn synth_platform.interfaces.api.app:app --reload --host 127.0.0.1 --port 8000
```

Alternative console script:

```bash
uv run synth-platform-api --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Frontend Local Setup

In a separate terminal:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`, so run backend and frontend in separate terminals for local development.

## Tests and Builds

Backend tests:

```bash
uv run --extra test --extra api --extra schema --extra parquet pytest
```

Frontend tests:

```bash
cd web
npm test -- --run
```

Frontend production build:

```bash
cd web
npm run build
```

## Backend Docker

Build the backend image:

```bash
docker build -t synthetic-data-twin-api:local .
```

Run it:

```bash
docker run --rm -p 8000:8000 synthetic-data-twin-api:local
```

Run it with persisted local runtime storage:

```bash
docker run --rm -p 8000:8000 -v "$PWD/.data:/app/.data" synthetic-data-twin-api:local
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Optional backend-only Compose:

```bash
docker compose up --build api
```

The Docker image installs backend dependencies with `uv`, exposes port `8000`, runs FastAPI through `uvicorn`, and excludes frontend build output, `node_modules`, caches, local databases, logs, and runtime artifacts from the build context.

## CI/CD Overview

GitHub Actions runs on pull requests, pushes to `main`/`master`, and manual dispatch:

- `backend-tests`: installs with `uv` and runs backend pytest.
- `frontend-tests-build`: runs React tests and production build.
- `docker-build`: builds the backend Docker image after backend and frontend checks pass, then smoke-tests `/api/health`.

The CI Docker job tags images as `synthetic-data-twin-api:ci` and `synthetic-data-twin-api:${{ github.sha }}`. It does not push to a registry. Future registry publishing should be added only after secrets, registry naming, and deployment ownership are decided.

The release workflow remains tag-based and publishes package/release assets.

## Troubleshooting

- **Templates API unavailable**: restart the backend from this checkout. An older process on port `8000` may be serving pre-template routes.
- **Project detail fails to load**: make sure the backend is running from this repository with `uv run`.
- **Frontend cannot reach backend**: confirm `curl http://127.0.0.1:8000/api/health` works, then restart `npm run dev`.
- **Docker port already in use**: stop the local backend or run Docker with a different host port, for example `-p 8001:8000`.
- **Generated data appears in git status**: local runtime state belongs under `.staging/`, `.data/`, or `output/`, all of which are ignored.

## Cleanup Notes

Generated caches, local staging data, local SQLite runtime files, frontend build output, and virtualenv folders are intentionally ignored. Required source, tests, docs, package config, frontend lock files, Docker config, and CI config should remain tracked.

## Development Workflow

1. Start the backend with `uv run uvicorn ...`.
2. Start the frontend with `npm run dev` in `web/`.
3. Make focused changes in `src/`, `web/src/`, tests, or docs.
4. Run backend tests, frontend tests, and frontend build.
5. For deployment-readiness changes, also run the backend Docker build and health smoke test.
