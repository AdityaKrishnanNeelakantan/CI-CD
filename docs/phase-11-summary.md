# Phase 11 Summary: React Schema Twin API Slice

## What Was Built

Phase 11 adds a new top-level React frontend under `web/` for the Schema Twin workflow only. It uses Vite, TypeScript, React, TanStack Query, and Vitest.

The first browser UI calls the Phase 10 FastAPI endpoints directly:

- `GET /api/settings` for generation defaults.
- `POST /api/schema/sessions` for Schema Twin sessions.
- `POST /api/schema/sessions/{id}/schema-file` for schema upload and parsed review data.
- `POST /api/schema/sessions/{id}/generate` to start generation.
- `GET /api/jobs/{job_id}` with polling for `queued`, `running`, `succeeded`, and `failed` states.
- `GET /api/schema/sessions/{id}/preview` for generated table preview.
- `GET /api/schema/sessions/{id}/validation` for validation report and highlights.
- `POST /api/schema/sessions/{id}/download` and `GET /api/downloads/{download_id}` for the gated ZIP download.

The page follows the intended step pattern:

- Input / Review
- Configure
- Generate
- Results

It shows specific API envelope errors for schema parse failures, `llm_policy_error`, and `transfer_blocked` instead of collapsing them into a generic failure state. A blocked download disables the download button and surfaces the transfer-gate reason.

## How To Run Locally

Install the Python API dependencies:

```bash
source .venv/bin/activate
pip install -e '.[api,schema,parquet]'
```

Run the FastAPI backend:

```bash
synth-platform-api --reload --host 127.0.0.1 --port 8000
```

Run the React frontend:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`. Streamlit remains available separately through the existing `synth-platform-ui --server.port 8502` path.

## Streamlit Compatibility

No Streamlit files were modified in this phase. Streamlit and the React frontend are parallel surfaces over the same backend/application core.

## Deferred

- Authentication and authorization.
- Shared app shell/navigation for other workflows.
- Database Twin, PDF Twin, and Customer Interaction Twin React pages.
- WebSocket or SSE job progress.
- Production deployment packaging.
- A broader design system beyond the Schema Twin page.

## Verification

Frontend tests:

```text
npm test
Test Files  1 passed (1)
Tests  3 passed (3)
Duration  2.75s
```

Frontend production build:

```text
npm run build
tsc --noEmit && vite build
✓ built in 1.01s
```

Python regression:

```text
1411 passed, 11 skipped, 74 warnings in 49.27s
```

The Phase 10 baseline was `1411 passed, 11 skipped`, so the Python suite count is unchanged.
