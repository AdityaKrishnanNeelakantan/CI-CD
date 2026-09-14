# Phase 10 Summary: FastAPI Foundation + Schema Twin API Slice

## What Was Built

Phase 10 adds the first working FastAPI vertical slice for the React migration path:

- A real FastAPI app in `src/synth_platform/interfaces/api/app.py`.
- Local React dev CORS origins for `localhost` and `127.0.0.1` on ports `3000` and `5173`.
- Standard JSON response envelopes:
  - Success: `{"data": ...}`
  - Error: `{"error": {"code": "...", "message": "..."}}`
- `GET /api/health` with service status, package version, and feature flags including Docling availability.
- File-backed API state under `SP_API_STATE_DIR` or `.staging/api` for workflow sessions, jobs, generated results, and download tickets.
- Generic workflow sessions:
  - `POST /api/workflows/{workflow}/sessions`
  - `GET /api/workflows/{workflow}/sessions/{session_id}`
- Generic jobs:
  - `POST /api/jobs`
  - `GET /api/jobs/{job_id}`
- Generic gated downloads:
  - `GET /api/downloads/{download_id}`

The job model supports `queued`, `running`, `succeeded`, and `failed`, plus progress messages and error payloads. Schema generation runs in a background thread and is polled through `GET /api/jobs/{job_id}`.

## Schema Twin API

The Schema Twin proof-of-concept API includes:

- `POST /api/schema/sessions`
- `POST /api/schema/sessions/{id}/schema-file`
- `POST /api/schema/sessions/{id}/generate`
- `GET /api/schema/sessions/{id}/preview`
- `GET /api/schema/sessions/{id}/validation`
- `POST /api/schema/sessions/{id}/download`

The upload endpoint calls the existing `load_schema_bytes`, `summarize_schema`, `schema_column_details`, and `list_llm_text_columns` workflow functions. The generation job calls the existing `generate_from_schema()` facade and passes `PlatformDB()` as both `history` and `product_settings`, so omitted row count and export format continue to resolve through `read_generation_defaults()` just like Streamlit. `LlmPolicyError` is surfaced as a failed job with code `llm_policy_error`.

One small implementation detail differs from a naive API-only design: the job packages the generated ZIP with `package_download(result)` immediately after a successful run and stores it as a local blob. `POST /api/schema/sessions/{id}/download` creates a download ticket for that packaged blob, and `GET /api/downloads/{download_id}` performs the actual transfer gate before returning bytes.

## Settings And Projects

Settings endpoints:

- `GET /api/settings`
- `PUT /api/settings`

Projects endpoints:

- `GET /api/projects`
- `GET /api/projects/{project_id}/runs`

The pure, non-Streamlit helper logic from `interfaces/streamlit/project_ui.py` was extracted to `src/synth_platform/infrastructure/persistence/project_views.py`. The old Streamlit helper module now re-exports the same names as a compatibility shim. This lets the API import the project/settings view helpers without importing a Streamlit interface module, while preserving existing Streamlit imports.

## Streamlit Compatibility

No Streamlit page file was modified. The Streamlit helper import surface remains available through `interfaces/streamlit/project_ui.py`, and the application/workflow core contracts used by Streamlit were not changed.

## Transfer Gate

Every API download path returns bytes only through `TransferService.downloadable_bytes()`. The generic `/api/downloads/{download_id}` endpoint records allowed and blocked attempts through `TransferService(transfer_recorder=get_platform_db())`, matching the existing Streamlit transfer-recording pattern.

## Explicitly Deferred

- Authentication and authorization.
- WebSocket or SSE progress streaming.
- Production-grade shared/durable object storage.
- Database Twin API endpoints.
- PDF Twin API endpoints.
- Customer Interaction Twin API endpoints.
- React frontend work.

## Verification

Focused Phase 10 API tests:

```text
4 passed, 2 warnings in 2.96s
```

Architecture layering contract:

```text
3 passed in 0.79s
```

Full regression suite:

```text
1411 passed, 11 skipped, 94 warnings in 59.47s
```

The requested baseline before this phase was `1407 passed, 11 skipped`; the four added Phase 10 API tests account for the increase to 1411 passing tests.
