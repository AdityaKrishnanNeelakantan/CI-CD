# Phase 5 Summary: Local Platform Persistence

## Scope

- Added a backend-only local SQLite persistence layer for project/run history and product defaults.
- The database is separate from workflow source/target/output databases.
- No Phase 1-4 validation, transfer, LLM policy, text-generation, or schema-validation internals were changed.
- Settings and My Projects UI pages were not built in this phase.

## Database Location

- Default file: `<Settings.staging_root>/platform.db`.
- With the current default `Settings.staging_root`, this resolves to `.staging/platform.db`.
- Tests can override the location with `SP_PLATFORM_DB_PATH` or by constructing `PlatformDB(path)`.

## Schema

- `projects`
  - `id TEXT PRIMARY KEY`
  - `name TEXT NOT NULL`
  - `workflow_type TEXT NOT NULL CHECK ('schema', 'database', 'pdf')`
  - `created_at TEXT NOT NULL`
  - `updated_at TEXT NOT NULL`
  - `status TEXT NOT NULL CHECK ('in_progress', 'completed', 'failed')`
  - `metadata_json TEXT NOT NULL DEFAULT '{}'`

- `runs`
  - `id TEXT PRIMARY KEY`
  - `project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE`
  - `workflow_type TEXT NOT NULL CHECK ('schema', 'database', 'pdf')`
  - `created_at TEXT NOT NULL`
  - `updated_at TEXT NOT NULL`
  - `status TEXT NOT NULL CHECK ('in_progress', 'completed', 'failed')`
  - `validation_status TEXT`
  - `validation_passed INTEGER CHECK (0, 1, NULL)`
  - `transfer_status TEXT NOT NULL DEFAULT 'not_attempted' CHECK ('not_attempted', 'success', 'blocked')`
  - `transfer_allowed INTEGER CHECK (0, 1, NULL)`
  - `transfer_attempted_at TEXT`
  - `output_id TEXT`
  - `metadata_json TEXT NOT NULL DEFAULT '{}'`

- `settings`
  - `key TEXT PRIMARY KEY`
  - `value_json TEXT NOT NULL`
  - `updated_at TEXT NOT NULL`

## Design Rationale

- Projects and runs are split so a future My Projects page can show a stable project while still preserving retries or repeated generation attempts.
- Settings are key-value JSON because the current product defaults are small and likely to evolve during Phase 6.
- Indexes cover project lists by workflow/status, run lists by project/workflow, and transfer updates by workflow/output id.

## Data-Access Layer

- Added `src/synth_platform/infrastructure/persistence/platform_db.py`.
- Main API:
  - `PlatformDB.initialize()`
  - `create_project()`, `list_projects()`, `update_project()`
  - `record_run()`, `list_runs()`, `record_transfer_attempt()`
  - `write_setting()`, `read_setting()`, `write_settings()`, `read_settings()`

## Recording Hooks

- Schema Twin:
  - `src/synth_platform/application/workflows/schema_twin.py`
  - `generate_from_schema(..., history=...)` records a run after the canonical schema pipeline returns validation/export readiness.
  - `src/synth_platform/interfaces/streamlit/pages/schema_twin.py` passes `get_platform_db()` into generation and transfer.

- Database Twin:
  - `src/synth_platform/application/workflows/database_twin.py`
  - Added `run_database_twin_pipeline(..., history=...)` for headless workflow recording after QA and target write.
  - `src/synth_platform/interfaces/streamlit/pages/database_twin.py` records after QA validation and passes the same platform DB into transfer gating.

- PDF Twin:
  - `src/synth_platform/application/workflows/pdf_twin.py`
  - `run_document_validation(..., history=...)` records a run after PDF validation writes `document_validation_report.json`.
  - `src/synth_platform/interfaces/streamlit/pages/pdf_twin.py` passes `get_platform_db()` into validation and transfer.

- Transfer gate outcome:
  - `src/synth_platform/application/services/transfer_service.py`
  - `TransferService(..., transfer_recorder=...)` now accepts an optional recorder and updates the matching run after both allowed and blocked gate attempts.
  - Transfer decisions and blocking behavior remain unchanged.

## Tests

- Added `tests/unit/test_platform_db.py`.
- Added `tests/integration/test_platform_persistence_hooks.py`.
- Focused verification:
  - `tests/contract/architecture/test_layering.py`: `3 passed`.
  - `tests/unit/test_platform_db.py tests/integration/test_platform_persistence_hooks.py`: `7 passed`.
  - `tests/unit/application/test_transfer_service.py tests/unit/schema/test_schema_mode.py`: `25 passed`.
  - `tests/integration/database_pdf/test_pipeline_runner.py tests/integration/database_pdf/test_render_and_validation_pipeline.py`: `4 passed`.
  - Streamlit workflow tests:
    - Schema: `1 passed, 1 skipped`.
    - PDF: `2 passed`.
    - Database: `2 passed`.
- Full-suite verification:
  - Command: `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
  - Result: `1385 passed, 11 skipped, 72 warnings in 50.03s`.
- Comparison to Phase 4 final result:
  - Phase 4: `1378 passed, 11 skipped`.
  - Phase 5: `1385 passed, 11 skipped`.
  - Net change: `+7` passing tests, no change in skipped tests, no regression in pass/skip counts.

## Phase 6 Notes

- The My Projects UI can read from `projects` joined to latest `runs`, using `workflow_type`, `created_at`, `status`, and transfer fields directly.
- Product Settings can map defaults to keys: `generation_mode`, `default_record_count`, `privacy_level`, and `default_output_format`.
- If Phase 6 needs custom project names before generation, the workflow APIs already allow the caller to supply richer metadata, but there is not yet a user-facing naming flow.
