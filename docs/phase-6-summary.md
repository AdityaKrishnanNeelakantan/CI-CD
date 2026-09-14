# Phase 6 Summary: Settings and My Projects UI

## Scope

- Added Streamlit pages for My Projects and Settings on top of the Phase 5 `platform.db`.
- Registered both pages in the Streamlit navigation under Platform.
- Left Schema Twin, Database Twin, PDF Twin generation controls and workflow entry points unchanged.
- Left Phase 1-5 validation, transfer gating, LLM policy, text generation, schema validation, and workflow recording behavior unchanged.

## My Projects

- New page: `src/synth_platform/interfaces/streamlit/pages/my_projects.py`.
- Lists projects from `platform.db` with their latest recorded run.
- Shows real tracked fields: name, workflow type, created date, project status, validation outcome, transfer outcome, and record count when a recorded run contains real `row_counts` or `row_counts_by_table` metadata.
- Provides tabs for All Projects, Schema, Database, and Document.
- Supports renaming a project through `PlatformDB.update_project()`.
- Shows run history/details for each project, including validation, transfer, output id, and records when tracked.
- Delete support was skipped because there is no existing data-access method for deletion and it was not required for this phase.

## Settings

- New page: `src/synth_platform/interfaces/streamlit/pages/settings.py`.
- Reads and writes these keys through the existing settings API:
  - `generation_mode`
  - `default_record_count`
  - `privacy_level`
  - `default_output_format`
- Uses a Save action to persist all settings consistently.
- Provides defaults when no setting has been saved yet.
- Settings are persisted only; they are not yet wired into Schema, Database, or PDF generation behavior.

## Supporting UI Logic

- Added `src/synth_platform/interfaces/streamlit/project_ui.py`.
- This module composes `PlatformDB.list_projects()` and `PlatformDB.list_runs()` into testable view models without changing the persistence schema or adding new DAL methods.
- `PlatformDB.record_run()` now auto-generates timestamped fallback project names, for example `Schema Twin - Sep 10, 2026 09:45 PM`, when no project name is supplied.

## Mockup Field Notes

- The original mockup included a Records column. Phase 5 does not have a first-class records column in `projects` or `runs`.
- The My Projects page shows records only when the latest run already recorded real row-count metadata.
- When row-count metadata is absent, the page displays `-` rather than guessing.
- No schema additions were made.

## Verification

- New focused tests:
  - `tests/unit/test_streamlit_project_ui.py`
  - Added auto-name coverage in `tests/unit/test_platform_db.py`.
- Run command:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest tests/unit/test_streamlit_project_ui.py tests/unit/test_platform_db.py`
- Result:
  - `9 passed in 0.13s`.
- Streamlit page smoke:
  - `SP_PLATFORM_DB_PATH=/private/tmp/synth-platform-ui-test.db XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python - <<'PY' ...`
  - New My Projects and Settings pages rendered without exceptions.
- App health-check smoke:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest tests/e2e/workflows/test_streamlit_smoke.py`
  - `1 passed in 1.02s`.
- Full-suite command:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
- Result:
  - `1390 passed, 11 skipped, 72 warnings in 49.61s`.
- Phase 5 baseline:
  - `1385 passed, 11 skipped`.
- Net change:
  - `+5` passing tests, no change in skipped tests.

## Follow-Up

- Wire saved settings into workflow defaults after product decisions are made for how each mode should interpret generation mode, privacy level, output format, and record count.
- Add a delete/archive project flow only after the persistence layer exposes an intentional lifecycle method.
