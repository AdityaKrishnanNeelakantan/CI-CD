# Phase 7 Summary: PostgreSQL Support in Database Twin UI

## Scope

- Added PostgreSQL as a first-class source option in the Database Twin Streamlit Connect step.
- Reused the existing `PostgresSource` connector for TLS enforcement, read-only health checks, schema/table allowlists, identifier validation, discovery, row counts, and sampling.
- Left Phase 1-6 transfer gating, LLM policy, text generation, schema validation, platform DB schema/DAL, My Projects, and Settings behavior unchanged except for recording non-sensitive source metadata on Database Twin runs.

## UI Changes

- Updated `src/synth_platform/interfaces/streamlit/pages/database_twin.py`.
- The Connect step now offers:
  - Use a generated sample database
  - Upload a SQLite file
  - Connect to PostgreSQL
- PostgreSQL connection fields:
  - Host
  - Port
  - Database name
  - Username
  - Password
  - SSL mode
  - Allowed schemas
  - Optional allowed tables
- PostgreSQL connection failures are shown through the existing user-facing error component, with technical detail available in the expandable details area.

## Supporting UI Logic

- Added `src/synth_platform/interfaces/streamlit/database_source_ui.py`.
- Added `src/synth_platform/interfaces/streamlit/postgres_source_adapter.py`.
- The PostgreSQL Streamlit adapter implements the existing generic `SourceAdapter` shape and delegates source security/validation behavior to `PostgresSource`.
- The adapter lives in the Streamlit interface layer so the engine layer does not import infrastructure, preserving the existing architecture contract.

## Security Confirmation

- `PostgresSource` validation/security logic was reused, not duplicated in the UI.
- Confirmed existing SDK support in `src/synth_platform/interfaces/sdk/client.py` already routes `postgresql` and `postgresql+psycopg` URLs to `PostgresSource`.
- Confirmed `PostgresSource` enforces:
  - TLS for non-local hosts when `sslmode` is absent or unsafe.
  - Read-only transactions through `SET TRANSACTION READ ONLY`.
  - Statement and lock timeouts.
  - Allowed schemas and optional allowed tables.
  - Safe SQL identifier validation.
- PostgreSQL credentials are held only in Streamlit session state while the workflow is active.
- PostgreSQL run records store only non-sensitive metadata such as `source_type: postgresql`; raw URLs, usernames, passwords, and connection strings are not written to `platform.db`.
- Existing SQLite uploaded files are still copied into the temporary workflow directory as before; no broader persistence change was made in this phase.

## Pipeline Findings

- Discovery, profiling, inference, training, generation, validation, target write, and transfer gating continue to consume the generic adapter interface.
- No downstream stage required PostgreSQL-specific branching.
- The headless `engine.discovery.database.adapters.registry` still only registers SQLite. Registering PostgreSQL there would require a lower-layer port/factory refactor because `PostgresSource` currently lives in infrastructure and the engine layer may not import infrastructure. This is a follow-up risk for CLI/YAML-driven Database Twin PostgreSQL parity, not a blocker for the requested Streamlit UI path.

## Dependency

- Added a `postgresql` optional dependency group in `pyproject.toml`:
  - `psycopg[binary]>=3.1`
- Installed it in `.venv-phase2`; the environment now has `psycopg 3.3.5`.

## Verification

- New and focused PostgreSQL/UI tests:
  - `tests/unit/database_pdf/test_postgres_adapter.py`
  - `tests/unit/test_streamlit_database_source_ui.py`
  - Added PostgreSQL validation coverage in `tests/e2e/workflows/test_database_twin_ui.py`.
- Targeted command:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest tests/contract/architecture/test_layering.py tests/e2e/workflows/test_database_twin_ui.py tests/unit/database_pdf/test_postgres_adapter.py tests/unit/test_streamlit_database_source_ui.py tests/unit/database_pdf/negative/test_negative_inputs.py`
- Result:
  - `23 passed in 7.19s`.
- Full-suite command:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
- Result:
  - `1395 passed, 11 skipped, 72 warnings in 49.56s`.
- Phase 6 baseline:
  - `1390 passed, 11 skipped`.
- Net change:
  - `+5` passing tests, no change in skipped tests.

## Follow-Up

- Add lower-layer PostgreSQL registration for headless CLI/YAML Database Twin runs after introducing an architecture-safe factory or port boundary.
- Add live PostgreSQL integration coverage when a managed test database is available in CI.
