# React Migration Scope: Streamlit to React + Real API

## Executive Summary

This is not a frontend-only migration. The current product is a Streamlit shell that calls Python application/workflow functions directly in the same process, stores step state in `st.session_state`, writes run artifacts to temporary local directories, and gates downloads with `TransferService` immediately before `st.download_button`.

The reusable core is strong: Schema, Database, PDF, and Interaction workflows mostly sit behind `src/synth_platform/application/workflows/*`. The missing layer is a real HTTP boundary with durable workflow sessions, file upload storage, job status, artifact download endpoints, validation-gated transfer handling, and deployment/security decisions. A realistic full-feature migration is multi-week at minimum and likely a 6-10 week effort for one experienced engineer, depending on auth/deployment requirements and how polished the React UX needs to be.

Recommendation: migrate incrementally, workflow by workflow, with both UIs coexisting temporarily. Schema Twin is the best first proof of concept; Database Twin and PDF Twin are much larger because they are multi-step, artifact-heavy workflows.

## Current Architecture Evidence

- Streamlit shell: `src/synth_platform/interfaces/streamlit/app.py::run` defines the sidebar navigation for Home, My Projects, Settings, Schema Mode, Database Twin, PDF Twin, and Customer Interaction Twin.
- API slice: `src/synth_platform/interfaces/api/app.py` contains the working local FastAPI proof-of-concept for cross-cutting sessions/jobs/downloads plus Schema Twin, Settings, and Projects endpoints. The split route modules under `src/synth_platform/interfaces/api/routes/` are still scaffold files and are not yet the active route surface.
- React slice: `web/` contains the Vite/React Schema Twin frontend that calls the FastAPI endpoints through `web/src/api.ts`.
- Persistence: `src/synth_platform/infrastructure/persistence/platform_db.py::PlatformDB` stores projects, runs, transfer attempts, and settings in local SQLite.
- Download gate: `src/synth_platform/application/services/transfer_service.py::TransferService` is the required validation gate for user-facing downloads.
- Product defaults: `src/synth_platform/domain/product_settings.py::read_generation_defaults` is read by Streamlit pages and workflow facades.

## Audit Method

I audited the active Streamlit entry point and every page under `src/synth_platform/interfaces/streamlit/pages/`, then traced each page import into the application workflow facades, platform persistence, transfer service, API scaffold, and Streamlit-only helper modules. This document is scoped to the user-facing Streamlit product surface that would need React/API parity; it does not inventory lower-level engine internals except where a page calls them through an application facade.

## File-by-File Evidence Index

- Product shell: `src/synth_platform/interfaces/streamlit/app.py:9` configures the Streamlit app; `src/synth_platform/interfaces/streamlit/app.py:16` creates page objects; `src/synth_platform/interfaces/streamlit/app.py:29` creates sidebar navigation.
- API implementation: `src/synth_platform/interfaces/api/app.py:1` defines the active FastAPI application and route registration. `src/synth_platform/interfaces/api/store.py:1` provides local file-backed API state. `src/synth_platform/interfaces/api/launcher.py:1` exposes the packaged API launcher. `src/synth_platform/interfaces/api/routes/generation.py:1`, `src/synth_platform/interfaces/api/routes/artifacts.py:1`, `src/synth_platform/interfaces/api/routes/training.py:1`, `src/synth_platform/interfaces/api/routes/validation.py:1`, and `src/synth_platform/interfaces/api/routes/connections.py:1` remain scaffold modules.
- Home page: `src/synth_platform/interfaces/streamlit/pages/home.py:17` lays out four workflow cards; `home.py:23`, `home.py:29`, `home.py:35`, and `home.py:41` are the four workflow navigation controls; `home.py:48` is the technical details expander.
- Schema Twin page: `src/synth_platform/interfaces/streamlit/pages/schema_twin.py:50` defines session-state keys; `schema_twin.py:64` creates a temp workdir; `schema_twin.py:82` reads product defaults; `schema_twin.py:107` captures intent; `schema_twin.py:127` uploads schema files; `schema_twin.py:136` parses uploaded schema bytes; `schema_twin.py:207` sets row count; `schema_twin.py:216` sets locale; `schema_twin.py:223` sets seed; `schema_twin.py:232` sets export format; `schema_twin.py:238` detects LLM-eligible columns; `schema_twin.py:246` captures the LLM toggle; `schema_twin.py:261` triggers generation; `schema_twin.py:269` calls `generate_from_schema`; `schema_twin.py:282` packages download bytes; `schema_twin.py:353` selects preview table; `schema_twin.py:364` computes validation highlights; `schema_twin.py:404` calls `TransferService.downloadable_bytes`; `schema_twin.py:411` serves the download button.
- Database Twin page: `src/synth_platform/interfaces/streamlit/pages/database_twin.py:85` defines session-state keys; `database_twin.py:118` creates a temp workdir; `database_twin.py:141` reads product defaults; `database_twin.py:143` defines downstream reset behavior; `database_twin.py:172` and `database_twin.py:182` manage `RunManifest`; `database_twin.py:219` captures intent; `database_twin.py:237` chooses source mode; `database_twin.py:245` controls sample size; `database_twin.py:246` triggers sample database creation; `database_twin.py:251` uploads SQLite; `database_twin.py:270` through `database_twin.py:278` capture PostgreSQL connection settings; `database_twin.py:280` triggers PostgreSQL connection; `database_twin.py:325` rebuilds the source adapter; `database_twin.py:342` triggers discovery; `database_twin.py:454` triggers profiling; `database_twin.py:469` selects a profile table; `database_twin.py:496` triggers semantic inference; `database_twin.py:529` approves the inferred contract; `database_twin.py:568` recommends a synthesizer; `database_twin.py:574` selects model type; `database_twin.py:591` controls DP epsilon; `database_twin.py:602` and `database_twin.py:606` capture DP public bounds; `database_twin.py:612` triggers training; `database_twin.py:649` exports the trained artifact; `database_twin.py:685` disconnects the source; `database_twin.py:699` loads the artifact; `database_twin.py:728` captures per-table row counts; `database_twin.py:750` through `database_twin.py:755` capture category override controls; `database_twin.py:768` captures optional LLM text use; `database_twin.py:778` triggers relational generation; `database_twin.py:784` calls `run_relational_generation`; `database_twin.py:793` calls the Streamlit helper `apply_llm_free_text_to_tables`; `database_twin.py:844` previews generated CSV output; `database_twin.py:883` triggers QA validation; `database_twin.py:897` records the run; `database_twin.py:958` records a blocked transfer attempt when QA fails; `database_twin.py:978` triggers target export; `database_twin.py:1010` gates target download with `TransferService.downloadable_file`; `database_twin.py:1018` serves the download.
- PDF Twin page: `src/synth_platform/interfaces/streamlit/pages/pdf_twin.py:59` defines session-state keys; `pdf_twin.py:81` creates a temp workdir; `pdf_twin.py:116` captures intent; `pdf_twin.py:130` uploads the PDF; `pdf_twin.py:142` creates a `RunManifest`; `pdf_twin.py:160` selects extraction engine; `pdf_twin.py:170` checks Docling availability; `pdf_twin.py:177` triggers profiling; `pdf_twin.py:182` calls `run_document_profiling`; `pdf_twin.py:227` triggers template compilation; `pdf_twin.py:281` triggers redaction; `pdf_twin.py:300` downloads the redacted PDF directly; `pdf_twin.py:314` triggers semantic binding; `pdf_twin.py:362` captures generation seed; `pdf_twin.py:368` triggers synthetic value generation; `pdf_twin.py:370` calls `run_value_generation`; `pdf_twin.py:422` triggers rendering; `pdf_twin.py:440` triggers validation; `pdf_twin.py:441` calls `run_document_validation`; `pdf_twin.py:457` records a blocked transfer before validation; `pdf_twin.py:507` gates validated PDF download; `pdf_twin.py:516` serves the PDF download button.
- Interaction Twin page: `src/synth_platform/interfaces/streamlit/pages/interaction_twin.py:33` defines session-state keys; `interaction_twin.py:45` creates a temp workdir; `interaction_twin.py:62` reads product defaults; `interaction_twin.py:67` uploads transcript files; `interaction_twin.py:72` decodes upload bytes; `interaction_twin.py:73` parses the transcript; `interaction_twin.py:103` captures interaction type; `interaction_twin.py:109` captures output format; `interaction_twin.py:115` captures seed; `interaction_twin.py:116` captures sensitive-info removal; `interaction_twin.py:130` triggers generation; `interaction_twin.py:137` calls `run_interaction_twin`; `interaction_twin.py:178` shows log preview; `interaction_twin.py:179` and `interaction_twin.py:181` expose JSON/report expanders; `interaction_twin.py:189` gates ZIP download with `TransferService.downloadable_bytes`; `interaction_twin.py:200` serves the ZIP download.
- My Projects page: `src/synth_platform/interfaces/streamlit/pages/my_projects.py:19` opens `platform.db`; `my_projects.py:21` creates workflow tabs; `my_projects.py:32` loads summaries; `my_projects.py:38` renders the project table; `my_projects.py:46` opens per-project details; `my_projects.py:53` creates the rename form; `my_projects.py:63` calls `db.update_project`; `my_projects.py:67` loads run rows.
- Settings page: `src/synth_platform/interfaces/streamlit/pages/settings.py:17` opens `platform.db`; `settings.py:18` reads settings; `settings.py:21` captures generation mode; `settings.py:28` captures default record count; `settings.py:35` captures privacy level; `settings.py:42` captures default output format; `settings.py:49` submits the form; `settings.py:52` writes settings.
- Reusable Schema facade: `src/synth_platform/application/workflows/schema_twin.py:237` parses schema bytes; `schema_twin.py:347` summarizes schema; `schema_twin.py:387` returns column details; `schema_twin.py:411` lists LLM text columns; `schema_twin.py:488` runs generation; `schema_twin.py:503` resolves product defaults; `schema_twin.py:532` delegates to `run_schema_pipeline`; `schema_twin.py:536` records history; `schema_twin.py:555` packages ZIP bytes; `schema_twin.py:565` flattens validation highlights.
- Reusable Database facade: `src/synth_platform/application/workflows/database_twin.py:9` through `database_twin.py:33` re-export the stage services; `database_twin.py:37` wraps the full pipeline with optional history recording; `database_twin.py:67` lists workflow stages.
- Reusable PDF facade: `src/synth_platform/application/workflows/pdf_twin.py:7` through `pdf_twin.py:37` re-export PDF stage services; `pdf_twin.py:41` wraps validation with optional history recording; `pdf_twin.py:68` lists workflow stages.
- Reusable Interaction facade: `src/synth_platform/application/workflows/interaction_twin.py:32` runs the interaction workflow; `interaction_twin.py:44` resolves defaults; `interaction_twin.py:48` applies privacy default mapping; `interaction_twin.py:59` calls the engine; `interaction_twin.py:64` records history.
- Platform persistence: `src/synth_platform/infrastructure/persistence/platform_db.py:113` initializes `projects`, `runs`, and `settings`; `platform_db.py:238` creates projects; `platform_db.py:269` lists projects; `platform_db.py:279` updates projects; `platform_db.py:308` records runs; `platform_db.py:370` lists runs; `platform_db.py:386` records transfer attempts; `platform_db.py:448` writes settings; `platform_db.py:460` reads settings.
- Transfer gate: `src/synth_platform/application/services/transfer_service.py:47` gates byte downloads; `transfer_service.py:64` gates file downloads; `transfer_service.py:108` logs and records approvals/blocks; `transfer_service.py:140` applies workflow-specific validation verdicts.

## Inventory: Home

File: `src/synth_platform/interfaces/streamlit/pages/home.py`

### User-Facing Controls

- Four `st.page_link` navigation actions:
  - Open Schema Mode.
  - Open Database Twin.
  - Open PDF Twin.
  - Open Customer Interaction Twin.
- One informational `st.expander` for technical details.

### Direct Backend Calls

- No workflow calls.
- Uses `platform_intro` from `src/synth_platform/interfaces/streamlit/components/common/ux.py`.

### State Dependencies

- None beyond Streamlit page navigation.

### Streamlit-Specific Behavior

- `st.page_link` resolves local Streamlit page files. React needs normal client routing and/or server routes.
- Home is static aside from Streamlit layout primitives.

## Inventory: Schema Twin

File: `src/synth_platform/interfaces/streamlit/pages/schema_twin.py`

### User-Facing Controls

- Intent `st.selectbox`: Development & testing, QA / automated tests, Demonstrations, Data pipeline development, Early project environments.
- Schema `st.file_uploader`: `.json`, `.yaml`, `.yml`, `.sql`.
- Schema review tables: metrics, schema table dataframe, expandable column/relationship details.
- Generation controls:
  - `Rows per table` number input, defaulted from `read_generation_defaults(get_platform_db())`.
  - `Locale` selectbox: `en_US`, `en_GB`, `en_IN`, `de_DE`, `fr_FR`.
  - `Random seed` number input.
  - `Export format` selectbox: `csv`, `parquet`, defaulted from settings when supported.
  - Optional `Use LLM for text-heavy columns only` toggle when `list_llm_text_columns` finds eligible columns.
- `Generate Synthetic Data` primary button.
- Preview table `st.selectbox` and dataframe preview.
- Validation report `st.expander`.
- `Download synthetic dataset (ZIP)` download button, disabled when `TransferService` blocks transfer.

### Direct Backend Calls

- `load_schema_bytes`, `summarize_schema`, `schema_column_details`, `list_llm_text_columns`, `generate_from_schema`, `package_download`, `validation_highlights` from `src/synth_platform/application/workflows/schema_twin.py`.
- `read_generation_defaults` from `src/synth_platform/domain/product_settings.py`.
- `get_platform_db` from `src/synth_platform/infrastructure/persistence/platform_db.py`.
- `TransferService.downloadable_bytes` from `src/synth_platform/application/services/transfer_service.py`.
- Error handling catches `LlmPolicyError` from `src/synth_platform/domain/privacy/llm_policy.py` and `TransferBlockedError` from `src/synth_platform/errors.py`.

### State Dependencies

- `st.session_state`: `schema_intent`, `schema_config`, `schema_summary`, `schema_file_id`, `schema_workdir`, `schema_result`, `schema_zip_bytes`, `schema_run_metrics`, `schema_llm_text`.
- A temporary work directory from `tempfile.mkdtemp(prefix="schema_twin_")`.
- Uploaded file identity is tracked as `name:size` to avoid reprocessing the same upload on rerun.
- `platform.db` provides generation defaults and receives project/run/transfer records through workflow history and `TransferService`.
- `SchemaModeResult` remains in memory after generation; preview tables and export paths are read from that object.

### Streamlit-Specific Behavior

- The whole script reruns after each interaction; file identity and result objects survive via `st.session_state`.
- `st.stop()` is used as step gating after missing schema, missing result, etc.
- `st.status`, `st.write`, and `measure_generation` create an in-process progress/status experience during generation.
- `st.download_button` serves ZIP bytes from memory after synchronous generation and transfer approval.
- React/API replacement needs a schema upload resource, parsed schema preview, generation job, persisted result metadata, preview endpoint, validation endpoint/report payload, and gated download endpoint.

## Inventory: Database Twin

File: `src/synth_platform/interfaces/streamlit/pages/database_twin.py`

### User-Facing Controls

- Intent `st.selectbox`: Development & testing, QA / automated tests, Demonstrations, Analytics prototyping, Safe sharing with partners.
- Source `st.radio`:
  - Use a generated sample database.
  - Upload a SQLite file.
  - Connect to PostgreSQL.
- Sample source controls:
  - `Customers to generate` slider.
  - `Generate sample database` button.
- SQLite upload control:
  - `SQLite database file` uploader for `.db`, `.sqlite`, `.sqlite3`.
- PostgreSQL controls:
  - Host, database name, username, password text inputs.
  - Port number input.
  - SSL mode selectbox: `require`, `verify-full`, `verify-ca`, `disable`.
  - Allowed schemas text input.
  - Allowed tables text input.
  - `Connect to PostgreSQL` button.
- `Discover structure` button; renders schema overview and expandable column details.
- Profiling controls:
  - `Process large tables in chunks` checkbox.
  - Conditional `Chunk size (rows)` number input.
  - `Understand values` button.
  - Profile table selectbox for value-profile preview.
- Semantic inference controls:
  - `Understand data` button.
  - Candidate semantic-type tables.
  - `Approve understanding` button. Current UI auto-approves proposed semantic types; there is no per-column editing control yet.
- Training controls:
  - `Twin type` selectbox: Standard twin or Twin with differential privacy.
  - Conditional DP epsilon slider.
  - Conditional numeric/datetime public bounds via date/number inputs.
  - `Train` button.
- Artifact controls:
  - `Download trained twin` button currently exports artifact and marks it ready; despite the label, the UI does not expose a separate artifact `st.download_button`.
  - `Disconnect source` button.
- Generation controls:
  - Per-table `Rows to generate` number inputs, defaulted from settings.
  - Per-category distribution tweak expanders, bar chart, `Override this distribution` checkbox, and per-category sliders.
  - Optional `Use LLM for text-heavy columns only` toggle.
  - `Generate synthetic database` button.
  - Preview table selectbox and CSV-backed dataframe preview.
- Validation/export controls:
  - `Validate results` button.
  - `Export to target database` button.
  - `Download Database B` download button, disabled when validation/transfer blocks.

### Direct Backend Calls

From `src/synth_platform/application/workflows/database_twin.py`:

- Source/sample/artifact: `build_sample_database`, `load_artifact`, `get_synthesizer_adapter_class`, `run_artifact_export`.
- Stage calls: `run_discovery`, `run_profiling`, `run_inference`, `run_contract_approval`, `run_training_and_sampling`, `run_relational_generation`, `run_qa_validation`, `run_target_write`.
- Loaders/helpers: `load_profile`, `load_dataset_contract`, `load_training_report`, `load_relational_generation_report`, `load_qa_report`, `load_target_write_report`, `recommend_synthesizer`, `RunManifest`, `PROFILE_FILENAME`.

Other direct calls:

- `build_source_adapter`, `postgres_connection_config`, `PostgresConnectionDetails` from `src/synth_platform/interfaces/streamlit/database_source_ui.py`.
- `TransferService.downloadable_bytes` and `TransferService.downloadable_file`.
- `get_platform_db().record_run` after QA validation.
- `read_generation_defaults(get_platform_db())`.
- `apply_llm_free_text_to_tables`, `list_contract_free_text_columns`, `profile_vs_synthetic_drift_rows`, `fidelity_drift_rows`, and other UI/helper functions from `src/synth_platform/interfaces/streamlit/components/common/ux.py`.

### State Dependencies

- `st.session_state`: `db_workdir`, `db_source_path`, `db_source_type`, `db_source_config`, `db_uploaded_file_id`, `db_manifest`, `db_discovery`, `db_profile`, `db_candidates`, `db_contract`, `db_model_type`, `db_dp_epsilon`, `db_dp_bounds`, `db_training_report`, `db_artifact_path`, `db_source_disconnected`, `db_loaded_artifact`, `db_row_counts`, `db_category_overrides`, `db_relational_report`, `db_qa_report`, `db_target_path`, `db_write_report`, `db_chunk_size`, `db_run_metrics`, `db_llm_text`, `db_llm_evidence`, `db_intent`.
- Temporary work directory from `tempfile.mkdtemp(prefix="db_twin_demo_")`.
- `RunManifest` rooted under the session workdir; `_manifest_for_new_outputs` creates a new run when output filenames would collide.
- Source adapter instances are rebuilt from session-stored source type/config on each rerun.
- PostgreSQL credentials live in session state as connection config URL after connection.
- Artifact path and loaded artifact object live in session state after training/export.
- Generated relational CSV paths are read directly for previews and drift analysis.
- `platform.db` records QA run history and transfer attempts.

### Streamlit-Specific Behavior

- Step gating uses reruns and `st.stop()` extensively.
- Upload identity guard avoids resetting downstream state every rerun for SQLite uploads.
- Buttons are momentary; completion is represented by persisted session keys.
- Long-running discovery/profile/inference/training/generation/export happen synchronously in request-thread-equivalent UI execution.
- Source disconnect is only a boolean in `st.session_state`; a real API must define what disconnect means for stored credentials and adapter lifetimes.
- Generated artifact/download files sit in local temp dirs; React/API needs durable run storage and cleanup.
- Preview and drift panels read CSV files directly from server filesystem; API needs table-preview and drift endpoints.

## Inventory: PDF Twin

File: `src/synth_platform/interfaces/streamlit/pages/pdf_twin.py`

### User-Facing Controls

- Intent `st.selectbox`: Document testing, QA / automation, Demonstrations, Privacy-safe sharing.
- PDF `st.file_uploader`.
- Extraction engine `st.selectbox`: Automatic (native/OCR), Docling; disabled after profiling.
- `Discover document` button.
- Template details expander after template compilation.
- `Build twin template` button.
- Optional redacted PDF:
  - `De-identify` button.
  - `Download redacted PDF` download button.
- `Understand fields` button and field-binding details expander.
- Value generation:
  - `Seed` number input.
  - `Generate values` button.
  - Generated value metrics/captions.
- Rendering/validation/download:
  - `Build twin document` button.
  - `Validate twin` button.
  - `Download synthetic twin PDF` download button, disabled until validation passes transfer gate.

### Direct Backend Calls

From `src/synth_platform/application/workflows/pdf_twin.py`:

- `RunManifest`, `PDFDocumentAdapter`, `is_docling_available`, `DOCUMENT_PROFILE_FILENAME`.
- `run_document_profiling`, `load_document_profile`.
- `run_template_compilation`, `load_document_template`.
- `run_document_deidentification`, `load_document_deidentification_report`.
- `run_semantic_binding`, `load_document_binding_map`.
- `run_value_generation`, `load_document_synthetic_values`.
- `run_document_rendering`, `load_document_ground_truth`.
- `run_document_validation`, `load_document_validation_report`.

Other direct calls:

- `TransferService.downloadable_file`.
- `get_platform_db()` for validation history and transfer recording.
- UI helpers such as `list_pdf_narrative_bindings`, `validation_badge`, `render_data_drift_panel`, `render_run_metrics`.

### State Dependencies

- `st.session_state`: `pdf_workdir`, `pdf_source_path`, `pdf_manifest`, `pdf_doc_id`, `pdf_profile`, `pdf_template`, `pdf_binding_map`, `pdf_values`, `pdf_ground_truth`, `pdf_rendered_path`, `pdf_validation_report`, `pdf_redacted_path`, `pdf_deidentification_report`, `pdf_run_metrics`, `pdf_llm_text`, `pdf_intent`, `pdf_extraction_engine`.
- Temporary work directory from `tempfile.mkdtemp(prefix="pdf_twin_demo_")`.
- Uploaded file is written to `source.pdf`.
- `RunManifest` persists stage output paths under the temp workdir.
- Validation history and transfer attempts are recorded to `platform.db`.

### Streamlit-Specific Behavior

- Each stage is synchronous and moves forward by setting session keys, then `st.rerun()`.
- `st.stop()` gates each later stage until prerequisite session keys are present.
- Download gating is tied directly to `st.download_button`; before validation, the PDF button is shown disabled after `TransferService` rejects missing validation.
- Extraction engine selectbox is disabled once profile exists; React must enforce immutable stage settings or offer explicit restart/reset behavior.
- Redacted PDF is an optional sibling path that starts after template compilation and has its own downloadable artifact.

## Inventory: Customer Interaction Twin

File: `src/synth_platform/interfaces/streamlit/pages/interaction_twin.py`

### User-Facing Controls

- Transcript `st.file_uploader`: `.txt`, `.log`.
- Configure controls:
  - `Interaction Type` selectbox: Customer Support, Billing Support, Account Support, Benefit Activation.
  - `Output Format` selectbox: Structured JSON + Synthetic Logs.
  - `Random seed` number input.
  - `Remove sensitive information` toggle, defaulted from product privacy level.
- Preview dataframe of first 10 parsed turns.
- `Generate Interaction Twin` primary button.
- Result views:
  - Synthetic log `st.text_area`.
  - Structured JSON expander.
  - Validation report expander.
  - `Download interaction twin (ZIP)` download button, disabled when transfer blocks.

### Direct Backend Calls

- `parse_transcript` and `run_interaction_twin` from `src/synth_platform/application/workflows/interaction_twin.py`.
- `privacy_level_removes_sensitive_information`, `read_generation_defaults` from `src/synth_platform/domain/product_settings.py`.
- `get_platform_db()` for settings, history, and transfer recording.
- `TransferService.downloadable_bytes`.

### State Dependencies

- `st.session_state`: `interaction_file_id`, `interaction_source_text`, `interaction_source_name`, `interaction_preview`, `interaction_result`, `interaction_workdir`, `interaction_run_metrics`.
- Temporary work directory from `tempfile.mkdtemp(prefix="interaction_twin_")`.
- Uploaded file identity is tracked as `name:size`.
- Parsed `Transcript` and `InteractionTwinResult` objects remain in memory.
- `platform.db` records run history and transfer attempts.

### Streamlit-Specific Behavior

- Uploaded file content is decoded and parsed during rerun.
- Generation is synchronous inside `st.status`.
- Result object and generated package bytes remain in session memory.
- `st.download_button` serves ZIP bytes after transfer approval.

## Inventory: My Projects

File: `src/synth_platform/interfaces/streamlit/pages/my_projects.py`

### User-Facing Controls

- Tabs: All Projects, Schema, Database, Document, Interaction.
- Projects dataframe per tab.
- Per-project expander showing metrics and run rows.
- Per-project rename form:
  - `Project name` text input.
  - `Rename project` submit button.

### Direct Backend Calls

- `get_platform_db`.
- `load_project_summaries`, `project_table_rows`, `project_run_rows` from `src/synth_platform/interfaces/streamlit/project_ui.py`.
- `db.update_project` directly on rename.

### State Dependencies

- Reads `platform.db` projects/runs.
- No explicit session state beyond Streamlit form state.

### Streamlit-Specific Behavior

- `st.tabs`, expanders, and forms provide local UI state.
- Successful rename calls `st.rerun()` to refresh project rows.
- React/API needs project listing, run listing, and project update endpoints.

## Inventory: Settings

File: `src/synth_platform/interfaces/streamlit/pages/settings.py`

### User-Facing Controls

- Settings form:
  - `Generation mode` selectbox: `schema_driven`, `source_driven`.
  - `Default record count` number input.
  - `Privacy level` selectbox: `standard`, `restricted`, `strict`.
  - `Default output format` selectbox: `csv`, `parquet`, `sqlite`, `pdf`.
  - `Save settings` submit button.

### Direct Backend Calls

- `get_platform_db`.
- `read_product_settings`, `write_product_settings` from `src/synth_platform/interfaces/streamlit/project_ui.py`.
- `PlatformDB.write_settings` indirectly through `write_product_settings`.

### State Dependencies

- Reads and writes the `settings` table in `platform.db`.

### Streamlit-Specific Behavior

- Form submission batches setting changes in-process.
- No client/server validation boundary exists today; invalid values are constrained by selectboxes and number input.

## Minimal API Surface for Feature Parity

The API should model workflow sessions and jobs explicitly. The shapes below are high-level, not OpenAPI-complete.

### Cross-Cutting

- `GET /api/health`
  - Response: service status, version, feature flags such as Docling availability.
  - Type: simple request/response.
- `POST /api/workflows/{workflow}/sessions`
  - Request: workflow type, intent, optional client-generated name.
  - Response: `session_id`, initial step state, defaults.
  - Type: simple request/response.
  - Guarantees: must create or later reconcile with `platform.db` project/run records.
- `GET /api/workflows/{workflow}/sessions/{session_id}`
  - Response: durable step state, available actions, stage outputs, validation status.
  - Type: simple request/response.
- `POST /api/jobs`
  - Request: action name, `session_id`, action-specific parameters.
  - Response: `job_id`, status URL.
  - Type: long-running job for discovery/profile/inference/training/generation/render/validation/export.
- `GET /api/jobs/{job_id}`
  - Response: `queued|running|succeeded|failed`, progress messages, errors, output references.
  - Type: polling minimum; SSE/WebSocket optional later.
- `GET /api/downloads/{download_id}`
  - Response: file/bytes stream.
  - Type: download.
  - Guarantees: must call `TransferService` before returning content and record allowed/blocked attempts.

### Schema Twin

- `POST /api/schema/sessions`
  - Request: intent.
  - Response: `session_id`, product defaults.
  - Type: simple.
- `POST /api/schema/sessions/{id}/schema-file`
  - Request: multipart schema file.
  - Response: parsed schema summary, column details, eligible LLM text columns.
  - Type: upload + parsing.
  - Existing functions: `load_schema_bytes`, `summarize_schema`, `schema_column_details`, `list_llm_text_columns`.
- `POST /api/schema/sessions/{id}/generate`
  - Request: row count, locale, seed, export format, LLM text enabled.
  - Response: `job_id`.
  - Type: long-running job, though small runs may complete quickly.
  - Existing function: `generate_from_schema`.
  - Guarantees: respect `LlmPolicy`; read product defaults when values omitted; record to `platform.db`.
- `GET /api/schema/sessions/{id}/preview?table=...&limit=...`
  - Response: table names, row counts, preview rows.
  - Type: simple.
- `GET /api/schema/sessions/{id}/validation`
  - Response: validation report and flattened highlights.
  - Type: simple.
- `POST /api/schema/sessions/{id}/download`
  - Response: `download_id` or immediate ZIP stream.
  - Type: gated download.
  - Existing functions: `package_download`, `TransferService.downloadable_bytes`.

### Database Twin

- `POST /api/database/sessions`
  - Request: intent.
  - Response: `session_id`, defaults.
  - Type: simple.
- `POST /api/database/sessions/{id}/sample-source`
  - Request: customer count.
  - Response: source summary.
  - Type: job or simple for current sample size.
  - Existing function: `build_sample_database`.
- `POST /api/database/sessions/{id}/sqlite-source`
  - Request: multipart SQLite file.
  - Response: source summary.
  - Type: upload.
- `POST /api/database/sessions/{id}/postgres-source`
  - Request: host, port, database, username, password, SSL mode, allowed schemas/tables.
  - Response: connection health and source handle.
  - Type: simple request/response, but must handle secret storage and cleanup explicitly.
  - Existing helpers: `postgres_connection_config`, `build_source_adapter`.
- `POST /api/database/sessions/{id}/discover`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_discovery`.
- `POST /api/database/sessions/{id}/profile`
  - Request: sample limit, chunked flag, chunk size.
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_profiling`.
- `POST /api/database/sessions/{id}/infer-semantics`
  - Request: sample limit, chunk size.
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_inference`.
- `POST /api/database/sessions/{id}/contract`
  - Request: semantic decisions by table/column.
  - Response: approved contract.
  - Type: simple or short job.
  - Existing function: `run_contract_approval`.
- `GET /api/database/sessions/{id}/training-recommendation`
  - Response: recommended model type and reasons.
  - Type: simple.
  - Existing function: `recommend_synthesizer`.
- `POST /api/database/sessions/{id}/train`
  - Request: model type, DP epsilon, public column bounds, model kwargs.
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_training_and_sampling`.
- `POST /api/database/sessions/{id}/artifact`
  - Request: artifact version.
  - Response: `job_id`, later artifact metadata.
  - Type: job + optional download if product wants trained artifact download parity.
  - Existing function: `run_artifact_export`.
- `POST /api/database/sessions/{id}/disconnect-source`
  - Request: none.
  - Response: source disconnected flag.
  - Type: simple.
  - Guarantee: must remove/expire source credentials or handles, not just flip a UI boolean.
- `POST /api/database/sessions/{id}/generate`
  - Request: row counts by table, category overrides, LLM text enabled.
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_relational_generation`; optional UI-owned `apply_llm_free_text_to_tables` may need an application-layer wrapper.
  - Guarantees: source-free after disconnect; respect `LlmPolicy`; settings default record counts apply.
- `GET /api/database/sessions/{id}/preview?table=...&limit=...`
  - Response: preview rows, row counts, drift signals.
  - Type: simple but reads generated CSV outputs.
- `POST /api/database/sessions/{id}/validate`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_qa_validation`.
  - Guarantees: record QA run to `platform.db`.
- `POST /api/database/sessions/{id}/export-target`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_target_write`.
- `POST /api/database/sessions/{id}/download-target`
  - Response: `download_id`.
  - Type: gated download.
  - Existing function: `TransferService.downloadable_file`.

### PDF Twin

- `POST /api/pdf/sessions`
  - Request: intent.
  - Response: `session_id`, doc id, Docling availability.
  - Type: simple.
- `POST /api/pdf/sessions/{id}/source-file`
  - Request: multipart PDF.
  - Response: source metadata.
  - Type: upload.
- `POST /api/pdf/sessions/{id}/profile`
  - Request: extraction engine.
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_document_profiling`.
- `POST /api/pdf/sessions/{id}/template`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_template_compilation`.
- `POST /api/pdf/sessions/{id}/redact`
  - Response: `job_id`.
  - Type: job + downloadable artifact.
  - Existing function: `run_document_deidentification`.
- `POST /api/pdf/sessions/{id}/bindings`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_semantic_binding`.
- `POST /api/pdf/sessions/{id}/values`
  - Request: seed, LLM text enabled if exposed later.
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_value_generation`.
- `POST /api/pdf/sessions/{id}/render`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_document_rendering`.
- `POST /api/pdf/sessions/{id}/validate`
  - Response: `job_id`.
  - Type: job.
  - Existing function: `run_document_validation`.
  - Guarantees: record to `platform.db`.
- `POST /api/pdf/sessions/{id}/download-synthetic`
  - Response: `download_id`.
  - Type: gated download.
  - Existing function: `TransferService.downloadable_file`.
- `POST /api/pdf/sessions/{id}/download-redacted`
  - Response: `download_id`.
  - Type: download. Product decision needed: whether redacted PDF should also pass through `TransferService` or have its own gate.

### Customer Interaction Twin

- `POST /api/interaction/sessions`
  - Request: optional intent/project name.
  - Response: `session_id`, product defaults.
  - Type: simple.
- `POST /api/interaction/sessions/{id}/transcript`
  - Request: multipart `.txt` or `.log`.
  - Response: parsed preview, turn count, speaker counts.
  - Type: upload + parsing.
  - Existing function: `parse_transcript`.
- `POST /api/interaction/sessions/{id}/generate`
  - Request: interaction type, output format, remove-sensitive flag, seed.
  - Response: `job_id` or immediate result for small transcripts.
  - Type: likely job for consistent UX.
  - Existing function: `run_interaction_twin`.
  - Guarantees: default privacy mapping from settings when omitted; record to `platform.db`.
- `GET /api/interaction/sessions/{id}/result`
  - Response: synthetic preview, structured JSON, validation report, redaction report.
  - Type: simple.
- `POST /api/interaction/sessions/{id}/download`
  - Response: `download_id`.
  - Type: gated download.
  - Existing function: `TransferService.downloadable_bytes`.

### Projects

- `GET /api/projects?workflow_type=...`
  - Response: project summaries equivalent to `project_table_rows`.
  - Type: simple.
  - Existing functions: `load_project_summaries`, `PlatformDB.list_projects`.
- `GET /api/projects/{project_id}/runs`
  - Response: run rows equivalent to `project_run_rows`.
  - Type: simple.
  - Existing functions: `project_run_rows`, `PlatformDB.list_runs`.
- `PATCH /api/projects/{project_id}`
  - Request: name and later status/metadata if needed.
  - Response: updated project.
  - Type: simple.
  - Existing function: `PlatformDB.update_project`.
  - Status: implemented for project name updates.

### Settings

- `GET /api/settings`
  - Response: generation mode, default record count, privacy level, default output format.
  - Type: simple.
  - Existing function: `read_product_settings`/`read_generation_defaults`.
- `PUT /api/settings`
  - Request: full settings payload.
  - Response: saved settings.
  - Type: simple.
  - Existing function: `write_product_settings`.
  - Guarantees: preserve Phase 9 semantics: settings are defaults, run-level controls override them.

## Hidden Costs Streamlit Currently Covers

- Authentication and sessions: Streamlit currently has no auth. A deployed React/API product must decide whether local-only no-auth is acceptable, or whether users, teams, permissions, and project ownership are required.
- Workflow session durability: `st.session_state` currently holds rich Python objects. API sessions need serializable state, storage, expiry, cleanup, and restart behavior.
- File upload handling: uploads currently arrive as in-memory Streamlit objects and are written to temp dirs. API upload handling needs size limits, MIME/extension validation, virus/security posture, temp/durable storage, cleanup, and path traversal protections.
- Long-running jobs: Streamlit blocks in-process with `st.status`, `st.spinner`, and reruns. API needs job queue semantics, progress logs, failure payloads, cancellation, retries, and polling/SSE/WebSocket.
- Artifact storage: generated CSVs, PDFs, ZIPs, and SQLite DBs currently live under temp workdirs. A real backend needs a storage model and lifecycle policy.
- Download authorization/gating: `st.download_button` currently appears only after direct `TransferService` calls. API download endpoints must guarantee every user-facing artifact passes the gate.
- Source credential handling: PostgreSQL credentials currently become an in-memory/session config URL. A real API needs secret handling, redaction in logs, expiration, and disconnect semantics.
- CORS/deployment topology: React and FastAPI may be separate origins/services. CORS, cookies/tokens, local development proxies, and production routing need explicit design.
- Error messaging: `show_user_error`, `st.error`, `st.warning`, and `st.info` currently render errors inline. API must standardize error codes, user-safe messages, technical diagnostics, and validation errors.
- UI state reset rules: each page has custom downstream reset behavior. API must encode stage dependencies so stale outputs cannot be mixed with newer inputs.
- Preview pagination: Streamlit reads pandas dataframes or CSVs directly. API needs bounded preview endpoints and probably pagination for large generated outputs.
- Progress metrics: `measure_generation` currently lives in the UI layer. API jobs need server-side timing/memory metrics if the React UI should keep run metrics.
- Environment capability flags: PDF Docling availability is checked in-process by `is_docling_available`. React needs capability endpoints.
- Local-only persistence assumption: `platform.db` is local SQLite. Multi-user deployment may require a server database and migrations, even if the domain API remains similar.

## Reusable vs. Needs Untangling

### Reusable Mostly As-Is

- Schema workflow facade: `src/synth_platform/application/workflows/schema_twin.py` already wraps schema parsing, summary, LLM column detection, generation, packaging, and validation highlights.
- Database workflow facade: `src/synth_platform/application/workflows/database_twin.py` re-exports stage functions and includes `run_database_twin_pipeline`.
- PDF workflow facade: `src/synth_platform/application/workflows/pdf_twin.py` re-exports stage functions and wraps validation history recording.
- Interaction workflow facade: `src/synth_platform/application/workflows/interaction_twin.py` wraps transcript parsing/generation defaults/history.
- Transfer gate: `src/synth_platform/application/services/transfer_service.py`.
- Product settings defaults: `src/synth_platform/domain/product_settings.py`.
- Platform persistence APIs: `PlatformDB` is usable for local/single-user API mode.

### Needs Untangling or New Application-Layer Wrappers

- Database page owns orchestration details: manifest collision handling (`_active_manifest`, `_manifest_for_new_outputs`), source disconnect flag, step reset order, generated CSV previews, and manual `get_platform_db().record_run`.
- Database optional LLM text rewrite is invoked through `apply_llm_free_text_to_tables` in the Streamlit UI helper module. That should move behind an application/API-safe service before React uses it.
- PostgreSQL source connection helpers live under `interfaces/streamlit/database_source_ui.py`; API should not import Streamlit interface modules for source handling.
- Project/settings view-model helpers live under `interfaces/streamlit/project_ui.py`. Their pure formatting logic can move to a shared presentation/API view-model module or be reimplemented in API serializers.
- Streamlit pages store non-serializable objects in memory: `SchemaConfig`, `SchemaModeResult`, `RunManifest`, loaded artifacts, parsed transcripts, and PDF dictionaries. API must persist references and serializable summaries instead.
- PDF redacted download currently bypasses an explicit validation report gate in the UI. The desired API behavior should be decided before exposing it.

## Effort Estimate

### Phase A: API Foundation and Workflow Session Model

- Covers: FastAPI app object, route registration, error envelope, settings/project routes, upload storage abstraction, workflow session records, job model, polling endpoint, download endpoint pattern, CORS/dev proxy.
- Size: Large, 1.5-2.5 weeks.
- Risks: auth decision can expand this substantially; replacing `st.session_state` with durable state is easy to underestimate.
- Incremental: yes; can exist beside Streamlit.

### Phase B: React Shell and Shared UX Patterns

- Covers: React app scaffold, routing, page shell, workflow stepper, forms, upload components, status polling, error/toast/inline validation patterns, tables/previews, download handling.
- Size: Medium-large, 1-2 weeks.
- Risks: design polish, table rendering, file upload progress, and accessibility can stretch timeline.
- Incremental: yes; initial shell can link only migrated workflows.

### Phase C: Schema Twin Proof of Concept

- Covers: schema upload/parse, review, configure, generate job, preview, validation, gated ZIP download.
- Size: Medium, 1-1.5 weeks after Phase A/B.
- Risks: preserving package/download semantics and LLM policy errors; large row count behavior.
- Incremental: yes; best first migration candidate.

### Phase D: Customer Interaction Twin

- Covers: transcript upload/parse, configuration, generation job/result preview, validation/redaction report, gated ZIP download.
- Size: Medium, 0.75-1.25 weeks.
- Risks: transcript validation edge cases and preserving default privacy behavior.
- Incremental: yes; can follow Schema as the second workflow.

### Phase E: PDF Twin

- Covers: PDF upload, profiling, template compilation, optional de-identification, semantic binding, value generation, rendering, validation, gated download, capability checks for Docling.
- Size: Large, 1.5-2.5 weeks.
- Risks: file sizes, OCR/Docling latency, progress reporting, artifact storage, redacted artifact policy.
- Incremental: yes, but should not start until the job/session model is proven.

### Phase F: Database Twin

- Covers: sample/SQLite/PostgreSQL source setup, discovery, profiling, inference, contract approval, model recommendation, DP controls, training, artifact export, disconnect, source-free generation, category overrides, optional LLM text, QA validation, target DB export, gated download.
- Size: Very large, 2.5-4 weeks.
- Risks: source credential security, long job durations, multi-stage state invalidation, generated table preview performance, artifact lifecycle, making disconnect real instead of a UI boolean.
- Incremental: yes, but this should be the last migrated workflow unless Database Twin is the business-critical path.

### Phase G: Production Hardening

- Covers: auth if required, deployment topology, CORS, environment config, storage cleanup jobs, observability, rate/size limits, security review, API tests, e2e React tests, migration documentation.
- Size: Large, 1.5-3 weeks.
- Risks: switching from local SQLite/temp dirs to shared production services can turn this into a larger platform project.
- Incremental: partly; some hardening can happen after initial internal React rollout, but auth/storage/download security should not be deferred for external deployment.

## Total Estimate

- Internal/local React + FastAPI parity with no auth and local filesystem/SQLite assumptions: roughly 6-8 weeks.
- Deployable multi-user React + FastAPI product with auth, durable storage, proper secret handling, observability, and hardened file handling: roughly 8-12+ weeks.
- A thin demo that migrates only Schema Twin could be built in about 3-4 weeks including API foundation and React shell, but that would not prove the hardest Database/PDF migration risks.

## Final Recommendation

Choose an incremental migration. Keep Streamlit running while the API and React shell mature, migrate Schema Twin first, then Customer Interaction Twin, then PDF Twin, then Database Twin. This order proves the shared upload/job/download patterns early while delaying the most stateful and security-sensitive workflows until the backend session model is stable.

A big-bang rewrite is higher risk because every workflow currently depends on Streamlit reruns, in-memory session objects, and temporary local artifact paths in slightly different ways. The existing application and engine layers are reusable enough that the right move is not to rewrite the product at once; it is to put a real HTTP/job/storage boundary around the tested workflows and retire Streamlit one workflow at a time.
