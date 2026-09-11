# Shared Product Workspace

## Purpose

The product workspace supports the common UI around Schema, Database,
Document/PDF, and Customer Interaction Twin without replacing their workflow
facades. It provides local sessions, progress, normalized result presentation,
projects, templates, settings, and artifact metadata.

```text
Streamlit shell
  -> WorkspaceService (metadata/state only)
  -> repository ports
  -> atomic local JSON/filesystem adapters

Specialized Streamlit page
  -> specialized application workflow
  -> engine/domain validation and artifacts
  -> shared result presenter
  -> WorkspaceService
```

`application/coordinator` is not in this persistence or execution path. It
continues to select capabilities only.

## Canonical screens

1. **Home / Chat** offers capability cards and deterministic prompt/attachment
   routing.
2. **Schema Twin** follows Input -> Review -> Generate -> Results.
3. **Database Twin** follows Connect -> Configure/approve -> Generate -> Results.
4. **Document Twin** follows Upload -> Configure -> Generate -> Results.
5. **Customer Interaction Twin** follows Upload -> Configure -> Generate ->
   Results.
6. **Generation progress** renders workflow-specific stage labels grouped under
   shared macro steps. It does not fabricate percentages.
7. **Results** separates execution, validation, and release state; exposes
   preview descriptors, metrics, lineage, downloads, regenerate, and save.
8. **My Projects** stores named collections of session/result/artifact IDs.
9. **Settings & Help** stores safe preferences, presents deployment controls as
   read-only, and links to each specialized workflow.

The sidebar also provides a dedicated Templates page and direct workflow access.

## Application contracts

`application/dto/workspace.py` defines:

- `WorkflowSession`: capability, safe SHA-256 source/configuration fingerprints,
  current macro step, execution state, stage events, and result/run references;
- `ProgressEvent`: specialized stage ID/label, macro step, one of
  `pending/running/succeeded/warn/failed/blocked`, timestamps, safe message, and
  optional counts;
- `WorkflowResultView`: separate execution status, validation status, nullable
  release verdict, blockers, warnings, metrics, preview descriptors, artifacts,
  and lineage;
- `ArtifactRef`: local approved-root path, media type, size, checksum, and
  download policy;
- `Project`: session/result/artifact references only;
- `Template`: named, versioned, workflow-scoped reusable definition;
- `UserPreferences`: effective Schema defaults for locale, row count, and table
  format;
- `RuntimeSettingsView`: read-only deployment-selected boundaries.

`application/ports/workspace_repositories.py` defines repository protocols.
`application/services/workspace.py` owns CRUD and state transitions. Operations
exposed through the shared UI cross `application/services/tool_gateway.py`,
which accepts only strict commands from `application/dto/tool_commands.py` and
checks local operation/resource permissions before delegation. Artifact reads
must match their session; delete is denied to the UI principal by default.
`application/services/result_presentation.py` adapts specialized outcomes for
the shared result screen; it does not rerun validation or derive a new release
policy.

## Persistence

`infrastructure/persistence/local_workspace.py` stores one JSON document per
session, result, project, or artifact and one preferences document. Writes use a
temporary file, flush/fsync, and atomic replacement. Result/session/project
completion uses a write-ahead journal and cross-process lock so an interrupted
commit is replayed when the workspace is reopened. IDs are restricted to a
cross-platform path-safe character set.

Artifact registration resolves the source path, requires a regular file below an
approved local root, computes SHA-256, and copies bytes into an immutable
content-addressed snapshot. Shared downloads resolve through the catalog and
verify both size and checksum before serving bytes.

The default root is:

```text
<SP_OUTPUT_ROOT>/workspace
  artifact_blobs/sha256/
  artifacts/
  projects/
  results/
  sessions/
  preferences.json
  runs/
```

Original SQLite/PDF inputs use permission-restricted process-temporary staging
rather than this durable workspace. Database staging is deleted when the source
is explicitly disconnected; PDF staging is deleted after result completion.
Current-process directories are removed at process exit, and creation performs a
24-hour stale-directory sweep to bound abandoned-session retention. Raw
Interaction text is never written, including to staging.

The adapters are intentionally local and synchronous. The supported UI launcher
binds Streamlit to `127.0.0.1`; this process-global workspace is single-user
localhost functionality, not an authenticated multi-user service. They are not
advertised as multi-host transaction storage. PostgreSQL, Redis, object storage,
queues, workers, MCP, and generic workflow agents were not added.

## Workflow ownership preserved

- **Schema:** the existing schema pipeline still owns generation, progress,
  validation, previews, and table exports. Shared code catalogs its outputs.
- **Database:** interactive discovery, inference, human semantic approval,
  training, source disconnect, generation, QA, and target write remain on the
  existing page/facade path. The headless auto-approval runner is not used.
- **Document:** extraction/profile, template, binding, generation, rendering,
  and validation remain specialized. De-identification remains optional and is
  not turned into a required generation stage.
- **Interaction:** sanitization occurs before optional model use; the model call
  remains zero-or-one with deterministic fallback; privacy/source-replay release
  checks and the exact four-file package remain authoritative.

## Privacy and settings boundaries

Persisted workspace models have no fields for source payloads, raw transcript
text, passwords, tokens, database URLs, or resolved credentials. Interaction
session titles are generic and only the source/configuration hashes are stored.
Its artifact catalog contains only release-admitted sanitized output paths.

Optional model use is guarded separately: input masking, prompt-injection,
scope, and deterministic content checks run before a loopback-only Ollama call;
output size/JSON/privacy/content checks run before workflow-specific grounding
and release validation. Guardrail reports contain categories/counts only. These
rules are conservative inspectable controls, not comprehensive moderation.

`Settings.from_env()` selects deployment controls:

- `SP_OUTPUT_ROOT`
- `SP_STAGING_ROOT`
- `SP_ALLOWED_BACKENDS` (comma separated)
- `SP_MAX_ROWS`
- `SP_MIN_FIDELITY`
- `SP_GUARDRAIL_POLICY_VERSION`
- `SP_GUARDRAIL_MAX_INPUT_CHARS`
- `SP_GUARDRAIL_MAX_OUTPUT_CHARS`
- `SP_LOCAL_MODEL_HOST` (must be loopback when model composition is used)
- existing seed/holdout settings

The Settings page can update only `UserPreferences`; it cannot write deployment
controls or credentials.

## Current limitation

Chat attachments are currently used for deterministic routing; specialized pages
still own ingestion and may ask the user to provide the source there. Carrying
bytes across the hand-off remains a separate interface enhancement so it cannot
bypass workflow-specific input validation.
