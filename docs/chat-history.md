# Chat History

## Project Direction

- Build one synthetic-data platform with multiple SSOT input workflows.
- Different input types can vary, but the platform behavior should stay consistent:
  schema input, database input, PDF/document input, and customer interaction input all lead to a twin, generation, validation, and downloadable artifacts.
- Avoid presenting the workflows as disconnected demos.
- Keep technical details optional and expandable in the UI.

## Database Twin

- Fixed import/runtime issues around `synth_platform` by ensuring local `src` is on the Streamlit launcher path.
- Removed hard dependency on SDV Gaussian Copula where it caused import problems and kept safe fallback adapters available.
- Added user-controlled row counts per table instead of a single inflexible default.
- Made source row counts reference-only. They do not constrain generation count.
- Added distribution override controls with editable numeric percentage fields.
- Preserved user-entered category percentages without automatic normalization.
- Added exact total validation so requested percentages must total 100%.
- Added learned/requested/generated distribution comparison support.
- Added visible ML training evidence for database mode.
- Improved profiler and PII/sensitive field handling so Faker-style generation is used where appropriate.

## Schema Mode

- Updated wording to say supported schema input is SQL DDL.
- Removed DBML wording unless parsing exists.
- Clarified schema files contain `CREATE TABLE` statements, fields, keys, and relationships.
- Added user-controlled row/table counts.
- Kept source-free quality wording honest: schema mode cannot claim source-distribution fidelity because no source records exist.

## PDF Twin

- Reworked positioning around one platform and shared identity/entity concepts.
- Required PDF values to come from the shared synthetic entity/database result rather than independently creating a second identity.
- Added cross-output validation expectations for fields such as name, account number, balance, and transactions.
- Reworded document quality claims to “preserves useful document structure” unless exact layout fidelity is proven.

## Customer Interactions Twin

- Renamed the workflow from Transcript Twin to Customer Interactions Twin.
- Updated the home page and app navigation to show Customer Interactions Twin.
- Changed the workflow language so users provide a customer interaction and receive a synthetic customer interaction.
- Simplified the UI:
  - “Add customer interaction”
  - “Preview”
  - “Build twin contract”
  - “Create customer conversation”
  - “Validate interaction”
  - “Download twin”
- Replaced technical output columns with user-friendly columns:
  - `Message #`
  - `Who`
  - `What they said`
- Changed generated speakers from `speaker_1` style labels to names like `Customer` and `Agent`.
- Made generated output read like a normal customer-support conversation.
- Added a generated twin preview before download.
- ZIP download now uses `customer_interactions_twin_artifact.zip`.

## NVIDIA NeMo

- Added optional NVIDIA SDK adapters instead of only documentation/probes.
- NeMo Curator can be used for PII redaction during customer interaction contract build when available.
- NeMo Guardrails can be used during validation when available and configured.
- If the current app Python environment cannot import the SDKs, the UI now says they are not available in this app environment rather than claiming they are not installed everywhere.
- The current checked app environment is Python 3.14, while the NVIDIA optional extra is guarded for Python versions below 3.14.

## UI / UX Corrections

- Made product wording less technical and more user-friendly.
- Moved advanced NVIDIA details into a collapsed section.
- Removed noisy SDK tables from the main customer interaction flow.
- Reduced visible technical JSON and moved it into expanders.
- Made validation messages clearer:
  - no source replay
  - no raw source retained
  - repeated phrasing is a warning, not an automatic hard failure

## Testing And Validation

Repeated validation runs included:

```text
tests/unit/test_transcript_ssot.py
tests/unit/test_nvidia_nemo_integration.py
tests/e2e/workflows/test_home_ui.py
tests/e2e/workflows/test_schema_twin_ui.py
tests/e2e/workflows/test_database_twin_ui.py
tests/e2e/workflows/test_pdf_twin_ui.py
tests/e2e/workflows/test_transcript_twin_ui.py
```

Recent passes included:

```text
8 passed
9 passed
6 passed, 1 skipped
compileall passed
git diff --check passed
```

## Current Naming Note

Some internal module and test filenames still use `transcript_twin` for stability. The user-facing product name is Customer Interactions Twin.
