# Phase 9 Summary: Persisted Settings Wired Into Workflow Defaults

## Scope

- Wired persisted `platform.db` settings into actual generation defaults where a setting has a clear workflow equivalent.
- Preserved per-run controls as overrides; settings now provide starting/default values, not forced policy.
- Added a shared settings resolver in `src/synth_platform/domain/product_settings.py` so application, engine, and Streamlit layers can use the same validated defaults without breaking layering.
- Did not change the `platform.db` schema.

## Current Defaults Audit

### Schema Twin

- Record count previously came from the Streamlit `Rows per table` input, hardcoded to `100`, and from required `row_count` in `generate_from_schema`.
- Privacy level had no direct generation setting. Schema Twin is source-free and already generates synthetic values from metadata; optional LLM text remains controlled per run and by `LlmPolicy`.
- Generation mode was hardcoded in the facade as `schema_driven`. The Schema Twin page does not execute source-driven generation.
- Output format previously defaulted to `csv` in both the Streamlit selectbox and the facade, with `parquet` also supported.

### Database Twin

- Record count previously came from the Streamlit per-table inputs, each hardcoded to `200`. The headless `DatabaseTwinPipelineConfig` required `row_counts_by_table`.
- Privacy level had no direct generation setting. Privacy behavior lives in profiling/training/safe synthesizer behavior and release validation rather than a single default toggle.
- Generation mode had no meaningful per-run equivalent. Database Twin is the source-trained database workflow.
- Output format had no global equivalent. The workflow writes relational samples as CSV internally and exports a validated target SQLite database.

### PDF Twin

- Record count had no meaningful equivalent. PDF Twin generates replacement values for discovered fields, tables, and inline spans; it does not generate a requested number of records.
- Privacy level had no safe single-toggle equivalent. The synthetic-twin path already generates from redacted templates and bindings, while redacted PDF output is an explicit sibling action.
- Generation mode had no meaningful equivalent.
- Output format had no meaningful equivalent. PDF Twin outputs PDF.

### Customer Interaction Twin

- Record count had no meaningful equivalent. The first version preserves the source transcript turn count.
- Privacy previously came from the `Remove sensitive information` toggle, defaulting to `True`.
- Generation mode had no meaningful equivalent. It uses deterministic templated synthesis.
- Output format previously defaulted to and only supported `Structured JSON + Synthetic Logs`.

## Wiring Implemented

- Schema Twin:
  - `default_record_count` now defaults `generate_from_schema(..., row_count=None)` and the Streamlit `Rows per table` control.
  - `default_output_format` now defaults Schema export when it is `csv` or `parquet`; unsupported formats fall back to `csv`.
  - Explicit per-run `row_count` and `export_format` still take precedence.
- Database Twin:
  - `DatabaseTwinPipelineConfig.row_counts_by_table` may now be `None`.
  - When omitted, the pipeline resolves every contract table to `default_record_count`.
  - Streamlit per-table row-count inputs now start from `default_record_count`; changed inputs still become explicit per-run overrides.
- Interaction Twin:
  - `privacy_level` now controls the default value of `remove_sensitive_information` when no per-run value is supplied.
  - The Streamlit toggle starts from the persisted privacy mapping and still overrides per run.
  - Unsupported global output formats do not alter the single supported interaction package format.
- PDF Twin:
  - No generation setting was wired because none maps cleanly without changing workflow semantics.

## Privacy-Level Decision

Phase 9 keeps privacy conservative. `standard`, `restricted`, and `strict` all default Interaction Twin to remove sensitive information. This preserves the existing safe behavior where uploaded transcript PII/PCI is redacted before synthesis. The current codebase does not yet expose separate detector sensitivity or masking-strength levels, so Phase 9 does not pretend those distinctions exist.

For Schema, Database, and PDF workflows, privacy handling is already embedded in source-free synthesis, safe database training/generation, redacted PDF templates, validation, and transfer gates. There is no single default knob that can honestly represent `standard` versus `restricted` versus `strict` yet.

## Generation Mode Decision

`generation_mode` remains persisted but intentionally unmapped in this phase:

- Schema Twin is explicitly schema-driven.
- Database Twin is explicitly database/artifact-driven.
- PDF Twin is document-template-driven.
- Interaction Twin is deterministic transcript-structure-preserving synthesis.

Changing this setting now does not silently switch workflows into incompatible modes.

## Verification

Focused settings-wiring and workflow regression command:

```bash
XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest tests/integration/test_settings_wiring.py tests/e2e/workflows/test_schema_twin_ui.py tests/e2e/workflows/test_database_twin_ui.py tests/e2e/workflows/test_pdf_twin_ui.py tests/e2e/workflows/test_interaction_twin_ui.py tests/unit/test_interaction_twin.py tests/unit/test_streamlit_project_ui.py tests/unit/test_platform_db.py tests/contract/architecture/test_layering.py
```

Result:

- `29 passed in 8.79s`

Concrete regression evidence:

- Schema Twin with `default_record_count=5` and `default_output_format=parquet` generated 5 rows and `.parquet` exports with no per-run override.
- Schema Twin with explicit `row_count=3` and `export_format=csv` generated 3 rows and `.csv` exports.
- Database Twin with `default_record_count=6` and no `row_counts_by_table` generated 6 `customers` and 6 `orders`.
- Database Twin with explicit `{"customers": 3, "orders": 4}` generated those exact counts.
- Interaction Twin with `privacy_level=strict` and no per-run override enabled redaction; explicit `remove_sensitive_information=False` disabled it for that run.
- PDF Twin has an explicit regression test documenting that value generation does not accept `product_settings`, `privacy_level`, or `output_format` because those settings do not map cleanly.

## Baseline Comparison

- Phase 8 baseline: `1403 passed, 11 skipped`.
- Phase 9 focused settings/workflow slice: `29 passed`.
- Phase 9 full suite: `1407 passed, 11 skipped, 92 warnings in 47.19s`.
- Net change: `+4` passing tests, no change in skipped tests.

## Follow-Up

- Add real privacy-level semantics only when there is a product-approved detector/masking sensitivity model for each workflow.
- Consider adding supported `json`/`log` values to Settings if Interaction Twin gains selectable output package shapes.
- Consider a separate Database Twin output-format setting if target export expands beyond SQLite.
