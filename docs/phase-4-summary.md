# Phase 4 Summary: Terminology and Documentation Alignment

## Scope

- This was a documentation-only cleanup for Bucket A naming drift.
- No source code, behavior, or tests were intentionally changed in this phase.
- No classes, functions, variables, or test assertions were renamed.

## Documentation Added

- Added `docs/terminology-map.md` as the canonical architecture-term-to-implementation map.
- Added a terminology alignment pointer in `docs/architecture.md`.
- Added a terminology-map link in `README.md`.
- Updated `docs/gap-analysis.md` to preserve the historical audit rows while marking Bucket A items closed/aligned in Phase 4.

## Mappings Closed

- `SeedReader` maps to existing seeded generation and sampling controls:
  - `Settings.seed` and `Settings.from_env()` in `src/synth_platform/settings.py:7-27`.
  - Schema Mode seed passed to `generate_from_schema()` in `src/synth_platform/interfaces/streamlit/pages/schema_twin.py:249-269`.
  - `prepare_schema_for_generation()` and `PipelineConfig(seed=...)` in `src/synth_platform/application/workflows/schema_twin.py:456-519`.
  - `PipelineConfig.seed` applied by `run_schema_pipeline()` in `src/synth_platform/application/orchestration/schema/schema_driven.py:76-89`.
  - Deterministic SQLite sampling support via `sp_seed_hash` registered in `SqliteSource.__init__()` at `src/synth_platform/infrastructure/sources/sqlite.py:19-33`.

- `CategorySampler` maps to existing categorical distribution and rebalancing behavior:
  - YAML `choices` / `probabilities` parsing in `_parse_column()` at `src/synth_platform/engine/inference/schema/yaml_schema.py:245-314`.
  - Database Twin category override UI in `src/synth_platform/interfaces/streamlit/pages/database_twin.py:642-720`.
  - `normalize_category_weights()`, `build_rebalanced_encoder()`, and `rebalance_column()` in `src/synth_platform/engine/training/database/adapters/category_rebalancing.py:35-68`.

- `Results page` maps to inline workflow result sections:
  - Schema Mode Preview, Validate, and Download sections in `src/synth_platform/interfaces/streamlit/pages/schema_twin.py:302-410`.
  - Database Twin preview, QA validation, export, and download sections in `src/synth_platform/interfaces/streamlit/pages/database_twin.py:752-920`.
  - PDF Twin generated values, validation, and download sections in `src/synth_platform/interfaces/streamlit/pages/pdf_twin.py:387-539`.

- Front-end workflow step pattern maps to existing step-based flows:
  - Schema Mode progress model in `src/synth_platform/interfaces/streamlit/pages/schema_twin.py:66-79`.
  - Database Twin progress model in `src/synth_platform/interfaces/streamlit/pages/database_twin.py:113-127`.
  - PDF Twin progress model in `src/synth_platform/interfaces/streamlit/pages/pdf_twin.py:84-98`.
  - Customer Interaction Twin remains a separate backlog item because that workflow is not built.

## Additional Bucket A-Style Mapping Found

- `TextColumn` YAML/code maps to the existing generic `Column(type="text")` model plus text distribution parameters and routing:
  - YAML text/categorical parsing in `_parse_column()` at `src/synth_platform/engine/inference/schema/yaml_schema.py:245-314`.
  - Text-heavy LLM eligibility in `assess_column_eligibility()` at `src/synth_platform/engine/generation/text/eligibility.py:127-170`.
  - Text routing in `DataSimulator` at `src/synth_platform/engine/generation/schema/simulator.py:1022-1056`.
  - Fallback/LLM text generation in `TextGenerationEngine.generate()` at `src/synth_platform/engine/generation/text/generator.py:60-95`.
  - The prior fallback crash was fixed in Phase 3; this phase only aligned naming.

## Scope Notes Not Marked Bucket A

- Broad `Document Twin` language currently maps to `PDF Twin` for the implemented Streamlit surface, but broader document support remains a real product-scope decision rather than a closed naming mismatch.
- `Governance Hub` and `Context Synthesizer` remain architecture decisions. Existing validation/context helpers are real, but there is no separate hub or top-level cross-domain reconciler by those names.

## Tests

- Requested command:
  - `python -m pytest`
  - Result: failed immediately because `/Users/sivasanker/.local/bin/python` does not have `pytest` installed.
- Full-suite verification using the Phase 3 virtualenv:
  - Command: `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
  - Result: `1378 passed, 11 skipped, 92 warnings in 48.42s`.
- Comparison to Phase 3 final result:
  - Phase 3: `1378 passed, 11 skipped`.
  - Phase 4: `1378 passed, 11 skipped`.
  - No regression in pass/skip counts.
