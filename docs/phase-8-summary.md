# Phase 8 Summary: Customer Interaction Twin Workflow

## Scope

- Added Customer Interaction Twin as the fourth workflow type alongside Schema, Database, and PDF Twin.
- The workflow accepts `.txt` and `.log` transcript uploads, supports interaction type selection, produces structured JSON plus synthetic logs, and optionally removes sensitive information.
- The workflow follows the Upload -> Configure -> Generate -> Results step pattern in Streamlit.

## Synthesis Approach

- Used deterministic templated synthesis, not LLM-assisted generation.
- This first version preserves the source transcript's speaker sequence and turn count, but replaces the content with new customer-support-style synthetic turns.
- No LLM provider is called, so Phase 2 `LlmPolicy` does not need to approve any interaction generation path in this phase.

## Engine Logic

- Added `src/synth_platform/engine/generation/interaction/`.
- Core behavior includes:
  - Parsing speaker-labeled transcript exports into structured turns.
  - Normalizing `internal`/agent-like speakers to `agent` and `external`/customer-like speakers to `customer`.
  - Detecting sensitive text with the existing rule-based PII detector from the PDF/document engine.
  - Adding transcript-specific detection for 4-digit PCI fragments, covering card numbers split across multiple conversational turns.
  - Producing validation-ready reports with explicit `hard_checks_passed`, `passed`, and `export_ready` fields.
- Output package includes:
  - `synthetic_interaction.json`
  - `synthetic_interaction.log`
  - `validation_report.json`
  - `redaction_report.json`

## Workflow And Transfer Gate

- Added `src/synth_platform/application/workflows/interaction_twin.py`.
- Runs are recorded to `platform.db` as workflow type `interaction`.
- Added `interaction_twin`/`interaction` support to the Phase 1 `TransferService` release gate.
- Downloads are blocked unless the interaction validation report explicitly passes and marks the output export-ready.
- Added a small `platform.db` migration path so existing local SQLite databases with the old three-workflow `CHECK` constraint can accept the new `interaction` workflow type.

## Streamlit UI

- Added `src/synth_platform/interfaces/streamlit/pages/interaction_twin.py`.
- Registered the page in `src/synth_platform/interfaces/streamlit/app.py`.
- Added the fourth Home page card in `src/synth_platform/interfaces/streamlit/pages/home.py`.
- Added Interaction filtering and labeling in My Projects via `src/synth_platform/interfaces/streamlit/project_ui.py` and `src/synth_platform/interfaces/streamlit/pages/my_projects.py`.

## Fixture Confirmation

- Used the existing real transcript fixtures under `tests/fixtures/local-data/`.
- Confirmed the directory is still gitignored:
  - `.gitignore:13:tests/fixtures/local-data/`
- PII/PCI regression coverage uses `Transcript3-HP.txt`.
- Specific evidence: the split card-number-like value `4199 4099 9799 8199` from `Transcript3-HP.txt` does not appear in the synthetic output, and each fragment is asserted absent.

## Verification

- Focused command:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest tests/unit/test_interaction_twin.py tests/e2e/workflows/test_interaction_twin_ui.py tests/unit/test_platform_db.py tests/unit/test_streamlit_project_ui.py tests/unit/application/test_transfer_service.py tests/contract/architecture/test_layering.py`
- Result:
  - `29 passed in 4.08s`.
- Full-suite command:
  - `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
- Result:
  - `1403 passed, 11 skipped, 72 warnings in 50.33s`.
- Phase 7 baseline:
  - `1395 passed, 11 skipped`.
- Net change:
  - `+8` passing tests, no change in skipped tests.

## Follow-Up

- Interaction Type currently adjusts only the opening support framing; richer type-specific dialogue templates can be added later.
- Output Format currently supports the requested first format: Structured JSON + Synthetic Logs.
- Name detection is intentionally conservative and context-based; broader person-name detection should be added only with a measured false-positive strategy.
