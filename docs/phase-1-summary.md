# Phase 1 Summary: Validation-Gated Transfers

## Verified Bypasses

- Schema Twin previously showed validation at `src/synth_platform/interfaces/streamlit/pages/schema_twin.py:342-369`, then packaged and exposed `st.download_button()` directly at `:374-384` even if validation had failed. It now calls `TransferService.downloadable_bytes()` at `:382-388`; blocked transfers render a disabled download button at `:397-404`.
- PDF Twin previously rendered `Download synthetic twin PDF` before `Validate twin`, making the PDF reachable immediately after rendering. The fixed order is `Validate twin` first at `src/synth_platform/interfaces/streamlit/pages/pdf_twin.py:438-453`, then a disabled, audited download while validation is absent at `:454-471`, and only a service-approved download after validation at `:504-519`.
- Database Twin already blocked target writes after failed QA at `src/synth_platform/interfaces/streamlit/pages/database_twin.py:849-850`, but once a target DB existed it opened the file directly and passed it to `st.download_button()` with no transfer approval/audit step. It now shows an explicit disabled transfer on failed QA at `:862-881`, and uses `TransferService.downloadable_file()` before exposing `Download Database B` at `:915-924`.

## Gate Design

- `TransferService` lives in `src/synth_platform/application/services/transfer_service.py`.
- For canonical `ValidationReport` objects, it reuses `domain.validation.release_gate.decide()` and allows only `Status.PASS`.
- For existing UI report shapes, it accepts only explicit pass evidence:
  - `schema_twin`: `passed is True` and `export_ready is True`.
  - `pdf_twin`: `hard_checks_passed is True`.
  - `database_twin`: `hard_checks_passed is True` and `release.decision == "PASS"`.
- Missing validation, malformed validation, failed validation, unknown workflow names, and missing output files all fail closed.
- `AuditLogger.log_transfer_attempt()` records every service-mediated transfer attempt with workflow, output id, validation status, reason, and non-sensitive metadata. Both allowed and blocked attempts use the same audit operation: `transfer_attempt`.

## Quarantine Decision

Quarantine remains separate from the transfer gate. The existing `Quarantine` implementation stages table dictionaries as CSV and promotes directories after a domain `ValidationReport` reaches `PASS`; the Streamlit download surfaces include ZIP bytes, rendered PDF files, and SQLite target DB files. Folding those into the current quarantine mechanism would either distort its table-staging contract or create a second hidden packaging path. This phase therefore makes transfer approval the final handoff gate and leaves quarantine promotion as a distinct staging concern for table-oriented publication flows.

## Notes And Non-Goals

- The Schema UI report is not the same domain `ValidationReport` object used by `publish_dataset()`, so the service includes a narrow legacy-dict adapter for Schema Mode rather than pretending `decide()` can always be called there.
- This phase does not change LLM provider selection, text-generation fallback behavior, or schema `hard_checks_passed` calculation.
- API download routes are not present yet; they should call `TransferService` when added.
- CLI publication parity remains through `publish_dataset()`; this phase specifically closes Streamlit handoff/download paths.
