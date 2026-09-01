# Docling PDF Pipeline Validation

Validation date: 2026-08-13 (America/Phoenix)

## Scope

This report covers the Docling integration added to PDF Twin and verifies that the new extraction path does not replace or bypass the existing template, semantic binding, synthetic-value generation, rendering, privacy, or validation stages.

The intended Docling data flow is:

```text
PDF
  -> PDF source validation / preflight
  -> Docling DocumentConverter
  -> document_profile.json (extraction_method=docling)
  -> DoclingLayoutBackend
  -> document_template.json
  -> document_binding_map.json
  -> document_synthetic_values.json
  -> synthetic.pdf + ground_truth.json
  -> document_validation_report.json
  -> optional redacted.pdf + deidentification report
```

## Validation performed in this workspace

### 1. Static/runtime import validation

- Active source and tests compile successfully with `python -m compileall`.
- The Docling module is imported lazily, so the rest of PDF Twin remains importable if the optional Docling runtime is absent.
- `docling` is registered as a real `LayoutBackend` alongside `native` and `ocr`.
- Explicit Docling extraction persists `extraction_method="docling"`; template compilation resolves that same backend name instead of silently switching extraction engines.

### 2. Architecture contract

Command:

```bash
pytest -q tests/contract/architecture
```

Result: **3 passed**.

### 3. Docling adapter / contract tests without Docling installed

Command:

```bash
pytest -q \
  tests/unit/database_pdf/test_docling_engine.py \
  tests/unit/database_pdf/test_layout_backend_registry.py \
  tests/contract/database_pdf/test_extraction_router_contract.py \
  tests/integration/database_pdf/test_docling_pdf_pipeline.py
```

Result in the current environment: **12 passed, 2 skipped**.

The two skipped cases deliberately require the real `docling` package. Unit and contract coverage still verifies document-model adaptation, normalized geometry, backend registration, and the normalized extraction artifact contract.

### 4. Full Docling application boundary using an API-compatible local test double

Because this execution sandbox cannot download the third-party Docling distribution or model artifacts, an ephemeral, non-committed test double implementing the subset of the current `DocumentConverter`/`DoclingDocument` API consumed by the adapter was placed on `PYTHONPATH`.

The following committed tests were then executed through that boundary:

```bash
pytest -q \
  tests/integration/database_pdf/test_docling_pdf_pipeline.py \
  tests/unit/database_pdf/test_layout_backend_registry.py \
  tests/unit/database_pdf/test_docling_engine.py \
  tests/contract/database_pdf/test_extraction_router_contract.py
```

Result: **14 passed**.

The integration test executes:

```text
profile -> template -> semantic binding -> synthetic values
        -> render -> validate -> de-identify
```

This validates the platform integration and artifact hand-offs. It is **not** evidence that the third-party Docling runtime itself executed in this sandbox.

### 5. Existing real-PDF regression

The repository's retained real PDF fixtures were restored and the existing PDF end-to-end regressions were run against the native/OCR implementation:

```bash
pytest -q \
  tests/e2e/workflows/test_real_pdf_fixtures_full_pipeline.py \
  tests/e2e/workflows/test_document_pipeline_e2e.py
```

Result: **11 passed**.

This verifies that adding Docling did not break the established PDF pipeline.

## Validation that still must run in connected CI

The current sandbox cannot resolve/download packages from PyPI, and the original repository did not contain a Docling installation. Therefore an actual Docling conversion could not be executed here.

The repository CI installs the `pdf` extra (which pins Docling) and explicitly runs:

```bash
pytest tests/integration/database_pdf/test_docling_pdf_pipeline.py -q
```

That test is the required release gate for the real third-party runtime. It is skipped only when Docling is absent, so the CI dependency installation must succeed before merge.

For a local connected environment, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ui,test,pdf]"
pytest tests/integration/database_pdf/test_docling_pdf_pipeline.py -q
```

Then run the full PDF regression:

```bash
pytest tests/unit/database_pdf \
       tests/contract/database_pdf \
       tests/integration/database_pdf \
       tests/e2e/workflows/test_document_pipeline_e2e.py \
       tests/e2e/workflows/test_real_pdf_fixtures_full_pipeline.py -q
```

## Offline / air-gapped deployment

Docling's standard PDF processing uses model artifacts. The adapter accepts a pre-fetched artifact directory through:

```text
DOCLING_SERVE_ARTIFACTS_PATH=/absolute/path/to/docling-models
```

When set, that value is passed to `PdfPipelineOptions(artifacts_path=...)`. Pre-fetch and approve model artifacts as part of the deployment process rather than relying on a production application to download them on first request.

## Release conclusion

The application integration is implemented and structurally validated. The existing PDF pipeline regression is green. The one remaining environment-specific release check is execution with the actual pinned Docling runtime and its model artifacts in connected CI or a prepared company environment.
