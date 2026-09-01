# Docling Integration

## Purpose

Docling is a first-class PDF extraction backend for PDF Twin. It is not used as a replacement for the platform's template, semantic binding, synthetic generation, rendering, privacy, or validation engines. It supplies richer document text/layout evidence to those existing stages.

## Dependency

`pyproject.toml` pins:

```text
docling==2.119.0
```

The dependency is part of the `pdf` extra.

For offline/air-gapped PDF processing, set `DOCLING_SERVE_ARTIFACTS_PATH` to a pre-fetched Docling model-artifact directory. The adapter passes that directory through `PdfPipelineOptions(artifacts_path=...)`. If no artifact directory is configured, Docling may fetch the PDF model artifacts on first use.

## Source ownership

- `engine/documents/pdf/docling_engine.py` — converts a PDF with `DocumentConverter`, builds page-level profiling text, and converts Docling provenance/table geometry into the platform span contract.
- `engine/documents/pdf/layout_backends/docling_backend.py` — implements the generic `LayoutBackend` interface.
- `engine/documents/pdf/extraction_router.py` — selects Docling when explicitly requested and persists `extraction_method="docling"`.
- `interfaces/streamlit/pages/pdf_twin.py` — exposes Docling vs Automatic extraction to the user.

## Internal data flow

```text
PDF
 ↓
PDFDocumentAdapter validation
 ↓
Docling DocumentConverter
 ↓
DoclingDocument
 ├─ text/provenance ─────────────→ document profiling
 └─ bbox/table cells
        ↓ normalize coordinates
        ↓
   LayoutBackend span contract
        ↓
   template compilation
        ↓
   semantic binding
        ↓
   value generation
        ↓
   rendering
        ↓
   validation
```

The adapter intentionally preserves PDF metadata and AcroForm-value privacy scanning even when Docling supplies the main content extraction.

## Tests

- `tests/unit/database_pdf/test_docling_engine.py` — page text, tables, geometry normalization.
- `tests/unit/database_pdf/test_layout_backend_registry.py` — backend registration and real extraction when Docling is installed.
- `tests/contract/database_pdf/test_extraction_router_contract.py` — normalized-document contract for the Docling route.
- `tests/integration/database_pdf/test_docling_pdf_pipeline.py` — complete Docling path from profiling through validation plus optional de-identification. It is skipped only when Docling is not installed.
