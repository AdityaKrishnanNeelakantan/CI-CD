# PDF Twin Workflow

## User flow

```text
Intent → Upload PDF → Select extraction engine → Discover document
       → Build twin template → Understand fields → Generate synthetic values
       → Build twin PDF → Validate → Download

                                      └─ optional: De-identify source PDF
```

## Runtime path

```text
interfaces/streamlit/pages/pdf_twin.py
  ↓
application/workflows/pdf_twin.py
  ↓
engine/documents/pdf/
  ├─ adapter / preflight / PDF classification
  ├─ extraction router
  │   ├─ docling_engine.py                 # Docling conversion + normalized spans
  │   ├─ native PDF extraction             # pypdf
  │   └─ OCR fallback                      # pytesseract/pdf2image path
  ├─ profiling
  ├─ template compilation
  │   └─ layout_backends/
  │       ├─ docling_backend.py
  │       ├─ native_backend.py
  │       └─ ocr_backend.py
  ├─ semantic binding
  ├─ synthetic value generation
  ├─ rendering
  └─ validation
```

## Extraction behavior

PDF Twin exposes two product choices:

- **Docling** — uses Docling for content/layout extraction. The resulting profile records `extraction_method="docling"`; template compilation then resolves the same method to `DoclingLayoutBackend`, so profiling and geometry use the same extraction lineage.
- **Automatic (native/OCR)** — preserves the original path: native PDF text is used when available and the existing OCR backend is selected for scanned/empty-text PDFs.

When the `pdf` dependency extra is installed, Docling is installed and becomes the default UI choice. An explicit Docling request never silently falls back to native/OCR: failure is surfaced so the run cannot claim a Docling lineage when another extractor actually ran.

Docling bounding boxes are converted to the platform layout contract before downstream use:

```text
{text, x0, y0, x1, y1, page_number, confidence}
```

Coordinates are fractional `[0, 1]` values with a top-left origin. Table-cell geometry from Docling is also converted into individual positioned spans so the existing table/region analyzer remains backend-independent.

## Stage dependencies

| Stage | Producer | Important output | Consumed by |
|---|---|---|---|
| Upload/preflight/profile | `run_document_profiling` | `document_profile.json` | Template compilation / UX review |
| Template compilation | `run_template_compilation` | `document_template.json` | Binding; optional de-identification |
| Optional de-identification | `run_document_deidentification` | redacted PDF + de-identification report | Independent downloadable output |
| Semantic binding | `run_semantic_binding` | `document_binding_map.json` | Value generation |
| Value generation | `run_value_generation` | `document_synthetic_values.json` | Rendering |
| Rendering | `run_document_rendering` | rendered PDF + `document_ground_truth.json` | Validation |
| Validation | `run_document_validation` | `document_validation_report.json` | Final release/download |

## Privacy behavior

Docling does not replace the platform's privacy checks. When Docling is selected, AcroForm values and the PDF Info dictionary are still merged into profiling text so privacy/PII detection retains the coverage of the original native adapter.

De-identification remains deliberately separate from synthetic-twin generation. A user may need a redacted derivative without generating a synthetic twin, so it is an optional sibling path rather than a mandatory generation stage.
