# Internal Dependencies and Data Flow

## Unified chat routing

```text
interfaces/streamlit/pages/chat.py
      │
      ▼
application/coordinator (pure capability selection)
      │
      ▼
interfaces/streamlit/capabilities.py (page hand-off only)
      │
      ├─► existing Schema workflow
      ├─► existing Database workflow
      ├─► existing PDF/Document workflow
      └─► existing Customer Interaction workflow
```

The coordinator does not import Streamlit, engine services, infrastructure, or workflow pages. Workflow orchestration starts only after capability selection; MCP and the live-agent plane are not active dependencies.

## Canonical workflow entry points

| Workflow | UI | Application facade |
|---|---|---|
| Schema | `interfaces/streamlit/pages/schema_twin.py` | `application/workflows/schema_twin.py` |
| Database | `interfaces/streamlit/pages/database_twin.py` | `application/workflows/database_twin.py` |
| PDF | `interfaces/streamlit/pages/pdf_twin.py` | `application/workflows/pdf_twin.py` |
| Customer Interaction | `interfaces/streamlit/pages/interaction_twin.py` | `application/workflows/interaction_twin.py` |

The UI should import these application facades instead of reaching into multiple low-level service packages.

## Database dependency chain

```text
SQLiteSourceAdapter
      │
      ▼
run_discovery ── discovery.json
      │
      ▼
run_profiling ── profile.json / cleaning report
      │
      ▼
run_inference ── semantic candidates
      │
      ▼
run_contract_approval ── approved dataset contract
      │
      ▼
run_training_and_sampling ── training report + trained adapters
      │
      ▼
run_artifact_export ── portable generator artifact
      │
      ▼
load_artifact (source can now be disconnected)
      │
      ▼
run_relational_generation ── generated table files/report
      │
      ▼
run_qa_validation ── qa_report.json
      │
      ▼
run_target_write ── target synthetic database
```

`RunManifest` provides run/stage lineage and immutable output paths across these stages.

## PDF dependency chain

```text
PDFDocumentAdapter / preflight validation
      │
      ▼
extraction_router
  ├─ Docling → docling_engine.py
  └─ Automatic → native / OCR
      │
      ▼
run_document_profiling
      │ persists extraction_method
      ▼
run_template_compilation
  └─ layout backend resolved from the same extraction_method
      │
      ├─────────────► run_document_deidentification (optional sibling)
      │
      ▼
run_semantic_binding
      │
      ▼
run_value_generation
      │
      ▼
run_document_rendering
      │
      ▼
run_document_validation
```

The binding map is the semantic bridge between learned/template fields and generated synthetic values; ground truth emitted by rendering is the input used to validate the final document.

## Schema dependency chain

```text
load_schema / normalize schema
      │
      ▼
prepare generation configuration
      │
      ▼
run_schema_pipeline
      ├─ schema inference / relational planning
      ├─ value + optional text generation
      ├─ export
      └─ validation metrics/report
      │
      ▼
SchemaModeResult (preview, row counts, validation, exports)
```

Schema Mode intentionally has no source-data training dependency.

## Customer Interaction dependency chain

```text
pasted transcript / .txt / .log
      │ raw source remains in memory only
      ▼
parse_and_sanitize_transcript
      ├─ typed sensitive-value replacement
      └─ participant pseudonymization
      │
      ├─────────────► optional one-shot ChatModel enhancement
      │               (sanitized source only; closed enum output)
      ▼
build_interaction_ssot
      │ deterministic fallback on model failure or invalid output
      ▼
validate_interaction_ssot
      ├─ strict schema and count integrity
      ├─ residual PII / secret checks
      └─ source replay / overlap checks
      │
      ▼
release gate → checksummed four-file ZIP package
```

The package contains only `sanitized_source.txt`, `interaction_ssot.json`, `validation_report.json`, and `manifest.json`. The workflow reuses the existing PDF PII/entity detectors, `ChatModel` port, `RunManifest`/`StageResult` lineage, and domain release gate without moving those responsibilities into the coordinator.

## Shared code versus workflow-specific code

- **Shared platform concepts:** `domain/`, `application/ports`, generic `infrastructure/`.
- **Shared stage ownership:** `engine/<stage>/`.
- **Database lineage implementations:** stage subpackages ending in `/database`.
- **Schema lineage implementations:** stage subpackages ending in `/schema` plus `generation/text`.
- **PDF lineage implementation:** `engine/documents/pdf`.
- **Interaction lineage implementation:** `engine/interactions` plus `domain/interactions`.
- **Workflow sequencing only:** `application/workflows/`.
- **Presentation only:** `interfaces/streamlit/`.

## Migration provenance

`docs/module-migration-map.json` contains the mechanical old-module → canonical-module mapping used for this reorganization. Code under `archive/legacy/` is not part of the active dependency graph.
