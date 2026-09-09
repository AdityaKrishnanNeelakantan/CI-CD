# Platform Architecture

## Purpose

This repository uses a single canonical package, `synth_platform`, for all four implemented user-facing workflows. The architecture separates **what the product does** (workflows/use cases/domain) from **how it talks to external systems** (infrastructure) and **how users invoke it** (interfaces).

The default Streamlit entry point is a unified Chat page backed by a thin, deterministic capability coordinator. The coordinator routes to Schema, Database, Document/PDF, or Customer Interaction Twin; it does not reproduce their generation, extraction, training, validation, privacy, or packaging logic.

The larger cross-domain target architecture also includes a separate live-agent plane, MCP tool gateway, shared governance/evaluation, and internal model platform. Those are transition targets rather than current runtime components. See `unified-chat-transition.md` for phase boundaries.

## Layers

### `interfaces/`
Entry points only: Streamlit, REST API, CLI, and SDK. Interface code translates user input into application calls and renders application results. It must not own statistical, training, generation, or validation algorithms.

### `application/`
Coordinates product behavior.

- `coordinator/` selects a registered product capability through protocol-independent contracts; it never executes workflow stages.
- `workflows/` exposes one stable facade per implemented product workflow.
- `use_cases/` exposes reusable application operations.
- `orchestration/` owns stage sequencing, state, and checkpoint coordination after a workflow has been selected.
- `ports/` defines abstractions that infrastructure can implement.
- `dto/` contains boundary data-transfer objects.

### `engine/`
Reusable executable capabilities grouped by product stage:

- `discovery/`
- `profiling/`
- `inference/`
- `training/`
- `generation/`
- `validation/`
- `documents/`
- `interactions/`

The migrated Database/PDF lineage intentionally retains subpackages such as `engine/*/database` and `engine/documents/pdf`; this keeps behavioral ownership explicit without copying algorithms into each UI workflow. Interaction parsing, sanitization, SSOT construction, and release validation are owned by `engine/interactions`.

### `domain/`
Business concepts and policies: schema, constraints, privacy, relational structure, document models, profiling/generation/training/validation concepts, artifacts, and run metadata. Domain code is kept free from dataframe, database, UI, PDF-library, and queue dependencies.

### `infrastructure/`
External implementations: source connectors, sinks, model backends, PDF/OCR implementations, extraction adapters, persistence, storage, artifacts, jobs, LLM clients, and observability.

## Dependency direction

```text
                         +------------------+
                         |    interfaces    |
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |   application    |
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |      engine      |
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |      domain      |
                         +------------------+

  infrastructure implements ports / external mechanisms and may depend inward;
  domain, engine, and application must not import infrastructure or interfaces.
```

`tests/contract/architecture/test_layering.py` enforces this direction with an AST scan.

## Workflow ownership

### Schema Mode

`interfaces/streamlit/pages/schema_twin.py`
→ `application/workflows/schema_twin.py`
→ schema orchestration + schema inference/generation/validation engines.

There is intentionally no mandatory model-training stage because Schema Mode is source-free. Calling this step "training" in the UX would misrepresent the implementation.

### Database Twin

`interfaces/streamlit/pages/database_twin.py`
→ `application/workflows/database_twin.py`
→ discovery → profiling → inference/approval → training/artifact → relational generation → QA validation → target write.

### PDF Twin

`interfaces/streamlit/pages/pdf_twin.py`
→ `application/workflows/pdf_twin.py`
→ extraction selection (Docling or automatic native/OCR) → document profiling → template compilation using the same extraction lineage → semantic binding → synthetic values → render → validation.

De-identification is an optional sibling operation after template construction; it is not a prerequisite for generating the twin.

### Customer Interaction Twin

`interfaces/streamlit/pages/interaction_twin.py`
→ `application/workflows/interaction_twin.py`
→ transcript parsing and sanitization → optional one-shot semantic enhancement → strict interaction SSOT → privacy/source-replay validation → checksummed ZIP packaging.

Raw source text is kept in memory only. The model, when enabled, receives only sanitized text and may return closed-vocabulary semantic labels; malformed, invalid, or unavailable model output falls back deterministically. The released package contains only `sanitized_source.txt`, `interaction_ssot.json`, `validation_report.json`, and `manifest.json`.

## Run artifacts and lineage

Database, PDF, and Customer Interaction stages use `RunManifest`-based run directories and named stage outputs. This gives stage-level lineage instead of one opaque pipeline result. The workflow documentation lists the important artifact filenames and their producer/consumer relationships.

## Legacy isolation

Previous/optional implementation lineages and historical reports are kept under `archive/legacy/`. They are not importable product packages under `src/` and should not be treated as active runtime ownership. The original-to-canonical module mapping is recorded in `docs/module-migration-map.json`.

## Adding a feature

1. Determine whether it is a domain concept, reusable engine capability, application orchestration concern, external integration, or UI concern.
2. Put reusable stage algorithms under the stage engine, not under a workflow page.
3. Expose cross-stage product behavior through an application workflow/use case.
4. Put vendor/database/filesystem/network details under infrastructure.
5. Add unit tests near the capability and integration/e2e tests under the relevant product workflow.
6. Run the architecture contract before merging.
