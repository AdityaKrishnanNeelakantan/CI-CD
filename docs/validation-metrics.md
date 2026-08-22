# Validation Metrics

Validation is workflow-specific but follows a common release principle: structural correctness comes first, then fidelity/privacy/utility evidence where applicable.

## Structured data

Relevant checks in the active codebase include:

- primary-key uniqueness and null behavior;
- foreign-key integrity/orphans;
- schema/type adherence;
- relational and business-rule constraints;
- distribution/fidelity comparisons;
- privacy-oriented metrics where configured;
- generation/readiness and artifact self-tests.

Database Twin writes a QA report after relational generation and before target write.

## Schema Mode

Schema Mode validates the generated data against the requested schema/rules and emits a structured validation report used by the UI before download.

## PDF Twin

PDF validation compares rendered document output with the generation ground truth/template expectations and emits `document_validation_report.json`.

## Release semantics

A failed required check must not be represented as a successful release. If a required capability is not configured or cannot run, report it explicitly as warning/not-run rather than manufacturing a pass.
