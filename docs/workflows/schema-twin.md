# Schema Mode Workflow

## User flow

```text
Intent → Provide schema → Review schema → Configure generation
       → Generate → Preview → Validate → Download
```

Schema Mode is **source-free**. It does not require a training stage because there is no source dataset to fit. The generation plan is derived from schema types, relationships, rules, distributions, and optional semantic/text settings.

## Runtime path

```text
interfaces/streamlit/pages/schema_twin.py
  ↓
application/workflows/schema_twin.py
  ↓
application/orchestration/schema/schema_driven.py
  ├─ engine/inference/schema/*
  ├─ engine/generation/schema/*
  ├─ engine/generation/text/*
  └─ engine/validation/schema/*
```

## Stage ownership

| Stage | Main ownership | Responsibility |
|---|---|---|
| Schema input/inference | `engine/inference/schema/` | Parse schema, normalize columns/types, relationships, semantic/domain hints |
| Configuration/planning | `application/orchestration/schema/`, `engine/inference/schema/` | Row counts, seed, rules, distributions, generation plan |
| Generation | `engine/generation/schema/`, `engine/generation/text/` | Generate relational/tabular and text values |
| Validation | `engine/validation/schema/` | Schema/type/rule/privacy/readiness/quality checks |
| Export | `application/orchestration/schema/export.py` and generation export helpers | Package generated outputs |

## UI boundary

`schema_twin.py` owns session state, controls, previews, and download actions. It calls the application facade for schema loading/preparation/generation rather than embedding core generators in the UI.
