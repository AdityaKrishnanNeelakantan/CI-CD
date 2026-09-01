# Database Twin Workflow

## User flow

```text
Intent → Connect SQLite → Discover → Understand values (profile)
       → Understand data (infer + approve contract)
       → Train twin → Export trained artifact → Disconnect source
       → Generate synthetic database → Validate → Export target database
```

## Runtime path

```text
interfaces/streamlit/pages/database_twin.py
  ↓
application/workflows/database_twin.py
  ├─ engine/discovery/database/
  ├─ engine/profiling/database/
  ├─ engine/inference/database/
  ├─ engine/training/database/
  ├─ engine/generation/database/
  └─ engine/validation/database/
```

## Stage dependencies

| Stage | Producer | Important output | Consumed by |
|---|---|---|---|
| Discovery | `run_discovery` | `discovery.json` | Profiling, inference |
| Profiling | `run_profiling` | `profile.json` plus cleaning/contract metadata | Inference, QA reference fidelity |
| Inference | `run_inference` + `run_contract_approval` | semantic candidates + approved dataset contract | Training, generation |
| Training | `run_training_and_sampling` | training report / per-table synthesizers | Artifact export |
| Artifact | `run_artifact_export` | portable generator artifact (`manifest.json`, checksums, self-test, model state) | Source-free generation |
| Generation | `run_relational_generation` | synthetic table files + relational generation report | QA |
| Validation | `run_qa_validation` | `qa_report.json` | Release decision / target write |
| Target write | `run_target_write` | target DB + `target_write_report.json` | Final output |

## Key internal dependencies

- Discovery uses source adapters; the Streamlit path currently uses `SQLiteSourceAdapter`.
- Profiling consumes discovery metadata plus source rows and emits reusable profile data.
- Inference consumes both discovery and profile evidence and produces candidate semantics; explicit approval creates the dataset contract.
- Training consumes the approved contract and source data. Adapter selection is routed through the training planner/registry.
- Generation consumes the approved contract plus trained/loaded table adapters and preserves relational constraints through the relational generator.
- QA consumes the generated report and can compare against the reference profile before release.
- Target write is downstream of QA and writes the validated synthetic result to the target database.

## Product boundary

Although the infrastructure package includes connector/sink abstractions beyond SQLite, the current Streamlit Database Twin should be presented as SQLite-capable until other connectors are wired and tested in this workflow.
