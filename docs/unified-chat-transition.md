# Unified Chat Transition

## Decision

The product exposes synthetic-twin workflows as specialized capabilities behind one thin conversational coordinator. Schema, Database, Document/PDF, and Customer Interaction remain application workflows with workflow-owned deterministic, statistical or optional model-backed, validation, and packaging stages. They are not converted into generic LLM agents.

The coordinator owns only:

1. understanding an explicit or structurally unambiguous request;
2. selecting a capability;
3. collecting missing input at the interface boundary;
4. handing control to the existing workflow UI/application facade; and
5. presenting workflow-owned results.

It does not generate rows, parse or render documents, train models, repair relationships, validate privacy, or package artifacts.

## Repository Reality

| Target capability | Current implementation | Initial transition status |
|---|---|---|
| Schema Twin | Mature Streamlit page and cohesive application workflow facade | Available |
| Database Twin | Mature interactive Streamlit workflow and headless engine pipeline | Available through the existing interactive workflow |
| Document Twin | Mature PDF extraction, binding, value generation, rendering, and validation workflow | Available as PDF Twin |
| Customer Interaction Twin | TXT/LOG parser, sanitization, structured SSOT, privacy/source-replay validation, artifact manifest, and ZIP workflow | Available |

The active repository does not currently contain NVIDIA NeMo Data Designer, MCP servers, a live-agent plane, a unified model registry, or a shared evaluation service. Those target components must not be represented as implemented.

## Plane Boundaries

The cross-domain target diagrams describe separate concerns:

- **Synthetic-data plane:** this repository's specialized twin workflows and their existing deterministic/statistical/model-backed stages.
- **Live-agent plane:** operational service agents and domain MCP servers. This is outside the first transition.
- **Governance and evaluation:** existing workflow-specific validators are preserved. A unified evaluator is a later convergence project.
- **Security and authorization:** deterministic infrastructure controls, not coordinator or LLM decisions.
- **Optimization loop:** workflow traces may support prompt improvement and post-training only after reliable evaluation and consent/retention policies exist.

Three coordinators remain distinct:

- **Chat coordinator:** selects a product capability.
- **Workflow orchestration:** sequences Schema, Database, Document, or Customer Interaction stages.
- **Generation/model orchestration:** performs generation or bounded semantic enhancement inside a workflow.

The first transition adds only the chat coordinator.

## Phase 1: Chat-First Product Shell

### Scope

- Make a unified Chat page the default Streamlit entry point.
- Add a pure application-level capability registry and deterministic router.
- Support explicit capability selection and unambiguous attachment-type routing.
- Hand control to the selected specialized workflow without copying workflow logic.
- Preserve direct Schema, Database, PDF, and Customer Interaction navigation.
- Route Customer Interaction to its workflow for pasted text or `.txt`/`.log` input.
- Keep chat state in Streamlit `session_state`; keep existing workflow state namespaces untouched.

### Routing policy

Routing precedence is auditable and deterministic:

1. explicit capability selected by the user;
2. an explicit command such as `/schema`, `/database`, `/document`, or `/interaction`;
3. exactly one unambiguous attachment type;
4. otherwise ask the user to choose a capability.

The first router intentionally does not use an LLM. Broad natural-language intent classification can be added later behind the same contracts, but ambiguous requests must still require confirmation.

### Attachment mapping

- Schema: `.sql`, `.json`, `.yaml`, `.yml`
- Database: `.db`, `.sqlite`, `.sqlite3`
- Document: `.pdf`
- Customer Interaction: `.txt`, `.log`

The transitional shell uses attachments for routing only. Specialized workflow pages own ingestion and validation, so the source is handled only after capability selection. Customer Interaction also accepts pasted transcript text directly on its page.

### Explicit non-goals

- No generic workflow agents.
- No MCP adapter or tool server.
- No PostgreSQL, Redis, MinIO, DuckDB, queue, or worker migration.
- No Data Designer integration claim.
- No replacement of existing validation, artifact, model, or orchestration code.
- No automated execution of the Database headless pipeline from chat; that path auto-approves semantics and is not equivalent to the interactive review workflow.
- No placeholder Interaction Twin that produces unvalidated artifacts.

### Implemented Customer Interaction flow

```text
transcript text / .txt / .log
        |
        v
parse + sanitize + participant pseudonymization
        |
        +--> optional one-shot model semantics (sanitized text only)
        |     `--> deterministic fallback on timeout, malformed, or invalid output
        v
strict interaction SSOT
        |
        v
privacy + source-replay validation
        |
        v
release gate --> checksummed ZIP
```

Raw transcript text is never written to the run directory. Immutable facts, participant IDs, counts, and release checks are deterministic. Optional model output is restricted to closed-vocabulary topic, issue, action, resolution, and sentiment fields; it cannot author transcript text or summaries. A released ZIP contains exactly `sanitized_source.txt`, `interaction_ssot.json`, `validation_report.json`, and `manifest.json`.

This implementation remains in the synthetic-data plane. It does not create a service agent, an MCP server, a database or queue dependency, or a Data Designer integration.

## Phase 2: Shared Product Workspace (implemented foundation)

The current branch adds the common product layer around the Phase 1 hand-off:

- local workflow sessions with source/configuration fingerprints only;
- workflow-specific stage events grouped into shared wizard macro-steps;
- normalized result views that keep execution, validation, and release separate;
- approved-root, content-addressed artifact snapshots with checksum-verified downloads;
- recoverable aggregate completion commits with a write-ahead journal and process lock;
- permission-restricted process-temporary SQLite/PDF source staging with
  disconnect/completion, process-exit, and 24-hour stale cleanup;
- My Projects, Results, real Schema Templates, effective Settings, and Help pages;
- atomic local JSON persistence under `<SP_OUTPUT_ROOT>/workspace`;
- a localhost-only installed UI launcher until authenticated tenancy exists;
- SHA-256 upload/configuration invalidation for Schema, PDF, and Interaction,
  plus SHA-256 SQLite upload identity.

This foundation does not flatten specialized workflow contracts. Database still
requires interactive semantic approval and source disconnect; PDF's
redaction/de-identification path remains optional; Interaction keeps its
sanitize-before-model release gate and exact package contract.

Still pending from the broader presenter phase:

- carry uploaded bytes from Chat into workflow-owned ingestion safely;
- render an entire specialized wizard inline inside the conversation;
- introduce durable asynchronous task execution only if synchronous Streamlit
  becomes an operational constraint.

## Guardrail foundation (implemented)

The current runtime now places explicit checks at the boundaries shown in the
guardrail architecture:

- Chat and model input: sensitive-data masking, prompt-injection indicators,
  scope/size validation, and conservative deterministic content policy;
- model output: JSON/shape and size validation, sensitive-data rejection or
  masking, content checks, and workflow-specific groundedness;
- local tools: strict command DTOs plus operation/resource permission checks
  before `WorkspaceService` access;
- provider composition: loopback-only Ollama behind `GuardedChatModel`, with
  deterministic fallback when unavailable or blocked.

Reports contain categories and counts, not raw prompts or matched values. This
foundation does not add a live service agent or MCP runtime. The local UI policy
is not multi-user RBAC; authentication remains a prerequisite for networked
operation. See `guardrails.md`.

## Phase 3: Protocol Adapters and Internal Deployment

- Add authentication/RBAC before exposing guarded operations beyond localhost.
- Introduce MCP only as a thin adapter over `GuardedWorkspaceTools`; application workflows remain protocol-independent and command policy cannot be bypassed.
- Add Redis/PostgreSQL/object storage only for demonstrated session, metadata, or artifact requirements.
- Move internal inference from development serving to vLLM/NIM without changing guarded model/workflow contracts.

## Phase 4: Evaluation and Optimization

- Normalize workflow traces: intent, selected capability, safe input contract, model identity, guardrail category/count decisions, output fingerprint, repairs, accepted artifact, validation verdicts, and latency. Do not persist raw prompts/completions by default.
- Establish privacy, retention, consent, and redaction controls before curating training data.
- Build workflow-specific offline evaluation gates.
- Improve prompts/rules first; evaluate QLoRA next; consider preference optimization only when reliable preference data exists.
- Register a candidate model only when it beats the current model on agreed quality, privacy, fidelity, and latency gates.

## Initial Code Shape

```text
interfaces/streamlit/pages/chat.py
        |
        v
application/coordinator/router.py
        |
        v
interfaces/streamlit/capabilities.py
        |
        +--> existing Schema page/workflow
        +--> existing Database page/workflow
        +--> existing PDF page/workflow
        `--> existing Customer Interaction page/workflow

specialized workflow outcome
        |
        v
application/services/result_presentation.py
        |
        v
application/services/workspace.py
        |
        v
application/ports/workspace_repositories.py
        ^
        |
infrastructure/persistence/local_workspace.py
```

The interface mapping is intentionally outside `application/`: the application router knows product capability IDs and availability, but it does not import Streamlit pages.

## Acceptance Criteria for Phase 1

- Chat is the default Streamlit page.
- Schema, Database, Document, and Customer Interaction can be selected explicitly from Chat.
- `.sql`/schema, SQLite, PDF, and `.txt`/`.log` attachments route deterministically.
- ambiguous or conflicting attachments produce clarification rather than a guess.
- Interaction selection invokes only the specialized Customer Interaction workflow.
- selecting an available capability offers a hand-off to the existing workflow implementation without copied stage logic.
- existing specialized pages remain directly accessible.
- existing architecture dependency contracts and relevant workflow checks continue to pass.
