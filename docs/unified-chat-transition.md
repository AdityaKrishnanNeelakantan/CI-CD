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

## Phase 2: Reusable Capability Presenters

- Extract the current page-owned presentation sequences into reusable capability presenters while leaving engine behavior unchanged.
- Define workflow-specific input/result contracts where current facade shapes are asymmetric.
- Carry uploaded bytes from chat into workflow-owned ingestion safely.
- Present validation reports and artifacts back in the conversation.
- Add durable task state only when synchronous Streamlit execution becomes an operational constraint.

## Phase 3: Tool Gateway and Internal Deployment

- Add an API/tool facade over stable application contracts.
- Introduce MCP only as an adapter below the coordinator; application workflows remain protocol-independent.
- Add authentication/RBAC and deterministic policy enforcement.
- Add Redis/PostgreSQL/object storage only for demonstrated session, metadata, or artifact requirements.
- Move internal inference from development serving to vLLM/NIM without changing workflow contracts.

## Phase 4: Evaluation and Optimization

- Normalize workflow traces: intent, selected capability, input contract, model identity, raw output, repairs, accepted artifact, validation verdicts, and latency.
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
