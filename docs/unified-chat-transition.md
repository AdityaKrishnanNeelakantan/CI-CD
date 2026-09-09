# Unified Chat Transition

## Decision

The product will expose synthetic-twin workflows as specialized capabilities behind one thin conversational coordinator. Schema, Database, and Document/PDF remain application workflows with deterministic, statistical, model-backed, validation, and packaging stages. They are not converted into generic LLM agents.

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
| Customer Interaction Twin | No active workflow, DTO, engine, page, or test in this repository | Planned; never routed to a substitute workflow |

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
- **Workflow orchestration:** sequences Schema, Database, or Document stages.
- **Generation/model orchestration:** performs generation inside a workflow.

The first transition adds only the chat coordinator.

## Phase 1: Chat-First Product Shell

### Scope

- Make a unified Chat page the default Streamlit entry point.
- Add a pure application-level capability registry and deterministic router.
- Support explicit capability selection and unambiguous attachment-type routing.
- Hand control to the selected existing workflow without copying workflow logic.
- Preserve direct Schema, Database, and PDF navigation during migration.
- Show Customer Interaction as planned/unavailable with a truthful explanation.
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
- Customer Interaction: `.txt`, `.log` (planned/unavailable until its workflow exists)

The transitional shell uses attachments for routing only. Existing workflow pages continue to own ingestion and validation, so users hand the source to the selected workflow after routing.

### Explicit non-goals

- No generic workflow agents.
- No MCP adapter or tool server.
- No PostgreSQL, Redis, MinIO, DuckDB, queue, or worker migration.
- No Data Designer integration claim.
- No replacement of existing validation, artifact, model, or orchestration code.
- No automated execution of the Database headless pipeline from chat; that path auto-approves semantics and is not equivalent to the interactive review workflow.
- No placeholder Interaction Twin that produces unvalidated artifacts.

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
        `--> explicit unavailable Interaction response
```

The interface mapping is intentionally outside `application/`: the application router knows product capability IDs and availability, but it does not import Streamlit pages.

## Acceptance Criteria for Phase 1

- Chat is the default Streamlit page.
- Schema, Database, and Document can be selected explicitly from Chat.
- `.sql`/schema, SQLite, and PDF attachments route deterministically.
- ambiguous or conflicting attachments produce clarification rather than a guess.
- Interaction selection returns a planned/unavailable result and never invokes another workflow.
- selecting an available capability offers a hand-off to the existing workflow implementation without copied stage logic.
- existing specialized pages remain directly accessible.
- existing architecture dependency contracts and relevant workflow checks continue to pass.
