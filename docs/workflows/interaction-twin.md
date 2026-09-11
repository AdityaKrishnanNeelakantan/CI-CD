# Customer Interaction Twin Workflow

## User flow

```text
Select Customer Interaction → Paste transcript or upload TXT/LOG
  → Parse and sanitize → Build one structured SSOT
  → Validate privacy and source replay → Download released ZIP
```

The capability converts one source transcript into one structured interaction system of record (SSOT). It does not synthesize a replacement conversation and does not invoke a model once per turn.

## Runtime path

```text
interfaces/streamlit/pages/interaction_twin.py
  ↓
application/workflows/interaction_twin.py
  ├─ optional ChatModel / Ollama semantic enhancement
  ├─ RunManifest + StageResult lineage
  └─ canonical artifact and ZIP packaging
  ↓
engine/interactions/service.py
  ├─ transcript parsing
  ├─ participant pseudonymization
  ├─ PII, identifier, and secret sanitization
  ├─ deterministic semantic extraction and fallback
  ├─ strict SSOT construction
  └─ privacy, integrity, and source-replay validation
  ↓
domain/interactions/models.py + domain/validation/release_gate.py
```

The chat coordinator only selects this capability and hands control to its page. Parsing, privacy, semantic extraction, validation, and packaging remain workflow-owned.

## Input and parsing

The page accepts pasted text and `.txt` or `.log` uploads. A transcript may use lines such as:

```text
[09:14:03] Customer: I cannot access my account.
Agent: I can help with the access issue.
```

Timestamps are optional. Continuation lines are appended to the preceding turn, and unlabeled content is retained under an unknown participant rather than guessed. Input is limited to 1,000,000 characters.

Raw participant labels are replaced with stable run-local IDs such as `participant_001`. Sensitive values are replaced with typed tokens such as `[EMAIL_001]`. Sanitization reuses the PDF workflow's PII and entity detectors and adds labeled account/member/customer/case/ticket/order/reference identifiers plus password, passcode, PIN, OTP, token, and secret patterns.

## Optional model behavior

The workflow is deterministic by default. When Ollama enhancement is enabled:

- the model receives only the sanitized transcript;
- the workflow performs zero or one `ChatModel.complete()` call for the whole transcript;
- input is capped at 20,000 characters;
- output must validate as a closed enum-only JSON object for topics, issue codes, action codes, resolution, and sentiment; and
- timeout, malformed JSON, schema failure, or invalid enum values trigger deterministic fallback.

Immutable facts, participant IDs, turn counts, summaries, privacy checks, and release decisions are never delegated to the model. Transcript content is treated as untrusted data, not as model instructions.

## Validation and release

The validation stage checks:

- strict SSOT schema conformance;
- participant and turn-count integrity;
- residual PII, identifiers, and secrets;
- absence of original sensitive spans and raw speaker labels;
- exact source-turn replay; and
- suspicious eight-word overlap with the source.

These checks feed the existing domain release gate. A ZIP is created only when the release verdict is not `FAIL`.

## Persistence and package contract

Raw transcript text remains in memory and is never written to the run directory. A released ZIP contains exactly:

| File | Purpose |
|---|---|
| `sanitized_source.txt` | Sanitized, pseudonymized transcript used for downstream processing |
| `interaction_ssot.json` | Strict structured interaction record without transcript text |
| `validation_report.json` | Privacy, replay, integrity metrics, and release verdict |
| `manifest.json` | Source fingerprint, run/model metadata, artifact sizes, and SHA-256 checksums |

ZIP entry names, order, permissions, and timestamps are deterministic. The manifest records checksums and sizes for the three data artifacts; packaging verifies them before writing the archive.

## Architecture boundaries

Customer Interaction Twin is part of the synthetic-data plane. It does not add or claim:

- a generic LLM sub-agent or live service agent;
- an MCP server or Faker-as-a-Tool;
- a Data Designer adapter;
- a database, queue, or object-store requirement; or
- a replacement shared governance/evaluation service.
