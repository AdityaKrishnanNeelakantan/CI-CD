# Guardrail Architecture

## Purpose

The platform implements input, output, and tool guardrails around existing
boundaries. Guardrails do not create a generic agent and do not move generation,
validation, or release ownership out of Schema, Database, Document, or Customer
Interaction Twin.

```text
Home / Chat --input checks--> deterministic capability router
                                      |
                                      v
                              specialized workflow
                                      |
                 input checks -> guarded ChatModel -> output checks
                                      |
                           workflow validation/release

local UI or future MCP adapter
        -> strict tool command schema
        -> local permission policy
        -> WorkspaceService / specialized operation
```

The capability coordinator remains deterministic and routing-only. MCP is not
implemented; a future MCP adapter may translate protocol messages into the
existing guarded command DTOs but must remain below the coordinator and may not
bypass application policy.

## Input guardrails

`engine/security/text_guardrails.py` applies local deterministic checks before a
model or routing boundary:

- sensitive-data detection and masking using the existing offset-based PII
  detector plus secret/token/private-key patterns;
- prompt-injection indicators such as instruction override, system-prompt
  extraction, tool invocation, and guardrail bypass requests;
- scope limits for maximum input size and workflow-specific prompt markers;
- conservative content-safety rules for clearly instructional dangerous,
  credential-abuse, self-harm, and child-sexual-abuse requests.

Reports contain fixed category names and counts only. They never contain prompt
fragments, matched values, secrets, or source text. The rules are intentionally
inspectable and conservative; they are not claimed to be a comprehensive
moderation classifier.

Home/Chat uses the input checks before retaining routing messages. Sensitive
values are masked, blocked content is represented by a generic placeholder, and
attachment history stores counts/types rather than raw filenames. The router
still selects capabilities by explicit commands, choices, and extensions; no
model participates in routing.

## Model boundary

`infrastructure/llm/guarded_chat.py` decorates the existing `ChatModel` port:

1. inspect and transform input;
2. block before network/model use when policy fails;
3. call the injected provider;
4. require bounded JSON/output shape where configured;
5. reject or mask sensitive output;
6. apply deterministic content checks;
7. return output to the specialized workflow for its authoritative schema,
   grounding, validation, and release checks.

`bootstrap.build_guarded_chat_model()` supplies named policy profiles for
Customer Interaction semantics, schema-guided extraction, and synthetic text.
The Ollama adapter accepts only a plain HTTP loopback origin (`localhost` or a
loopback IP), bypasses environment proxies, and rejects HTTP redirects so a
local endpoint cannot forward model content to another origin. Active
composition does not select direct hosted OpenAI/Groq paths: the legacy
smart-value helper is instantiated with hosted providers disabled, while
model-enabled synthetic text accepts only an explicitly injected guarded
completion model and otherwise uses deterministic fallbacks.

Groundedness remains workflow-specific rather than a fabricated universal
score. Customer Interaction admits only closed semantic labels supported by its
deterministic source evidence. Schema-guided document extraction drops scalar
values that do not have normalized source evidence. Synthetic narrative text is
not source-grounded by design; its relevant checks are schema/format, privacy,
content policy, and source-replay validation.

## Tool boundary

`application/dto/tool_commands.py` defines Pydantic commands with unknown fields
forbidden, bounded text, and path-safe identifiers. `GuardedWorkspaceTools` in
`application/services/tool_gateway.py` evaluates each read/write/update/delete
against `LocalToolPermissionPolicy` before delegating to `WorkspaceService`.

The current localhost UI uses a process-local principal with workspace read and
write permission. Artifact reads additionally prove artifact/session ownership.
Project/session updates validate both resource IDs. Delete remains denied unless
the principal has delete capability and the command carries explicit
confirmation; the UI principal deliberately has no delete capability.

This is operation and resource-scope enforcement, not authentication or
multi-user RBAC. The supported launcher remains bound to `127.0.0.1`.

## Policy configuration

The following deployment settings are read-only in the browser:

- `SP_GUARDRAIL_POLICY_VERSION`
- `SP_GUARDRAIL_MAX_INPUT_CHARS`
- `SP_GUARDRAIL_MAX_OUTPUT_CHARS`
- `SP_LOCAL_MODEL_HOST` (validated as loopback when a model is composed)

Users cannot disable model or tool guardrails from Settings. Guardrail reports
are transient unless a specialized workflow explicitly includes source-free
category/count evidence in its result; raw prompts and completions are not added
to workspace persistence.
