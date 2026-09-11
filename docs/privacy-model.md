# Privacy Model

The platform treats privacy as an explicit policy/evaluation concern rather than making blanket anonymity guarantees.

## Policy concepts

The domain privacy layer models actions such as preserving safe public categories, transforming direct identifiers, handling rare/quasi-identifying values, excluding unsuitable free text, and redacting document content where required.

## Evaluation

Validation code includes privacy-oriented evaluation such as distance-to-closest-record style checks and membership-inference-style metrics where configured. Required checks that do not run should be surfaced as `NOT_RUN`/`SKIPPED` rather than silently reported as passing.

## Differential privacy

Do not claim formal differential privacy merely because noise, sanitization, or privacy metrics are present. A formal DP claim requires the DP mechanism/accountant and its report to have actually run with defined parameters and guarantees.

## PDF handling

PDF Twin separates source de-identification/redaction from synthetic generation. The redacted-source operation is optional and produces its own report; synthetic rendering has separate validation and ground-truth artifacts.


## Shared workspace persistence

The local product workspace persists safe SHA-256 source/configuration fingerprints, workflow stage events, normalized result metadata, projects, preferences, lineage, and artifact references. Its DTOs have no fields for source payloads, passwords, tokens, database URLs, or resolved credentials. Artifact paths are resolved and accepted only below deployment-approved local roots.

Customer Interaction is stricter: raw transcript text remains in the active Streamlit process only. Pre-validation sanitized/structured files use permission-restricted temporary staging; only artifacts admitted by a non-`FAIL` release verdict are promoted into the durable workspace and cataloged, while blocked outputs are removed. Session/project/result records contain no transcript or source filename, and blocked results retain only safe validation/lineage metadata without downloadable artifacts. Execution status is stored separately from validation and release verdict so persistence cannot turn a completed run into a privacy claim.


Original SQLite and PDF uploads use permission-restricted process-temporary staging outside the durable workspace. Staging creation fails unless private permissions can be established, and explicit cleanup reports failure unless removal is verified. At the Database disconnect gate, the UI first deletes every persisted protected `profile.json` intermediate and aborts the disconnect if profile or source cleanup cannot be verified; source-derived profile statistics remain in the active process only for post-generation QA, while the portable artifact contains the export-sanitized profile. It then removes the staged SQLite source. Document removes its staged source after result completion. Current-process source staging is removed at process exit, and a 24-hour stale-directory sweep bounds abandoned-session retention. Shared result downloads use immutable content-addressed snapshots and re-verify size and SHA-256 before serving. The installed UI launcher binds to localhost because authenticated multi-user workspace access is not implemented.

## Guardrail evidence and model privacy

Chat/model input masking uses offset-based PII detection plus secret/token/key
patterns before model use. Prompt-injection, scope, and conservative content
checks execute locally. Model output is bounded and rescanned; sensitive JSON is
rejected rather than repaired, while non-JSON text can be masked. Persistable
`GuardrailReport` values contain only policy identifiers, fixed finding codes,
categories, counts, and decisions—never prompt fragments or matched values.

Customer Interaction still sanitizes and release-gates independently; its model
may return only source-grounded closed labels. Ollama composition accepts only a
loopback HTTP origin. The deterministic safety patterns are an inspectable
baseline rather than a comprehensive classifier, and local tool permission
checks are not represented as authenticated multi-user RBAC.
