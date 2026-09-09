# Synth Platform

Synth Platform is a synthetic-data intelligence platform with a chat-first entry
point for specialized Schema, Database, Document/PDF, and Customer Interaction
Twin workflows. A thin, deterministic coordinator selects a workflow; the
workflow remains responsible for generation or structured extraction, repair,
validation, privacy controls, and artifact packaging.

```text
Chat UI -> Capability Coordinator -> Existing Workflow -> Validation -> Artifacts
```

The platform lifecycle is:

```text
Discover -> Understand -> Contract -> Learn -> Generate -> Validate -> Package
```

## Current Capability Status

| Capability | Status | Current responsibility |
|---|---|---|
| Unified Chat | Implemented | Explicit commands, capability selection, attachment-type routing, conversation state, and safe workflow hand-off |
| Schema Twin | Implemented | Schema normalization, deterministic generation, relational validation, and CSV/Parquet packaging |
| Database Twin | Implemented | Discovery, profiling, statistical training, portable artifacts, relational generation, QA, and target write |
| Document Twin | Implemented for PDF | Extraction, template compilation, semantic binding, synthetic values, rendering, and layout/content validation |
| Customer Interaction Twin | Implemented | TXT/LOG parsing, source sanitization, one structured SSOT, privacy/source-replay validation, and checksummed ZIP packaging |
| Production readiness CLI | Implemented | Local dependency, version, workflow, storage, signing, OCR, and optional-capability checks |
| NVIDIA/Data Designer adapter | Planned | Dependencies can be installed, but no application adapter is implemented yet |
| MCP/Faker-as-a-Tool and live service agents | Planned | Target architecture only; not part of the active runtime |
| Unified governance/evaluation and enterprise security plane | Planned | Existing workflow validators and artifact signing remain authoritative until shared services are implemented |

Customer Interaction Twin turns one pasted or uploaded `.txt`/`.log` transcript into one structured interaction SSOT. It sanitizes sensitive values before optional model use, validates privacy and source-replay risks, and releases a checksummed ZIP containing only the sanitized source, SSOT, validation report, and manifest. Raw transcript text is not persisted. See [`docs/workflows/interaction-twin.md`](docs/workflows/interaction-twin.md).

The attached cross-domain, air-gapped architecture diagrams are treated as a
target architecture. Installing an NVIDIA, NeMo, MCP, or agent dependency does
not cause readiness to report an application integration that does not exist.
See [`docs/unified-chat-transition.md`](docs/unified-chat-transition.md) for the
phased transition and [`docs/architecture.md`](docs/architecture.md) for current
layer ownership.

## Run From Source

Python 3.11 or newer is required. Install the current chat-first implementation
from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[ui,database,pdf,schema,parquet]'
synth-platform-ui --server.port 8502
```

Open `http://localhost:8502`. Chat is the default page; the existing specialized
workflow pages remain directly accessible during the transition.

The published v3.0.2 wheel predates the chat-first transition. Use the source
installation above to evaluate the changes in this branch.

## Optional NVIDIA Dependencies

The `nvidia` extra declares the local packages anticipated by the target
synthetic-data and governance architecture:

```bash
pip install -e '.[ui,database,pdf,schema,parquet,nvidia]'
```

Python 3.12 is recommended when evaluating the complete NVIDIA extra because the
`nemo-retriever[nemotron-parse]` dependency is enabled only on Python 3.12.
These packages do **not** activate a Data Designer adapter, NeMo workflow, MCP
gateway, service-agent plane, or unified evaluator by themselves.

For an air-gapped installation, stage this project and every required wheel in a
local directory, then disable package-index access explicitly:

```bash
python -m pip install --no-index \
  --find-links=/opt/synth-platform/wheels \
  -e '.[ui,database,pdf,schema,parquet,nvidia]'
```

## Production Readiness

The readiness command is intentionally local and air-gap safe. It inspects
installed distribution metadata, declared version constraints, local workflow
facades, output paths, signing-key configuration, and local OCR executables. It
does not open sockets, invoke models, process source data, or call an external
service.

```bash
# Human-readable report
synth-platform-readiness

# Stable JSON report
synth-platform-readiness --json

# Require every locally verifiable optional dependency/executable group
synth-platform-readiness --strict

# Require selected capabilities
synth-platform-readiness --require schema --require nvidia

# Override local artifact paths
synth-platform-readiness \
  --output-root /srv/synth/output \
  --staging-root /srv/synth/staging
```

Readiness states:

- `READY`: all required and selected locally verifiable checks pass.
- `DEGRADED`: required checks pass, but optional capabilities are unavailable or
  source execution cannot be matched to installed distribution metadata.
- `NOT_READY`: a required, strict, or explicitly selected capability check fails.
- `NOT_CHECKED`: a service-health question such as Ollama endpoint/model health
  was deliberately not probed. It becomes fatal only when explicitly required.
- `PLANNED`: the target component has no active application integration and is
  never promoted merely because a package is installed.

Exit codes are `0` for `READY` or `DEGRADED`, `1` for `NOT_READY`, `2` for invalid
CLI arguments, and `3` for a redacted internal checker failure.

Production artifact signing requires a compatible Ed25519 private/public pair:

```text
SYNTH_SIGNING_PRIVATE_KEY_HEX
SYNTH_TRUSTED_PUBLIC_KEY_HEX
```

Each variable must contain 32 bytes encoded as hexadecimal. The readiness report
shows only configuration and compatibility booleans; it never emits key values.

## Repository Documentation

- [`docs/architecture.md`](docs/architecture.md): current layers and workflow ownership
- [`docs/internal-dependencies.md`](docs/internal-dependencies.md): active dependency chains
- [`docs/unified-chat-transition.md`](docs/unified-chat-transition.md): target-plane boundaries and phases
- [`docs/ci.md`](docs/ci.md): CI/CD checks
- [`docs/code-map.md`](docs/code-map.md): active source inventory
