# Synth Platform

Unified synthetic-data intelligence platform with multiple SSOT input workflows
for SQL schema, database, PDF/document, and customer interaction sources.

The platform centers on one stable lifecycle:

```text
Connect -> Understand -> Build twin -> Generate -> Validate -> Package
```

See `docs/architecture.md` for the architecture and `docs/ci.md` for the
current CI/CD checks.

## Workflows

- **Schema Twin** accepts SQL DDL with `CREATE TABLE` statements and generates
  synthetic tables from declared fields, keys, constraints, and relationships.
- **Database Twin** is a retrainable ML-model workflow: connect a source
  database, approve the understanding, train table models, export a portable
  artifact, generate from the artifact, validate results, and retrain when the
  SSOT or model settings change. PII/PHI/sensitive fields are not learned as raw
  values; they are replaced with Faker/context-aware synthetic values.
- **PDF Twin** preserves useful document structure and binds PDF values to the
  shared synthetic entity/database result so identities stay consistent across
  outputs.
- **Customer Interactions Twin** turns a source interaction into a twin contract,
  then creates synthetic customer conversations with context-aware Faker
  replacements for direct identifiers, PHI, and sensitive values.

## Verification

Recent local readiness checks:

```text
tests/e2e/workflows: 23 passed, 10 skipped
tests/unit: 1134 passed, 3 skipped
focused SSOT correctness checks: 54 passed
compileall src tests: passed
git diff --check: passed
```

## Run The Streamlit UI

Install the released wheel into a virtual environment:

```bash
python -m venv .venv-release-ui
source .venv-release-ui/bin/activate
python -m pip install --upgrade pip
pip install 'synth-platform[ui,database,pdf] @ https://github.com/AdityaKrishnanNeelakantan/CI-CD/releases/download/v3.0.2/synth_platform-3.0.2-py3-none-any.whl'
```

Start the UI:

```bash
synth-platform-ui --server.port 8502
```

Then open:

```text
http://localhost:8502
```

The `ui,database,pdf` extras are needed because the Streamlit app loads the
database and PDF source-input pages. For customer interaction and SDK-backed
generation features, install the optional extras supported by your Python
version.

SDK-backed generation is configured through an OpenAI-compatible model provider
endpoint. One shared SLM runtime is used by Schema Mode, PDF Twin, and Customer
Interactions Twin. Local development defaults to the Ollama-compatible
`synth-platform-slm` alias. In customer environments this should point to an
internally hosted NIM, vLLM, TGI, or OpenAI-compatible gateway:

```bash
export SP_PLATFORM_SLM_ENDPOINT="http://internal-llm.example.com/v1"
export SP_PLATFORM_SLM_PROVIDER="internal"
export SP_PLATFORM_SLM_MODEL="synth-platform-slm"
```

For local Ollama development, use `http://localhost:11434/v1` and set
`SP_NEMO_DATA_DESIGNER_API_KEY=ollama`. If an internal gateway requires
authentication, set `SP_PLATFORM_SLM_API_KEY_ENV` to the environment-variable
name holding the key.
The older `SP_NEMO_DATA_DESIGNER_*` names are still accepted as compatibility
fallbacks.

For Customer Interactions/Transcript Twin on a local SLM, keep each SDK request
small and let Data Designer process one turn per seed row:

```bash
export SP_TRANSCRIPT_DATA_DESIGNER_MAX_TOKENS="96"
export SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT="180"
export SP_TRANSCRIPT_DATA_DESIGNER_MAX_PARALLEL_REQUESTS="4"
```

To inspect raw OpenAI-compatible payloads from Data Designer, run the local
logging proxy and point the endpoint at the proxy:

```powershell
.\.venv\Scripts\python.exe scripts\openai_compatible_proxy.py --target http://127.0.0.1:11434 --port 18000
$env:SP_NEMO_DATA_DESIGNER_ENDPOINT="http://127.0.0.1:18000/v1"
```

Schema upload in the Streamlit UI is qualified for SQL DDL `.sql` files with
`CREATE TABLE` statements, fields, keys, defaults, constraints, and explicit
PK/FK relationships.

## Run From Source

For local development, install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[ui,database,pdf,schema,parquet]'
synth-platform-ui --server.port 8502
```
