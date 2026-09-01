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
endpoint. In customer environments this should point to an internally hosted
NIM/LLM service such as NIM, vLLM, TGI, OpenRouter-compatible gateway,
Together-compatible gateway, or another OpenAI-compatible endpoint:

```bash
export SP_NEMO_DATA_DESIGNER_ENDPOINT="http://internal-llm.example.com/v1"
export SP_NEMO_DATA_DESIGNER_PROVIDER="internal"
export SP_NEMO_DATA_DESIGNER_MODEL="customer/slm"
```

No generation key is required for local endpoints such as
`http://localhost:8000/v1`. If an internal gateway requires authentication, set
`SP_NEMO_DATA_DESIGNER_API_KEY` in the runtime environment.

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
