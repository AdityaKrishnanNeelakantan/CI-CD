# Synth Platform

Unified synthetic-data intelligence platform for schema, database, and document
twin workflows.

The platform centers on a stable lifecycle:

```text
Discover -> Understand -> Contract -> Learn -> Generate -> Validate -> Package
```

See `docs/architecture.md` for the architecture and `docs/ci.md` for the
current CI/CD checks.

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
database twin and PDF twin pages.

## Run From Source

For local development, install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[ui,database,pdf,schema,parquet]'
synth-platform-ui --server.port 8502
```
