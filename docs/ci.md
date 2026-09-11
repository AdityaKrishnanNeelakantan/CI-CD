# CI/CD

The project keeps backend, frontend, packaging, artifact, and release checks
separate so failures point to the layer that needs attention.

## Local Commands

Install the minimum test environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"
```

Install the default CI environment:

```bash
python -m pip install -e ".[test,schema,database,pdf,parquet]"
```

Install the heavy integration environment:

```bash
python -m pip install -e ".[test,ui,schema,database,pdf,parquet,docling,ocr,llm]"
```

Run the core checks:

```bash
export XDG_CACHE_HOME="${PWD}/.cache"
python -m compileall -q src tests streamlit_app.py
pytest tests/contract -q
pytest tests/unit -q
pytest tests/security -q
```

Run lighter integrations:

```bash
pytest tests/integration/database_pdf \
  --ignore=tests/integration/database_pdf/test_docling_pdf_pipeline.py -q
pytest tests/integration/test_conditional_generation.py \
  tests/integration/test_document_extraction.py \
  tests/integration/test_relational_generation.py \
  tests/integration/test_two_source_directive.py -q
```

Run heavy checks manually or on schedule:

```bash
pytest tests/integration/schema tests/integration/database_pdf/test_docling_pdf_pipeline.py tests/e2e -q
```

Build and verify package artifacts:

```bash
python -m pip install --upgrade build
python -m build
python -m venv .venv-wheel
.venv-wheel/bin/python -m pip install --upgrade pip
.venv-wheel/bin/python -m pip install dist/*.whl
.venv-wheel/bin/python -c "import synth_platform; print(synth_platform.__version__)"
.venv-wheel/bin/synth-platform --help
```

Build and verify a portable twin delivery:

```bash
python scripts/build_artifact_delivery.py --output-dir artifact-delivery
```

Smoke test the installed Streamlit UI launcher:

```bash
python scripts/smoke_streamlit_ui.py
```

Create a release:

```bash
git tag v3.0.0
git push origin v3.0.0
```

Tag pushes matching `v*` run `.github/workflows/release.yml`. The release
workflow builds package distributions, verifies the wheel, builds a portable
twin delivery, creates checksums, and publishes a GitHub Release.

## CI Jobs

- `package-smoke`: validates packaging, editable install, imports, compile, and
  CLI help.
- `architecture`: enforces dependency direction and cross-module contracts.
- `unit`: runs focused unit tests.
- `security`: runs privacy/security guard tests.
- `integration-light`: exercises lifecycle-oriented integrations without the
  heaviest model/provider paths.
- `integration-heavy`: manual or scheduled job for Docling, Streamlit, OCR,
  broader schema/e2e coverage, and future provider-backed tests.
- `build-package`: builds the wheel and source distribution, installs the wheel
  in a fresh environment, runs package smoke checks, and uploads `dist/`.
- `artifact-delivery`: trains a small portable twin, deletes the source
  database, generates synthetic tables from the artifact only, validates the
  result, checks source canaries do not leak, and uploads the delivery bundle.
- `ui-smoke`: installs the package with UI dependencies, starts
  `synth-platform-ui`, and verifies the Streamlit health endpoint.
- `release`: runs on version tags, builds verified package artifacts and a
  portable twin delivery archive, generates checksums, and publishes a GitHub
  Release.

## Current Boundaries

CI verifies tests, package builds, frontend builds, Docker image builds, and
release artifacts. Publishing to a package registry, container registry, or
deployment target requires project-specific credentials and ownership decisions.
