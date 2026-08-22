# CI/CD Phase 1

Phase 1 stabilizes the project as an installable Python package and makes CI
truthful about which layer failed. Deployment is intentionally out of scope.

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

## Phase 1 Boundaries

This phase does not deploy Streamlit, publish packages, build Docker images,
create an artifact registry, or automate releases. Those belong to later phases
after the package and CI quality gates are stable.
