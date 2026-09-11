# Phase 3 Summary: Text Fallback and Schema Validation Fail-Closed

## C4: Text Generation Fallback Crash

- Current bug location before this fix: `src/synth_platform/engine/generation/text/generator.py:74-85`.
- The ineligible-column fallback branch called `null_rng.random()` before `null_rng` was initialized later in `TextGenerationEngine.generate()`.

Before:

```python
params = column.distribution_params or {}
eligibility = assess_column_eligibility(column, table_name=table_name)

if not eligibility.eligible:
    values = [
        fallback_text(
            ...
            rng_value=null_rng.random(),
        )
        for i in range(size)
    ]

...
null_rng = np.random.default_rng(
    (self.config.seed or 0) + hash(f"{table_name}.{column.name}") % (2**32)
)
```

After:

```python
params = column.distribution_params or {}
null_rng = np.random.default_rng(
    (self.config.seed or 0) + hash(f"{table_name}.{column.name}") % (2**32)
)
eligibility = assess_column_eligibility(column, table_name=table_name)

if not eligibility.eligible:
    values = [
        fallback_text(
            ...
            rng_value=null_rng.random(),
        )
        for i in range(size)
    ]
```

- Final fixed location: `src/synth_platform/engine/generation/text/generator.py:69-95`.
- Regression test added at `tests/integration/schema/test_text_generation.py:173`.
- The test calls `generate_text_column()` with an integer column and verifies fallback strings are returned with ineligible evidence instead of raising `UnboundLocalError`.

## C5: Schema Validation Fail-Closed

- Current bug location before this fix: `src/synth_platform/application/workflows/schema_twin.py:77-87`.
- `SchemaModeResult.hard_checks_passed` returned `True` when the validation report had no recognized pass/fail fields.

Before:

```python
def validation_report(self) -> Dict[str, Any]:
    return self.pipeline.validation_report or {}

def hard_checks_passed(self) -> bool:
    report = self.validation_report
    if "hard_checks_passed" in report:
        return bool(report["hard_checks_passed"])
    if "passed" in report:
        return bool(report["passed"])
    status = report.get("status")
    if isinstance(status, str):
        return status.lower() in {"pass", "passed", "ok", "success"}
    return True
```

After:

```python
def validation_report(self) -> Dict[str, Any]:
    report = self.pipeline.validation_report
    return report if isinstance(report, Mapping) else {}

def hard_checks_passed(self) -> bool:
    report = self.validation_report
    if not isinstance(report, Mapping) or not report:
        return False
    if "hard_checks_passed" in report:
        return report["hard_checks_passed"] is True
    if "passed" in report:
        return report["passed"] is True
    status = report.get("status")
    if isinstance(status, str):
        return status.lower() in {"pass", "passed", "ok", "success"}
    return False
```

- Final fixed location: `src/synth_platform/application/workflows/schema_twin.py:68-89`.
- Missing, empty, malformed, unknown, or non-explicit validation statuses now return `False`.
- Explicit pass/fail behavior is preserved for recognized reports: `hard_checks_passed is True`, `passed is True`, and positive string statuses still pass; false or absent values do not.
- `validation_report` now normalizes non-mapping pipeline payloads to `{}` so `validation_highlights()` also fails closed instead of crashing on malformed reports.

## Transfer Gate Impact

- Phase 1 `TransferService` core structure was not changed.
- The Schema Twin adapter path is now strictly safer because a malformed `SchemaModeResult` can no longer synthesize a passing `passed=True` value from an unknown validation report.
- Regression coverage in `tests/unit/application/test_transfer_service.py:140` confirms a Schema Twin transfer derived from an unknown `SchemaModeResult.hard_checks_passed` value is blocked.
- Existing behavior for valid Schema Twin transfer reports is unchanged: `{"passed": True, "export_ready": True}` still passes, and failed or absent validation still blocks.

## Tests

- Dependency install:
  - First command: `.venv-phase2/bin/python -m pip install -e '.[test,database,pdf,ui,schema,ocr,parquet]'`
  - Initial sandboxed result: failed because DNS to `pypi.org` was blocked while resolving build dependencies.
  - Elevated rerun result: succeeded; editable `synth-platform==3.0.2` was rebuilt and installed, with dependencies already satisfied.
- C4 focused regression:
  - Command: `.venv-phase2/bin/python -m pytest tests/integration/schema/test_text_generation.py::test_generate_text_column_for_ineligible_integer_returns_fallback_values -v`
  - Included in focused run result: `10 passed in 2.38s`.
- C5 focused regressions:
  - Command included `tests/unit/schema/test_schema_mode.py::test_schema_mode_hard_checks_fail_closed_for_missing_or_malformed_validation`, `tests/unit/schema/test_schema_mode.py::test_schema_mode_hard_checks_still_accept_explicit_pass_signals`, and `tests/unit/application/test_transfer_service.py::test_schema_transfer_blocks_when_schema_mode_validation_status_is_unknown`.
  - Focused run result: `10 passed in 2.38s`.
- Broader touched-area run:
  - Command: `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest tests/integration/schema/test_text_generation.py tests/unit/schema/test_schema_mode.py tests/unit/application/test_transfer_service.py -v`
  - Result: `56 passed in 2.67s`.
- Full suite:
  - First non-elevated command: `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
  - Result: `1371 passed, 11 skipped, 7 failed, 92 warnings in 53.11s`.
  - Failures were environment-related: readonly default audit database writes and denied localhost socket bind.
  - Elevated rerun command: `XDG_CACHE_HOME=/private/tmp/synth-platform-cache .venv-phase2/bin/python -m pytest`
  - Final result: `1378 passed, 11 skipped, 72 warnings in 47.41s`.

## Test Changes

- Added new regression tests only.
- No existing test assertions were changed to accommodate the corrected behavior.

## Follow-Up Risks

- The default audit logger can still target a location that is readonly in the restricted sandbox, which affects non-elevated e2e runs. This was observed during verification but was not changed in Phase 3 because it is outside C4/C5.
