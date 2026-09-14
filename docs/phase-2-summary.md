# Phase 2 Summary: LLM Egress Policy

## Confirmed Previous Behavior

- `TextGenerationConfig` defaulted to external provider `"openai"` in `src/synth_platform/engine/generation/text/generator.py:34` before this change. Its provider call path read `OPENAI_API_KEY`/`GROQ_API_KEY` and used Groq's public OpenAI-compatible URL by default in the same file's `_call_provider()` implementation.
- `SmartValueGenerator` defaulted to external provider `"groq"` in `src/synth_platform/engine/generation/schema/smart_values.py:676` before this change, and initialized Groq/OpenAI clients from environment-backed API keys.
- `OllamaChatModel` was already local by default, using `http://localhost:11434` in `src/synth_platform/infrastructure/llm/ollama_chat.py:16`.
- The disk cache in `SmartValueGenerator` is a local JSON cache. It uses `Path.home() / ".mvp" / "value_cache"` when no cache directory is supplied, loads from local files, and writes JSON with `json.dump()`; no network behavior is present in cache load/save.

## Policy Enforcement

- The single source of truth now lives in `src/synth_platform/domain/privacy/llm_policy.py`.
- `LlmPolicy` classifies providers as local/off/external. `ollama` and `local` are allowed by default. `off`, `none`, and `disabled` are non-egress disabled states. `openai` and `groq` are blocked unless `SYNTH_ALLOW_EXTERNAL_LLM=true` is set.
- `TextGenerationEngine._call_provider()` consults `LlmPolicy` before constructing any client or reading external provider configuration.
- `SmartValueGenerator._get_client()` consults `LlmPolicy` before constructing any Groq/OpenAI client.
- Policy blocks raise `LlmPolicyError`, which is not swallowed by the text-generation fallback path. Ordinary provider/runtime failures still fall back where the previous behavior did.

## Updated Defaults

- `TextGenerationConfig.provider` now defaults to `"ollama"` in `src/synth_platform/engine/generation/text/generator.py:36`.
- `SmartValueGenerator.provider` now defaults to `"ollama"` in `src/synth_platform/engine/generation/schema/smart_values.py:678`.
- `DataSimulator.llm_provider` and the legacy `llm_text.py` helper defaults were also changed from `"openai"` to `"ollama"` so compatibility paths inherit the same air-gapped-safe default.

## UI Messaging

- Schema Twin catches `LlmPolicyError` and displays a specific air-gapped policy message with the `SYNTH_ALLOW_EXTERNAL_LLM` opt-in guidance in `src/synth_platform/interfaces/streamlit/pages/schema_twin.py:281`.
- Database Twin catches the same policy error around the optional free-text rewrite post-pass and displays the same guidance in `src/synth_platform/interfaces/streamlit/pages/database_twin.py:731`.

## Design Decisions

- I used `SYNTH_ALLOW_EXTERNAL_LLM=true` as the deliberate opt-in flag. API keys alone are no longer sufficient.
- I added Ollama support to the shared text engine provider call path so the new default is actually usable when a local Ollama daemon is running.
- Local Ollama connection failures still fall back through the existing provider-failure behavior; only external-provider policy violations fail closed.

## Contradictions / Extra Findings

- The original description was accurate for the two named defaults. I also found additional `"openai"` defaults in `DataSimulator` and the legacy `llm_text.py` wrapper, which could feed the same shared text-generation path. Those are now `"ollama"`.
- This phase intentionally did not fix the known ineligible text fallback `null_rng` crash in `TextGenerationEngine.generate()` or the schema `hard_checks_passed`/transfer gating work from Phase 1.

## Regression Coverage

- Added `tests/unit/test_llm_policy.py` covering blocked external provider attempts without opt-in, allowed external provider calls with opt-in, local Ollama behavior without opt-in, and local disk cache read/write.
- Installed project dependencies into `.venv-phase2` with `.[test,database,pdf,ui,schema,ocr,parquet]`.
- `python -m pytest tests/unit/test_llm_policy.py -v` result, run via `.venv-phase2/bin/python`: **6 passed in 0.16s**.
- Full suite result, run via `.venv-phase2/bin/python` with `XDG_CACHE_HOME=/private/tmp/synth-platform-cache` and elevated sandbox permissions for localhost/socket and cache access: **1368 passed, 11 skipped, 92 warnings in 54.97s**.
- The first full-suite run surfaced a Phase 2 architecture-contract violation because the engine imported `infrastructure.llm.policy` and `OllamaChatModel` directly. I fixed this by moving the policy to the domain layer and removing direct engine-to-infrastructure LLM imports.
- The restricted sandbox blocked tests that bind localhost and tests that write the default `~/.cache/synth-platform` cache. The passing full-suite run used a writable `XDG_CACHE_HOME` and elevated sandbox permissions for the Streamlit health-check socket.

## Real-Fixture Verification

- Added `tests/fixtures/local-data/` to `.gitignore`; `git check-ignore` confirms files under it are ignored.
- Copied the three local PDFs into `tests/fixtures/local-data/`: `healthcare_patient_summary.pdf`, `clinical_trial_master_report.pdf`, and `sample_clinical_study_records.pdf`.
- Copied four customer-service transcripts into the same ignored directory for future Customer Interaction Twin work. They are intentionally not processed in this phase.
- Copied `schema.sql` and created a trimmed SQLite fixture `banking_trimmed.db` from the 191MB source database instead of copying the full DB. The trimmed DB is about 1.3MB and contains FK-consistent rows: 429 branches, 992 customers, 1000 accounts, 1320 cards, 3183 merchants, 5000 transactions, and 500 loans.
- Replaced the old missing `dummy_statement.pdf` Streamlit UI fixture path with `tests/fixtures/local-data/healthcare_patient_summary.pdf`.
- Added backend healthcare PDF verification that runs profiling, template compilation, semantic binding, value generation, rendering, validation, de-identification, and the Phase 1 transfer gate. The real data surfaced that structured `pii_findings` detects the SSN-like value and phone numbers, while the patient full name and DOB are removed by the template/entity redaction path rather than listed in `pii_findings`.
- Added backend banking DB verification that runs discover, profile, infer, approve, train, generate, validate, target write, FK checking, and Phase 1 transfer gate checks against the trimmed fixture.
- Focused real-fixture result: `tests/e2e/workflows/test_pdf_twin_ui.py`, the local healthcare PDF backend test, and the local banking DB backend test: **4 passed in 5.12s**.
- `git diff --check` still reports trailing whitespace in pre-existing Phase 1 transfer-gating hunks (`audit.py` and `database_twin.py`). I did not change those as part of Phase 2.
