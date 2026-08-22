# Active Code Map

This is the active `src/synth_platform` Python inventory after consolidation. Files under `archive/legacy/` are intentionally excluded.

## Application workflows

**4 Python files**

```text
application/workflows/__init__.py
application/workflows/database_twin.py
application/workflows/pdf_twin.py
application/workflows/schema_twin.py
```

## Application use cases

**21 Python files**

```text
application/use_cases/__init__.py
application/use_cases/compile_learning_plan.py
application/use_cases/discover_source.py
application/use_cases/export_artifact.py
application/use_cases/generate_and_publish.py
application/use_cases/generate_dataset.py
application/use_cases/generate_from_artifact.py
application/use_cases/generate_pdf.py
application/use_cases/infer_schema.py
application/use_cases/load_artifact.py
application/use_cases/materialize_connector.py
application/use_cases/profile_source.py
application/use_cases/propose_pdf_schema.py
application/use_cases/publish_dataset.py
application/use_cases/render_synthetic_pdfs.py
application/use_cases/run_full_pipeline.py
application/use_cases/train_from_database.py
application/use_cases/train_from_pdf.py
application/use_cases/train_model.py
application/use_cases/validate_dataset.py
application/use_cases/validate_extraction.py
```

## Application orchestration

**14 Python files**

```text
application/orchestration/__init__.py
application/orchestration/checkpointing.py
application/orchestration/schema/__init__.py
application/orchestration/schema/cache.py
application/orchestration/schema/config.py
application/orchestration/schema/export.py
application/orchestration/schema/result.py
application/orchestration/schema/schema_driven.py
application/orchestration/schema/session.py
application/orchestration/schema/source_driven.py
application/orchestration/schema/summary.py
application/orchestration/stages.py
application/orchestration/state_machine.py
application/orchestration/workflow.py
```

## Domain

**63 Python files**

```text
domain/__init__.py
domain/artifacts/__init__.py
domain/artifacts/bundle.py
domain/artifacts/checksums.py
domain/artifacts/guards.py
domain/artifacts/manifest.py
domain/artifacts/versions.py
domain/constraints/__init__.py
domain/constraints/compiler.py
domain/constraints/enforcement.py
domain/constraints/evaluation.py
domain/constraints/models.py
domain/constraints/repair.py
domain/documents/__init__.py
domain/documents/models.py
domain/extraction/__init__.py
domain/extraction/lowering.py
domain/extraction/quality.py
domain/extraction/target_schema.py
domain/generation/__init__.py
domain/generation/conditional.py
domain/generation/models.py
domain/inference/__init__.py
domain/planning/__init__.py
domain/planning/compiler.py
domain/planning/models.py
domain/planning/routing.py
domain/privacy/__init__.py
domain/privacy/classification.py
domain/privacy/models.py
domain/privacy/policies.py
domain/privacy/redaction.py
domain/privacy/verifier.py
domain/profiling/__init__.py
domain/profiling/distributions.py
domain/profiling/missingness.py
domain/profiling/models.py
domain/profiling/sensitivity.py
domain/profiling/statistics.py
domain/relational/__init__.py
domain/relational/cardinality.py
domain/relational/cycles.py
domain/relational/dag.py
domain/relational/key_allocation.py
domain/relational/models.py
domain/runs/__init__.py
domain/runs/models.py
domain/schema/__init__.py
domain/schema/compatibility.py
domain/schema/graph.py
domain/schema/models.py
domain/schema/source.py
domain/semantics/__init__.py
domain/semantics/confidence.py
domain/semantics/dependencies.py
domain/semantics/evidence.py
domain/semantics/models.py
domain/training/__init__.py
domain/validation/__init__.py
domain/validation/metric.py
domain/validation/models.py
domain/validation/release_gate.py
domain/validation/thresholds.py
```

## Discovery engine

**11 Python files**

```text
engine/discovery/__init__.py
engine/discovery/database/__init__.py
engine/discovery/database/adapters/__init__.py
engine/discovery/database/adapters/base.py
engine/discovery/database/adapters/csv_sqlite_builder.py
engine/discovery/database/adapters/errors.py
engine/discovery/database/adapters/registry.py
engine/discovery/database/adapters/sqlite_adapter.py
engine/discovery/database/demo/sample_database.py
engine/discovery/database/service.py
engine/discovery/service.py
```

## Profiling engine

**12 Python files**

```text
engine/profiling/__init__.py
engine/profiling/database/__init__.py
engine/profiling/database/cleaning/__init__.py
engine/profiling/database/cleaning/cleaner.py
engine/profiling/database/cleaning/service.py
engine/profiling/database/profiler.py
engine/profiling/database/service.py
engine/profiling/registry.py
engine/profiling/schema/__init__.py
engine/profiling/schema/profiler.py
engine/profiling/schema/profiles.py
engine/profiling/service.py
```

## Inference engine

**23 Python files**

```text
engine/inference/__init__.py
engine/inference/database/__init__.py
engine/inference/database/contract.py
engine/inference/database/engine.py
engine/inference/database/service.py
engine/inference/platform/__init__.py
engine/inference/platform/cross_table.py
engine/inference/platform/service.py
engine/inference/schema/__init__.py
engine/inference/schema/compat.py
engine/inference/schema/ddl.py
engine/inference/schema/distribution_rules.py
engine/inference/schema/domain_capsule.py
engine/inference/schema/domain_priors.py
engine/inference/schema/formulas.py
engine/inference/schema/introspect.py
engine/inference/schema/planning.py
engine/inference/schema/schema.py
engine/inference/schema/schema_columns.py
engine/inference/schema/semantic.py
engine/inference/schema/source_driven.py
engine/inference/schema/source_overlay.py
engine/inference/schema/yaml_schema.py
```

## Training engine

**22 Python files**

```text
engine/training/__init__.py
engine/training/backend_registry.py
engine/training/database/__init__.py
engine/training/database/adapters/__init__.py
engine/training/database/adapters/category_rebalancing.py
engine/training/database/adapters/context_generators.py
engine/training/database/adapters/copula_encoders.py
engine/training/database/adapters/dp_copula_adapter.py
engine/training/database/adapters/safe_copula_adapter.py
engine/training/database/adapters/sdv_adapter.py
engine/training/database/artifact/__init__.py
engine/training/database/artifact/builder.py
engine/training/database/artifact/errors.py
engine/training/database/artifact/loader.py
engine/training/database/artifact/manifest.py
engine/training/database/artifact/self_test.py
engine/training/database/artifact/service.py
engine/training/database/base.py
engine/training/database/planner.py
engine/training/database/registry.py
engine/training/database/service.py
engine/training/service.py
```

## Generation engine

**65 Python files**

```text
engine/generation/__init__.py
engine/generation/column_generator.py
engine/generation/constraint_executor.py
engine/generation/database/__init__.py
engine/generation/database/constraint_engine.py
engine/generation/database/key_store.py
engine/generation/database/relational_generator.py
engine/generation/database/relational_service.py
engine/generation/database/schema_graph.py
engine/generation/database/streaming_generator.py
engine/generation/database/target_write_service.py
engine/generation/database/target_writer.py
engine/generation/pii.py
engine/generation/relational_executor.py
engine/generation/schema/__init__.py
engine/generation/schema/assets.py
engine/generation/schema/chunked_export.py
engine/generation/schema/codegen.py
engine/generation/schema/constraints.py
engine/generation/schema/context.py
engine/generation/schema/customization.py
engine/generation/schema/datetime_utils.py
engine/generation/schema/db.py
engine/generation/schema/demos/__init__.py
engine/generation/schema/demos/banking.py
engine/generation/schema/duplicate_guard.py
engine/generation/schema/engines/__init__.py
engine/generation/schema/engines/fact_engine.py
engine/generation/schema/exceptions.py
engine/generation/schema/export.py
engine/generation/schema/generators/__init__.py
engine/generation/schema/generators/base.py
engine/generation/schema/generators/copula.py
engine/generation/schema/generators_legacy.py
engine/generation/schema/llm_text.py
engine/generation/schema/locales/__init__.py
engine/generation/schema/locales/detector.py
engine/generation/schema/locales/packs.py
engine/generation/schema/locales/registry.py
engine/generation/schema/noise.py
engine/generation/schema/pii_columns.py
engine/generation/schema/realism.py
engine/generation/schema/recipes.py
engine/generation/schema/reference_data.py
engine/generation/schema/simulator.py
engine/generation/schema/smart_values.py
engine/generation/schema/templates/__init__.py
engine/generation/schema/templates/library.py
engine/generation/schema/testing.py
engine/generation/schema/timeseries.py
engine/generation/schema/vocab_seeds.py
engine/generation/schema/vocabulary.py
engine/generation/schema/workflows.py
engine/generation/service.py
engine/generation/table_generator.py
engine/generation/text/__init__.py
engine/generation/text/context.py
engine/generation/text/eligibility.py
engine/generation/text/evidence.py
engine/generation/text/faker_fallback.py
engine/generation/text/fallback.py
engine/generation/text/generator.py
engine/generation/text/prompts.py
engine/generation/text/validators.py
engine/generation/text/vocabulary.py
```

## Validation engine

**34 Python files**

```text
engine/validation/__init__.py
engine/validation/database/__init__.py
engine/validation/database/qa_report.py
engine/validation/database/qa_service.py
engine/validation/database/release_manager.py
engine/validation/encoding.py
engine/validation/holdout_evaluator.py
engine/validation/metric_registry.py
engine/validation/metrics.py
engine/validation/privacy.py
engine/validation/schema/__init__.py
engine/validation/schema/audit.py
engine/validation/schema/export_validation.py
engine/validation/schema/format_checks.py
engine/validation/schema/metrics/__init__.py
engine/validation/schema/metrics/business.py
engine/validation/schema/metrics/distributions.py
engine/validation/schema/metrics/privacy.py
engine/validation/schema/metrics/readiness.py
engine/validation/schema/metrics/rules.py
engine/validation/schema/metrics/schema_match.py
engine/validation/schema/metrics/type_adherence.py
engine/validation/schema/performance/__init__.py
engine/validation/schema/performance/memory.py
engine/validation/schema/performance/report.py
engine/validation/schema/performance/timing.py
engine/validation/schema/pm_compare.py
engine/validation/schema/quality.py
engine/validation/schema/reporting.py
engine/validation/schema/throughput.py
engine/validation/schema/validation.py
engine/validation/schema/validation_contract.py
engine/validation/service.py
engine/validation/utility.py
```

## Document engine

**45 Python files**

```text
engine/documents/__init__.py
engine/documents/pdf/__init__.py
engine/documents/pdf/archetypes/__init__.py
engine/documents/pdf/archetypes/bank_statement.py
engine/documents/pdf/archetypes/base.py
engine/documents/pdf/archetypes/registry.py
engine/documents/pdf/base.py
engine/documents/pdf/binding_engine.py
engine/documents/pdf/binding_service.py
engine/documents/pdf/classifier.py
engine/documents/pdf/deidentification_service.py
engine/documents/pdf/docling_engine.py
engine/documents/pdf/duplicates.py
engine/documents/pdf/entities.py
engine/documents/pdf/errors.py
engine/documents/pdf/extraction_router.py
engine/documents/pdf/field_semantics.py
engine/documents/pdf/generation_service.py
engine/documents/pdf/image_redaction_service.py
engine/documents/pdf/language.py
engine/documents/pdf/layout_backends/__init__.py
engine/documents/pdf/layout_backends/base.py
engine/documents/pdf/layout_backends/docling_backend.py
engine/documents/pdf/layout_backends/native_backend.py
engine/documents/pdf/layout_backends/ocr_backend.py
engine/documents/pdf/layout_backends/registry.py
engine/documents/pdf/layout_engine.py
engine/documents/pdf/native_layout.py
engine/documents/pdf/ocr_engine.py
engine/documents/pdf/pdf_adapter.py
engine/documents/pdf/pdf_classifier.py
engine/documents/pdf/pdf_renderer.py
engine/documents/pdf/pii.py
engine/documents/pdf/pii_policy.py
engine/documents/pdf/preflight.py
engine/documents/pdf/render_service.py
engine/documents/pdf/render_validator.py
engine/documents/pdf/service.py
engine/documents/pdf/shape.py
engine/documents/pdf/template_compiler.py
engine/documents/pdf/template_service.py
engine/documents/pdf/text_stats.py
engine/documents/pdf/txt_adapter.py
engine/documents/pdf/validation_service.py
engine/documents/pdf/value_generator.py
```

## Shared engine support

**21 Python files**

```text
engine/common/__init__.py
engine/common/database/__init__.py
engine/common/database/config.py
engine/common/database/core/__init__.py
engine/common/database/core/domain_model.py
engine/common/database/core/fingerprint.py
engine/common/database/core/observability.py
engine/common/database/core/pipeline_runner.py
engine/common/database/core/run_manifest.py
engine/common/database/core/scaling.py
engine/common/database/core/stage_result.py
engine/common/database/core/workflow_orchestrator.py
engine/common/database/privacy/__init__.py
engine/common/database/privacy/context_fields.py
engine/common/database/privacy/deterministic_masker.py
engine/common/database/privacy/dp_primitives.py
engine/common/database/privacy/dp_statistics.py
engine/common/database/privacy/fit_guard.py
engine/common/database/privacy/privacy_accountant.py
engine/common/database/privacy/profile_sanitizer.py
engine/common/database/privacy/rare_category.py
```

## Infrastructure

**59 Python files**

```text
infrastructure/__init__.py
infrastructure/artifacts/__init__.py
infrastructure/artifacts/package_reader.py
infrastructure/artifacts/package_writer.py
infrastructure/artifacts/signing.py
infrastructure/artifacts/store.py
infrastructure/artifacts/tensor_store.py
infrastructure/artifacts/verifier.py
infrastructure/backends/__init__.py
infrastructure/backends/neural/__init__.py
infrastructure/backends/neural/backend.py
infrastructure/backends/neural/models.py
infrastructure/backends/random_forest/__init__.py
infrastructure/backends/random_forest/backend.py
infrastructure/backends/random_forest/models.py
infrastructure/backends/statistical/__init__.py
infrastructure/backends/statistical/backend.py
infrastructure/backends/statistical/models.py
infrastructure/documents/__init__.py
infrastructure/documents/model.py
infrastructure/documents/pdf_loader.py
infrastructure/documents/pdf_router.py
infrastructure/documents/reportlab_renderer.py
infrastructure/documents/tesseract_ocr.py
infrastructure/extraction/__init__.py
infrastructure/extraction/dataframe_pass.py
infrastructure/extraction/document_pass.py
infrastructure/extraction/llm_pass.py
infrastructure/extraction/sqlite_pass.py
infrastructure/jobs/__init__.py
infrastructure/jobs/celery.py
infrastructure/jobs/inline.py
infrastructure/llm/__init__.py
infrastructure/llm/ollama_chat.py
infrastructure/observability/__init__.py
infrastructure/observability/audit.py
infrastructure/observability/metrics.py
infrastructure/observability/tracing.py
infrastructure/persistence/__init__.py
infrastructure/persistence/database.py
infrastructure/persistence/metadata_repository.py
infrastructure/persistence/models.py
infrastructure/sinks/__init__.py
infrastructure/sinks/csv_sink.py
infrastructure/sinks/database_sink.py
infrastructure/sinks/parquet_sink.py
infrastructure/sinks/postgres_sink.py
infrastructure/sinks/registry.py
infrastructure/sinks/staging.py
infrastructure/sources/__init__.py
infrastructure/sources/files.py
infrastructure/sources/frame_connector.py
infrastructure/sources/postgres.py
infrastructure/sources/sqlalchemy_base.py
infrastructure/sources/sqlite.py
infrastructure/storage/__init__.py
infrastructure/storage/local.py
infrastructure/storage/minio.py
infrastructure/storage/s3.py
```

## Interfaces

**34 Python files**

```text
interfaces/__init__.py
interfaces/api/__init__.py
interfaces/api/app.py
interfaces/api/dependencies.py
interfaces/api/error_handlers.py
interfaces/api/routes/__init__.py
interfaces/api/routes/artifacts.py
interfaces/api/routes/connections.py
interfaces/api/routes/generation.py
interfaces/api/routes/training.py
interfaces/api/routes/validation.py
interfaces/cli/__init__.py
interfaces/cli/app.py
interfaces/cli/commands/__init__.py
interfaces/cli/commands/generate.py
interfaces/cli/commands/train.py
interfaces/cli/commands/validate.py
interfaces/sdk/__init__.py
interfaces/sdk/client.py
interfaces/sdk/models.py
interfaces/streamlit/__init__.py
interfaces/streamlit/app.py
interfaces/streamlit/components/common/__init__.py
interfaces/streamlit/components/common/intent_presets.py
interfaces/streamlit/components/common/ux.py
interfaces/streamlit/components/database/__init__.py
interfaces/streamlit/components/pdf/__init__.py
interfaces/streamlit/components/schema/__init__.py
interfaces/streamlit/pages/__init__.py
interfaces/streamlit/pages/database_twin.py
interfaces/streamlit/pages/home.py
interfaces/streamlit/pages/pdf_twin.py
interfaces/streamlit/pages/schema_twin.py
interfaces/streamlit/state.py
```
