"""PDF twin: upload a source PDF -> profile/extract -> compile template ->
bind semantics -> generate synthetic values -> render twin -> validate.

UI-focused: every step below calls straight into the already-tested
src/documents pipeline modules. No business logic lives in this file.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    CompleteSessionCommand,
    GetResultCommand,
    GetSessionCommand,
    SetConfigurationFingerprintCommand,
    StartSessionCommand,
)
from synth_platform.application.dto.workspace import (
    PreviewDescriptor,
    ProgressStatus,
    WorkflowKind,
)
from synth_platform.application.services.result_presentation import (
    stage_workflow_result_view,
)
from synth_platform.application.workflows.pdf_twin import (
    DOCUMENT_PROFILE_FILENAME,
    PDFDocumentAdapter,
    RunManifest,
    is_docling_available,
    load_document_binding_map,
    load_document_deidentification_report,
    load_document_ground_truth,
    load_document_profile,
    load_document_synthetic_values,
    load_document_template,
    load_document_validation_report,
    run_document_deidentification,
    run_document_profiling,
    run_document_rendering,
    run_document_validation,
    run_semantic_binding,
    run_template_compilation,
    run_value_generation,
)
from synth_platform.bootstrap import build_guarded_chat_model
from synth_platform.domain.validation.models import Status
from synth_platform.infrastructure.storage.source_staging import (
    create_source_staging,
    remove_source_staging,
    touch_source_staging,
)
from synth_platform.interfaces.streamlit.components.common.ux import (
    list_pdf_narrative_bindings,
    measure_generation,
    mode_outcomes,
    platform_intro,
    render_data_drift_panel,
    render_progress,
    render_run_metrics,
    show_user_error,
    step_guide,
    step_header,
    validation_badge,
)
from synth_platform.interfaces.streamlit.workspace import (
    ensure_workflow_session,
    fingerprint_configuration,
    get_workspace_tools,
    record_failure,
    record_progress_once,
    render_backend_progress,
    render_project_save,
    render_result_summary,
    workflow_run_dir,
)

st.title("PDF Twin")
platform_intro()
st.caption(
    "Upload a document and create a layout-preserving synthetic twin — "
    "structure and labels kept; sensitive values replaced."
)

defaults = {
    "pdf_workdir": None,
    "pdf_source_path": None,
    "pdf_manifest": None,
    "pdf_doc_id": "doc1",
    "pdf_profile": None,
    "pdf_template": None,
    "pdf_binding_map": None,
    "pdf_values": None,
    "pdf_ground_truth": None,
    "pdf_rendered_path": None,
    "pdf_validation_report": None,
    "pdf_redacted_path": None,
    "pdf_deidentification_report": None,
    "pdf_run_metrics": None,
    "pdf_llm_text": False,
    "pdf_intent": "Document testing",
    "pdf_extraction_engine": "Automatic (native/OCR)",
    "pdf_source_hash": None,
    "pdf_workspace_session_id": None,
    "pdf_configuration_hash": None,
    "pdf_result_view_id": None,
    "pdf_source_staging_dir": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def _progress_steps() -> list[tuple[str, bool]]:
    return [
        ("Intent", True),
        ("Upload", st.session_state.pdf_source_path is not None),
        ("Understand", st.session_state.pdf_binding_map is not None),
        ("Generate", st.session_state.pdf_values is not None),
        ("Validate", st.session_state.pdf_validation_report is not None),
        ("Download", st.session_state.pdf_rendered_path is not None),
    ]


render_progress(
    _progress_steps(),
    current_hint="Upload → Understand → Generate → Validate → Download",
)

# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(1, "Intent", True)
    step_guide(
        what="How you plan to use the synthetic document (informational).",
        next_step="Upload a PDF.",
    )
    intent_options = [
        "Document testing",
        "QA / automation",
        "Demonstrations",
        "Privacy-safe sharing",
    ]
    st.session_state.pdf_intent = st.selectbox(
        "What will you use the synthetic PDF for?",
        options=intent_options,
        index=intent_options.index(st.session_state.pdf_intent)
        if st.session_state.pdf_intent in intent_options
        else 0,
    )

# ---------------------------------------------------------------------------
# Step 1 - Upload
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(2, "Upload", st.session_state.pdf_source_path is not None)
    step_guide(what="Provide the source PDF to twin.", next_step="Discover document structure.")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        source_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        if source_hash != st.session_state.pdf_source_hash:
            for key in (
                "pdf_manifest",
                "pdf_profile",
                "pdf_template",
                "pdf_binding_map",
                "pdf_values",
                "pdf_ground_truth",
                "pdf_rendered_path",
                "pdf_validation_report",
                "pdf_redacted_path",
                "pdf_deidentification_report",
                "pdf_run_metrics",
                "pdf_result_view_id",
                "pdf_configuration_hash",
            ):
                st.session_state[key] = defaults[key]
            remove_source_staging(st.session_state.pdf_source_staging_dir)
            staging_dir = create_source_staging("document")
            st.session_state.pdf_source_staging_dir = staging_dir
            st.session_state.pdf_source_hash = source_hash
            pdf_session = ensure_workflow_session(
                state_key="pdf_workspace_session_id",
                workflow=WorkflowKind.DOCUMENT,
                title="Document Twin",
                source_fingerprint=source_hash,
                current_step="upload",
            )
            workdir = workflow_run_dir(pdf_session.session_id, WorkflowKind.DOCUMENT)
            path = staging_dir / "source.pdf"
            path.write_bytes(uploaded_bytes)
            st.session_state.pdf_workdir = workdir
            st.session_state.pdf_source_path = path
            record_progress_once(
                pdf_session.session_id,
                macro_step="upload",
                stage_id="document_upload",
                stage_label="Upload document",
                status=ProgressStatus.SUCCEEDED,
                message="Document source fingerprint recorded.",
            )

if st.session_state.pdf_source_path is None:
    st.stop()

touch_source_staging(st.session_state.pdf_source_staging_dir)
workspace_tools = get_workspace_tools()
pdf_session = workspace_tools.get_session(
    GetSessionCommand(session_id=st.session_state.pdf_workspace_session_id or "missing")
)
if pdf_session is None:
    pdf_session = ensure_workflow_session(
        state_key="pdf_workspace_session_id",
        workflow=WorkflowKind.DOCUMENT,
        title="Document Twin",
        source_fingerprint=st.session_state.pdf_source_hash,
        current_step="upload",
    )
    st.session_state.pdf_workdir = workflow_run_dir(
        pdf_session.session_id, WorkflowKind.DOCUMENT
    )

manifest = st.session_state.pdf_manifest
if manifest is None:
    manifest = RunManifest.create(runs_dir=st.session_state.pdf_workdir / "runs")
    st.session_state.pdf_manifest = manifest
doc_id = st.session_state.pdf_doc_id
render_backend_progress(pdf_session.session_id)

# ---------------------------------------------------------------------------
# Step 2 - Preflight / extraction / OCR + profiling
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(3, "Discover document", st.session_state.pdf_profile is not None)
    step_guide(
        what="Process the PDF so the platform can understand its structure.",
        next_step="Build the twin template.",
    )
    extraction_options = ["Automatic (native/OCR)", "Docling"]
    if st.session_state.pdf_extraction_engine not in extraction_options:
        st.session_state.pdf_extraction_engine = extraction_options[0]
    st.session_state.pdf_extraction_engine = st.selectbox(
        "Extraction engine",
        extraction_options,
        index=extraction_options.index(st.session_state.pdf_extraction_engine),
        disabled=st.session_state.pdf_profile is not None,
        help=(
            "Docling performs document-layout and table-aware extraction and may initialize model artifacts on first use. "
            "Automatic preserves the native-text path and falls back to OCR for scanned PDFs when OCR tooling is installed."
        ),
    )
    if st.session_state.pdf_extraction_engine == "Docling" and not is_docling_available():
        st.warning(
            "Docling is not installed in this environment. Install the PDF extras with "
            "`pip install -e \".[pdf]\"` or choose Automatic."
        )

    if st.session_state.pdf_profile is None:
        if st.button("Discover document", icon=":material/document_scanner:"):
            workspace_tools.start_session(
                StartSessionCommand(
                    session_id=pdf_session.session_id,
                    run_id=manifest.run_id,
                )
            )
            record_progress_once(
                pdf_session.session_id,
                macro_step="configure",
                stage_id="document_profiling",
                stage_label="Discover document",
                status=ProgressStatus.RUNNING,
            )
            preferred_method = (
                "docling" if st.session_state.pdf_extraction_engine == "Docling" else None
            )
            with st.spinner("Processing document..."):
                result = run_document_profiling(
                    PDFDocumentAdapter(), st.session_state.pdf_source_path, doc_id, manifest,
                    st.session_state.pdf_workdir / "metadata",
                    preferred_extraction_method=preferred_method,
                )
            if not result.is_success():
                record_failure(
                    pdf_session.session_id,
                    macro_step="configure",
                    stage_id="document_profiling",
                    stage_label="Discover document",
                )
                show_user_error(
                    "We couldn't discover the document structure.",
                    technical=result.errors,
                    next_action="Try another PDF, or check that the file is not corrupted.",
                )
                st.stop()
            st.session_state.pdf_profile = load_document_profile(
                manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}")
            )
            record_progress_once(
                pdf_session.session_id,
                macro_step="configure",
                stage_id="document_profiling",
                stage_label="Discover document",
                status=ProgressStatus.SUCCEEDED,
                counts={"privacy_findings": len(st.session_state.pdf_profile["pii_findings"])},
            )
            st.rerun()
    else:
        profile = st.session_state.pdf_profile
        c1, c2, c3 = st.columns(3)
        c1.metric("Extraction method", profile["extraction_method"])
        c2.metric("Language", profile["language"].get("language") or "unknown")
        c3.metric("Privacy findings", len(profile["pii_findings"]))
        st.caption(
            f"Document type: {profile['classification']['document_type']} "
            f"(status: {profile['classification']['status']})"
        )
        with st.expander("Technical details — extraction", expanded=False):
            st.caption(
                f"Method `{profile['extraction_method']}` produced the profiling text and "
                "is reused by template compilation for layout extraction."
            )

if st.session_state.pdf_profile is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 3 - Compile template
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(4, "Build twin template", st.session_state.pdf_template is not None)
    step_guide(
        what="Capture structure and labels for the twin (sensitive values stay redacted).",
        next_step="Understand fields, then generate values.",
    )
    if st.session_state.pdf_template is None:
        if st.button("Build twin template", icon=":material/view_quilt:"):
            result = run_template_compilation(
                st.session_state.pdf_source_path, doc_id, manifest,
                extraction_method=st.session_state.pdf_profile["extraction_method"],
            )
            if not result.is_success():
                record_failure(
                    pdf_session.session_id,
                    macro_step="configure",
                    stage_id="template_compilation",
                    stage_label="Build twin template",
                )
                show_user_error(
                    "We couldn't build the twin template.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.pdf_template = load_document_template(
                manifest.output_path(f"documents/{doc_id}/document_template.json")
            )
            record_progress_once(
                pdf_session.session_id,
                macro_step="configure",
                stage_id="template_compilation",
                stage_label="Build twin template",
                status=ProgressStatus.SUCCEEDED,
            )
            st.rerun()
    else:
        template = st.session_state.pdf_template
        region_types = [r["region_type"] for page in template["pages"] for r in page["regions"]]
        from collections import Counter

        st.write({k: v for k, v in Counter(region_types).items()})
        st.caption("Structure and labels only — sensitive values are redacted at this stage.")
        with st.expander("Template schema details", expanded=False):
            detail_rows = []
            for page in template.get("pages") or []:
                for region in page.get("regions") or []:
                    detail_rows.append(
                        {
                            "page": page.get("page_number"),
                            "region": region.get("region_id"),
                            "type": region.get("region_type"),
                            "label": region.get("label") or "—",
                            "value type": region.get("value_type") or "—",
                        }
                    )
            if detail_rows:
                st.dataframe(detail_rows, hide_index=True, width="stretch")
            st.caption(
                "Template stores regions, labels, and value shapes — not raw source field values."
            )

if st.session_state.pdf_template is None:
    st.stop()

# ---------------------------------------------------------------------------
# Mode 2 - De-identify (redacted PDF, no synthetic generation) - an
# independent sibling of the Mode 3 chain below (steps 4-6): both start
# from the same document_template.json, but this mode never fabricates a
# replacement value, only renders the template's own masked previews.
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader(":material/visibility_off: Optional — redacted PDF", anchor=False)
    step_guide(what="Download a redacted copy without generating a synthetic twin.")
    if st.session_state.pdf_redacted_path is None:
        if st.button("De-identify", icon=":material/visibility_off:"):
            result = run_document_deidentification(
                st.session_state.pdf_template, doc_id, manifest,
                template_reference=str(st.session_state.pdf_source_path),
            )
            if not result.is_success():
                record_failure(
                    pdf_session.session_id,
                    macro_step="configure",
                    stage_id="document_deidentification",
                    stage_label="De-identify document",
                )
                show_user_error(
                    "De-identification failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.pdf_redacted_path = manifest.output_path(f"documents/{doc_id}/redacted.pdf")
            st.session_state.pdf_deidentification_report = load_document_deidentification_report(
                manifest.output_path(f"documents/{doc_id}/document_deidentification_report.json")
            )
            st.rerun()
    else:
        with open(st.session_state.pdf_redacted_path, "rb") as f:
            redacted_bytes = f.read()
        st.download_button(
            "Download redacted PDF", redacted_bytes, file_name="redacted.pdf", icon=":material/download:"
        )

# ---------------------------------------------------------------------------
# Step 4 - Semantic binding
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(5, "Understand fields", st.session_state.pdf_binding_map is not None)
    step_guide(
        what="Map labels to value roles so generation stays realistic and consistent.",
        next_step="Generate synthetic values.",
    )
    if st.session_state.pdf_binding_map is None:
        if st.button("Understand fields", icon=":material/hub:"):
            result = run_semantic_binding(
                st.session_state.pdf_template, doc_id, manifest,
                template_reference=str(st.session_state.pdf_source_path),
            )
            if not result.is_success():
                record_failure(
                    pdf_session.session_id,
                    macro_step="configure",
                    stage_id="semantic_binding",
                    stage_label="Understand fields",
                )
                show_user_error(
                    "We couldn't understand document fields.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.pdf_binding_map = load_document_binding_map(
                manifest.output_path(f"documents/{doc_id}/document_binding_map.json")
            )
            record_progress_once(
                pdf_session.session_id,
                macro_step="configure",
                stage_id="semantic_binding",
                stage_label="Understand fields",
                status=ProgressStatus.SUCCEEDED,
                counts={"bindings": len(st.session_state.pdf_binding_map.get("bindings") or [])},
            )
            st.rerun()
    else:
        field_rows = [
            {"label": b["label"], "role": b["semantic_role"]}
            for b in st.session_state.pdf_binding_map["bindings"]
            if b["binding_type"] == "field"
        ]
        if field_rows:
            st.dataframe(field_rows, hide_index=True, width="stretch")
        with st.expander("Technical details — field bindings", expanded=False):
            st.json(
                [
                    {
                        "label": b["label"],
                        "role": b["semantic_role"],
                        "strategy": b["generator_strategy"],
                    }
                    for b in st.session_state.pdf_binding_map["bindings"]
                    if b["binding_type"] == "field"
                ]
            )

if st.session_state.pdf_binding_map is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 5 - Generate synthetic values
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(6, "Generate synthetic values", st.session_state.pdf_values is not None)
    step_guide(
        what="Replace field values with privacy-safe synthetic content matched to each role.",
        next_step="Build and download the twin PDF.",
    )
    seed = st.number_input("Seed", value=42, key="pdf_seed")
    narrative_fields = list_pdf_narrative_bindings(st.session_state.pdf_binding_map)
    st.caption(
        "Names, addresses, dates, and notes use role-matched generators. "
        "Raw source values are not used."
    )
    if narrative_fields:
        st.dataframe(narrative_fields, hide_index=True, width="stretch")
        st.session_state.pdf_llm_text = st.toggle(
            "Use local LLM for narrative fields only",
            value=bool(st.session_state.pdf_llm_text),
            help=(
                "Uses guarded loopback Ollama and falls back to deterministic "
                "synthetic text if the model is unavailable or rejected."
            ),
        )
    else:
        st.session_state.pdf_llm_text = False
        st.caption("No eligible narrative fields were found in this document.")
    pdf_configuration_hash = fingerprint_configuration(
        {
            "extraction_engine": st.session_state.pdf_extraction_engine,
            "seed": int(seed),
            "llm_text_enabled": bool(st.session_state.pdf_llm_text),
        }
    )
    previous_configuration_hash = st.session_state.pdf_configuration_hash
    if previous_configuration_hash is None:
        workspace_tools.set_configuration_fingerprint(
            SetConfigurationFingerprintCommand(
                session_id=pdf_session.session_id,
                configuration_fingerprint=pdf_configuration_hash,
            )
        )
    elif previous_configuration_hash != pdf_configuration_hash:
        for key in (
            "pdf_values",
            "pdf_ground_truth",
            "pdf_rendered_path",
            "pdf_validation_report",
            "pdf_run_metrics",
            "pdf_result_view_id",
        ):
            st.session_state[key] = defaults[key]
        pdf_session = ensure_workflow_session(
            state_key="pdf_workspace_session_id",
            workflow=WorkflowKind.DOCUMENT,
            title="Document Twin",
            source_fingerprint=st.session_state.pdf_source_hash,
            configuration_fingerprint=pdf_configuration_hash,
            current_step="configure",
        )
        st.session_state.pdf_workdir = workflow_run_dir(
            pdf_session.session_id, WorkflowKind.DOCUMENT
        )
        manifest = RunManifest.create(
            runs_dir=st.session_state.pdf_workdir / "runs"
        )
        st.session_state.pdf_manifest = manifest
        for stage_id, stage_label in (
            ("document_upload", "Upload document"),
            ("document_profiling", "Discover document"),
            ("template_compilation", "Build twin template"),
            ("semantic_binding", "Understand fields"),
        ):
            record_progress_once(
                pdf_session.session_id,
                macro_step="configure",
                stage_id=stage_id,
                stage_label=stage_label,
                status=ProgressStatus.SUCCEEDED,
                message="Reused reviewed in-memory configuration input.",
            )
    st.session_state.pdf_configuration_hash = pdf_configuration_hash
    record_progress_once(
        pdf_session.session_id,
        macro_step="configure",
        stage_id="document_configuration",
        stage_label="Configure generation",
        status=ProgressStatus.SUCCEEDED,
    )
    if st.button("Generate values", icon=":material/casino:"):
        record_progress_once(
            pdf_session.session_id,
            macro_step="generate",
            stage_id="value_generation",
            stage_label="Generate synthetic values",
            status=ProgressStatus.RUNNING,
        )
        text_model = (
            build_guarded_chat_model("synthetic_text")
            if st.session_state.pdf_llm_text
            else None
        )
        with measure_generation() as run_stats:
            result = run_value_generation(
                st.session_state.pdf_template, st.session_state.pdf_binding_map, doc_id, manifest,
                binding_map_reference=str(st.session_state.pdf_source_path), seed=int(seed),
                llm_text_enabled=bool(st.session_state.pdf_llm_text),
                text_model=text_model,
            )
        if not result.is_success():
            record_failure(
                pdf_session.session_id,
                macro_step="generate",
                stage_id="value_generation",
                stage_label="Generate synthetic values",
            )
            show_user_error(
                "Synthetic value generation failed.",
                technical=result.errors,
            )
            st.stop()
        st.session_state.pdf_run_metrics = dict(run_stats)
        st.session_state.pdf_values = load_document_synthetic_values(
            manifest.output_path(f"documents/{doc_id}/document_synthetic_values.json")
        )
        st.session_state.pdf_rendered_path = None
        generated_values = st.session_state.pdf_values
        record_progress_once(
            pdf_session.session_id,
            macro_step="generate",
            stage_id="value_generation",
            stage_label="Generate synthetic values",
            status=ProgressStatus.SUCCEEDED,
            counts={
                "fields": len(generated_values.get("fields") or {}),
                "tables": len(generated_values.get("tables") or {}),
            },
        )
        st.rerun()

    if st.session_state.pdf_values is not None:
        values = st.session_state.pdf_values
        fields = values.get("fields") or {}
        tables = values.get("tables") or {}
        inline_spans = values.get("inline_spans") or {}
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Fields", len(fields))
        g2.metric("Tables", len(tables))
        g3.metric("Inline spans", len(inline_spans))
        g4.metric("Seed", int(seed))
        metrics = st.session_state.pdf_run_metrics or {}
        render_run_metrics(
            seconds=metrics.get("seconds"),
            peak_memory_mb=metrics.get("peak_memory_mb"),
            rows_per_second=None,
            extra={"scope": "PDF value generation"},
        )
        for region_id, field in fields.items():
            value = field.get("value", field.get("formatted"))
            st.caption(f"{field.get('label') or region_id}: **{value}**")

if st.session_state.pdf_values is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 6 - Render + validate
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(7, "Build twin document and validate", st.session_state.pdf_validation_report is not None)
    step_guide(
        what="Render the synthetic PDF, then check field match and layout fit.",
        next_step="Download the twin when validation looks good.",
    )
    if st.session_state.pdf_rendered_path is None:
        if st.button("Build twin document", icon=":material/picture_as_pdf:", type="primary"):
            result = run_document_rendering(
                st.session_state.pdf_template, st.session_state.pdf_binding_map, st.session_state.pdf_values,
                doc_id, manifest, synthetic_values_reference=str(st.session_state.pdf_source_path),
            )
            if not result.is_success():
                record_failure(
                    pdf_session.session_id,
                    macro_step="generate",
                    stage_id="document_rendering",
                    stage_label="Render twin document",
                )
                show_user_error(
                    "Building the twin document failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.pdf_rendered_path = manifest.output_path(f"documents/{doc_id}/rendered.pdf")
            st.session_state.pdf_ground_truth = load_document_ground_truth(
                manifest.output_path(f"documents/{doc_id}/document_ground_truth.json")
            )
            record_progress_once(
                pdf_session.session_id,
                macro_step="generate",
                stage_id="document_rendering",
                stage_label="Render twin document",
                status=ProgressStatus.SUCCEEDED,
            )
            st.rerun()
    else:
        with open(st.session_state.pdf_rendered_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("Download synthetic twin PDF", pdf_bytes, file_name="synthetic_twin.pdf", icon=":material/download:")

        if st.session_state.pdf_validation_report is None:
            if st.button("Validate twin", icon=":material/fact_check:"):
                result = run_document_validation(
                    st.session_state.pdf_rendered_path, st.session_state.pdf_ground_truth, doc_id, manifest,
                    ground_truth_reference=str(st.session_state.pdf_source_path),
                )
                if not result.is_success():
                    record_failure(
                        pdf_session.session_id,
                        macro_step="results",
                        stage_id="document_validation",
                        stage_label="Validate twin document",
                    )
                    show_user_error(
                        "Twin validation failed.",
                        technical=result.errors,
                    )
                    st.stop()
                st.session_state.pdf_validation_report = load_document_validation_report(
                    manifest.output_path(f"documents/{doc_id}/document_validation_report.json")
                )
                report = st.session_state.pdf_validation_report["report"]
                validation_status = Status.PASS if report["hard_checks_passed"] else Status.FAIL
                record_progress_once(
                    pdf_session.session_id,
                    macro_step="results",
                    stage_id="document_validation",
                    stage_label="Validate twin document",
                    status=(
                        ProgressStatus.SUCCEEDED
                        if report["hard_checks_passed"]
                        else ProgressStatus.BLOCKED
                    ),
                )
                artifact_paths = [
                    ("Synthetic twin PDF", st.session_state.pdf_rendered_path, True),
                    (
                        "Document validation report",
                        manifest.output_path(f"documents/{doc_id}/document_validation_report.json"),
                        True,
                    ),
                    (
                        "Generated ground truth",
                        manifest.output_path(f"documents/{doc_id}/document_ground_truth.json"),
                        True,
                    ),
                ]
                if st.session_state.pdf_redacted_path is not None:
                    artifact_paths.append(
                        ("Optional redacted PDF", st.session_state.pdf_redacted_path, True)
                    )
                result_view = stage_workflow_result_view(
                    workspace_tools,
                    pdf_session.session_id,
                    workflow=WorkflowKind.DOCUMENT,
                    manifest=manifest,
                    validation_status=validation_status,
                    release_verdict=None,
                    blockers=([] if report["hard_checks_passed"] else ["Document hard checks failed"]),
                    metrics={
                        "field_accuracy": report.get("field_accuracy"),
                        "matched_fields": report.get("matched_fields"),
                        "total_fields": report.get("total_fields"),
                        "overflow_failure_count": report.get("overflow_failure_count"),
                    },
                    previews=[
                        PreviewDescriptor(
                            label="Synthetic twin PDF",
                            kind="document",
                            artifact_id=None,
                            description="Layout-preserving rendered document.",
                        )
                    ],
                    artifacts=artifact_paths,
                )
                workspace_tools.complete_session(
                    CompleteSessionCommand(result=result_view)
                )
                st.session_state.pdf_result_view_id = result_view.result_id
                remove_source_staging(st.session_state.pdf_source_staging_dir)
                st.session_state.pdf_source_staging_dir = None
                st.rerun()
        else:
            report = st.session_state.pdf_validation_report["report"]
            if report["hard_checks_passed"]:
                validation_badge(True, passed_label="Validation passed")
            else:
                validation_badge(False)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Integrity checks", "passed" if report["hard_checks_passed"] else "failed")
            c2.metric("Field accuracy", f"{report['field_accuracy']:.0%}")
            matched = report.get("matched_fields")
            total = report.get("total_fields")
            if matched is not None and total is not None:
                c3.metric("Fields matched", f"{matched}/{total}")
            else:
                c3.metric("Fields matched", "n/a")
            c4.metric("Overflow failures", report["overflow_failure_count"])
            render_data_drift_panel(
                mode="pdf",
                rows=[
                    {
                        "signal": "Field accuracy",
                        "value": f"{report['field_accuracy']:.0%}",
                        "meaning": "How well rendered twin fields match generated ground truth",
                    },
                    {
                        "signal": "Overflow failures",
                        "value": report["overflow_failure_count"],
                        "meaning": "Layout cells that could not fit their synthetic value",
                    },
                ],
            )
            mode_outcomes(
                [
                    "Synthetic twin PDF",
                    "Validation report (field accuracy / overflow)",
                    "Optional redacted PDF if you ran de-identify",
                ]
            )
            st.caption(
                f"Intent: {st.session_state.pdf_intent}. "
                "Layout-preserving synthetic document with privacy-aware values."
            )


result_view_id = st.session_state.pdf_result_view_id
if result_view_id:
    persisted_result = workspace_tools.get_result(
        GetResultCommand(
            session_id=pdf_session.session_id,
            result_id=result_view_id,
        )
    )
    if persisted_result is not None:
        render_result_summary(persisted_result)
        render_project_save(pdf_session.session_id)
