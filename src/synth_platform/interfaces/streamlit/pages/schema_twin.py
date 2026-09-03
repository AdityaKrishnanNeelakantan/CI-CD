"""Schema Mode Streamlit page."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from synth_platform.application.workflows.schema_twin import (
    generate_from_schema,
    load_schema_bytes,
    package_download,
    schema_column_details,
    summarize_schema,
    validation_highlights,
)
from synth_platform.application.workflows.schema_prompt_draft import (
    draft_schema_from_prompt as draft_schema_from_prompt_workflow,
)
from synth_platform.interfaces.streamlit.components.common.ux import (
    measure_generation,
    platform_intro,
    show_user_error,
    step_guide,
    step_header,
)
from synth_platform.interfaces.streamlit.ui_config import ui_step, ui_value
from synth_platform.infrastructure.slm_runtime import (
    apply_local_endpoint_network_policy,
    data_designer_max_tokens,
    data_designer_skip_health_check,
    is_local_endpoint,
    normalize_model_endpoint,
    PlatformSLMRuntime,
    preflight_slm_endpoint,
    resolve_platform_slm_runtime,
)
from synth_platform.infrastructure.integrations.data_designer_provider import (
    build_data_designer_model_config,
    build_data_designer_provider,
)
from synth_platform.settings import Settings

SETTINGS = Settings.from_env()
SCHEMA_UI = ui_value("schema", default={})
PLATFORM_SLM = resolve_platform_slm_runtime()

PROVIDER_DEFAULTS = {
    "Configured AI provider": {
        "provider": PLATFORM_SLM.provider,
        "endpoint": PLATFORM_SLM.endpoint,
        "model": PLATFORM_SLM.model_id,
        "api_key_env": PLATFORM_SLM.api_key_env,
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": 768,
        "extra_body": None,
    }
}


def _init_state() -> None:
    defaults = {
        "schema_intent": SCHEMA_UI.get("default_intent", "Development & testing"),
        "schema_config": None,
        "schema_summary": None,
        "schema_file_id": None,
        "schema_ai_prompt": "",
        "schema_ai_ddl": "",
        "schema_ai_draft_cache_hit": False,
        "schema_ai_draft_repairs": [],
        "schema_rules": "",
        "schema_rule_rows": [],
        "schema_seed_df": None,
        "schema_result": None,
        "schema_zip_bytes": None,
        "schema_run_metrics": None,
        "schema_workdir": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.schema_workdir is None:
        st.session_state.schema_workdir = Path(tempfile.mkdtemp(prefix="schema_data_designer_"))


def _reset_result() -> None:
    st.session_state.schema_result = None
    st.session_state.schema_zip_bytes = None
    st.session_state.schema_run_metrics = None


def _active_rules() -> list[str]:
    rules = []
    global_rules = str(st.session_state.get("schema_rules") or "").strip()
    if global_rules:
        rules.append(global_rules)
    for row in st.session_state.get("schema_rule_rows") or []:
        rule = str(row.get("rule") or "").strip()
        if rule:
            rules.append(f"{row.get('scope')}: {rule}")
    return rules


def _import_data_designer():
    if importlib.util.find_spec("data_designer.config") is None:
        raise RuntimeError("The schema generation dependency is not installed.")
    try:
        dd = importlib.import_module("data_designer.config")
        interface = importlib.import_module("data_designer.interface")
        return dd, interface.DataDesigner
    except ImportError as exc:
        raise RuntimeError(
            "The generation package is installed but could not be imported. "
            "Install this project with the `schema` extra and restart the app."
        ) from exc


def _is_local_endpoint(endpoint: str) -> bool:
    return is_local_endpoint(endpoint)


def _api_key(provider_label: str, api_key_env: str, api_key_value: str | None = None) -> str | None:
    fallback_env = PROVIDER_DEFAULTS[provider_label]["api_key_env"]
    target_env = api_key_env.strip() or fallback_env
    explicit = str(api_key_value or "").strip()
    if explicit:
        os.environ[target_env] = explicit
        return explicit
    configured = os.getenv(target_env) or os.getenv(fallback_env)
    resolved = configured
    if resolved and not os.getenv(target_env):
        os.environ[target_env] = resolved
    return resolved


def _requires_api_key(endpoint: str) -> bool:
    return not is_local_endpoint(endpoint)


def _preflight_schema_slm(provider: str, endpoint: str, model: str, api_key_env: str) -> None:
    runtime = PlatformSLMRuntime(
        model_id=model,
        provider=provider,
        endpoint=normalize_model_endpoint(endpoint),
        api_key_env=api_key_env,
        model_family=PLATFORM_SLM.model_family,
    )
    preflight_slm_endpoint(runtime)


def _provider_extra_body(provider_label: str, endpoint: str) -> Mapping[str, Any] | None:
    if is_local_endpoint(endpoint):
        return None
    return PROVIDER_DEFAULTS[provider_label].get("extra_body")


def _apply_health_check_policy(skip_health_check: bool) -> None:
    if skip_health_check or data_designer_skip_health_check():
        os.environ["DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS"] = "1"


def _env_skip_health_check() -> bool:
    return data_designer_skip_health_check()


def _model_provider(dd: Any, provider: str, endpoint: str, api_key_env: str) -> Any:
    apply_local_endpoint_network_policy(endpoint)
    runtime = PlatformSLMRuntime(
        model_id=PLATFORM_SLM.model_id,
        provider=provider,
        endpoint=normalize_model_endpoint(endpoint),
        api_key_env=api_key_env,
        model_family=PLATFORM_SLM.model_family,
    )
    return build_data_designer_provider(dd, runtime)


def _model_config(
    dd: Any,
    *,
    alias: str,
    provider: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Mapping[str, Any] | None,
    skip_health_check: bool,
    workflow: str,
) -> Any:
    runtime = PlatformSLMRuntime(
        model_id=model,
        provider=provider,
        endpoint=PLATFORM_SLM.endpoint,
        api_key_env=PLATFORM_SLM.api_key_env,
        model_family=PLATFORM_SLM.model_family,
    )
    return build_data_designer_model_config(
        dd,
        alias=alias,
        workflow=workflow,
        runtime=runtime,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        extra_body=dict(extra_body) if extra_body else None,
        skip_health_check=bool(skip_health_check or data_designer_skip_health_check()),
    )


def _hidden_provider_config(prefix: str, *, fallback_prefix: str | None = None) -> tuple[str, str, str, str, str, str, float, float, int, bool]:
    source = fallback_prefix or prefix
    provider_label = st.session_state.get(f"{source}_provider_label", "Configured AI provider")
    if provider_label not in PROVIDER_DEFAULTS:
        provider_label = "Configured AI provider"
    defaults = PROVIDER_DEFAULTS[provider_label]
    return (
        provider_label,
        str(st.session_state.get(f"{source}_provider", defaults["provider"])),
        normalize_model_endpoint(str(st.session_state.get(f"{source}_endpoint", defaults["endpoint"]))),
        str(st.session_state.get(f"{source}_model", defaults["model"])),
        str(st.session_state.get(f"{source}_api_key_env", defaults["api_key_env"])),
        str(st.session_state.get(f"{source}_api_key_value", "")),
        float(st.session_state.get(f"{source}_temperature", defaults["temperature"])),
        float(st.session_state.get(f"{source}_top_p", defaults["top_p"])),
        int(st.session_state.get(f"{source}_max_tokens", defaults["max_tokens"])),
        bool(st.session_state.get(f"{source}_skip_health_check", _env_skip_health_check())),
    )


def _preview_to_frame(preview: Any) -> pd.DataFrame:
    dataset = getattr(preview, "dataset", preview)
    if isinstance(dataset, pd.DataFrame):
        return dataset
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    if hasattr(dataset, "to_dataframe"):
        return dataset.to_dataframe()
    if isinstance(dataset, list):
        return pd.DataFrame(dataset)
    raise RuntimeError("Preview returned an unsupported dataset shape.")


def _explain_empty_dataset_failure(exc: Exception, *, provider: str, model: str) -> RuntimeError:
    message = str(exc)
    if "Dataset is empty" in message or "kind=connection" in message or "Connection to model" in message:
        return RuntimeError(
            "The preview engine ran, but no successful rows were returned. "
            f"The configured model `{model}` reported a column-generation failure. "
            f"Raw warning/error: {message}"
        )
    return RuntimeError(message)


def _schema_draft_output_format() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["name", "tables"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "domain": {"type": "string"},
            "tables": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "columns"],
                    "properties": {
                        "name": {"type": "string"},
                        "row_count": {"type": "integer", "minimum": 1},
                        "description": {"type": "string"},
                        "columns": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["name", "type"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "int",
                                            "float",
                                            "date",
                                            "time",
                                            "datetime",
                                            "categorical",
                                            "foreign_key",
                                            "text",
                                            "boolean",
                                            "uuid",
                                            "email",
                                            "phone",
                                            "url",
                                            "address",
                                            "currency",
                                            "money",
                                            "decimal",
                                            "bank_account",
                                            "account_number",
                                            "routing_number",
                                            "ssn",
                                            "aadhaar",
                                            "iban",
                                            "swift_bic",
                                            "ifsc_code",
                                            "credit_card",
                                        ],
                                    },
                                    "unique": {"type": "boolean"},
                                    "nullable": {"type": "boolean"},
                                    "description": {"type": "string"},
                                    "params": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "required": ["parent_table", "parent_key", "child_table", "child_key"],
                            "properties": {
                                "parent_table": {"type": "string"},
                                "parent_key": {"type": "string"},
                                "child_table": {"type": "string"},
                                "child_key": {"type": "string"},
                            },
                        },
                    ]
                },
            },
        },
    }


def _coerce_structured_schema_value(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, Mapping):
            return payload
    raise RuntimeError(f"Schema draft returned an unsupported structured value: {value!r}")


def _draft_schema_json(
    *,
    prompt: str,
    provider_label: str,
    provider: str,
    endpoint: str,
    model: str,
    api_key_env: str,
    api_key_value: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    skip_health_check: bool,
) -> str:
    if _requires_api_key(endpoint) and not _api_key(provider_label, api_key_env, api_key_value):
        raise RuntimeError(f"Set `{api_key_env}` before asking AI to draft a schema.")
    previous_env = {
        "SP_PLATFORM_SLM_PROVIDER": os.environ.get("SP_PLATFORM_SLM_PROVIDER"),
        "SP_PLATFORM_SLM_ENDPOINT": os.environ.get("SP_PLATFORM_SLM_ENDPOINT"),
        "SP_PLATFORM_SLM_MODEL": os.environ.get("SP_PLATFORM_SLM_MODEL"),
        "SP_PLATFORM_SLM_API_KEY_ENV": os.environ.get("SP_PLATFORM_SLM_API_KEY_ENV"),
        "SP_SCHEMA_DRAFT_MAX_TOKENS": os.environ.get("SP_SCHEMA_DRAFT_MAX_TOKENS"),
        "DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS": os.environ.get("DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS"),
        api_key_env: os.environ.get(api_key_env),
    }
    try:
        os.environ["SP_PLATFORM_SLM_PROVIDER"] = provider
        os.environ["SP_PLATFORM_SLM_ENDPOINT"] = normalize_model_endpoint(endpoint)
        os.environ["SP_PLATFORM_SLM_MODEL"] = model
        os.environ["SP_PLATFORM_SLM_API_KEY_ENV"] = api_key_env
        os.environ["SP_SCHEMA_DRAFT_MAX_TOKENS"] = str(min(int(max_tokens), 768))
        if api_key_value:
            os.environ[api_key_env] = api_key_value
        if skip_health_check:
            os.environ["DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS"] = "1"

        draft = draft_schema_from_prompt_workflow(prompt)
        st.session_state.schema_ai_draft_cache_hit = draft.cache_hit
        st.session_state.schema_ai_draft_repairs = draft.repairs
        return draft.schema_json
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _column_prompt(*, table_name: str, column: Any, rules: list[str]) -> str:
    nullable = "nullable" if bool(getattr(column, "nullable", False)) else "required"
    unique = "unique" if bool(getattr(column, "unique", False)) else "not necessarily unique"
    return (
        f"Generate a realistic value for `{table_name}.{column.name}`.\n"
        f"Column type: {getattr(column, 'type', 'text')}. Constraint: {nullable}, {unique}.\n"
        f"Column description: {getattr(column, 'description', None) or 'none'}.\n"
        "User-defined rules:\n"
        + ("\n".join(f"- {rule}" for rule in rules) if rules else "- Follow the schema.")
        + "\nReturn only the generated value."
    )


def _category_values(column: Any) -> list[str]:
    params = dict(getattr(column, "distribution_params", {}) or {})
    values = params.get("choices") or params.get("values")
    if values:
        return [str(value) for value in values]
    name = column.name.lower()
    if "status" in name:
        return ["pending", "active", "completed", "cancelled"]
    if "type" in name:
        return ["standard", "premium", "enterprise"]
    return ["alpha", "beta", "gamma"]


def _category_weights(column: Any) -> list[float] | None:
    params = dict(getattr(column, "distribution_params", {}) or {})
    values = _category_values(column)
    weights = params.get("weights") or params.get("probabilities")
    if not isinstance(weights, list) or len(weights) != len(values):
        return None
    total = sum(float(weight) for weight in weights)
    if total <= 0:
        return None
    return [float(weight) / total for weight in weights]


def _datetime_convert_format(column_name: str, column_type: str) -> str:
    normalized_name = column_name.lower()
    normalized_type = column_type.lower()
    if normalized_type == "date" or "date" in normalized_name:
        return "%Y-%m-%d"
    if normalized_type == "time" or normalized_name.endswith("_time"):
        return "%H:%M:%S"
    return "%Y-%m-%dT%H:%M:%S"


def _add_column_from_schema(dd: Any, builder: Any, *, table_name: str, column: Any, rules: list[str], model_alias: str) -> None:
    name = column.name
    col_type = str(getattr(column, "type", "text") or "text").lower()
    params = dict(getattr(column, "distribution_params", {}) or {})

    if bool(getattr(column, "unique", False)) or col_type in {"foreign_key"} or name.lower().endswith("_id"):
        builder.add_column(
            dd.SamplerColumnConfig(
                name=name,
                sampler_type=dd.SamplerType.UUID,
                params=dd.UUIDSamplerParams(prefix=f"{table_name}_{name}_", short_form=True),
                convert_to="str",
            )
        )
        return

    if col_type in {"int", "integer"}:
        builder.add_column(
            dd.SamplerColumnConfig(
                name=name,
                sampler_type=dd.SamplerType.UNIFORM,
                params=dd.UniformSamplerParams(
                    low=float(params.get("min", 1)),
                    high=float(params.get("max", 1000)),
                    decimal_places=0,
                ),
                convert_to="int",
            )
        )
        return

    if col_type in {"float", "decimal", "real", "money", "currency"}:
        builder.add_column(
            dd.SamplerColumnConfig(
                name=name,
                sampler_type=dd.SamplerType.UNIFORM,
                params=dd.UniformSamplerParams(
                    low=float(params.get("min", 10)),
                    high=float(params.get("max", 5000)),
                    decimal_places=int(params.get("decimals", 2)),
                ),
                convert_to="float",
            )
        )
        return

    if col_type in {"boolean", "bool"}:
        builder.add_column(
            dd.SamplerColumnConfig(
                name=name,
                sampler_type=dd.SamplerType.BERNOULLI,
                params=dd.BernoulliSamplerParams(p=float(params.get("probability", 0.5))),
                convert_to="bool",
            )
        )
        return

    if col_type in {"date", "datetime", "timestamp", "time"} or any(
        token in name.lower() for token in ("date", "_at", "time")
    ):
        builder.add_column(
            dd.SamplerColumnConfig(
                name=name,
                sampler_type=dd.SamplerType.DATETIME,
                params=dd.DatetimeSamplerParams(
                    start=str(params.get("start", "2023-01-01")),
                    end=str(params.get("end", "2026-12-31")),
                    unit="D",
                ),
                convert_to=_datetime_convert_format(name, col_type),
            )
        )
        return

    if col_type in {"categorical", "category", "enum"} or name.lower() in {"status", "segment", "tier"}:
        builder.add_column(
            dd.SamplerColumnConfig(
                name=name,
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=_category_values(column), weights=_category_weights(column)),
                convert_to="str",
            )
        )
        return

    builder.add_column(
        dd.LLMTextColumnConfig(
            name=name,
            prompt=_column_prompt(table_name=table_name, column=column, rules=rules),
            model_alias=model_alias,
        )
    )


def _seed_for_table(seed_df: pd.DataFrame | None, table_name: str) -> pd.DataFrame | None:
    if seed_df is None:
        return None
    if "table" in seed_df.columns:
        filtered = seed_df[seed_df["table"].astype(str) == table_name]
        return filtered.drop(columns=["table"]) if not filtered.empty else None
    return seed_df


def _build_table_recipe(
    *,
    schema: Any,
    table_name: str,
    rules: list[str],
    provider_label: str,
    provider: str,
    endpoint: str,
    model: str,
    api_key_env: str,
    api_key_value: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    skip_health_check: bool,
    seed_df: pd.DataFrame | None,
) -> Any:
    dd, _data_designer_cls = _import_data_designer()
    if _requires_api_key(endpoint) and not _api_key(provider_label, api_key_env, api_key_value):
        raise RuntimeError(f"Set `{api_key_env}` before running preview.")
    _apply_health_check_policy(skip_health_check)
    _preflight_schema_slm(provider, endpoint, model, api_key_env)

    model_alias = "schema-text-generator"
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            _model_config(
                dd,
                alias=model_alias,
                provider=provider,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=_provider_extra_body(provider_label, endpoint),
                skip_health_check=skip_health_check,
                workflow="schema-row-generation",
            )
        ]
    )
    table_seed = _seed_for_table(seed_df, table_name)
    if table_seed is not None:
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=table_seed))
    for column in schema.columns.get(table_name, []) or []:
        _add_column_from_schema(dd, builder, table_name=table_name, column=column, rules=rules, model_alias=model_alias)
    return builder


def _preview_table(
    builder: Any,
    *,
    rows: int,
    provider: str,
    endpoint: str,
    model: str,
    api_key_env: str,
) -> pd.DataFrame:
    _dd, data_designer_cls = _import_data_designer()
    try:
        return _preview_to_frame(
            data_designer_cls(model_providers=[_model_provider(_dd, provider, endpoint, api_key_env)]).preview(
                builder,
                num_records=int(rows),
            )
        )
    except Exception as exc:
        raise _explain_empty_dataset_failure(exc, provider=provider, model=model) from exc


def _config_to_dict(builder: Any) -> dict[str, Any]:
    config = builder.build()
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if hasattr(config, "dict"):
        return config.dict()
    return {"repr": repr(config)}


def _package_result(result: Mapping[str, Any]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for table_name, frame in result["preview_tables"].items():
            archive.writestr(f"{table_name}.csv", frame.to_csv(index=False))
        archive.writestr("rules.txt", "\n".join(result["rules"]))
        for table_name, config in result["data_designer_configs"].items():
            archive.writestr(f"generation_config_{table_name}.json", json.dumps(config, indent=2))
    return buffer.getvalue()


_init_state()

st.title(SCHEMA_UI.get("title", "Schema Mode"))
platform_intro()
st.caption("Define generation rules and preview synthetic records directly from the schema.")

with st.container(border=True):
    step = ui_step("schema", "intent")
    step_header(1, step.get("title", "Intent"), True)
    step_guide(what="Choose the business context for this schema.", next_step="Upload a SQL schema.")
    options = SCHEMA_UI.get("intent_options", ["Development & testing"])
    st.session_state.schema_intent = st.selectbox(
        "What will you use the synthetic data for?",
        options=options,
        index=options.index(st.session_state.schema_intent) if st.session_state.schema_intent in options else 0,
    )

with st.container(border=True):
    step = ui_step("schema", "provide_schema")
    step_header(2, step.get("title", "Provide schema"), st.session_state.schema_config is not None)
    step_guide(
        what="Ask AI to draft a schema from English, or upload SQL DDL directly.",
        next_step="Review the generated schema.",
    )
    st.markdown("**Ask AI to define the schema**")
    st.session_state.schema_ai_prompt = st.text_area(
        "Describe the dataset you need",
        value=st.session_state.schema_ai_prompt,
        height=120,
        placeholder=(
            "Example: Build an ecommerce dataset with customers, orders, order items, products, "
            "payments, and shipments. Include relationships and realistic operational fields."
        ),
    )
    (
        draft_provider_label,
        draft_provider,
        draft_endpoint,
        draft_model,
        draft_api_key_env,
        draft_api_key_value,
        draft_temperature,
        draft_top_p,
        draft_max_tokens,
        draft_skip_health_check,
    ) = _hidden_provider_config("schema_draft")
    draft_disabled = not str(st.session_state.schema_ai_prompt or "").strip()
    if st.button("Ask AI to draft schema", type="primary", disabled=draft_disabled):
        with st.status("Drafting schema...", expanded=True) as status:
            try:
                st.session_state.schema_ai_ddl = _draft_schema_json(
                    prompt=st.session_state.schema_ai_prompt,
                    provider_label=draft_provider_label,
                    provider=draft_provider,
                    endpoint=draft_endpoint,
                    model=draft_model,
                    api_key_env=draft_api_key_env,
                    api_key_value=draft_api_key_value,
                    temperature=draft_temperature,
                    top_p=draft_top_p,
                    max_tokens=draft_max_tokens,
                    skip_health_check=draft_skip_health_check,
                )
                status.update(label="Schema draft ready", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Schema draft failed", state="error")
                st.error("AI schema draft failed.")
                st.caption(
                    "The draft creates a structured schema contract and validates it before use. "
                    "Inspect the raw model detail below."
                )
                with st.expander("Technical details", expanded=True):
                    st.code(str(exc))

    if st.session_state.schema_ai_ddl:
        if st.session_state.schema_ai_draft_cache_hit:
            st.caption("Loaded from local schema draft cache.")
        elif st.session_state.schema_ai_draft_repairs:
            st.caption(f"Schema draft normalized with {len(st.session_state.schema_ai_draft_repairs)} repair action(s).")
        edited_ddl = st.text_area(
            "Review or edit AI-generated JSON schema",
            value=st.session_state.schema_ai_ddl,
            height=240,
            key="schema_ai_ddl_editor",
        )
        if st.button("Use this AI schema"):
            try:
                schema = load_schema_bytes(edited_ddl.encode("utf-8"), "ai_schema.json")
                st.session_state.schema_config = schema
                st.session_state.schema_summary = summarize_schema(schema)
                st.session_state.schema_file_id = "ai_schema"
                st.session_state.schema_ai_ddl = edited_ddl
                st.session_state.schema_rule_rows = []
                _reset_result()
                st.rerun()
            except Exception as exc:
                st.session_state.schema_config = None
                st.session_state.schema_summary = None
                show_user_error(
                    "We couldn't parse the AI-generated schema.",
                    technical=exc,
                    next_action="Edit the SQL so it contains valid CREATE TABLE statements, then try again.",
                )

    st.divider()
    st.markdown("**Or upload SQL directly**")
    uploaded = st.file_uploader("SQL schema file", type=["sql"])
    if uploaded is not None:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if file_id != st.session_state.schema_file_id:
            try:
                schema = load_schema_bytes(uploaded.getvalue(), uploaded.name)
                st.session_state.schema_config = schema
                st.session_state.schema_summary = summarize_schema(schema)
                st.session_state.schema_file_id = file_id
                st.session_state.schema_rule_rows = []
                _reset_result()
            except Exception as exc:
                st.session_state.schema_config = None
                st.session_state.schema_summary = None
                show_user_error(
                    "We couldn't load your schema file.",
                    technical=exc,
                    next_action="Use a .sql file with CREATE TABLE statements.",
                )
                st.stop()

if st.session_state.schema_config is None:
    st.info("Upload or draft a schema to continue.")
    st.stop()

schema = st.session_state.schema_config
summary = st.session_state.schema_summary or summarize_schema(schema)
st.session_state.schema_summary = summary

with st.container(border=True):
    step = ui_step("schema", "review_schema")
    step_header(3, step.get("title", "Review schema"), True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tables", summary.table_count)
    c2.metric("Columns", summary.column_count)
    c3.metric("Relationships", summary.relationship_count)
    c4.metric("Rows planned", f"{sum(int(getattr(table, 'row_count', 0) or 0) for table in schema.tables):,}")
    st.dataframe(summary.tables, width="stretch", hide_index=True)
    with st.expander("Column contracts", expanded=False):
        st.dataframe(schema_column_details(schema), width="stretch", hide_index=True)

with st.container(border=True):
    step_header(4, "Define rules", bool(_active_rules()))
    step_guide(
        what="Write plain-English constraints for generated records and fields.",
        next_step="Set preview rows.",
    )
    st.session_state.schema_rules = st.text_area(
        "Global generation rules",
        value=st.session_state.schema_rules,
        height=120,
        placeholder=(
            "Example:\n"
            "- Generate realistic customer/order records.\n"
            "- Names should match the selected business domain.\n"
            "- Emails must be fictional.\n"
            "- Status values should read like production workflow states."
        ),
    )
    rule_rows = []
    for table in schema.tables:
        with st.expander(f"{table.name} rules", expanded=False):
            table_rule = st.text_area(
                "Table rule",
                key=f"schema_rule_table_{table.name}",
                placeholder=f"Rules that apply to all `{table.name}` records.",
            )
            if table_rule.strip():
                rule_rows.append({"scope": table.name, "rule": table_rule.strip()})
            for column in schema.columns.get(table.name, []) or []:
                column_rule = st.text_input(
                    column.name,
                    key=f"schema_rule_col_{table.name}_{column.name}",
                    placeholder=f"Rule for `{table.name}.{column.name}`",
                )
                if column_rule.strip():
                    rule_rows.append({"scope": f"{table.name}.{column.name}", "rule": column_rule.strip()})
    st.session_state.schema_rule_rows = rule_rows

with st.container(border=True):
    step_header(5, "Configure preview", True)
    step_guide(
        what="Set preview row counts and optional seed data.",
        next_step="Run preview.",
    )
    provider_label, provider, endpoint, model, api_key_env, api_key_value, temperature, top_p, max_tokens, skip_health_check = (
        _hidden_provider_config("schema_preview", fallback_prefix="schema_draft")
    )

    st.markdown("**Rows to preview**")
    row_counts = {}
    columns = st.columns(min(4, max(1, len(schema.tables))))
    for idx, table in enumerate(schema.tables):
        with columns[idx % len(columns)]:
            row_counts[table.name] = int(
                st.number_input(
                    table.name,
                    min_value=1,
                    max_value=500,
                    value=min(int(getattr(table, "row_count", None) or SETTINGS.schema_default_rows_per_table), 25),
                    step=1,
                    key=f"schema_dd_rows_{table.name}",
                )
            )

    seed_upload = st.file_uploader("Optional seed dataset (.csv)", type=["csv"], key="schema_seed_upload")
    if seed_upload is not None:
        st.session_state.schema_seed_df = pd.read_csv(seed_upload)
        st.caption(f"Loaded seed dataset: {len(st.session_state.schema_seed_df):,} row(s).")

    rules = _active_rules()
    if not rules:
        st.warning("Add at least one global, table, or column rule before previewing.")

    if st.button("Preview records", type="primary", width="stretch", disabled=not rules):
        with st.status("Generating preview...", expanded=True) as status:
            try:
                status.write("Preparing schema contract...")
                with measure_generation() as run_stats:
                    mode_result = generate_from_schema(
                        schema,
                        row_count=max(row_counts.values()),
                        table_row_counts=row_counts,
                        seed=int(os.getenv("SP_SCHEMA_SEED", "7")),
                        output_dir=Path(st.session_state.schema_workdir) / "exports",
                        export_format="csv",
                        preview_rows=max(row_counts.values()),
                    )
                result = {
                    "preview_tables": mode_result.preview_tables,
                    "data_designer_configs": {
                        "schema_contract": {
                            "generation": "schema_pipeline",
                            "validation": validation_highlights(mode_result),
                            "rules": rules,
                        }
                    },
                    "rules": rules,
                    "summary": {
                        "provider": provider,
                        "model": model,
                        "tables": dict(mode_result.row_counts),
                    },
                }
                st.session_state.schema_result = result
                st.session_state.schema_run_metrics = dict(run_stats)
                st.session_state.schema_zip_bytes = package_download(mode_result)
                status.update(label="Preview complete", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:
                _reset_result()
                status.update(label="Preview failed", state="error")
                show_user_error(
                    "Preview failed.",
                    technical=exc,
                    next_action=(
                        "Confirm the configured credentials are available in the environment, then try a smaller preview."
                    ),
                )

result = st.session_state.schema_result
if result is None:
    st.stop()

with st.container(border=True):
    step_header(6, "Preview", True)
    total_rows = sum(len(frame) for frame in result["preview_tables"].values())
    m1, m2 = st.columns(2)
    m1.metric("Tables", len(result["preview_tables"]))
    m2.metric("Preview rows", total_rows)
    table_name = st.selectbox("Preview table", list(result["preview_tables"]))
    st.dataframe(result["preview_tables"][table_name], width="stretch", hide_index=True)
    with st.expander("Generation config", expanded=False):
        st.json(result["data_designer_configs"].get(table_name) or result["data_designer_configs"].get("schema_contract", {}))

with st.container(border=True):
    step_header(7, "Download", st.session_state.schema_zip_bytes is not None)
    st.download_button(
        "Download preview package",
        data=st.session_state.schema_zip_bytes,
        file_name=f"{summary.name.replace(' ', '_').lower()}_preview_package.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )
