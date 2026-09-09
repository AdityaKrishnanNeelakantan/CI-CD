"""Shared Data Designer provider/model config for the single internal SLM."""
from __future__ import annotations

import os
import importlib
from typing import Any

from synth_platform.engine.generation.slm_runtime import (
    PlatformSLMRuntime,
    data_designer_max_parallel_requests,
    data_designer_max_tokens,
    data_designer_provider_api_key,
    data_designer_skip_health_check,
    data_designer_timeout,
    resolve_platform_slm_runtime,
)


WORKFLOW_TOKEN_DEFAULTS = {
    "provider-smoke-test": 64,
    "schema-draft": 768,
    "schema-row-generation": 768,
    "pdf-value-batch": 512,
    "transcript-ssot": 768,
    "transcript-twin-structured": 1536,
    "transcript-conversation-structured": 768,
    "transcript-conversation-row": 96,
    "database-row-generation": 1024,
}

WORKFLOW_ENV_NAMES = {
    "schema-draft": "SP_SCHEMA_DRAFT_MAX_TOKENS",
    "schema-row-generation": "SP_SCHEMA_DATA_DESIGNER_MAX_TOKENS",
    "pdf-value-batch": "SP_PDF_DATA_DESIGNER_MAX_TOKENS",
    "transcript-ssot": "SP_TRANSCRIPT_SSOT_MAX_TOKENS",
    "transcript-twin-structured": "SP_TRANSCRIPT_DATA_DESIGNER_MAX_TOKENS",
    "transcript-conversation-structured": "SP_TRANSCRIPT_DATA_DESIGNER_MAX_TOKENS",
    "transcript-conversation-row": "SP_TRANSCRIPT_DATA_DESIGNER_MAX_TOKENS",
    "database-row-generation": "SP_DATABASE_DATA_DESIGNER_MAX_TOKENS",
}


def load_data_designer_sdk() -> tuple[Any, Any]:
    """Load the optional Data Designer SDK at runtime."""
    dd = importlib.import_module("data_designer.config")
    interface = importlib.import_module("data_designer.interface")
    return dd, interface.DataDesigner


def build_data_designer_provider(dd: Any, runtime: PlatformSLMRuntime | None = None) -> Any:
    """Build the one OpenAI-compatible provider used by SDK-backed workflows."""
    runtime = runtime or resolve_platform_slm_runtime()
    return dd.ModelProvider(
        name=runtime.provider,
        endpoint=runtime.endpoint,
        provider_type="openai",
        api_key=data_designer_provider_api_key(runtime),
    )


def get_data_designer_provider_config(dd: Any, runtime: PlatformSLMRuntime | None = None) -> Any:
    """Compatibility wrapper for the central Data Designer provider config."""
    return build_data_designer_provider(dd, runtime)


def build_data_designer_model_config(
    dd: Any,
    alias: str,
    *,
    workflow: str,
    runtime: PlatformSLMRuntime | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout_env: str | None = None,
    parallel_env: str | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    skip_health_check: bool | None = None,
) -> Any:
    """Build a workflow alias that points to the same physical SLM."""
    runtime = runtime or resolve_platform_slm_runtime()
    return dd.ModelConfig(
        alias=alias,
        model=runtime.model_id,
        provider=runtime.provider,
        skip_health_check=bool(data_designer_skip_health_check() if skip_health_check is None else skip_health_check),
        inference_parameters=build_data_designer_inference_params(
            dd,
            workflow,
            temperature=temperature,
            top_p=top_p,
            timeout_env=timeout_env,
            parallel_env=parallel_env,
            max_tokens=max_tokens,
            extra_body=extra_body,
        ),
    )


def get_data_designer_model_config(
    dd: Any,
    alias: str,
    *,
    workflow: str,
    runtime: PlatformSLMRuntime | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout_env: str | None = None,
    parallel_env: str | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    skip_health_check: bool | None = None,
) -> Any:
    """Compatibility wrapper for workflow aliases pointing at one physical model."""
    return build_data_designer_model_config(
        dd,
        alias,
        workflow=workflow,
        runtime=runtime,
        temperature=temperature,
        top_p=top_p,
        timeout_env=timeout_env,
        parallel_env=parallel_env,
        max_tokens=max_tokens,
        extra_body=extra_body,
        skip_health_check=skip_health_check,
    )


def build_data_designer_inference_params(
    dd: Any,
    workflow: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout_env: str | None = None,
    parallel_env: str | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Any:
    default_max_tokens = WORKFLOW_TOKEN_DEFAULTS.get(workflow, 1024)
    workflow_env = WORKFLOW_ENV_NAMES.get(workflow)
    resolved_max_tokens = data_designer_max_tokens(default_max_tokens, workflow_env=workflow_env)
    if max_tokens is not None:
        resolved_max_tokens = min(int(max_tokens), resolved_max_tokens)
    params: dict[str, Any] = {
        "max_parallel_requests": data_designer_max_parallel_requests(workflow_env=parallel_env),
        "timeout": data_designer_timeout(workflow_env=timeout_env),
        "temperature": _workflow_temperature(workflow, temperature),
        "top_p": _workflow_top_p(workflow, top_p),
        "max_tokens": resolved_max_tokens,
    }
    params["extra_body"] = dict(extra_body) if extra_body else None
    return dd.ChatCompletionInferenceParams(**params)


def get_data_designer_inference_params(
    dd: Any,
    workflow: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout_env: str | None = None,
    parallel_env: str | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Any:
    """Compatibility wrapper for workflow-specific local SLM inference params."""
    return build_data_designer_inference_params(
        dd,
        workflow,
        temperature=temperature,
        top_p=top_p,
        timeout_env=timeout_env,
        parallel_env=parallel_env,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def _workflow_temperature(workflow: str, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    env_name = {
        "schema-draft": "SP_SCHEMA_DRAFT_TEMPERATURE",
        "schema-row-generation": "SP_SCHEMA_DATA_DESIGNER_TEMPERATURE",
        "pdf-value-batch": "SP_PDF_DATA_DESIGNER_TEMPERATURE",
        "transcript-ssot": "SP_TRANSCRIPT_SSOT_TEMPERATURE",
        "transcript-twin-structured": "SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE",
        "transcript-conversation-structured": "SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE",
        "transcript-conversation-row": "SP_TRANSCRIPT_DATA_DESIGNER_TEMPERATURE",
        "database-row-generation": "SP_DATABASE_DATA_DESIGNER_TEMPERATURE",
    }.get(workflow, "SP_NEMO_DATA_DESIGNER_TEMPERATURE")
    fallback = {
        "schema-draft": "0.1",
        "schema-row-generation": "0.35",
        "pdf-value-batch": "0.2",
        "transcript-ssot": "0.1",
        "transcript-twin-structured": "0.2",
        "transcript-conversation-structured": "0.2",
        "transcript-conversation-row": "0.2",
        "database-row-generation": "0.35",
    }.get(workflow, "0.35")
    return float(os.getenv(env_name, os.getenv("SP_NEMO_DATA_DESIGNER_TEMPERATURE", fallback)))


def _workflow_top_p(workflow: str, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    env_name = {
        "schema-draft": "SP_SCHEMA_DRAFT_TOP_P",
        "schema-row-generation": "SP_SCHEMA_DATA_DESIGNER_TOP_P",
        "pdf-value-batch": "SP_PDF_DATA_DESIGNER_TOP_P",
        "transcript-ssot": "SP_TRANSCRIPT_SSOT_TOP_P",
        "transcript-twin-structured": "SP_TRANSCRIPT_DATA_DESIGNER_TOP_P",
        "transcript-conversation-structured": "SP_TRANSCRIPT_DATA_DESIGNER_TOP_P",
        "transcript-conversation-row": "SP_TRANSCRIPT_DATA_DESIGNER_TOP_P",
        "database-row-generation": "SP_DATABASE_DATA_DESIGNER_TOP_P",
    }.get(workflow, "SP_NEMO_DATA_DESIGNER_TOP_P")
    fallback = {
        "schema-draft": "0.8",
        "schema-row-generation": "0.9",
        "pdf-value-batch": "0.8",
        "transcript-ssot": "0.8",
        "transcript-twin-structured": "0.9",
        "transcript-conversation-structured": "0.9",
        "transcript-conversation-row": "0.9",
        "database-row-generation": "0.9",
    }.get(workflow, "0.9")
    return float(os.getenv(env_name, os.getenv("SP_NEMO_DATA_DESIGNER_TOP_P", fallback)))
