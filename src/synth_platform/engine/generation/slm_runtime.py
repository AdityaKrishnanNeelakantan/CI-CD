"""Shared OpenAI-compatible SLM runtime settings for Data Designer workflows."""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True)
class PlatformSLMRuntime:
    model_id: str
    provider: str
    endpoint: str
    api_key_env: str
    model_family: str = "open_source_slm"


def resolve_platform_slm_runtime() -> PlatformSLMRuntime:
    """Resolve the one SLM used by Schema, PDF, and Transcript twin workflows."""
    return PlatformSLMRuntime(
        model_id=os.getenv("SP_PLATFORM_SLM_MODEL", os.getenv("SP_DIGITAL_TWIN_SLM_MODEL", "synth-platform-slm")),
        provider=os.getenv(
            "SP_PLATFORM_SLM_PROVIDER",
            os.getenv("SP_DIGITAL_TWIN_SLM_PROVIDER", os.getenv("SP_NEMO_DATA_DESIGNER_PROVIDER", "internal")),
        ),
        endpoint=normalize_model_endpoint(
            os.getenv(
                "SP_PLATFORM_SLM_ENDPOINT",
                os.getenv(
                    "SP_DIGITAL_TWIN_SLM_ENDPOINT",
                    os.getenv("SP_NEMO_DATA_DESIGNER_ENDPOINT", "http://localhost:11434/v1"),
                ),
            )
        ),
        api_key_env=os.getenv(
            "SP_PLATFORM_SLM_API_KEY_ENV",
            os.getenv(
                "SP_DIGITAL_TWIN_SLM_API_KEY_ENV",
                os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY_ENV", "SP_NEMO_DATA_DESIGNER_API_KEY"),
            ),
        ),
        model_family=os.getenv(
            "SP_PLATFORM_SLM_MODEL_FAMILY",
            os.getenv("SP_DIGITAL_TWIN_MODEL_FAMILY", "open_source_slm"),
        ),
    )


def normalize_model_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if (parsed.hostname or "").lower() != "localhost":
        return endpoint
    netloc = "127.0.0.1"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}


def apply_local_endpoint_network_policy(endpoint: str) -> None:
    if not is_local_endpoint(endpoint):
        return
    no_proxy_values = ["localhost", "127.0.0.1", "::1"]
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = [part.strip() for part in os.getenv(env_name, "").split(",") if part.strip()]
        merged = existing + [value for value in no_proxy_values if value not in existing]
        os.environ[env_name] = ",".join(merged)


def data_designer_provider_api_key(runtime: PlatformSLMRuntime) -> str | None:
    apply_local_endpoint_network_policy(runtime.endpoint)
    if is_local_endpoint(runtime.endpoint):
        return None
    key = os.getenv(runtime.api_key_env) or os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY")
    if key:
        if not os.getenv(runtime.api_key_env):
            os.environ[runtime.api_key_env] = key
        return runtime.api_key_env
    return runtime.api_key_env


def require_data_designer_api_key(runtime: PlatformSLMRuntime) -> None:
    key = os.getenv(runtime.api_key_env) or os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY")
    if key:
        if not os.getenv(runtime.api_key_env):
            os.environ[runtime.api_key_env] = key
        return
    if is_local_endpoint(runtime.endpoint):
        return
    env_names = list(dict.fromkeys([runtime.api_key_env, "SP_NEMO_DATA_DESIGNER_API_KEY"]))
    raise RuntimeError(
        "Set one of these environment variables before running this workflow: "
        f"{', '.join(env_names)}."
    )


def data_designer_skip_health_check() -> bool:
    return os.getenv("SP_NEMO_DATA_DESIGNER_SKIP_HEALTH_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def data_designer_timeout(default: int = 300, *, workflow_env: str | None = None) -> int:
    return _positive_int_env(
        "SP_PLATFORM_SLM_TIMEOUT",
        "SP_NEMO_DATA_DESIGNER_TIMEOUT",
        workflow_env,
        default=default,
    )


def data_designer_max_parallel_requests(default: int = 1, *, workflow_env: str | None = None) -> int:
    return _positive_int_env(
        "SP_PLATFORM_SLM_MAX_PARALLEL_REQUESTS",
        "SP_NEMO_DATA_DESIGNER_MAX_PARALLEL_REQUESTS",
        workflow_env,
        default=default,
    )


def data_designer_max_tokens(default: int = 2048, *, workflow_env: str | None = None) -> int:
    return _positive_int_env(
        "SP_PLATFORM_SLM_MAX_TOKENS",
        "SP_NEMO_DATA_DESIGNER_MAX_TOKENS",
        workflow_env,
        default=default,
    )


def preflight_slm_endpoint(runtime: PlatformSLMRuntime, *, timeout: float = 5.0) -> None:
    """Fail fast when the configured OpenAI-compatible endpoint is unreachable."""
    if os.getenv("SP_PLATFORM_SLM_PREFLIGHT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    if os.getenv("SP_NEMO_DATA_DESIGNER_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    apply_local_endpoint_network_policy(runtime.endpoint)
    models_url = _endpoint_child_url(runtime.endpoint, "models")
    headers: dict[str, str] = {}
    key = os.getenv(runtime.api_key_env) or os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(512)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 404, 405}:
            return
        raise RuntimeError(_preflight_error_message(runtime, str(exc))) from exc
    except Exception as exc:
        raise RuntimeError(_preflight_error_message(runtime, str(exc))) from exc


def _endpoint_child_url(endpoint: str, child: str) -> str:
    return f"{endpoint.rstrip('/')}/{child.lstrip('/')}"


def _positive_int_env(*names: str | None, default: int) -> int:
    for name in names:
        if not name:
            continue
        raw_value = os.getenv(name)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value > 0:
            return value
    return default


def _preflight_error_message(runtime: PlatformSLMRuntime, cause: str) -> str:
    return (
        "The configured SLM endpoint is not reachable from this Python process. "
        f"endpoint={runtime.endpoint}; provider={runtime.provider}; model={runtime.model_id}; cause={cause}. "
        "Start the local OpenAI-compatible SLM server or update SP_PLATFORM_SLM_ENDPOINT."
    )
