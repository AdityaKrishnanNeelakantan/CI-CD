"""Production readiness checks for client-facing deployments."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import util
from typing import Any, Mapping

from synth_platform.engine.generation.slm_runtime import (
    is_local_endpoint,
    resolve_platform_slm_runtime,
)

_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    message: str
    remediation: str = ""


@dataclass(frozen=True)
class ProductionReadinessReport:
    profile: str
    ready: bool
    checks: tuple[ReadinessCheck, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "ready": self.ready,
            "checks": [check.__dict__ for check in self.checks],
        }


def deployment_profile() -> str:
    return os.getenv("SP_DEPLOYMENT_PROFILE", os.getenv("SP_ENV", "development")).strip().lower() or "development"


def is_production_profile(profile: str | None = None, ui_config: Mapping[str, Any] | None = None) -> bool:
    production_aliases = set(_config_list(ui_config, "production", "profiles", "production_aliases", default=["prod", "production", "client"]))
    return (profile or deployment_profile()).strip().lower() in production_aliases


def build_production_readiness_report(ui_config: Mapping[str, Any] | None = None) -> ProductionReadinessReport:
    profile = deployment_profile()
    production = is_production_profile(profile, ui_config)
    checks = [
        _profile_check(profile, ui_config),
        _mock_mode_check(production, ui_config),
        _slm_endpoint_check(production, ui_config),
        _api_key_check(production, ui_config),
        _data_designer_check(production, ui_config),
    ]
    checks.extend(_ui_mode_checks(ui_config or {}, production))
    ready = all(check.status != "fail" for check in checks)
    return ProductionReadinessReport(profile=profile, ready=ready, checks=tuple(checks))


def _profile_check(profile: str, ui_config: Mapping[str, Any] | None = None) -> ReadinessCheck:
    known = set(_config_list(ui_config, "production", "profiles", "known", default=["development", "demo", "staging", "prod", "production", "client"]))
    if profile in known:
        return ReadinessCheck("deployment_profile", "pass", f"Deployment profile is {profile}.")
    return ReadinessCheck(
        "deployment_profile",
        "warn",
        f"Deployment profile {profile!r} is not a standard profile.",
        _config_text(ui_config, "production", "checks", "unknown_profile_remediation"),
    )


def _mock_mode_check(production: bool, ui_config: Mapping[str, Any] | None = None) -> ReadinessCheck:
    enabled = os.getenv("SP_NEMO_DATA_DESIGNER_MOCK", "").strip().lower() in _TRUE_VALUES
    if production and enabled:
        return ReadinessCheck(
            "mock_generation",
            "fail",
            "Data Designer mock mode is enabled in production.",
            _config_text(ui_config, "production", "checks", "mock_generation_remediation"),
        )
    if enabled:
        return ReadinessCheck("mock_generation", "warn", "Data Designer mock mode is enabled.")
    return ReadinessCheck("mock_generation", "pass", "Mock generation is disabled.")


def _slm_endpoint_check(production: bool, ui_config: Mapping[str, Any] | None = None) -> ReadinessCheck:
    runtime = resolve_platform_slm_runtime()
    local_allowed = os.getenv("SP_PRODUCTION_ALLOW_LOCAL_SLM", "").strip().lower() in _TRUE_VALUES
    if production and is_local_endpoint(runtime.endpoint) and not local_allowed:
        return ReadinessCheck(
            "slm_endpoint",
            "fail",
            f"Production profile points at local SLM endpoint {runtime.endpoint}.",
            _config_text(ui_config, "production", "checks", "local_slm_remediation"),
        )
    return ReadinessCheck("slm_endpoint", "pass", f"SLM endpoint configured: {runtime.endpoint}.")


def _api_key_check(production: bool, ui_config: Mapping[str, Any] | None = None) -> ReadinessCheck:
    runtime = resolve_platform_slm_runtime()
    if is_local_endpoint(runtime.endpoint):
        return ReadinessCheck("slm_api_key", "pass", "Local SLM endpoint does not require an API key.")
    if os.getenv(runtime.api_key_env) or os.getenv("SP_NEMO_DATA_DESIGNER_API_KEY"):
        return ReadinessCheck("slm_api_key", "pass", f"SLM API key is supplied through {runtime.api_key_env}.")
    status = "fail" if production else "warn"
    return ReadinessCheck(
        "slm_api_key",
        status,
        f"No API key is configured for non-local SLM endpoint {runtime.endpoint}.",
        _config_text(ui_config, "production", "checks", "remote_api_key_remediation"),
    )


def _data_designer_check(production: bool, ui_config: Mapping[str, Any] | None = None) -> ReadinessCheck:
    installed = util.find_spec("data_designer") is not None
    if installed:
        return ReadinessCheck("data_designer_sdk", "pass", "Open-source Data Designer SDK is importable.")
    status = "fail" if production else "warn"
    return ReadinessCheck(
        "data_designer_sdk",
        status,
        "Open-source Data Designer SDK is not importable.",
        _config_text(ui_config, "production", "checks", "data_designer_remediation"),
    )


def _ui_mode_checks(ui_config: Mapping[str, Any], production: bool) -> list[ReadinessCheck]:
    modes = (
        ((ui_config.get("home") or {}).get("chat") or {}).get("modes")
        if isinstance(ui_config.get("home"), Mapping)
        else []
    )
    checks: list[ReadinessCheck] = []
    for mode in modes or []:
        if not isinstance(mode, Mapping):
            continue
        key = str(mode.get("key") or "unknown")
        ready_status = _config_text(ui_config, "production", "statuses", "ready") or "ready"
        production_status = str(mode.get("production_status") or ready_status).strip().lower()
        enabled = bool(mode.get("available_in_production", production_status == ready_status))
        if production and not enabled:
            checks.append(
                ReadinessCheck(
                    f"ui_mode_{key}",
                    "warn",
                    f"{mode.get('title') or key} is visible but marked {production_status}.",
                    _config_text(ui_config, "production", "checks", "wip_mode_remediation"),
                )
            )
        elif production and production_status != ready_status:
            checks.append(
                ReadinessCheck(
                    f"ui_mode_{key}",
                    "fail",
                    f"{mode.get('title') or key} is not marked production-ready.",
                    _config_text(ui_config, "production", "checks", "not_ready_mode_remediation"),
                )
            )
        else:
            checks.append(ReadinessCheck(f"ui_mode_{key}", "pass", f"{mode.get('title') or key} UI status is {production_status}."))
    return checks


def _config_value(config: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    value: Any = config or {}
    for part in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(part, default)
    return value


def _config_text(config: Mapping[str, Any] | None, *path: str) -> str:
    value = _config_value(config, *path, default="")
    return str(value) if value is not None else ""


def _config_list(config: Mapping[str, Any] | None, *path: str, default: list[str]) -> list[str]:
    value = _config_value(config, *path, default=default)
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return list(default)
