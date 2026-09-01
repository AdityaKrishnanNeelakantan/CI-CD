"""Loader for config/project.yaml.

Non-secret settings live in the YAML file itself. Any string value of the
form "${VAR_NAME}" is resolved from the environment at load time; a missing
environment variable is a hard failure rather than a silent empty string, so
misconfiguration surfaces immediately instead of producing a confusing
downstream connection error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(Exception):
    """Raised when project.yaml is missing, malformed, or references an unset env var."""


def _interpolate_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.match(value)
        if match:
            var_name = match.group(1)
            resolved = os.environ.get(var_name)
            if resolved is None:
                raise ConfigError(
                    f"project.yaml references environment variable '{var_name}' "
                    "which is not set."
                )
            return resolved
        return value
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


@dataclass(frozen=True)
class SamplingConfig:
    default_limit: int
    max_limit: int


@dataclass(frozen=True)
class SourceConfig:
    type: str
    connection: dict[str, Any]
    sampling: SamplingConfig


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str
    metadata_dir: str


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    dataset_id: str
    source: SourceConfig
    output: OutputConfig
    raw: dict[str, Any]


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file did not parse to a mapping: {config_path}")

    try:
        project = raw["project"]
        source = raw["source"]
        output = raw["output"]
        sampling = source["sampling"]
    except KeyError as exc:
        raise ConfigError(f"Config file missing required key: {exc}") from exc

    connection = _interpolate_env_vars(source.get("connection", {}))

    try:
        sampling_config = SamplingConfig(
            default_limit=int(sampling["default_limit"]),
            max_limit=int(sampling["max_limit"]),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"source.sampling.default_limit and max_limit must be integers: {exc}"
        ) from exc

    return ProjectConfig(
        name=project["name"],
        dataset_id=project["dataset_id"],
        source=SourceConfig(
            type=source["type"],
            connection=connection,
            sampling=sampling_config,
        ),
        output=OutputConfig(
            runs_dir=output["runs_dir"],
            metadata_dir=output["metadata_dir"],
        ),
        raw=raw,
    )
