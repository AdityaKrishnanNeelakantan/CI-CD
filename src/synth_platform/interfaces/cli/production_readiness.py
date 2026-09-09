"""Air-gap-safe production readiness checks for Synth Platform.

The command inspects only local metadata, files, environment-variable presence,
and executables on ``PATH``. It never opens a socket, imports optional runtime
packages, executes a model, or processes user data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

PACKAGE_NAME = "synth-platform"
SCHEMA_VERSION = "1.0"

CORE_REQUIREMENTS: dict[str, str] = {
    "pydantic": ">=2",
    "pandas": ">=2.0",
    "numpy": ">=1.23",
    "safetensors": ">=0.4",
    "SQLAlchemy": ">=2.0",
    "cryptography": ">=41",
    "pdfplumber": ">=0.11",
    "pypdf": ">=5",
    "reportlab": ">=4",
    "PyYAML": ">=6.0",
    "Faker": ">=20",
}

CAPABILITY_REQUIREMENTS: dict[str, dict[str, str | None]] = {
    "schema": {
        "data-designer": ">=0.9.1",
        "duckdb": ">=0.9",
        "python-json-logger": ">=3.0.0,<4",
        "scipy": ">=1.10",
        "simpleeval": ">=1.0.7",
    },
    "database": {"sdv": ">=1.37.4", "copulas": ">=0.12"},
    "pdf": {"fpdf2": ">=2.8.7", "langdetect": ">=1.0.9"},
    "ui": {"streamlit": "==1.60.0"},
    "parquet": {"pyarrow": ">=12"},
    "docling": {"docling": "==2.119.0"},
    "ocr": {"pdf2image": ">=1.17", "Pillow": ">=10", "pytesseract": ">=0.3"},
    "llm-openai": {"openai": ">=1.0.0"},
    "llm-groq": {"groq": ">=0.4.0"},
}

CHECK_STATUSES = ("pass", "warn", "fail", "unavailable", "planned", "not_checked")

PLANNED_CAPABILITIES = (
    (
        "planned.interaction_twin",
        "Customer Interaction Twin",
        "No workflow implementation exists; requests fail closed rather than using a substitute.",
    ),
    (
        "planned.data_designer_integration",
        "NVIDIA Data Designer integration",
        "Dependencies may be installed, but no application adapter is implemented yet.",
    ),
    (
        "planned.mcp_gateway",
        "MCP tool gateway and Faker-as-a-Tool",
        "The target architecture includes this plane; the active runtime does not.",
    ),
    (
        "planned.live_agent_plane",
        "Live multi-agent service plane",
        "Operational service agents and session infrastructure are not implemented here.",
    ),
    (
        "planned.unified_evaluation",
        "Unified governance and evaluation hub",
        "Workflow-specific validation exists; a shared evaluation service does not.",
    ),
    (
        "planned.security_plane",
        "Authentication, RBAC, gateway, and policy plane",
        "Artifact signing exists; the target security and authorization plane is not implemented.",
    ),
    (
        "planned.enterprise_state",
        "Redis, PostgreSQL, and object-storage deployment",
        "These remain deliberate later-phase infrastructure choices.",
    ),
)

REQUIRE_CHOICES = tuple(sorted((*CAPABILITY_REQUIREMENTS, "nvidia", "ollama")))


@dataclass(frozen=True)
class CheckResult:
    """One deterministic readiness observation."""

    id: str
    dimension: str
    requirement: str
    status: str
    summary: str
    details: dict[str, object] = field(default_factory=dict)
    capability: str | None = None


@dataclass(frozen=True)
class ReadinessReport:
    """Stable, JSON-serializable readiness report."""

    schema_version: str
    command: str
    package: dict[str, str]
    runtime: dict[str, str]
    air_gap_safe: bool
    policy: dict[str, object]
    overall_status: str
    summary: dict[str, int]
    checks: tuple[CheckResult, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_project_version() -> str:
    pyproject = _source_root().parents[1] / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        return str(project["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


def _distribution_contains_runtime(distribution: metadata.Distribution) -> bool:
    runtime_root = _source_root().resolve()
    candidates: list[Path] = []
    try:
        candidates.append(Path(distribution.locate_file("")).resolve())
    except (OSError, TypeError):
        pass

    try:
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            parsed = urlparse(str(payload.get("url", "")))
            if parsed.scheme == "file":
                candidates.append(Path(unquote(parsed.path)).resolve())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    return any(
        runtime_root == candidate or runtime_root.is_relative_to(candidate)
        for candidate in candidates
    )


def _project_metadata() -> tuple[str, str]:
    """Return the executing package version without trusting unrelated metadata."""

    try:
        distribution = metadata.distribution(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return _source_project_version(), "source_pyproject"

    if _distribution_contains_runtime(distribution):
        return distribution.version, "installed_distribution"
    return _source_project_version(), "source_pyproject_unrelated_distribution"


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _release_key(version: str) -> tuple[tuple[int, ...], int] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)(.*)$", version)
    if not match:
        return None
    release = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2).lower()
    normalized_suffix = suffix.lstrip(".-_")
    phase = -1 if normalized_suffix.startswith(("a", "b", "rc", "dev")) else 0
    if "post" in normalized_suffix:
        phase = 1
    return release, phase


def _compare_versions(left: str, right: str) -> int | None:
    left_key = _release_key(left)
    right_key = _release_key(right)
    if left_key is None or right_key is None:
        return None
    width = max(len(left_key[0]), len(right_key[0]))
    left_normalized = (*left_key[0], *((0,) * (width - len(left_key[0]))), left_key[1])
    right_normalized = (
        *right_key[0],
        *((0,) * (width - len(right_key[0]))),
        right_key[1],
    )
    return (left_normalized > right_normalized) - (left_normalized < right_normalized)


def _version_satisfies(version: str, specification: str | None) -> bool:
    """Evaluate the simple numeric constraints declared by this project."""

    if not specification:
        return True
    for clause in specification.split(","):
        match = re.fullmatch(r"\s*(>=|<=|==|>|<)\s*([0-9][0-9.]*)\s*", clause)
        if not match:
            return False
        comparison = _compare_versions(version, match.group(2))
        if comparison is None:
            return False
        operator = match.group(1)
        if operator == ">=" and comparison < 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
        if operator == "==" and comparison != 0:
            return False
    return True


def _python_check() -> CheckResult:
    supported = sys.version_info >= (3, 11)
    return CheckResult(
        id="core.python",
        dimension="runtime",
        requirement="required",
        status="pass" if supported else "fail",
        summary=(
            "Python runtime satisfies >=3.11."
            if supported
            else "Python 3.11 or newer is required."
        ),
        details={
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "required": ">=3.11",
        },
    )


def _package_check(version: str, version_source: str) -> CheckResult:
    installed = version_source == "installed_distribution"
    if installed:
        summary = (
            f"Installed {PACKAGE_NAME} distribution matches the executing package."
        )
    elif version_source == "source_pyproject_unrelated_distribution":
        summary = (
            "Running source does not match the visible installed distribution metadata."
        )
    else:
        summary = "Running from source; installed distribution metadata is unavailable."
    return CheckResult(
        id="core.package_metadata",
        dimension="runtime",
        requirement="required",
        status="pass" if installed else "warn",
        summary=summary,
        details={"name": PACKAGE_NAME, "version": version, "source": version_source},
    )


def _dependency_check(
    name: str,
    specification: str | None,
    *,
    requirement: str,
    dimension: str,
    capability: str | None = None,
) -> CheckResult:
    version = _distribution_version(name)
    present = version is not None
    compatible = present and _version_satisfies(version, specification)
    status = (
        "pass"
        if compatible
        else ("fail" if requirement == "required" else "unavailable")
    )
    if not present:
        summary = f"Distribution {name} is not installed."
    elif not compatible:
        summary = f"Distribution {name} {version} does not satisfy {specification}."
    else:
        summary = (
            f"Distribution {name} {version} satisfies {specification or 'any version'}."
        )
    return CheckResult(
        id=f"{dimension}.dependency.{name.lower()}",
        dimension=dimension,
        requirement=requirement,
        status=status,
        summary=summary,
        details={
            "distribution": name,
            "installed": present,
            "version": version,
            "required_specification": specification,
            "compatible": compatible,
        },
        capability=capability,
    )


def _workflow_check(name: str, relative_path: str) -> CheckResult:
    path = _source_root() / relative_path
    exists = path.is_file()
    return CheckResult(
        id=f"workflow.{name}",
        dimension="synthetic_data_plane",
        requirement="required",
        status="pass" if exists else "fail",
        summary=(
            f"{name.title()} workflow facade is present."
            if exists
            else f"{name.title()} workflow facade is missing."
        ),
        details={"implemented": exists, "module_path": relative_path},
        capability=name,
    )


def _path_check(check_id: str, path_value: str) -> CheckResult:
    path = Path(path_value).expanduser()
    if path.exists():
        is_directory = path.is_dir()
        writable = is_directory and os.access(path, os.W_OK | os.X_OK)
        reason = "existing_directory" if is_directory else "path_is_not_directory"
    else:
        ancestor = path.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        is_directory = False
        writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
        reason = "writable_existing_parent" if writable else "parent_not_writable"

    return CheckResult(
        id=check_id,
        dimension="storage",
        requirement="required",
        status="pass" if writable else "fail",
        summary=(
            f"Configured path {path_value!r} is locally writable."
            if writable
            else f"Configured path {path_value!r} is not usable for local output."
        ),
        details={
            "path": str(path),
            "exists": path.exists(),
            "is_directory": is_directory,
            "writable": writable,
            "reason": reason,
        },
    )


def _decode_signing_key(variable: str) -> bytes | None:
    raw = os.environ.get(variable)
    if not raw:
        return None
    try:
        value = bytes.fromhex(raw)
    except ValueError:
        return None
    return value if len(value) == 32 else None


def _signing_key_check(variable: str, check_id: str) -> CheckResult:
    configured = bool(os.environ.get(variable))
    valid = _decode_signing_key(variable) is not None
    return CheckResult(
        id=check_id,
        dimension="security",
        requirement="required",
        status="pass" if valid else "fail",
        summary=(
            f"{variable} is configured with a valid Ed25519 key length."
            if valid
            else f"{variable} must contain exactly 32 bytes encoded as hexadecimal."
        ),
        details={
            "environment_variable": variable,
            "configured": configured,
            "valid": valid,
        },
    )


def _signing_pair_matches(private_seed: bytes, trusted_public_key: bytes) -> bool:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    derived = (
        Ed25519PrivateKey.from_private_bytes(private_seed)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    return derived == trusted_public_key


def _signing_pair_check() -> CheckResult:
    private_seed = _decode_signing_key("SYNTH_SIGNING_PRIVATE_KEY_HEX")
    public_key = _decode_signing_key("SYNTH_TRUSTED_PUBLIC_KEY_HEX")
    compatible = False
    verification_available = False
    if private_seed is not None and public_key is not None:
        try:
            compatible = _signing_pair_matches(private_seed, public_key)
            verification_available = True
        except (ImportError, TypeError, ValueError):
            compatible = False
    return CheckResult(
        id="security.signing_key_pair",
        dimension="security",
        requirement="required",
        status="pass" if compatible else "fail",
        summary=(
            "Configured signing private key and trusted public key are compatible."
            if compatible
            else "Signing private key and trusted public key are not a verified pair."
        ),
        details={
            "private_key_valid": private_seed is not None,
            "public_key_valid": public_key is not None,
            "verification_available": verification_available,
            "compatible": compatible,
        },
    )


def _capability_dependency_check(
    capability: str, requirements: Mapping[str, str | None]
) -> CheckResult:
    installed = {name: _distribution_version(name) for name in requirements}
    missing = [name for name, version in installed.items() if version is None]
    incompatible = [
        name
        for name, version in installed.items()
        if version is not None and not _version_satisfies(version, requirements[name])
    ]
    available = not missing and not incompatible
    return CheckResult(
        id=f"capability.{capability}.dependencies",
        dimension="optional_capabilities",
        requirement="optional",
        status="pass" if available else "unavailable",
        summary=(
            f"Declared dependency set for {capability} is installed and compatible."
            if available
            else (
                f"{capability} has {len(missing)} missing and "
                f"{len(incompatible)} incompatible optional distribution(s)."
            )
        ),
        details={
            "required_specifications": dict(requirements),
            "installed_versions": {
                name: version
                for name, version in installed.items()
                if version is not None
            },
            "missing_distributions": missing,
            "incompatible_distributions": incompatible,
        },
        capability=capability,
    )


def _nvidia_requirements() -> dict[str, str | None]:
    requirements: dict[str, str | None] = {
        "data-designer": ">=0.9.1",
        "nvidia-nat": None,
        "nemoguardrails": None,
        "nemo-curator": None,
    }
    if sys.version_info[:2] == (3, 12):
        requirements["nemo-retriever"] = None
    return requirements


def _ocr_runtime_check() -> CheckResult:
    tesseract_env = os.environ.get("TESSERACT_CMD")
    configured_tesseract = Path(tesseract_env) if tesseract_env else None
    override_usable = bool(
        configured_tesseract
        and configured_tesseract.is_file()
        and os.access(configured_tesseract, os.X_OK)
    )
    if configured_tesseract is not None:
        tesseract = str(configured_tesseract) if override_usable else None
    else:
        tesseract = shutil.which("tesseract")
    pdftoppm = shutil.which("pdftoppm")
    ready = bool(tesseract and pdftoppm)
    return CheckResult(
        id="capability.ocr.executables",
        dimension="optional_capabilities",
        requirement="optional",
        status="pass" if ready else "unavailable",
        summary=(
            "OCR executables are available locally."
            if ready
            else "OCR requires local tesseract and pdftoppm executables."
        ),
        details={
            "tesseract_available": bool(tesseract),
            "pdftoppm_available": bool(pdftoppm),
            "tesseract_configured_by_environment": bool(tesseract_env),
            "tesseract_override_usable": override_usable if tesseract_env else None,
        },
        capability="ocr",
    )


def _ollama_check() -> CheckResult:
    adapter = _source_root() / "infrastructure/llm/ollama_chat.py"
    return CheckResult(
        id="capability.ollama.runtime",
        dimension="optional_capabilities",
        requirement="optional",
        status="not_checked",
        summary=(
            "Ollama adapter is present; endpoint and model health are intentionally not probed."
            if adapter.is_file()
            else "Ollama adapter is not present."
        ),
        details={
            "adapter_present": adapter.is_file(),
            "network_probe_performed": False,
            "default_endpoint": "http://localhost:11434",
        },
        capability="ollama",
    )


def _planned_checks() -> list[CheckResult]:
    return [
        CheckResult(
            id=check_id,
            dimension="target_architecture",
            requirement="planned",
            status="planned",
            summary=summary,
            details={"implemented": False, "boundary": detail},
        )
        for check_id, summary, detail in PLANNED_CAPABILITIES
    ]


def _is_fatal(
    check: CheckResult, *, strict: bool, required_capabilities: frozenset[str]
) -> bool:
    if check.requirement == "planned":
        return False
    if check.status == "fail":
        return True
    if check.requirement == "optional" and check.status != "pass":
        if check.status == "not_checked":
            return check.capability in required_capabilities
        return strict or check.capability in required_capabilities
    return False


def collect_report(
    *,
    strict: bool = False,
    required_capabilities: Sequence[str] = (),
    output_root: str = "output",
    staging_root: str = ".staging",
) -> ReadinessReport:
    """Collect a readiness report without importing optional packages or using the network."""

    required = frozenset(required_capabilities)
    version, version_source = _project_metadata()
    checks: list[CheckResult] = [
        _python_check(),
        _package_check(version, version_source),
    ]
    checks.extend(
        _dependency_check(
            distribution,
            specification,
            requirement="required",
            dimension="core",
        )
        for distribution, specification in CORE_REQUIREMENTS.items()
    )
    checks.extend(
        (
            _workflow_check("schema", "application/workflows/schema_twin.py"),
            _workflow_check("database", "application/workflows/database_twin.py"),
            _workflow_check("document", "application/workflows/pdf_twin.py"),
            _path_check("storage.output_root", output_root),
            _path_check("storage.staging_root", staging_root),
            _signing_key_check(
                "SYNTH_SIGNING_PRIVATE_KEY_HEX", "security.signing_private_key"
            ),
            _signing_key_check(
                "SYNTH_TRUSTED_PUBLIC_KEY_HEX", "security.trusted_public_key"
            ),
            _signing_pair_check(),
        )
    )
    checks.extend(
        _capability_dependency_check(capability, requirements)
        for capability, requirements in CAPABILITY_REQUIREMENTS.items()
    )
    checks.append(_ocr_runtime_check())
    checks.append(_capability_dependency_check("nvidia", _nvidia_requirements()))
    checks.append(_ollama_check())
    checks.extend(_planned_checks())

    fatal = any(
        _is_fatal(check, strict=strict, required_capabilities=required)
        for check in checks
    )
    degraded = any(
        check.requirement != "planned" and check.status in {"warn", "unavailable"}
        for check in checks
    )
    overall_status = "not_ready" if fatal else ("degraded" if degraded else "ready")
    counts = Counter(check.status for check in checks)

    return ReadinessReport(
        schema_version=SCHEMA_VERSION,
        command="synth-platform-readiness",
        package={"name": PACKAGE_NAME, "version": version},
        runtime={
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        air_gap_safe=True,
        policy={
            "strict": strict,
            "required_capabilities": sorted(required),
            "network_probes": False,
        },
        overall_status=overall_status,
        summary={status: counts.get(status, 0) for status in CHECK_STATUSES},
        checks=tuple(checks),
    )


def render_text(report: ReadinessReport) -> str:
    lines = [
        "Synth Platform production readiness",
        f"overall: {report.overall_status.upper()}",
        "air-gap safe: yes (no network or model probes)",
    ]
    for check in report.checks:
        lines.append(f"[{check.status.upper():11}] {check.id} - {check.summary}")
    counts = ", ".join(f"{key}={value}" for key, value in report.summary.items())
    lines.append(f"summary: {counts}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    version, _ = _project_metadata()
    parser = argparse.ArgumentParser(
        prog="synth-platform-readiness",
        description=(
            "Inspect local production prerequisites without network, service, or model probes."
        ),
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat unavailable locally verifiable optional dependencies and executables "
            "as readiness failures. Service health marked not_checked remains informational."
        ),
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        choices=REQUIRE_CHOICES,
        metavar="CAPABILITY",
        help="Require one optional capability; may be repeated.",
    )
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--staging-root", default=".staging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = collect_report(
            strict=args.strict,
            required_capabilities=args.require,
            output_root=args.output_root,
            staging_root=args.staging_root,
        )
        if args.json_output:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_text(report))
        return 0 if report.overall_status != "not_ready" else 1
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - CLI boundary
        print(
            f"synth-platform-readiness internal error: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
