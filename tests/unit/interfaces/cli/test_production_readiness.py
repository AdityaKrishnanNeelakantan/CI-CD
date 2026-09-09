from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from synth_platform.interfaces.cli import production_readiness as readiness

pytestmark = pytest.mark.unit

_VALID_KEY = "00" * 32
_COMPATIBLE_VERSIONS = {
    "pydantic": "2.10.0",
    "pandas": "2.2.0",
    "numpy": "2.1.0",
    "safetensors": "0.5.0",
    "SQLAlchemy": "2.0.0",
    "cryptography": "44.0.0",
    "pdfplumber": "0.11.0",
    "pypdf": "5.0.0",
    "reportlab": "4.0.0",
    "PyYAML": "6.0.0",
    "Faker": "30.0.0",
    "data-designer": "0.9.1",
    "duckdb": "0.9.0",
    "python-json-logger": "3.0.0",
    "scipy": "1.10.0",
    "simpleeval": "1.0.7",
    "sdv": "1.37.4",
    "copulas": "0.12.0",
    "fpdf2": "2.8.7",
    "langdetect": "1.0.9",
    "streamlit": "1.60.0",
    "pyarrow": "12.0.0",
    "docling": "2.119.0",
    "pdf2image": "1.17.0",
    "Pillow": "10.0.0",
    "pytesseract": "0.3.0",
    "openai": "1.0.0",
    "groq": "0.4.0",
    "nvidia-nat": "1.0.0",
    "nemoguardrails": "1.0.0",
    "nemo-curator": "1.0.0",
    "nemo-retriever": "1.0.0",
}


def _configure_ready_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    monkeypatch.setattr(
        readiness, "_project_metadata", lambda: ("3.0.2", "installed_distribution")
    )
    monkeypatch.setattr(
        readiness,
        "_distribution_version",
        lambda name: _COMPATIBLE_VERSIONS[name],
    )
    monkeypatch.setattr(readiness, "_signing_pair_matches", lambda *_args: True)
    monkeypatch.setattr(readiness.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("SYNTH_SIGNING_PRIVATE_KEY_HEX", _VALID_KEY)
    monkeypatch.setenv("SYNTH_TRUSTED_PUBLIC_KEY_HEX", _VALID_KEY)
    output = tmp_path / "output"
    staging = tmp_path / ".staging"
    output.mkdir()
    staging.mkdir()
    return output, staging


def _check(report: readiness.ReadinessReport, check_id: str) -> readiness.CheckResult:
    return next(check for check in report.checks if check.id == check_id)


def test_collect_report_is_air_gap_safe_and_truthful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)

    report = readiness.collect_report(
        output_root=str(output),
        staging_root=str(staging),
    )

    assert report.air_gap_safe is True
    assert report.policy["network_probes"] is False
    assert report.overall_status == "ready"
    assert _check(report, "workflow.schema").status == "pass"
    assert _check(report, "workflow.database").status == "pass"
    assert _check(report, "workflow.document").status == "pass"
    assert _check(report, "capability.ollama.runtime").status == "not_checked"
    assert _check(report, "planned.mcp_gateway").status == "planned"
    assert _check(report, "planned.live_agent_plane").status == "planned"


def test_missing_core_dependency_is_not_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "_distribution_version",
        lambda name: None if name == "pydantic" else _COMPATIBLE_VERSIONS[name],
    )

    report = readiness.collect_report(
        output_root=str(output),
        staging_root=str(staging),
    )

    assert report.overall_status == "not_ready"
    assert _check(report, "core.dependency.pydantic").status == "fail"


def test_missing_or_invalid_signing_keys_fail_without_leaking_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)
    invalid_secret = "not-a-valid-private-key"
    monkeypatch.setenv("SYNTH_SIGNING_PRIVATE_KEY_HEX", invalid_secret)
    monkeypatch.delenv("SYNTH_TRUSTED_PUBLIC_KEY_HEX")

    report = readiness.collect_report(
        output_root=str(output),
        staging_root=str(staging),
    )
    payload = json.dumps(report.to_dict())

    assert report.overall_status == "not_ready"
    assert invalid_secret not in payload
    assert _check(report, "security.signing_private_key").details == {
        "environment_variable": "SYNTH_SIGNING_PRIVATE_KEY_HEX",
        "configured": True,
        "valid": False,
    }
    assert _check(report, "security.trusted_public_key").details["configured"] is False


def test_optional_capability_is_nonfatal_unless_required_or_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "_distribution_version",
        lambda name: None if name == "data-designer" else _COMPATIBLE_VERSIONS[name],
    )

    ordinary = readiness.collect_report(
        output_root=str(output), staging_root=str(staging)
    )
    required = readiness.collect_report(
        required_capabilities=["schema"],
        output_root=str(output),
        staging_root=str(staging),
    )
    strict = readiness.collect_report(
        strict=True,
        output_root=str(output),
        staging_root=str(staging),
    )

    assert ordinary.overall_status == "degraded"
    assert required.overall_status == "not_ready"
    assert strict.overall_status == "not_ready"


def test_planned_capabilities_never_become_ready_from_installed_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)

    report = readiness.collect_report(
        output_root=str(output), staging_root=str(staging)
    )
    planned = [check for check in report.checks if check.requirement == "planned"]

    assert planned
    assert all(check.status == "planned" for check in planned)
    assert all(check.details["implemented"] is False for check in planned)


def test_path_checks_do_not_create_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_ready_environment(monkeypatch, tmp_path)
    output = tmp_path / "new" / "output"
    staging = tmp_path / "new" / "staging"

    report = readiness.collect_report(
        output_root=str(output), staging_root=str(staging)
    )

    assert _check(report, "storage.output_root").status == "pass"
    assert _check(report, "storage.staging_root").status == "pass"
    assert not output.exists()
    assert not staging.exists()


def test_json_output_is_deterministic_and_contains_no_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)

    exit_code = readiness.main(
        ["--json", "--output-root", str(output), "--staging-root", str(staging)]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["schema_version"] == "1.0"
    assert payload["air_gap_safe"] is True
    assert payload["overall_status"] == "ready"
    assert _VALID_KEY not in captured.out


def test_internal_error_returns_three_and_redacts_exception_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "private-value-must-not-leak"

    def fail(**_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(readiness, "collect_report", fail)
    exit_code = readiness.main([])
    captured = capsys.readouterr()

    assert exit_code == 3
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert secret not in captured.err


def test_invalid_required_capability_uses_argparse_exit_two() -> None:
    with pytest.raises(SystemExit) as exc_info:
        readiness.main(["--require", "not-a-capability"])

    assert exc_info.value.code == 2


def test_package_and_readiness_import_without_eager_optional_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    command = (
        "import sys; import synth_platform; "
        "assert 'pydantic' not in sys.modules; "
        "import synth_platform.interfaces.cli.production_readiness; "
        "assert 'pandas' not in sys.modules; "
        "assert 'streamlit' not in sys.modules; "
        "assert 'openai' not in sys.modules; "
        "assert 'data_designer' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_incompatible_dependency_version_fails_required_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "_distribution_version",
        lambda name: "1.9.0" if name == "pydantic" else _COMPATIBLE_VERSIONS[name],
    )

    report = readiness.collect_report(
        output_root=str(output), staging_root=str(staging)
    )
    check = _check(report, "core.dependency.pydantic")

    assert report.overall_status == "not_ready"
    assert check.status == "fail"
    assert check.details["compatible"] is False
    assert check.details["required_specification"] == ">=2"


def test_unrelated_installed_metadata_is_not_attributed_to_source_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class UnrelatedDistribution:
        version = "9.9.9"

        def locate_file(self, _path: str) -> Path:
            return tmp_path / "unrelated-site-packages"

        def read_text(self, _name: str) -> None:
            return None

    monkeypatch.setattr(
        readiness.metadata,
        "distribution",
        lambda _name: UnrelatedDistribution(),
    )

    version, source = readiness._project_metadata()

    assert version == "3.0.2"
    assert source == "source_pyproject_unrelated_distribution"


def test_mismatched_signing_pair_fails_even_when_both_keys_have_valid_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(readiness, "_signing_pair_matches", lambda *_args: False)

    report = readiness.collect_report(
        output_root=str(output), staging_root=str(staging)
    )
    check = _check(report, "security.signing_key_pair")

    assert report.overall_status == "not_ready"
    assert check.status == "fail"
    assert check.details["compatible"] is False


def test_ollama_uncertainty_is_neutral_unless_explicitly_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)

    strict = readiness.collect_report(
        strict=True,
        output_root=str(output),
        staging_root=str(staging),
    )
    required = readiness.collect_report(
        required_capabilities=["ollama"],
        output_root=str(output),
        staging_root=str(staging),
    )

    assert strict.overall_status == "ready"
    assert required.overall_status == "not_ready"
    assert _check(strict, "capability.ollama.runtime").status == "not_checked"


def test_non_executable_tesseract_override_does_not_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_tesseract = tmp_path / "tesseract"
    fake_tesseract.write_text("not executable")
    fake_tesseract.chmod(0o600)
    monkeypatch.setenv("TESSERACT_CMD", str(fake_tesseract))
    monkeypatch.setattr(
        readiness.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    check = readiness._ocr_runtime_check()

    assert check.status == "unavailable"
    assert check.details["tesseract_available"] is False
    assert str(fake_tesseract) not in json.dumps(check.details)


def test_summary_always_contains_the_fixed_status_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)

    report = readiness.collect_report(
        output_root=str(output), staging_root=str(staging)
    )

    assert tuple(report.summary) == readiness.CHECK_STATUSES
    assert all(
        isinstance(report.summary[status], int) for status in readiness.CHECK_STATUSES
    )


def test_cli_returns_one_for_failed_required_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output, staging = _configure_ready_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "_distribution_version",
        lambda name: None if name == "pydantic" else _COMPATIBLE_VERSIONS[name],
    )

    exit_code = readiness.main(
        ["--json", "--output-root", str(output), "--staging-root", str(staging)]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["overall_status"] == "not_ready"


def test_lazy_public_export_preserves_existing_import_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import synth_platform

    sentinel = object()
    synth_platform.__dict__.pop("SyntheticDataPlatform", None)
    monkeypatch.setattr(
        synth_platform,
        "import_module",
        lambda _name: type("SDK", (), {"SyntheticDataPlatform": sentinel}),
    )

    assert synth_platform.SyntheticDataPlatform is sentinel
    assert "SyntheticDataPlatform" in synth_platform.__all__


def test_real_ed25519_signing_pair_derivation() -> None:
    pytest.importorskip("cryptography")
    private_seed = bytes.fromhex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
    )
    matching_public_key = bytes.fromhex(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    )
    mismatched_public_key = bytes(32)

    assert readiness._signing_pair_matches(private_seed, matching_public_key) is True
    assert readiness._signing_pair_matches(private_seed, mismatched_public_key) is False
