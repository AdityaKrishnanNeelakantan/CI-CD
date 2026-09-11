"""Validation-gated transfer service for user-facing downloads."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_platform.domain.validation.models import Status, ValidationReport
from synth_platform.domain.validation.release_gate import decide
from synth_platform.engine.validation.schema.audit import AuditLogger, get_audit_logger
from synth_platform.errors import TransferBlockedError


@dataclass(frozen=True)
class TransferApproval:
    workflow: str
    output_id: str
    validation_status: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransferBytes:
    data: bytes
    approval: TransferApproval


@dataclass(frozen=True)
class TransferFile:
    path: Path
    approval: TransferApproval


class TransferService:
    """Single gate for handing generated artifacts to users.

    The service intentionally logs both allowed and blocked attempts. It never
    infers success from a missing or truthy-looking report; accepted report
    shapes must carry an explicit PASS signal.
    """

    def __init__(self, audit_logger: AuditLogger | None = None, transfer_recorder: Any | None = None):
        self.audit_logger = audit_logger or get_audit_logger()
        self.transfer_recorder = transfer_recorder

    def downloadable_bytes(
        self,
        *,
        workflow: str,
        output_id: str,
        validation_report: ValidationReport | dict[str, Any] | None,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> TransferBytes:
        approval = self._approve_or_raise(
            workflow=workflow,
            output_id=output_id,
            validation_report=validation_report,
            metadata={**(metadata or {}), "payload_type": "bytes", "payload_size": len(data)},
        )
        return TransferBytes(data=data, approval=approval)

    def downloadable_file(
        self,
        *,
        workflow: str,
        output_id: str,
        validation_report: ValidationReport | dict[str, Any] | None,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> TransferFile:
        file_path = Path(path)
        if not file_path.is_file():
            metadata = {
                **(metadata or {}),
                "payload_type": "file",
                "file_path": str(file_path),
                "payload_size": None,
            }
            self.audit_logger.log_transfer_attempt(
                workflow=workflow,
                output_id=output_id,
                status="blocked",
                validation_status="MISSING_OUTPUT",
                reason=f"output file not found: {file_path}",
                metadata=metadata,
            )
            self._record_transfer_attempt(
                workflow=workflow, output_id=output_id, allowed=False,
                validation_status="MISSING_OUTPUT", reason=f"output file not found: {file_path}",
                metadata=metadata,
            )
            raise TransferBlockedError(f"transfer blocked: output file not found: {file_path}")
        approval = self._approve_or_raise(
            workflow=workflow,
            output_id=output_id,
            validation_report=validation_report,
            metadata={
                **(metadata or {}),
                "payload_type": "file",
                "file_path": str(file_path),
                "payload_size": file_path.stat().st_size if file_path.is_file() else None,
            },
        )
        return TransferFile(path=file_path, approval=approval)

    def _approve_or_raise(
        self,
        *,
        workflow: str,
        output_id: str,
        validation_report: ValidationReport | dict[str, Any] | None,
        metadata: dict[str, Any],
    ) -> TransferApproval:
        allowed, validation_status, reason = self._validation_verdict(workflow, validation_report)
        status = "success" if allowed else "blocked"
        self.audit_logger.log_transfer_attempt(
            workflow=workflow,
            output_id=output_id,
            status=status,
            validation_status=validation_status,
            reason=reason,
            metadata=metadata,
        )
        self._record_transfer_attempt(
            workflow=workflow, output_id=output_id, allowed=allowed,
            validation_status=validation_status, reason=reason, metadata=metadata,
        )
        if not allowed:
            raise TransferBlockedError(f"transfer blocked: {reason}")
        return TransferApproval(
            workflow=workflow,
            output_id=output_id,
            validation_status=validation_status,
            reason=reason,
            metadata=metadata,
        )

    def _validation_verdict(
        self,
        workflow: str,
        validation_report: ValidationReport | dict[str, Any] | None,
    ) -> tuple[bool, str, str]:
        if validation_report is None:
            return False, "MISSING", "validation has not run"

        if isinstance(validation_report, ValidationReport):
            decision = decide(validation_report)
            if decision.verdict == Status.PASS:
                return True, Status.PASS.value, "canonical release gate passed"
            return (
                False,
                decision.verdict.value,
                f"canonical release gate returned {decision.verdict.value}; blocking={decision.blocking}",
            )

        if not isinstance(validation_report, dict):
            return False, "MALFORMED", "validation report is not a recognized object"

        if workflow == "database_twin":
            release = validation_report.get("release")
            if not isinstance(release, dict):
                return False, "MALFORMED", "database QA report is missing release decision"
            decision = release.get("decision")
            hard_checks = validation_report.get("hard_checks_passed")
            if decision == Status.PASS.value and hard_checks is True:
                return True, Status.PASS.value, release.get("reason") or "database QA release passed"
            return (
                False,
                str(decision or "MALFORMED"),
                release.get("reason") or "database QA release did not pass",
            )

        if workflow == "pdf_twin":
            if "hard_checks_passed" not in validation_report:
                return False, "MALFORMED", "PDF validation report is missing hard_checks_passed"
            if validation_report.get("hard_checks_passed") is True:
                return True, Status.PASS.value, "PDF hard validation checks passed"
            return False, Status.FAIL.value, "PDF hard validation checks failed"

        if workflow == "schema_twin":
            passed = validation_report.get("passed")
            export_ready = validation_report.get("export_ready")
            if passed is True and export_ready is True:
                return True, Status.PASS.value, "schema validation and export readiness passed"
            if passed is not True:
                return False, Status.FAIL.value, "schema hard validation did not pass"
            return False, Status.WARN.value, "schema output is not marked export-ready"

        if workflow in {"interaction", "interaction_twin"}:
            if validation_report.get("hard_checks_passed") is True and validation_report.get("export_ready") is True:
                return True, Status.PASS.value, "interaction validation and export readiness passed"
            return False, Status.FAIL.value, "interaction hard validation did not pass"

        return False, "MALFORMED", f"unknown transfer workflow {workflow!r}"

    def _record_transfer_attempt(
        self,
        *,
        workflow: str,
        output_id: str,
        allowed: bool,
        validation_status: str,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        if self.transfer_recorder is None:
            return
        try:
            self.transfer_recorder.record_transfer_attempt(
                workflow=workflow,
                output_id=output_id,
                allowed=allowed,
                validation_status=validation_status,
                reason=reason,
                metadata=metadata,
            )
        except Exception:
            pass
