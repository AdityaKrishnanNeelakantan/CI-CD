"""Validation contract: hard checks vs advisory fidelity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass
class ValidationContract:
    """Separates export-blocking hard checks from sampled advisory fidelity."""

    hard_validation_status: ValidationStatus = ValidationStatus.REVIEW
    advisory_fidelity_status: ValidationStatus = ValidationStatus.NOT_AVAILABLE
    privacy_status: ValidationStatus = ValidationStatus.NOT_AVAILABLE
    rule_status: ValidationStatus = ValidationStatus.NOT_AVAILABLE
    artifact_status: ValidationStatus = ValidationStatus.NOT_AVAILABLE
    validation_scope: str = "sampled"  # full_export | chunks | sampled
    sampled_rows: Optional[int] = None
    exact_rows: Optional[int] = None
    failure_reason: Optional[str] = None
    fix_suggestion: Optional[str] = None
    hard_fail_reasons: List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    export_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hard_validation_status"] = self.hard_validation_status.value
        payload["advisory_fidelity_status"] = self.advisory_fidelity_status.value
        payload["privacy_status"] = self.privacy_status.value
        payload["rule_status"] = self.rule_status.value
        payload["artifact_status"] = self.artifact_status.value
        return payload


def _status_from_bool(passed: bool, *, skipped: bool = False) -> ValidationStatus:
    if skipped:
        return ValidationStatus.NOT_AVAILABLE
    return ValidationStatus.PASS if passed else ValidationStatus.FAIL


def _check_result(value: Any, *, default: bool = True) -> bool:
    """Treat explicit None as 'not checked' rather than failure."""
    if value is None:
        return default
    return bool(value)


def build_validation_contract(
    *,
    hard_checks: Optional[Dict[str, Any]] = None,
    advisory_fidelity: Optional[Dict[str, Any]] = None,
    privacy: Optional[Dict[str, Any]] = None,
    rule_evidence: Optional[List[Dict[str, Any]]] = None,
    export_validation: Optional[Dict[str, Any]] = None,
    validation_scope: str = "sampled",
    sampled_rows: Optional[int] = None,
    exact_rows: Optional[int] = None,
    generation_mode: str = "schema_driven",
) -> ValidationContract:
    """Build a validation contract from component check results."""
    contract = ValidationContract(
        validation_scope=validation_scope,
        sampled_rows=sampled_rows,
        exact_rows=exact_rows,
    )
    hard = hard_checks or {}
    export_val = export_validation or {}

    if export_val:
        contract.artifact_status = _status_from_bool(bool(export_val.get("passed", False)))
        if not export_val.get("passed", False):
            contract.hard_fail_reasons.append(
                export_val.get("failure_reason") or "export artifact validation failed"
            )
    else:
        contract.artifact_status = ValidationStatus.NOT_AVAILABLE

    row_ok = _check_result(hard.get("row_count_passed"), default=bool(export_val.get("row_count_passed", True)))
    pk_ok = _check_result(hard.get("pk_passed"), default=bool(export_val.get("pk_passed", True)))
    fk_ok = _check_result(hard.get("fk_passed"), default=bool(export_val.get("fk_passed", True)))

    if export_val.get("passed"):
        hard_passed = True
        pk_ok = export_val.get("pk_passed", True)
        fk_ok = export_val.get("fk_passed", True)
        row_ok = export_val.get("row_count_passed", True)
    elif validation_scope == "full_export" and export_val:
        hard_passed = bool(export_val.get("passed"))
    elif validation_scope in {"full_export", "chunks"}:
        hard_passed = bool(row_ok and pk_ok and fk_ok)
    else:
        in_memory_fk = hard.get("fk_passed")
        if in_memory_fk is False and validation_scope == "sampled":
            contract.review_reasons.append(
                "In-memory sample FK check failed; export-path validation used for hard gate."
            )
            hard_passed = bool(export_val.get("passed")) if export_val else bool(row_ok and pk_ok)
        else:
            hard_passed = bool(
                not hard.get("has_errors", False)
                and row_ok
                and pk_ok
                and (fk_ok if fk_ok is not False else True)
            )

    if hard.get("has_errors"):
        contract.hard_fail_reasons.append("validation report has errors")
    if not row_ok:
        contract.hard_fail_reasons.append("row count mismatch")
    if pk_ok is False and not export_val.get("passed"):
        contract.hard_fail_reasons.append("primary key uniqueness failed")
    if fk_ok is False and validation_scope == "full_export" and not export_val.get("passed"):
        contract.hard_fail_reasons.append("foreign key integrity failed")

    if contract.hard_fail_reasons:
        contract.hard_validation_status = ValidationStatus.FAIL
    elif contract.review_reasons and not hard_passed:
        contract.hard_validation_status = ValidationStatus.REVIEW
    else:
        contract.hard_validation_status = ValidationStatus.PASS if hard_passed else ValidationStatus.FAIL

    if privacy:
        replay = int(privacy.get("total_replay_values", privacy.get("total_replay", -1)))
        if replay < 0:
            contract.privacy_status = ValidationStatus.NOT_AVAILABLE
        elif replay == 0:
            contract.privacy_status = ValidationStatus.PASS
        else:
            contract.privacy_status = ValidationStatus.FAIL
            contract.hard_fail_reasons.append(f"PII replay detected ({replay} values)")
    else:
        contract.privacy_status = ValidationStatus.NOT_AVAILABLE

    if rule_evidence:
        total = len(rule_evidence)
        passed = sum(1 for item in rule_evidence if item.get("passed"))
        weighted = any(item.get("rule_type") == "weighted" for item in rule_evidence)
        if total == 0:
            contract.rule_status = ValidationStatus.NOT_AVAILABLE
        elif passed == total:
            contract.rule_status = ValidationStatus.PASS
        elif weighted and passed >= total * 0.8:
            contract.rule_status = ValidationStatus.REVIEW
            contract.review_reasons.append("Some weighted rules approximate target multiplier")
        else:
            contract.rule_status = ValidationStatus.FAIL
            contract.hard_fail_reasons.append(f"distribution rules: {passed}/{total} passed")
    else:
        contract.rule_status = ValidationStatus.NOT_AVAILABLE

    if advisory_fidelity is None:
        contract.advisory_fidelity_status = ValidationStatus.NOT_AVAILABLE
    elif validation_scope == "sampled":
        passed = advisory_fidelity.get("passed")
        if passed is True:
            contract.advisory_fidelity_status = ValidationStatus.PASS
        elif passed is False:
            contract.advisory_fidelity_status = ValidationStatus.REVIEW
            contract.review_reasons.append(
                advisory_fidelity.get("note")
                or "Sampled advisory fidelity, not exact full-data validation."
            )
        else:
            contract.advisory_fidelity_status = ValidationStatus.NOT_AVAILABLE
    else:
        contract.advisory_fidelity_status = _status_from_bool(
            bool(advisory_fidelity.get("passed", False)),
            skipped=advisory_fidelity.get("skipped", False),
        )

    contract.export_ready = (
        contract.hard_validation_status == ValidationStatus.PASS
        and contract.privacy_status in {ValidationStatus.PASS, ValidationStatus.NOT_AVAILABLE}
        and contract.rule_status in {ValidationStatus.PASS, ValidationStatus.REVIEW, ValidationStatus.NOT_AVAILABLE}
        and contract.artifact_status in {ValidationStatus.PASS, ValidationStatus.NOT_AVAILABLE}
        and not contract.hard_fail_reasons
    )

    if contract.hard_fail_reasons:
        contract.failure_reason = "; ".join(contract.hard_fail_reasons[:3])
        contract.fix_suggestion = contract.fix_suggestion or _default_fix_suggestion(generation_mode)

    return contract


def _default_fix_suggestion(mode: str) -> str:
    if mode == "source_driven":
        return "Review export-path validation, identifier uniqueness, and privacy replay before exporting."
    return "Review export-path FK/PK validation and final rule evidence before exporting."
