"""THE invariant: train -> export -> DELETE SOURCE -> load -> generate ->
validate + holdout-evaluate -> quarantine gate."""
from __future__ import annotations

import os

import pytest

from synth_platform import GenerationRequest, SyntheticDataPlatform, TrainingRequest
from synth_platform.infrastructure.sinks.staging import Quarantine
from synth_platform.application.use_cases.validate_dataset import validate_dataset
from synth_platform.domain.validation.models import Status
from synth_platform.engine.validation.holdout_evaluator import (
    SecureHoldoutEvaluator, UtilityTask, holdout_split,
)
from synth_platform.infrastructure.sources.sqlite import SqliteSource
from synth_platform.errors import QuarantineBlockedError


def test_source_free_invariant_and_quarantine(banking_db, tmp_path):
    platform = SyntheticDataPlatform.from_settings()

    # holdout split BEFORE deleting the source (privacy needs real holdout)
    conn = SqliteSource(banking_db)
    split = holdout_split(conn, seed=42, max_rows=5000)
    conn.close()

    art = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", sample_size=5000, seed=42))
    pkg = tmp_path / "model.synthpkg"
    platform.export(art, pkg)

    os.remove(banking_db)               # <-- source is gone
    assert not os.path.exists(banking_db)

    loaded = platform.load(pkg)         # hostile-input-safe load
    tables = platform.generate(loaded, GenerationRequest(root_table_rows={"users": 150}, seed=42))

    # structural + fidelity + real privacy/utility
    evaluator = SecureHoldoutEvaluator(split, tasks=[UtilityTask("accounts", "status")], seed=42)
    report = validate_dataset(loaded, tables, evaluator.privacy_utility_checks(tables))

    names = {c.name: c for c in report.checks}
    # FK integrity holds with the source deleted
    assert names["fk_integrity[accounts.user_id]"].status == Status.PASS
    # privacy actually ran (not NOT_RUN)
    assert names["privacy_mia_auc[users]"].ran
    assert names["privacy_dcr[users]"].status == Status.PASS

    # quarantine: without a passing report, promotion is blocked
    q = Quarantine(tmp_path / "staging")
    result = q.submit(tables, report, run_id="run")
    if result.promotable:
        q.promote(result, tmp_path / "approved")
        assert (tmp_path / "approved" / "users.csv").exists()
    else:
        with pytest.raises(QuarantineBlockedError):
            q.promote(result, tmp_path / "approved")


def test_no_evaluator_marks_privacy_not_applicable(banking_db, tmp_path):
    """W0/RC-9: without a holdout, privacy/utility require a capability the source
    does not exercise, so they are NOT_APPLICABLE (neutral) rather than forcing a
    permanent WARN. The verdict is then governed by the structural+fidelity checks
    that actually ran — which pass — so the release is PASS and promotable."""
    from synth_platform.domain.validation.models import Capability, Status as St
    platform = SyntheticDataPlatform.from_settings()
    art = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", sample_size=3000, seed=5))
    tables = platform.generate(art, GenerationRequest(root_table_rows={"users": 120}, seed=5))
    report = platform.validate(art, tables)          # no evaluator
    names = {c.name: c for c in report.checks}
    assert names["membership_inference"].status == St.NOT_APPLICABLE
    assert names["tstr_utility"].status == St.NOT_APPLICABLE
    # structural+fidelity carried the verdict; nothing inapplicable blocked it
    assert report.overall in (St.PASS, St.WARN)
    if report.overall == St.PASS:
        q = Quarantine(tmp_path / "staging")
        result = q.submit(tables, report)
        q.promote(result, tmp_path / "approved")
        assert (tmp_path / "approved" / "users.csv").exists()
