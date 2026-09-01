"""Distribution rule pass-rate metric."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def compute_rule_pass_rate(rule_evidence: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute pass rate from distribution rule evaluation evidence."""
    if not rule_evidence:
        return {
            "metric": "rule_pass_rate",
            "score": 1.0,
            "target": 1.0,
            "passed": True,
            "passed_rules": 0,
            "total_rules": 0,
            "rules": [],
        }

    passed_rules = 0
    rows: List[Dict[str, Any]] = []
    for item in rule_evidence:
        passed = bool(item.get("passed", item.get("status") == "pass"))
        if passed:
            passed_rules += 1
        rows.append(
            {
                "rule": item.get("rule") or item.get("description") or item.get("name"),
                "passed": passed,
                "target_percent": item.get("target_percent"),
                "actual_percent": item.get("actual_percent"),
                "target_rows": item.get("target_rows"),
                "actual_rows": item.get("actual_rows"),
            }
        )
    total = len(rule_evidence)
    score = passed_rules / total if total else 1.0
    return {
        "metric": "rule_pass_rate",
        "score": round(score, 6),
        "target": 1.0,
        "passed": score >= 1.0,
        "passed_rules": passed_rules,
        "total_rules": total,
        "rules": rows,
    }
