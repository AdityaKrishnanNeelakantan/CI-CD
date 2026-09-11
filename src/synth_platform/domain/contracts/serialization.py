"""Serialization, canonical ordering, and fingerprinting for contracts."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from synth_platform.domain.contracts.models import CanonicalContract
from synth_platform.domain.contracts.versioning import assert_supported_contract_version


def _sorted_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        if all(isinstance(item, dict) and _sort_key(item) for item in value):
            return [_sorted_payload(item) for item in sorted(value, key=_sort_key)]
        return [_sorted_payload(item) for item in value]
    return value


def _sort_key(item: dict[str, Any]) -> str:
    for key in (
        "field_id",
        "entity_id",
        "relationship_id",
        "dependency_id",
        "event_id",
        "name",
    ):
        raw = item.get(key)
        if raw is not None:
            return str(raw)
    return ""


def contract_to_dict(contract: CanonicalContract) -> dict[str, Any]:
    assert_supported_contract_version(contract.contract_version)
    return contract.model_dump(mode="json", exclude_none=True)


def contract_to_json(contract: CanonicalContract) -> str:
    return json.dumps(contract_to_dict(contract), indent=2, sort_keys=True)


def contract_from_json(payload: str) -> CanonicalContract:
    contract = CanonicalContract.model_validate_json(payload)
    assert_supported_contract_version(contract.contract_version)
    return contract


def canonical_fingerprint(contract: CanonicalContract) -> str:
    assert_supported_contract_version(contract.contract_version)
    payload = json.dumps(
        _sorted_payload(contract.model_dump(mode="json", exclude_none=True)),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
