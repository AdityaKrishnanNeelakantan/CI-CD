"""Canonical cross-modality contract models."""
from synth_platform.domain.contracts.models import (
    CanonicalContract,
    CanonicalEntity,
    CanonicalEvent,
    CanonicalField,
    CanonicalProvenance,
    CanonicalRelationship,
    DependencyEdge,
    SourceDescriptor,
)
from synth_platform.domain.contracts.serialization import (
    canonical_fingerprint,
    contract_from_json,
    contract_to_dict,
    contract_to_json,
)
from synth_platform.domain.contracts.dataset_contract_adapter import dataset_contract_to_canonical_contract
from synth_platform.domain.contracts.dependency_adapter import dependency_profile_to_contract_edges
from synth_platform.domain.contracts.schema_adapter import database_schema_to_contract
from synth_platform.domain.contracts.versioning import (
    CANONICAL_CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    assert_supported_contract_version,
    is_supported_contract_version,
)

__all__ = [
    "CANONICAL_CONTRACT_VERSION",
    "SUPPORTED_CONTRACT_VERSIONS",
    "CanonicalContract",
    "CanonicalEntity",
    "CanonicalEvent",
    "CanonicalField",
    "CanonicalProvenance",
    "CanonicalRelationship",
    "DependencyEdge",
    "SourceDescriptor",
    "assert_supported_contract_version",
    "canonical_fingerprint",
    "contract_from_json",
    "contract_to_dict",
    "contract_to_json",
    "database_schema_to_contract",
    "dataset_contract_to_canonical_contract",
    "dependency_profile_to_contract_edges",
    "is_supported_contract_version",
]
