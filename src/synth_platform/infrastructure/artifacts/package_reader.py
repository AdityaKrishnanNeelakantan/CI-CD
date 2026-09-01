"""Read, verify, and reconstruct a signed source-free .synthpkg."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from safetensors.numpy import load as load_safetensors

from synth_platform.infrastructure.artifacts.signing import verify as verify_signature
from synth_platform.infrastructure.artifacts.verifier import (
    check_member_name, check_member_size, reject_pickle_bytes,
)
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.artifacts.checksums import parse_checksum_file, sha256_bytes
from synth_platform.domain.artifacts.manifest import ArtifactManifest
from synth_platform.domain.artifacts.versions import is_supported
from synth_platform.domain.constraints.models import ConstraintSet
from synth_platform.domain.contracts.models import CanonicalContract
from synth_platform.domain.documents.models import DocumentTemplateIR
from synth_platform.domain.entities.graph import EntityGraph
from synth_platform.domain.planning.models import LearningPlan
from synth_platform.domain.privacy.models import PrivacyPolicy
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.domain.relational.models import RelationalPlan
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.domain.semantics.dependencies import DependencyProfile
from synth_platform.errors import ArtifactCompatibilityError, ArtifactIntegrityError

_MODELS = {
    "canonical_contract.json": CanonicalContract,
    "entity_graph.json": EntityGraph,
    "schema.json": DatabaseSchema,
    "relational_plan.json": RelationalPlan,
    "semantic_profile.json": DataProfile,
    "dependency_profile.json": DependencyProfile,
    "learning_plan.json": LearningPlan,
    "privacy_policy.json": PrivacyPolicy,
    "constraints.json": ConstraintSet,
}


def read_package(path: str | Path, verify: bool = True) -> SynthArtifact:
    with zipfile.ZipFile(Path(path), "r") as archive:
        raw: dict[str, bytes] = {}
        for info in archive.infolist():
            check_member_name(info.filename)
            check_member_size(info.filename, info.file_size)
            blob = archive.read(info.filename)
            reject_pickle_bytes(blob)
            raw[info.filename] = blob

    required = {"manifest.json", "checksums.sha256", "signature.ed25519",
                "weights/model.safetensors"}
    missing = required - set(raw)
    if missing:
        raise ArtifactIntegrityError(f"missing required artifact members: {sorted(missing)}")

    if verify:
        declared = parse_checksum_file(raw["checksums.sha256"].decode("utf-8"))
        for name, blob in raw.items():
            if name in {"checksums.sha256", "signature.ed25519"}:
                continue
            if declared.get(name) != sha256_bytes(blob):
                raise ArtifactIntegrityError(f"checksum mismatch: {name}")
        try:
            verify_signature(raw["signature.ed25519"],
                             hashlib.sha256(raw["checksums.sha256"]).digest())
        except InvalidSignature as exc:
            raise ArtifactIntegrityError("artifact signature verification failed") from exc
        try:
            load_safetensors(raw["weights/model.safetensors"])
        except Exception as exc:
            raise ArtifactIntegrityError("invalid safetensors payload") from exc

    manifest = ArtifactManifest.model_validate_json(raw["manifest.json"])
    if not is_supported(manifest.format_version):
        raise ArtifactCompatibilityError(f"unsupported format {manifest.format_version!r}")
    parts = {}
    for fname, attr in SynthArtifact.MEMBERS.items():
        if fname not in raw:
            if fname in {"canonical_contract.json", "entity_graph.json", "dependency_profile.json"}:
                parts[attr] = None
                continue
            raise ArtifactIntegrityError(f"missing member: {fname}")
        parts[attr] = _MODELS[fname].model_validate_json(raw[fname])
    template = (DocumentTemplateIR.model_validate_json(raw["document_template.json"])
                if "document_template.json" in raw else None)
    return SynthArtifact(manifest=manifest, document_template=template, **parts)
