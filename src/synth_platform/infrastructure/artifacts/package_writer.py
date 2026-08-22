"""Write signed .synthpkg bundles with JSON metadata and safetensors weights."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save as save_safetensors

from synth_platform.infrastructure.artifacts.signing import sign
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.artifacts.checksums import render_checksum_file, sha256_bytes


def _member_bytes(artifact: SynthArtifact, attr: str) -> bytes:
    return getattr(artifact, attr).model_dump_json().encode("utf-8")


def _tensor_payload(artifact: SynthArtifact) -> bytes:
    tensors: dict[str, np.ndarray] = {"format_version": np.asarray([1], dtype=np.int64)}
    for table, tprof in artifact.data_profile.tables.items():
        for column, cprof in tprof.columns.items():
            prefix = f"{table}__{column}".replace("/", "_")
            if cprof.numeric is not None:
                tensors[f"{prefix}__quantiles"] = np.asarray(cprof.numeric.quantiles, dtype=np.float64)
                tensors[f"{prefix}__range"] = np.asarray(
                    [cprof.numeric.minimum, cprof.numeric.maximum, cprof.numeric.mean], dtype=np.float64)
            if cprof.categorical is not None:
                tensors[f"{prefix}__probabilities"] = np.asarray(
                    cprof.categorical.probabilities, dtype=np.float64)
    for edge, model in artifact.relational_plan.cardinality_models.items():
        prefix = edge.replace("/", "_").replace("->", "__to__")
        tensors[f"cardinality__{prefix}__values"] = np.asarray(model.values, dtype=np.int64)
        tensors[f"cardinality__{prefix}__probabilities"] = np.asarray(
            model.probabilities, dtype=np.float64)
    return save_safetensors(tensors)


def write_package(artifact: SynthArtifact, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    members: dict[str, bytes] = {
        fname: _member_bytes(artifact, attr)
        for fname, attr in SynthArtifact.MEMBERS.items()
    }
    if artifact.document_template is not None:
        members["document_template.json"] = artifact.document_template.model_dump_json().encode()
    members["weights/model.safetensors"] = _tensor_payload(artifact)

    artifact.manifest.members = sorted([*members, "manifest.json", "checksums.sha256", "signature.ed25519"])
    members["manifest.json"] = artifact.manifest.model_dump_json().encode("utf-8")
    checks = {name: sha256_bytes(blob) for name, blob in members.items()}
    checksum_bytes = render_checksum_file(checks).encode("utf-8")
    members["checksums.sha256"] = checksum_bytes
    members["signature.ed25519"] = sign(hashlib.sha256(checksum_bytes).digest())

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    return path
