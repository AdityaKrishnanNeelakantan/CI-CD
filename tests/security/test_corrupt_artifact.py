import zipfile
import pytest
from synth_platform import SyntheticDataPlatform, TrainingRequest
from synth_platform.infrastructure.artifacts.package_reader import read_package
from synth_platform.errors import ArtifactIntegrityError


def test_checksum_tamper_rejected(banking_db, tmp_path):
    platform = SyntheticDataPlatform.from_settings()
    art = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", sample_size=2000, seed=1))
    pkg = tmp_path / "m.synthpkg"
    platform.export(art, pkg)
    # rewrite schema.json member without fixing checksums
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(pkg, "r") as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}
    data["schema.json"] = b'{"format_version":"1.0.0","source_kind":"tampered","tables":{},"foreign_keys":[],"composite_keys":[]}'
    with zipfile.ZipFile(pkg, "w") as zout:
        for n, b in data.items():
            zout.writestr(n, b)
    with pytest.raises(ArtifactIntegrityError):
        read_package(pkg, verify=True)
