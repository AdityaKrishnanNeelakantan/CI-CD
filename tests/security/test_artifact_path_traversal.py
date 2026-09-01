import zipfile
import pytest
from synth_platform.infrastructure.artifacts.package_reader import read_package
from synth_platform.errors import UnsafePathError


def test_path_traversal_member_rejected(tmp_path):
    p = tmp_path / "evil.synthpkg"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("../../etc/evil.json", "{}")
        z.writestr("manifest.json", "{}")
        z.writestr("checksums.sha256", "")
    with pytest.raises(UnsafePathError):
        read_package(p)
