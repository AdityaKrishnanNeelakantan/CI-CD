import pickle
import zipfile
import pytest
from synth_platform.infrastructure.artifacts.package_reader import read_package
from synth_platform.errors import PickleRejectedError


def test_pickle_member_rejected(tmp_path):
    p = tmp_path / "evil.synthpkg"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("weights/model.pkl", pickle.dumps({"x": 1}))
        z.writestr("manifest.json", "{}")
        z.writestr("checksums.sha256", "")
    with pytest.raises(PickleRejectedError):
        read_package(p)
