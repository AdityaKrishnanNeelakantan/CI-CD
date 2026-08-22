"""Bundle round-trips through the .synthpkg store."""
from synth_platform.infrastructure.artifacts.store import SynthpkgStore


def test_synthpkg_roundtrip(banking_db, platform, tmp_path):
    art = platform.train.__self__.train.__func__(platform, __import__(
        "synth_platform").TrainingRequest(source=f"sqlite:///{banking_db}", sample_size=5000, seed=42))
    store = SynthpkgStore()
    pkg = tmp_path / "m.synthpkg"
    store.write(art, pkg)
    loaded = store.read(pkg, verify=True)
    assert loaded.manifest.artifact_id == art.manifest.artifact_id
    assert set(loaded.schema_.tables) == {"users", "accounts"}
