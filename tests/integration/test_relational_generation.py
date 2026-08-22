"""Generation preserves PK uniqueness and FK integrity across tables."""
from synth_platform import GenerationRequest, SyntheticDataPlatform, TrainingRequest


def test_fk_integrity(banking_db):
    platform = SyntheticDataPlatform.from_settings()
    art = platform.train(TrainingRequest(source=f"sqlite:///{banking_db}", sample_size=5000, seed=3))
    tables = platform.generate(art, GenerationRequest(root_table_rows={"users": 150}, seed=3))
    users, accounts = tables["users"], tables["accounts"]
    assert users["id"].is_unique
    assert accounts["user_id"].isin(set(users["id"])).all()  # zero orphans
