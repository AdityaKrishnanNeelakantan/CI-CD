from __future__ import annotations

import numpy as np
import pytest

from synth_platform.engine.generation.schema.realism import RealisticTextGenerator

pytestmark = pytest.mark.unit


def test_branch_and_manager_names_do_not_use_product_description_fallback() -> None:
    generator = RealisticTextGenerator(rng=np.random.default_rng(7))

    branch_names = generator.generate("branch_name", "branches", 20)
    manager_names = generator.generate("manager_name", "branches", 20)

    assert all(any(suffix in value for suffix in ("Branch", "Office", "Financial Center", "Service Center", "Operations Center")) for value in branch_names)
    assert all(len(str(value).split()) >= 2 for value in manager_names)
    assert not any("performance" in str(value).lower() or "materials" in str(value).lower() for value in branch_names)
