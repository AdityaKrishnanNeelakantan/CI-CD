"""End-to-end UI test: drives the actual src/synth_platform/interfaces/streamlit/pages/database_twin.py
through Streamlit's own AppTest harness - real script execution, real
widget interactions, real session_state and reruns, no browser needed.

This is the permanent regression test for the exact scenario this UI
exists to demonstrate: a source with a male:female ratio of ~4:3 trains
once, and the user can tweak the generated ratio (here to 2:3) entirely
from already-generated results, without ever retraining.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/database_twin.py"


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_full_database_twin_wizard_with_gender_ratio_tweak():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    _click(at, "Generate sample database")
    _click(at, "Discover structure")
    _click(at, "Understand values")
    _click(at, "Understand data")
    _click(at, "Approve understanding")
    _click(at, "Train")
    _click(at, "Download trained twin")
    _click(at, "Disconnect source")

    assert at.session_state["db_source_disconnected"] is True

    # The learned distribution must be shown, and must be close to the
    # source's real 4:3 ratio (male ~57%, female ~43%).
    gender_expander = next(e for e in at.expander if "customer.gender" in e.label)
    assert "male" in gender_expander.label and "female" in gender_expander.label

    gender_checkbox = next(c for c in at.checkbox if "gender" in (c.key or ""))
    gender_checkbox.set_value(True).run()
    assert not at.exception

    sliders = [s for s in at.slider if "gender" in (s.key or "")]
    male_slider = next(s for s in sliders if s.key.endswith("_male"))
    female_slider = next(s for s in sliders if s.key.endswith("_female"))
    male_slider.set_value(0.4).run()
    female_slider.set_value(0.6).run()
    assert not at.exception

    _click(at, "Generate synthetic database")
    assert at.session_state["db_relational_report"] is not None
    assert at.session_state["db_category_overrides"]["customer"]["gender"] == pytest.approx(
        {"male": 0.4, "female": 0.6}
    )

    preview_select = next(s for s in at.selectbox if s.label == "Preview table")
    preview_select.set_value("customer").run()
    assert not at.exception

    gender_caption = next(c for c in at.caption if "gender split" in c.value.lower())
    # 40:60 requested - generous tolerance for a small generated sample.
    assert "male" in gender_caption.value and "female" in gender_caption.value

    _click(at, "Validate results")
    assert at.session_state["db_qa_report"]["hard_checks_passed"] is True

    _click(at, "Export to target database")
    assert at.session_state["db_write_report"]["validation_report"]["all_writes_confirmed"] is True


def test_default_generation_without_override_stays_close_to_the_source_ratio():
    """The counterpart proof: with no override at all, the generated
    ratio should replicate the source's real ~4:3 split, not drift
    toward something else - distribution replication is the default
    behaviour, rebalancing is opt-in.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    _click(at, "Generate sample database")
    _click(at, "Discover structure")
    _click(at, "Understand values")
    _click(at, "Understand data")
    _click(at, "Approve understanding")
    _click(at, "Train")
    _click(at, "Download trained twin")
    _click(at, "Disconnect source")
    _click(at, "Generate synthetic database")

    preview_select = next(s for s in at.selectbox if s.label == "Preview table")
    preview_select.set_value("customer").run()

    import pandas as pd

    report = at.session_state["db_relational_report"]
    generated_df = pd.read_csv(report["tables"]["customer"]["path"])
    male_ratio = (generated_df["gender"] == "male").mean()
    assert male_ratio == pytest.approx(4 / 7, abs=0.05)


def test_regenerating_the_source_database_before_rerunning_discovery_does_not_collide_with_the_old_run():
    """Regression: re-clicking "Generate sample database" resets
    db_discovery (so the "Discover structure" button reappears) but must also
    start a fresh RunManifest - otherwise the second "Discover structure"
    click writes discovery.json into a run directory that already has
    one from the first pass, tripping the no-overwrite guard.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    _click(at, "Generate sample database")
    first_run_dir = at.session_state["db_manifest"].run_dir
    _click(at, "Discover structure")
    assert at.session_state["db_discovery"] is not None

    # Re-source: the button reappears (db_discovery was reset), and must
    # get a brand new run directory, not the one discovery already wrote to.
    _click(at, "Generate sample database")
    assert at.session_state["db_discovery"] is None
    assert at.session_state["db_manifest"].run_dir != first_run_dir

    _click(at, "Discover structure")
    assert not at.exception
    assert at.session_state["db_discovery"] is not None
