from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.rare_category import RARE_CATEGORY_LABEL, replace_rare_categories, suppress_rare_categories

pytestmark = pytest.mark.unit


def test_categories_at_or_above_threshold_are_unchanged():
    frequencies = {"active": 100, "inactive": 50}
    result = suppress_rare_categories(frequencies, minimum_support=20)
    assert result == {"active": 100, "inactive": 50}


def test_categories_below_threshold_are_merged_into_rare_bucket():
    frequencies = {"active": 100, "onetime_special_2024": 1, "another_rare_one": 2}
    result = suppress_rare_categories(frequencies, minimum_support=20)
    assert result == {"active": 100, RARE_CATEGORY_LABEL: 3}


def test_no_rare_categories_means_no_rare_bucket_added():
    frequencies = {"a": 50, "b": 30}
    result = suppress_rare_categories(frequencies, minimum_support=20)
    assert RARE_CATEGORY_LABEL not in result


def test_rare_values_never_survive_as_dict_keys():
    frequencies = {"customer_singleton_id_884231": 1, "common": 1000}
    result = suppress_rare_categories(frequencies, minimum_support=20)
    assert "customer_singleton_id_884231" not in result
    assert result == {"common": 1000, RARE_CATEGORY_LABEL: 1}


def test_replace_rare_categories_masks_rare_rows_in_a_series():
    series = pd.Series(["common"] * 30 + ["rare_one", "rare_two"])
    result = replace_rare_categories(series, minimum_support=20)
    assert (result[:30] == "common").all()
    assert (result[30:] == RARE_CATEGORY_LABEL).all()


def test_replace_rare_categories_leaves_series_unchanged_when_nothing_is_rare():
    series = pd.Series(["a"] * 25 + ["b"] * 25)
    result = replace_rare_categories(series, minimum_support=20)
    assert list(result) == list(series)


def test_replace_rare_categories_does_not_mutate_the_original_series():
    series = pd.Series(["common"] * 25 + ["rare"])
    original = series.copy()
    replace_rare_categories(series, minimum_support=20)
    assert (series == original).all()
