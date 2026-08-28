import numpy as np
import pandas as pd

from patbot.fast_risk import (
    _draft_day_status_probability,
    _history_components,
    _is_serious_sleeper_status,
    _norm_id,
)


def test_norm_id_removes_csv_float_suffix():
    assert _norm_id(1234.0) == "1234"
    assert _norm_id("5678.0") == "5678"
    assert _norm_id("abc") == "abc"


def test_norm_id_handles_missing_values():
    assert _norm_id(None) == ""
    assert _norm_id(np.nan) == ""


def test_history_components_reuses_cached_slow_layer():
    row = pd.Series(
        {
            "history_missed_rate": 0.20,
            "history_signal_scale": 0.65,
            "age_tail_bonus": 0.03,
        }
    )
    missed, scale, age = _history_components(row)
    assert missed == 0.20
    assert scale == 0.65
    assert age == 0.03


def test_history_components_defaults_are_safe():
    missed, scale, age = _history_components(pd.Series(dtype=object))
    assert missed == 0.0
    assert scale == 1.0
    assert age == 0.0


def test_uncorroborated_questionable_is_ignored_on_draft_day():
    status, probability, source = _draft_day_status_probability(None, "Questionable")
    assert status == "Questionable"
    assert probability == 1.0
    assert source == "sleeper_ignored"


def test_uncorroborated_probable_is_ignored_on_draft_day():
    status, probability, source = _draft_day_status_probability(None, "Probable")
    assert status == "Probable"
    assert probability == 1.0
    assert source == "sleeper_ignored"


def test_hard_sleeper_status_still_matters_without_fantasypros():
    status, probability, source = _draft_day_status_probability(None, "PUP")
    assert status == "PUP"
    assert probability == 0.20
    assert source == "sleeper_hard"
    assert _is_serious_sleeper_status(status)


def test_fantasypros_injury_corroboration_takes_priority():
    item = {"status": "Questionable", "probability_of_playing": 0.62}
    status, probability, source = _draft_day_status_probability(item, "Questionable")
    assert status == "Questionable"
    assert probability == 0.62
    assert source == "fantasypros"
