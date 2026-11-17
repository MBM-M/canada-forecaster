"""Model and backtest correctness: no look-ahead, no target leakage."""
import numpy as np
import pandas as pd
import pytest

from service.backtest import backtest, evaluable_years, first_evaluable_year
from service.data import TARGETS, ready_at, visible
from service.model import DEFAULT_SPEC, Spec, build_features, feature_names


@pytest.mark.parametrize("target", list(TARGETS))
def test_features_use_only_earlier_years(raw, target):
    """Every feature of a training row is computed from earlier years; the target year is never an input."""
    t = TARGETS[target]
    vis = visible(raw, target, ready_at(target, 2023))
    train, rows = build_features(target, vis, 2024)
    assert set(rows["year"]) == {2024} and train["year"].max() == 2023
    frame = raw[t.frame].set_index([t.group_col, "year"])
    for r in train.sample(min(40, len(train)), random_state=0).itertuples():
        g, y = getattr(r, t.group_col), r.year
        assert r.lag1 == pytest.approx(frame.loc[(g, y - 1), t.value_col])
        assert r.lag2 == pytest.approx(frame.loc[(g, y - 2), t.value_col])
        assert r.growth_lag1 == pytest.approx(r.lag1 / r.lag2 - 1)
        for c in t.exog:
            assert getattr(r, f"{c}_lag1") == pytest.approx(frame.loc[(g, y - 1), c])


@pytest.mark.parametrize("target", list(TARGETS))
def test_no_feature_contains_the_target(raw, target):
    """Regression test for the original `*_yoy` leak: no feature may equal this year's growth of the target."""
    t = TARGETS[target]
    vis = visible(raw, target, ready_at(target, 2024))
    train, _ = build_features(target, vis[vis["year"] < 2024], 2024)
    own_growth = train[t.value_col] / train["lag1"] - 1
    for col in feature_names(t):
        assert not np.allclose(train[col], own_growth), f"{col} equals the target's own-year growth"


@pytest.mark.parametrize("target", list(TARGETS))
def test_backtest_never_sees_the_future(raw, target):
    """Destroy every value from the test year on: the prediction for that year must not change."""
    t = TARGETS[target]
    year = 2022
    clean = backtest(raw, target, DEFAULT_SPEC, [year])
    corrupted = {k: v.copy() for k, v in raw.items()}
    frame = corrupted[t.frame]
    cols = [t.value_col, *t.exog]
    frame.loc[frame["year"] >= year, cols] = 1e9
    broken = backtest(corrupted, target, DEFAULT_SPEC, [year])
    assert np.allclose(clean["pred"], broken["pred"])
    assert not np.allclose(clean["actual"], broken["actual"])  # only the scoring truth changes


def test_visible_follows_publication_lags(raw):
    for target in TARGETS:
        before = visible(raw, target, ready_at(target, 2022) - pd.Timedelta(minutes=1))
        after = visible(raw, target, ready_at(target, 2022))
        assert before["year"].max() == 2021 and after["year"].max() == 2022


def test_evaluable_years_respect_as_of_and_history(raw):
    ys = evaluable_years(raw, "wages", ready_at("wages", 2023), 10)
    assert ys[-1] == 2023 and ys[0] >= first_evaluable_year(raw, "wages")
    assert evaluable_years(raw, "wages", ready_at("wages", 2023) - pd.Timedelta(days=1), 1) == [2022]


def test_spec_roundtrip():
    s = Spec("rf", max_depth=4, min_samples_leaf=3)
    assert Spec.from_dict(s.to_dict()) == s
