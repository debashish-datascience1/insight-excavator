"""
Unit tests for analyses/library.py.
Every test uses synthetic data with known statistical properties so results are predictable.
"""

import numpy as np
import pandas as pd
import pytest
from app.analyses.library import (
    correlation, group_difference, association, trend, anomaly, missingness
)

rng = np.random.default_rng(42)

REQUIRED_KEYS = ("test", "stat", "effect_size", "effect_size_label", "p_value", "n", "summary_numbers")


# ─── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def strong_corr_df():
    n = 300
    x = rng.normal(50, 10, n)
    y = 3.0 * x + rng.normal(0, 5, n)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def no_corr_df():
    n = 300
    return pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})


@pytest.fixture
def group_diff_df():
    n = 150
    return pd.DataFrame({
        "value": np.concatenate([rng.normal(10, 2, n), rng.normal(25, 2, n)]),
        "group": ["A"] * n + ["B"] * n,
    })


@pytest.fixture
def multigroup_df():
    n = 100
    return pd.DataFrame({
        "value": np.concatenate([rng.normal(5, 1, n), rng.normal(15, 1, n), rng.normal(30, 1, n)]),
        "group": ["A"] * n + ["B"] * n + ["C"] * n,
    })


@pytest.fixture
def assoc_df():
    n = 400
    a = rng.choice(["X", "Y"], n, p=[0.6, 0.4])
    b = np.where(a == "X", rng.choice(["P", "Q"], n, p=[0.85, 0.15]),
                 rng.choice(["P", "Q"], n, p=[0.15, 0.85]))
    return pd.DataFrame({"cat_a": a, "cat_b": b})


@pytest.fixture
def trend_df():
    n = 200
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    values = np.arange(n, dtype=float) * 0.8 + rng.normal(0, 3, n)
    return pd.DataFrame({"value": values, "date": dates})


@pytest.fixture
def anomaly_df():
    n = 400
    normal = rng.normal(50, 5, n)
    spikes = np.array([200.0, 210.0, -80.0, -90.0, 220.0, 230.0])
    return pd.DataFrame({"x": np.concatenate([normal, spikes])})


# ─── Correlation ──────────────────────────────────────────────

class TestCorrelation:
    def test_strong_corr_significant(self, strong_corr_df):
        r = correlation(strong_corr_df, "x", "y")
        assert r is not None
        assert r["p_value"] < 0.05
        assert r["effect_size"] > 0.8
        assert r["n"] == 300

    def test_no_corr_low_effect(self, no_corr_df):
        r = correlation(no_corr_df, "x", "y")
        assert r is not None
        assert r["effect_size"] < 0.3

    def test_required_keys(self, strong_corr_df):
        r = correlation(strong_corr_df, "x", "y")
        for k in REQUIRED_KEYS:
            assert k in r

    def test_too_few_rows_returns_none(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        assert correlation(df, "x", "y") is None

    def test_effect_size_in_range(self, strong_corr_df):
        r = correlation(strong_corr_df, "x", "y")
        assert 0 <= r["effect_size"] <= 1


# ─── Group Difference ─────────────────────────────────────────

class TestGroupDifference:
    def test_large_diff_detected(self, group_diff_df):
        r = group_difference(group_diff_df, "value", "group")
        assert r is not None
        assert r["p_value"] < 0.05
        assert r["effect_size"] > 1.0

    def test_required_keys(self, group_diff_df):
        r = group_difference(group_diff_df, "value", "group")
        for k in REQUIRED_KEYS:
            assert k in r

    def test_multigroup_anova_or_kruskal(self, multigroup_df):
        r = group_difference(multigroup_df, "value", "group")
        assert r is not None
        assert r["p_value"] < 0.05
        assert r["test"] in ("one_way_anova", "kruskal_wallis")

    def test_group_means_present(self, group_diff_df):
        r = group_difference(group_diff_df, "value", "group")
        assert "group_means" in r["summary_numbers"]
        assert set(r["summary_numbers"]["group_means"].keys()) == {"A", "B"}


# ─── Association ──────────────────────────────────────────────

class TestAssociation:
    def test_dependent_cats_detected(self, assoc_df):
        r = association(assoc_df, "cat_a", "cat_b")
        assert r is not None
        assert r["p_value"] < 0.05
        assert r["effect_size"] > 0.1

    def test_cramers_v_in_0_1(self, assoc_df):
        r = association(assoc_df, "cat_a", "cat_b")
        assert 0 <= r["effect_size"] <= 1

    def test_required_keys(self, assoc_df):
        r = association(assoc_df, "cat_a", "cat_b")
        for k in REQUIRED_KEYS:
            assert k in r


# ─── Trend ────────────────────────────────────────────────────

class TestTrend:
    def test_increasing_trend_detected(self, trend_df):
        r = trend(trend_df, "value", "date")
        assert r is not None
        assert r["p_value"] < 0.05
        assert r["summary_numbers"]["direction"] == "increasing"

    def test_r_squared_reasonable(self, trend_df):
        r = trend(trend_df, "value", "date")
        assert r["summary_numbers"]["r_squared"] > 0.5

    def test_required_keys(self, trend_df):
        r = trend(trend_df, "value", "date")
        for k in REQUIRED_KEYS:
            assert k in r


# ─── Anomaly ──────────────────────────────────────────────────

class TestAnomaly:
    def test_spikes_detected(self, anomaly_df):
        r = anomaly(anomaly_df, ["x"])
        assert r is not None
        assert r["summary_numbers"]["anomaly_count"] >= 4

    def test_clean_data_low_anomaly_share(self):
        df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
        r = anomaly(df, ["x"])
        assert r["summary_numbers"]["anomaly_share"] < 0.05

    def test_required_keys(self, anomaly_df):
        r = anomaly(anomaly_df, ["x"])
        for k in REQUIRED_KEYS:
            assert k in r

    def test_fences_present(self, anomaly_df):
        r = anomaly(anomaly_df, ["x"])
        assert "lower_fence" in r["summary_numbers"]
        assert "upper_fence" in r["summary_numbers"]


# ─── Missingness ──────────────────────────────────────────────

class TestMissingness:
    def test_structured_missingness(self):
        n = 300
        x = rng.normal(0, 1, n)
        y = np.where(x > 1.0, np.nan, rng.normal(5, 1, n))
        df = pd.DataFrame({"x": x, "y": y})
        r = missingness(df, "y")
        assert r is not None
        assert r["summary_numbers"]["missing_count"] > 0

    def test_no_missing_returns_none(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert missingness(df, "x") is None

    def test_required_keys(self):
        n = 100
        vals = np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n))
        df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": vals})
        r = missingness(df, "b")
        if r is not None:
            for k in REQUIRED_KEYS:
                assert k in r
