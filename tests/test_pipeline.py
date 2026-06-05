"""
End-to-end pipeline tests on a small synthetic fixture — no LLM required.
Tests the deterministic stages (profiler, verifier, gate) independently.
"""

import numpy as np
import pandas as pd
import pytest
import uuid
from app.pipeline.ingest import profile_dataframe
from app.pipeline.verify import run_verify
from app.pipeline.gate import run_gate
from app.models import RunState, Hypothesis

rng = np.random.default_rng(42)


@pytest.fixture
def fixture_df():
    n = 300
    group = rng.choice(["premium", "standard"], n, p=[0.4, 0.6])
    spend = np.where(group == "premium", rng.normal(500, 50, n), rng.normal(200, 40, n))
    age = rng.integers(18, 65, n).astype(float)
    score = 0.4 * spend / spend.max() * 100 + rng.normal(0, 8, n)
    return pd.DataFrame({"group": group, "spend": spend, "age": age, "score": score})


# ─── Profiler ─────────────────────────────────────────────────

def test_profiler_row_count(fixture_df):
    profile = profile_dataframe(fixture_df)
    assert profile.row_count == 300


def test_profiler_col_types(fixture_df):
    profile = profile_dataframe(fixture_df)
    assert "group" in profile.categorical_cols
    assert "spend" in profile.numeric_cols
    assert "age" in profile.numeric_cols
    assert "score" in profile.numeric_cols


def test_profiler_stats_present(fixture_df):
    profile = profile_dataframe(fixture_df)
    spend_col = next(c for c in profile.columns if c.name == "spend")
    assert "mean" in spend_col.stats
    assert "min" in spend_col.stats


# ─── Verifier ─────────────────────────────────────────────────

def test_verify_routes_all_hypotheses(fixture_df):
    state = RunState(run_id=str(uuid.uuid4()))
    state.hypotheses = [
        Hypothesis(type="group_difference", columns=["spend", "group"],
                   description="premium vs standard spend"),
        Hypothesis(type="correlation", columns=["spend", "score"],
                   description="spend-score link"),
    ]
    state = run_verify(state, fixture_df)
    assert len(state.findings) == 2
    for f in state.findings:
        assert f.n > 0
        assert f.p_value >= 0


def test_verify_bad_hypothesis_skipped(fixture_df):
    state = RunState(run_id=str(uuid.uuid4()))
    state.hypotheses = [
        Hypothesis(type="correlation", columns=["nonexistent_col", "spend"],
                   description="bad hyp"),
    ]
    state = run_verify(state, fixture_df)
    assert len(state.findings) == 0


# ─── Gate ─────────────────────────────────────────────────────

def test_gate_passes_strong_finding(fixture_df):
    state = RunState(run_id=str(uuid.uuid4()))
    state.hypotheses = [
        Hypothesis(type="group_difference", columns=["spend", "group"],
                   description="premium vs standard spend"),
    ]
    state = run_verify(state, fixture_df)
    state = run_gate(state)
    assert len(state.verified_findings) >= 1
    for f in state.verified_findings:
        assert f.significant
        assert f.p_value < 0.05
        assert f.surprise_score > 0


def test_gate_rejects_random_noise():
    n = 200
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "group": rng.choice(["A", "B"], n),
    })
    state = RunState(run_id=str(uuid.uuid4()))
    state.hypotheses = [
        Hypothesis(type="group_difference", columns=["x", "group"],
                   description="random group diff"),
    ]
    state = run_verify(state, df)
    state = run_gate(state)
    for f in state.verified_findings:
        assert f.p_value < 0.05


def test_gate_deduplicates_same_pair(fixture_df):
    state = RunState(run_id=str(uuid.uuid4()))
    # Same pair twice — gate should keep only one
    state.hypotheses = [
        Hypothesis(type="group_difference", columns=["spend", "group"], description="first"),
        Hypothesis(type="group_difference", columns=["spend", "group"], description="second"),
    ]
    state = run_verify(state, fixture_df)
    state = run_gate(state)
    assert len(state.verified_findings) <= 1
