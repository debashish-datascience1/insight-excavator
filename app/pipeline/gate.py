"""
Significance gate — Phase 5.

Keeps findings only if:
  - p_value < SIGNIFICANCE_THRESHOLD
  - effect_size >= per-test minimum
  - n >= MIN_N
  - not trivial (self-correlation, near-perfect duplicate correlation)

Survivors are ranked by surprise_score = effect_size × (1 - p_value).
Deduplication: one finding per (column-set, hypothesis-type) pair — highest surprise wins.
"""

import time
from app.models import Finding, RunState
import app.config as cfg

_MIN_EFFECTS: dict[str, float] = {
    "pearson_correlation":       cfg.MIN_EFFECT_CORRELATION,
    "spearman_correlation":      cfg.MIN_EFFECT_CORRELATION,
    "independent_t_test":        cfg.MIN_EFFECT_COHENS_D,
    "mann_whitney_u":            cfg.MIN_EFFECT_COHENS_D,
    "one_way_anova":             cfg.MIN_EFFECT_ETA_SQUARED,
    "kruskal_wallis":            cfg.MIN_EFFECT_ETA_SQUARED,
    "chi_square":                cfg.MIN_EFFECT_CRAMERS_V,
    "linear_regression_trend":   cfg.MIN_EFFECT_CORRELATION,
    "iqr_anomaly":               cfg.MIN_EFFECT_ANOMALY_SHARE,
    "isolation_forest":          cfg.MIN_EFFECT_ANOMALY_SHARE,
    "missingness_correlation":   cfg.MIN_EFFECT_CORRELATION,
}


def _is_trivial(f: Finding) -> bool:
    cols = f.hypothesis.columns
    if len(cols) >= 2 and cols[0] == cols[1]:
        return True
    if f.test_name in ("pearson_correlation", "spearman_correlation") and f.effect_size > 0.99:
        return True
    return False


def _surprise(f: Finding) -> float:
    return f.effect_size * max(0.0, 1.0 - f.p_value)


def run_gate(state: RunState) -> RunState:
    t0 = time.time()
    state.stage = "gating"

    passed: list[Finding] = []
    for f in state.findings:
        if f.p_value >= cfg.SIGNIFICANCE_THRESHOLD:
            continue
        if f.effect_size < _MIN_EFFECTS.get(f.test_name, 0.05):
            continue
        if f.n < cfg.MIN_N:
            continue
        if _is_trivial(f):
            continue
        f.significant = True
        f.surprise_score = _surprise(f)
        passed.append(f)

    # Deduplicate: keep highest-surprise per (frozen column set, hypothesis type)
    best: dict[tuple, Finding] = {}
    for f in passed:
        key = (frozenset(f.hypothesis.columns), f.hypothesis.type)
        if key not in best or f.surprise_score > best[key].surprise_score:
            best[key] = f

    state.verified_findings = sorted(best.values(), key=lambda x: x.surprise_score, reverse=True)
    state.stage_timings["gate"] = round(time.time() - t0, 2)
    return state
