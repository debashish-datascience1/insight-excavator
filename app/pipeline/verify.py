"""
Verifier — Phase 5.

Routes each Hypothesis to the matching library function and collects raw results.
The gate (gate.py) decides what survives — nothing is filtered here.
"""

import time
import pandas as pd
from app.models import Hypothesis, Finding, RunState
from app.analyses import library


def _route(hyp: Hypothesis, df: pd.DataFrame) -> dict | None:
    cols = hyp.columns
    try:
        if hyp.type == "correlation" and len(cols) == 2:
            return library.correlation(df, cols[0], cols[1])
        if hyp.type == "group_difference" and len(cols) == 2:
            return library.group_difference(df, cols[0], cols[1])
        if hyp.type == "association" and len(cols) == 2:
            return library.association(df, cols[0], cols[1])
        if hyp.type == "trend" and len(cols) == 2:
            return library.trend(df, cols[0], cols[1])
        if hyp.type == "anomaly" and len(cols) >= 1:
            return library.anomaly(df, cols)
        if hyp.type == "missingness" and len(cols) == 1:
            return library.missingness(df, cols[0])
    except Exception:
        pass
    return None


def run_verify(state: RunState, df: pd.DataFrame) -> RunState:
    t0 = time.time()
    state.stage = "verifying"

    findings: list[Finding] = []
    for hyp in state.hypotheses:
        result = _route(hyp, df)
        if result is None:
            continue
        findings.append(Finding(
            hypothesis=hyp,
            test_name=result["test"],
            stat=result["stat"],
            effect_size=result["effect_size"],
            effect_size_label=result.get("effect_size_label", ""),
            p_value=result["p_value"],
            n=result["n"],
            significant=False,
            summary_numbers=result.get("summary_numbers", {}),
        ))

    state.findings = findings
    state.stage_timings["verify"] = round(time.time() - t0, 2)
    return state
