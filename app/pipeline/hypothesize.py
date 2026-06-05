"""
Hypothesis generator — Phase 4.

LLM receives the profile + data sample and returns list[Hypothesis] as strict JSON.
Pydantic validates every item. Unknown types and missing columns are silently dropped.
"""

import time
import json
import pandas as pd
from app.models import Hypothesis, RunState, Profile
from app.llm import call_llm

_SYSTEM_PROMPT = """You are an exploratory data analyst. Given a dataset profile and sample rows,
propose hypotheses about potentially interesting patterns.

Each hypothesis must have:
  type    — one of: "correlation" | "group_difference" | "association" | "trend" | "anomaly"
  columns — list of column names that exist in the profile
  description — what you expect and why it would be surprising if confirmed

Column type rules (strictly enforced downstream):
  correlation      → both columns must be numeric
  group_difference → columns[0] numeric, columns[1] categorical
  association      → both columns categorical
  trend            → columns[0] numeric, columns[1] datetime
  anomaly          → one or two numeric columns

Aim for 10-15 hypotheses. Favour surprising, non-obvious relationships over trivial ones.
Avoid: id columns, near-duplicate columns, trivial (X correlates with X).

Respond with ONLY:
{"hypotheses": [{"type": "...", "columns": ["col_a", "col_b"], "description": "..."}]}"""


def _profile_summary(profile: Profile) -> str:
    lines = [f"Dataset: {profile.row_count} rows, {profile.col_count} columns\n\nColumns:"]
    for c in profile.columns:
        stat = ""
        if c.kind == "numeric" and c.stats:
            stat = f" [min={c.stats.get('min','?')}, max={c.stats.get('max','?')}, mean={c.stats.get('mean','?')}]"
        elif c.kind == "categorical" and c.stats:
            tops = list(c.stats.get("top_values", {}).keys())[:3]
            stat = f" [top: {', '.join(tops)}]"
        elif c.kind == "datetime" and c.stats:
            stat = f" [{c.stats.get('min','')} → {c.stats.get('max','')}]"
        lines.append(f"  {c.name} ({c.kind}, {c.null_pct*100:.1f}% null){stat}")
    return "\n".join(lines)


def run_hypothesize(
    state: RunState,
    df: pd.DataFrame,
    round_number: int = 1,
    prior_findings: list | None = None,
) -> RunState:
    t0 = time.time()
    state.stage = "hypothesizing"

    summary = _profile_summary(state.profile)
    sample_json = df.sample(min(20, len(df)), random_state=42).to_json(
        orient="records", date_format="iso"
    )

    extra = ""
    if prior_findings:
        confirmed = [f.hypothesis.description for f in prior_findings[:5]]
        extra = (
            f"\n\nRound {round_number} — already confirmed:\n"
            + "\n".join(f"  - {d}" for d in confirmed)
            + "\nNow go deeper or find related patterns."
        )

    result = call_llm(_SYSTEM_PROMPT, f"{summary}\n\nSample rows:\n{sample_json}{extra}")

    valid_cols = set(df.columns)
    for item in result.get("hypotheses", []):
        try:
            cols = [c for c in item.get("columns", []) if c in valid_cols]
            if not cols:
                continue
            state.hypotheses.append(Hypothesis(
                type=item["type"],
                columns=cols,
                description=item.get("description", ""),
                round_number=round_number,
            ))
        except Exception:
            continue

    state.stage_timings[f"hypothesize_r{round_number}"] = round(time.time() - t0, 2)
    return state
