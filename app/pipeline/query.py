"""
Natural Language Query Engine.

User asks a plain-English question → LLM parses it into a Hypothesis →
statistical verification → verified answer with chart and proof numbers.

This reuses the entire existing pipeline (verify, gate, narrate) — the only new
piece is the NL → Hypothesis parsing step.
"""

import uuid
import pandas as pd
from app.models import Hypothesis, RunState, Profile
from app.llm import call_llm
from app.pipeline.verify import run_verify
from app.pipeline.gate import run_gate
from app.pipeline.narrate import run_narrate, _build_chart


_PARSE_SYSTEM_PROMPT = """You are a statistical analyst. Parse a plain-English question about a dataset
into a structured statistical hypothesis, using only columns that exist in the dataset.

Return JSON:
{
  "type": "correlation" | "group_difference" | "association" | "trend" | "anomaly",
  "columns": ["col1", "col2"],
  "description": "what is being tested"
}

Column type rules (strictly enforced):
  correlation      → both columns must be numeric
  group_difference → columns[0] numeric, columns[1] categorical
  association      → both columns categorical
  trend            → columns[0] numeric, columns[1] datetime
  anomaly          → one or two numeric columns

If the question cannot be answered with the available columns return:
{"error": "short explanation of why"}"""


def _col_summary(profile: Profile) -> str:
    return "\n".join(f"  {c.name} ({c.kind})" for c in profile.columns)


def answer_question(
    question: str,
    df: pd.DataFrame,
    profile: Profile,
) -> dict:
    """
    Parse a natural language question, verify it statistically, return a result dict:

    On success:
      {
        verified: bool,
        answer: str,          # plain-English sentence
        test_name: str,
        effect_size: float,
        effect_size_label: str,
        p_value: float,
        n: int,
        summary_numbers: dict,
        chart_json: str,
        hypothesis: Hypothesis,
      }

    On failure:
      { error: str }
    """
    # Step 1 — Parse the question into a structured hypothesis
    parse_result = call_llm(
        _PARSE_SYSTEM_PROMPT,
        f"Question: {question}\n\nAvailable columns:\n{_col_summary(profile)}",
    )

    if "error" in parse_result:
        return {"error": parse_result["error"]}

    valid_cols = set(df.columns)
    cols = [c for c in parse_result.get("columns", []) if c in valid_cols]
    if not cols:
        return {"error": "Could not identify relevant columns in your dataset for this question."}

    try:
        hyp = Hypothesis(
            type=parse_result["type"],
            columns=cols,
            description=parse_result.get("description", question),
        )
    except Exception as e:
        return {"error": f"Could not build a hypothesis from the question: {e}"}

    # Step 2 — Verify using the existing pipeline
    q_state = RunState(run_id=str(uuid.uuid4()))
    q_state.profile = profile
    q_state.hypotheses = [hyp]

    q_state = run_verify(q_state, df)

    if not q_state.findings:
        return {"error": "Could not compute a statistical test for this question. Check that the columns have enough data."}

    raw_finding = q_state.findings[0]

    q_state = run_gate(q_state)

    # Step 3 — Narrate (even if not significant, narrate the raw result)
    if q_state.verified_findings:
        q_state = run_narrate(q_state, df)
        f = q_state.verified_findings[0]
        return {
            "verified": True,
            "answer": f.insight_text or f.hypothesis.description,
            "test_name": f.test_name,
            "effect_size": f.effect_size,
            "effect_size_label": f.effect_size_label,
            "p_value": f.p_value,
            "n": f.n,
            "summary_numbers": f.summary_numbers,
            "chart_json": f.chart_json,
            "hypothesis": f.hypothesis,
        }
    else:
        # Not statistically significant — still show the numbers honestly
        f = raw_finding
        chart = _build_chart(f, df)
        return {
            "verified": False,
            "answer": (
                f"No significant relationship found between {' and '.join(f.hypothesis.columns)}. "
                f"p = {f.p_value:.4f}, effect size = {f.effect_size:.4f} ({f.effect_size_label}) — "
                f"below the significance threshold."
            ),
            "test_name": f.test_name,
            "effect_size": f.effect_size,
            "effect_size_label": f.effect_size_label,
            "p_value": f.p_value,
            "n": f.n,
            "summary_numbers": f.summary_numbers,
            "chart_json": chart,
            "hypothesis": f.hypothesis,
        }
